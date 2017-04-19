#include "IndexIVFScalarQuantizer.h"

#include <cstdio>
#include <omp.h>

#include <immintrin.h>

#include "utils.h"

#include "FaissAssert.h"

namespace faiss {




/*******************************************************************
 * IndexIVFScalarQuantizer implementation
 ********************************************************************/

IndexIVFScalarQuantizer::IndexIVFScalarQuantizer
          (Index *quantizer, size_t d, size_t nlist,
           MetricType metric, QuantizerType qtype):
    IndexIVF (quantizer, d, nlist, metric), qtype (qtype)
{
    switch (qtype) {
    case QT_8bit: case QT_8bit_uniform:
        code_size = d;
        break;
    case QT_4bit: case QT_4bit_uniform:
        code_size = (d + 1) / 2;
        break;
    }
    codes.resize(nlist);
}

namespace {

typedef Index::idx_t idx_t;
typedef IndexIVFScalarQuantizer::QuantizerType QuantizerType;


/*******************************************************************
 * Codec: takes value in [0, 1], outputs codes
 */

struct Codec8bit {

    static void encode_component (float x, uint8_t *code, int i) {
        code[i] = (int)(255 * x);
    }

    static float decode_component (const uint8_t *code, int i) {
        return (code[i] + 0.5f) / 255.0f;
    }

    static __m256 decode_8_components (const uint8_t *code, int i) {
        uint64_t c8 = *(uint64_t*)(code + i);
        __m128i c4lo = _mm_cvtepu8_epi32 (_mm_set1_epi32(c8));
        __m128i c4hi = _mm_cvtepu8_epi32 (_mm_set1_epi32(c8 >> 32));
        // __m256i i8 = _mm256_set_m128i(c4lo, c4hi);
        __m256i i8 = _mm256_castsi128_si256 (c4lo);
        i8 = _mm256_insertf128_si256 (i8, c4hi, 1);
        __m256 f8 = _mm256_cvtepi32_ps (i8);
        __m256 half = _mm256_set1_ps (0.5f);
        f8 += half;
        __m256 one_255 = _mm256_set1_ps (1.f / 255.f);
        return f8 * one_255;
    }

};


struct Codec4bit {

    static void encode_component (float x, uint8_t *code, int i) {
        code [i / 2] |= (int)(x * 15.0) << ((i & 1) << 2);
    }

    static float decode_component (const uint8_t *code, int i) {
        return (((code[i / 2] >> ((i & 1) << 2)) & 0xf) + 0.5f) / 15.0f;
    }

};


/*******************************************************************
 * Similarity: gets vector components and computes a similarity wrt. a
 * query vector stored in the object
 */

struct SimilarityL2 {
    const float *y, *yi;
    float accu;

    SimilarityL2 (const float * y): y(y) {}

    void begin () {
        accu = 0;
        yi = y;
    }

    void add_component (float x) {
        float tmp = *yi++ - x;
        accu += tmp * tmp;
    }

    float result () {
        return accu;
    }

};

struct SimilarityIP {
    const float *y, *yi;
    const float accu0;
    float accu;

    SimilarityIP (const float * y, float accu0):
        y (y), accu0 (accu0) {}

    void begin () {
        accu = accu0;
        yi = y;
    }

    void add_component (float x) {
        accu +=  *yi++ * x;
    }

    float result () {
        return accu;
    }

};


template<class Quantizer, class Similarity>
float compute_distance(const Quantizer & quant, Similarity & sim, const uint8_t *code)
{
    sim.begin();
    for (size_t i = 0; i < quant.d; i++) {
        float xi = quant.reconstruct_component (code, i);
        sim.add_component (xi);
    }
    return sim.result();
}



/*******************************************************************
 * Quantizer: normalizes scalar vector components, then passes them through a codec
 */



struct ScalarQuantizer {
    virtual void encode_vector(const float *x, uint8_t *code) const = 0;
    virtual void decode_vector(const uint8_t *code, float *x) const = 0;

    virtual float compute_distance_L2 (SimilarityL2 &sim, const uint8_t * codes) const = 0;
    virtual float compute_distance_IP (SimilarityIP &sim, const uint8_t * codes) const = 0;

    virtual ~ScalarQuantizer() {}
};



void train_Uniform(idx_t n, int d, const float *x, std::vector<float> & trained)
{
    trained.resize (2);
    float & vmin = trained[0];
    float & vmax = trained[1];
    vmin = HUGE_VAL; vmax = -HUGE_VAL;
    for (size_t i = 0; i < n * d; i++) {
        if (x[i] < vmin) vmin = x[i];
        if (x[i] > vmax) vmax = x[i];
    }
}


template<class Codec>
struct QuantizerUniform: ScalarQuantizer {
    const size_t d;
    const float vmin, vmax;
    QuantizerUniform(size_t d, const std::vector<float> &trained):
        d(d), vmin(trained[0]), vmax(trained[1]) {}


    virtual void encode_vector(const float *x, uint8_t *code) const
    {
        for (size_t i = 0; i < d; i++) {
            float xi = (x[i] - vmin) / (vmax - vmin);
            if (xi < 0) xi = 0;
            if (xi > 1.0) xi = 1.0;
            Codec::encode_component (xi, code, i);
        }
    }

    virtual void decode_vector(const uint8_t *code, float *x) const
    {
        for (size_t i = 0; i < d; i++) {
            float xi = Codec::decode_component (code, i);
            x[i] = vmin + xi * (vmax - vmin);
        }
    }

    float reconstruct_component (const uint8_t * code, int i) const
    {
        float xi = Codec::decode_component (code, i);
        return vmin + xi * (vmax - vmin);
    }

    virtual float compute_distance_L2 (SimilarityL2 &sim, const uint8_t * codes) const
    { return compute_distance(*this, sim, codes); }

    virtual float compute_distance_IP (SimilarityIP &sim, const uint8_t * codes) const
    { return compute_distance(*this, sim, codes); }
};


void train_NonUniform(idx_t n, int d, const float *x, std::vector<float> & trained)
{
    trained.resize (2 * d);
    float * vmin = trained.data();
    float * vmax = trained.data() + d;
    memcpy (vmin, x, sizeof(*x) * d);
    memcpy (vmax, x, sizeof(*x) * d);
    for (size_t i = 1; i < n; i++) {
        const float *xi = x + i * d;
            for (size_t j = 0; j < d; j++) {
                if (xi[j] < vmin[j]) vmin[j] = xi[j];
                if (xi[j] > vmax[j]) vmax[j] = xi[j];
            }
    }
}


template<class Codec>
struct QuantizerNonUniform: ScalarQuantizer {
    const size_t d;
    const float *vmin, *vmax;

    QuantizerNonUniform(size_t d, const std::vector<float> &trained):
        d(d), vmin(trained.data()), vmax(trained.data() + d) {}

    virtual void encode_vector(const float *x, uint8_t *code) const
    {
        for (size_t i = 0; i < d; i++) {
            float xi = (x[i] - vmin[i]) / (vmax[i] - vmin[i]);
            if (xi < 0) xi = 0;
            if (xi > 1.0) xi = 1.0;
            Codec::encode_component (xi, code, i);
        }
    }

    virtual void decode_vector(const uint8_t *code, float *x) const
    {
        for (size_t i = 0; i < d; i++) {
            float xi = Codec::decode_component (code, i);
            x[i] = vmin[i] + xi * (vmax[i] - vmin[i]);
        }
    }

    float reconstruct_component (const uint8_t * code, int i) const
    {
        float xi = Codec::decode_component (code, i);
        return vmin[i] + xi * (vmax[i] - vmin[i]);
    }

    virtual float compute_distance_L2 (SimilarityL2 &sim, const uint8_t * codes) const
    { return compute_distance(*this, sim, codes); }

    virtual float compute_distance_IP (SimilarityIP &sim, const uint8_t * codes) const
    { return compute_distance(*this, sim, codes); }
};


/************************** AVX-optimized version ************/


ScalarQuantizer *select_quantizer(
       IndexIVFScalarQuantizer::QuantizerType qtype,
       size_t d, const std::vector<float> & trained)
{
    switch(qtype) {
    case IndexIVFScalarQuantizer::QT_8bit:

        return new QuantizerNonUniform<Codec8bit>(d, trained);
    case IndexIVFScalarQuantizer::QT_4bit:
        return new QuantizerNonUniform<Codec4bit>(d, trained);
    case IndexIVFScalarQuantizer::QT_8bit_uniform:
        return new QuantizerUniform<Codec8bit>(d, trained);
    case IndexIVFScalarQuantizer::QT_4bit_uniform:
        return new QuantizerUniform<Codec4bit>(d, trained);
    }
    FAISS_ASSERT(!"should not happen");
    return nullptr;
}


} // anonymous namespace

/*******************************************************************
 * Implementation of the IndexIVFScalarQuantizer
 */


void IndexIVFScalarQuantizer::train_residual (idx_t n, const float *x)
{
    long * idx = new long [n];
    quantizer->assign (n, x, idx);
    float *residuals = new float [n * d];
#pragma omp parallel for
    for (idx_t i = 0; i < n; i++)
        quantizer->compute_residual (x + i * d, residuals + i * d, idx[i]);

    switch (qtype) {
    case QT_4bit_uniform: case QT_8bit_uniform:
        train_Uniform (n, d, residuals, trained);
        break;
    case QT_4bit: case QT_8bit:
        train_NonUniform (n, d, residuals, trained);
        break;
    }

    delete idx;
    delete residuals;
}


void IndexIVFScalarQuantizer::add_with_ids
       (idx_t n, const float * x, const long *xids)
{
    FAISS_ASSERT (is_trained);
    long * idx = new long [n];
    quantizer->assign (n, x, idx);
    size_t nadd = 0;
    ScalarQuantizer *squant = select_quantizer(qtype, d, trained);

#pragma omp parallel reduction(+: nadd)
    {
        std::vector<float> residual (d);
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        for (size_t i = 0; i < n; i++) {

            long list_no = idx [i];
            if (list_no >= 0 && list_no % nt == rank) {
                long id = xids ? xids[i] : ntotal + i;

                FAISS_ASSERT (list_no < nlist);

                ids[list_no].push_back (id);
                nadd++;
                quantizer->compute_residual (x + i * d, residual.data(), list_no);

                size_t cur_size = codes[list_no].size();
                codes[list_no].resize (cur_size + code_size);

                squant->encode_vector (residual.data(),
                                       codes[list_no].data() + cur_size);
            }
        }
    }
    ntotal += nadd;
    delete squant;
    delete idx;
}


void search_with_probes_ip (const IndexIVFScalarQuantizer & index,
                            const float *x,
                            const idx_t *cent_ids, const float *cent_dis,
                            const ScalarQuantizer & quant,
                            int k, float *simi, idx_t *idxi)
{
    int nprobe = index.nprobe;
    size_t code_size = index.code_size;
    size_t d = index.d;
    std::vector<float> decoded(d);
    minheap_heapify (k, simi, idxi);
    for (int i = 0; i < nprobe; i++) {
        idx_t list_no = cent_ids[i];
        if (list_no < 0) break;
        float accu0 = cent_dis[i];

        const std::vector<idx_t> & ids = index.ids[list_no];
        const uint8_t* codes = index.codes[list_no].data();

        SimilarityIP sim(x, accu0);

        for (size_t j = 0; j < ids.size(); j++) {

            float accu = quant.compute_distance_IP(sim, codes);

            if (accu > simi [0]) {
                minheap_pop (k, simi, idxi);
                minheap_push (k, simi, idxi, accu, ids[j]);
            }
            codes += code_size;
        }

    }
    minheap_reorder (k, simi, idxi);
}

void search_with_probes_L2 (const IndexIVFScalarQuantizer & index,
                            const float *x_in,
                            const idx_t *cent_ids,
                            const Index *quantizer,
                            const ScalarQuantizer & quant,
                            int k, float *simi, idx_t *idxi)
{
    int nprobe = index.nprobe;
    size_t code_size = index.code_size;
    size_t d = index.d;
    std::vector<float> decoded(d), x(d);
    maxheap_heapify (k, simi, idxi);
    for (int i = 0; i < nprobe; i++) {
        idx_t list_no = cent_ids[i];
        if (list_no < 0) break;

        const std::vector<idx_t> & ids = index.ids[list_no];
        const uint8_t* codes = index.codes[list_no].data();

        // shift of x_in wrt centroid
        quantizer->compute_residual (x_in, x.data(), list_no);

        SimilarityL2 sim(x.data());

        for (size_t j = 0; j < ids.size(); j++) {

            float dis = quant.compute_distance_L2 (sim, codes);

            if (dis < simi [0]) {
                maxheap_pop (k, simi, idxi);
                maxheap_push (k, simi, idxi, dis, ids[j]);
            }
            codes += code_size;
        }
    }
    maxheap_reorder (k, simi, idxi);
}


void IndexIVFScalarQuantizer::search (idx_t n, const float *x, idx_t k,
                                      float *distances, idx_t *labels) const
{
    FAISS_ASSERT (is_trained);
    idx_t * idx = new idx_t [n * nprobe];
    float *dis = new float [n * nprobe];
    quantizer->search (n, x, nprobe, dis, idx);

    ScalarQuantizer *squant = select_quantizer(qtype, d, trained);

    if (metric_type == METRIC_INNER_PRODUCT) {
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            search_with_probes_ip (*this, x + i * d,
                                   idx + i * nprobe, dis + i * nprobe, *squant,
                                   k, distances + i * k, labels + i * k);
        }
    } else {
#pragma omp parallel for
        for (size_t i = 0; i < n; i++) {
            search_with_probes_L2 (*this, x + i * d,
                                   idx + i * nprobe, quantizer, *squant,
                                   k, distances + i * k, labels + i * k);
        }
    }

    delete squant;
    delete dis;
    delete idx;
}

void IndexIVFScalarQuantizer::set_typename() {
    FAISS_ASSERT(!"not implemented");
}

void IndexIVFScalarQuantizer::merge_from_residuals (IndexIVF &other) {
    FAISS_ASSERT(!"not implemented");
}

    void test_avx () {
        uint8_t code[8] = {12, 13, 14, 15, 16, 17, 18, 19};
        __m256 vec = Codec8bit::decode_8_components(code, 0);
        float vf[8];
        _mm256_storeu_ps(vf, vec);
        printf("vec=[");
        for(int i= 0; i < 8; i++) printf("%g ", vf[i] * 255);
        printf("]\n");
    }


}
