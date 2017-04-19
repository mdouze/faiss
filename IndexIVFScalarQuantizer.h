

#ifndef FAISS_INDEX_IVF_SCALAR_QUANTIZER_H
#define FAISS_INDEX_IVF_SCALAR_QUANTIZER_H

#include <stdint.h>


#include <vector>


#include "IndexIVF.h"


namespace faiss {



struct IndexIVFScalarQuantizer:IndexIVF {

    enum QuantizerType {
        QT_8bit,
        QT_4bit,
        QT_8bit_uniform,
        QT_4bit_uniform,
    };

    QuantizerType qtype;
    size_t code_size;

    std::vector<float> trained;

    std::vector<std::vector<uint8_t> > codes;

    IndexIVFScalarQuantizer(Index *quantizer, size_t d, size_t nlist,
                            MetricType metric, QuantizerType qtype);

    virtual void train_residual (idx_t n, const float *x) override;

    virtual void add_with_ids (idx_t n, const float * x, const long *xids)
        override;

    virtual void search (idx_t n, const float *x, idx_t k,
                         float *distances, idx_t *labels) const override;

    virtual void set_typename();

    virtual void merge_from_residuals (IndexIVF &other);


};



 void test_avx ();
}


#endif
