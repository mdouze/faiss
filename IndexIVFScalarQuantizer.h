

#ifndef FAISS_INDEX_IVF_SCALAR_QUANTIZER_H
#define FAISS_INDEX_IVF_SCALAR_QUANTIZER_H

#include <stdint.h>


#include <vector>


#include "IndexIVF.h"


namespace faiss {

/** An IVF implementation where the components of the residuals are
 * encoded with a scalar uniform quantizer.
 *
 */

struct IndexIVFScalarQuantizer:IndexIVF {

    enum QuantizerType {
        QT_8bit,             ///< 8 bits per component
        QT_4bit,             ///< 4 bits per component
        QT_8bit_uniform,     ///< same, shared range for all dimensions
        QT_4bit_uniform,
    };

    QuantizerType qtype;

    // how to define the range of the unform encoder
    enum RangeStat {
        RS_minmax,           ///< [min, max]
        RS_meanstd,          ///< [mean - std * rs, mean + std * rs]
        RS_quantiles,        ///< [Q(rs), Q(1-rs)]
    };

    RangeStat rangestat;
    float rangestat_arg;

    /// bytes per vector
    size_t code_size;

    /// output training
    std::vector<float> trained;

    std::vector<std::vector<uint8_t> > codes;

    IndexIVFScalarQuantizer(Index *quantizer, size_t d, size_t nlist,
                            QuantizerType qtype, MetricType metric = METRIC_L2);

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
