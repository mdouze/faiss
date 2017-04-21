#!/usr/bin/env python2

import os
import time
import numpy as np
import pdb


import faiss

#################################################################
# I/O functions
#################################################################

def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


#################################################################
#  Main program
#################################################################

print "load data"

xt = fvecs_read("sift1M/sift_learn.fvecs")
xb = fvecs_read("sift1M/sift_base.fvecs")
xq = fvecs_read("sift1M/sift_query.fvecs")

# xq = xq[:1000]
# xb = xb[:100000]

nq, d = xq.shape

print "load GT"
gt = ivecs_read("sift1M/sift_groundtruth.ivecs")

# gt = gt[:1000]


ncent = 256

variants = [(name, getattr(faiss.IndexIVFScalarQuantizer, name))
            for name in dir(faiss.IndexIVFScalarQuantizer)
            if name.startswith('QT_')]



quantizer = faiss.IndexFlatL2(d)
# quantizer.add(np.zeros((1, d), dtype='float32'))

if False:
    for name, qtype in [('flat', 0)] + variants:

        print "============== test", name
        t0 = time.time()

        if name == 'flat':
            index = faiss.IndexIVFFlat(quantizer, d, ncent,
                                       faiss.METRIC_L2)
        else:
            index = faiss.IndexIVFScalarQuantizer(quantizer, d, ncent,
                                                  qtype, faiss.METRIC_L2)

        index.nprobe = 16
        print "[%.3f s] train" % (time.time() - t0)
        index.train(xt)
        print "[%.3f s] add" % (time.time() - t0)
        index.add(xb)
        print "[%.3f s] search" % (time.time() - t0)
        D, I = index.search(xq, 100)
        print "[%.3f s] eval" % (time.time() - t0)

        for rank in 1, 10, 100:
            n_ok = (I[:, :rank] == gt[:, :1]).sum()
            print "%.4f" % (n_ok / float(nq)),
        print



if True:
    for name, qtype in variants:

        print "============== test", name

        for rsname, vals in [('RS_minmax',
                              [-0.4, -0.2, -0.1, -0.05, 0.0, 0.1, 0.5]),
                             ('RS_meanstd', [0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]),
                             ('RS_quantiles', [0.02, 0.05, 0.1, 0.15])]:
            for val in vals:
                print "%-15s %5g    " % (rsname, val),
                index = faiss.IndexIVFScalarQuantizer(quantizer, d, ncent,
                                                      qtype, faiss.METRIC_L2)
                index.nprobe = 16
                index.rangestat = getattr(faiss.IndexIVFScalarQuantizer,
                                          rsname)
                index.rangestat_arg = val

                index.train(xt)
                index.add(xb)
                t0 = time.time()
                D, I = index.search(xq, 100)
                t1 = time.time()

                for rank in 1, 10, 100:
                    n_ok = (I[:, :rank] == gt[:, :1]).sum()
                    print "%.4f" % (n_ok / float(nq)),
                print "   %.3f s" % (t1 - t0)
