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

xq = xq[:1000]
# xb = xb[:100000]

nq, d = xq.shape

print "load GT"
gt = ivecs_read("sift1M/sift_groundtruth.ivecs")

gt = gt[:1000]


ncent = 256

variants = [(name, getattr(faiss.IndexIVFScalarQuantizer, name))
            for name in dir(faiss.IndexIVFScalarQuantizer)
            if name.startswith('QT_')]



quantizer = faiss.IndexFlatL2(d)
# quantizer.add(np.zeros((1, d), dtype='float32'))


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
