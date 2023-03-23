# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import sys
import time
import numpy as np
import struct


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    # Wenqi: Format of ground truth (for 10000 query vectors):
    #   1000(topK), [1000 ids]
    #   1000(topK), [1000 ids]
    #        ...     ...
    #   1000(topK), [1000 ids]
    # 10000 rows in total, 10000 * 1001 elements, 10000 * 1001 * 4 bytes
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def load_sift1M():
    print("Loading sift1M...", end='', file=sys.stderr)
    xt = fvecs_read("sift1M/sift_learn.fvecs")
    xb = fvecs_read("sift1M/sift_base.fvecs")
    xq = fvecs_read("sift1M/sift_query.fvecs")
    gt = ivecs_read("sift1M/sift_groundtruth.ivecs")
    print("done", file=sys.stderr)

    return xb, xq, xt, gt


def evaluate(index, xq, gt, k):
    nq = xq.shape[0]
    t0 = time.time()
    D, I = index.search(xq, k)  # noqa: E741
    t1 = time.time()

    recalls = {}
    i = 1
    while i <= k:
        recalls[i] = (I[:, :i] == gt[:, :1]).sum() / float(nq)
        i *= 10

    return (t1 - t0) * 1000.0 / nq, recalls


def mmap_bvecs_FB(fname, num_vec=int(1e9)):
    """
    for both FB SimNetSearch query and base vectors
    8 bytes header, uint8 representation
    first 4 bytes: number of vec (int32) == 1e9
    second 4 bytes: number of dim (int32) == 256
    """
    x = np.memmap(fname, dtype='uint8', mode='r')
    d = x[4:8].view('int32')[0]
    x = x[8: 8 + num_vec * d]
    return x.reshape(-1, d)

def mmap_bvecs_SBERT(fname, num_vec=int(1e6)):
    """
    SBERT, 384 dim, no header
    """
    d = 384
    x = np.memmap(fname, dtype='float32', mode='r')
    return x.reshape(-1, d)

def mmap_bvecs_GNN(fname, num_vec=int(1e6)):
    """
    SBERT, 384 dim, no header
    """
    d = 256
    x = np.memmap(fname, dtype='float32', mode='r')
    return x.reshape(-1, d)

def mmap_bvecs_Journal(fname, num_vec=int(1e6)):
    """
    SBERT, 384 dim, no header
    """
    d = 100
    x = np.memmap(fname, dtype='float32', mode='r')
    return x.reshape(-1, d)

def read_deep_fbin(filename):
    """
    Read *.fbin file that contains float32 vectors

    All embedding data is stored in .fbin format:
    [num_vectors (uint32), vector_dim (uint32), vector_array (float32)]

    The groundtruth is stored in .ibin format:
    [num_vectors (uint32), vector_dim (uint32), vector_array (int32)]

    https://research.yandex.com/datasets/biganns
    https://pastebin.com/BAf6bM5L
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)

    arr = np.memmap(filename, dtype=np.float32, offset=8, mode='r')
    return arr.reshape(nvecs, dim)

def read_deep_ibin(filename, dtype='int32'):
    """
    Read *.ibin file that contains int32, uint32 or int64 vectors

    All embedding data is stored in .fbin format:
    [num_vectors (uint32), vector_dim (uint32), vector_array (float32)]

    The groundtruth is stored in .ibin format:
    [num_vectors (uint32), vector_dim (uint32), vector_array (int32)]

    https://research.yandex.com/datasets/biganns
    https://pastebin.com/BAf6bM5L
    """
    with open(filename, "rb") as f:
        nvecs, dim = np.fromfile(f, count=2, dtype=np.int32)
    arr = np.fromfile(filename, dtype=dtype, offset=8)
    return arr.reshape(nvecs, dim)


def write_deep_fbin(filename, vecs):
    """ Write an array of float32 vectors to *.fbin file
    Args:s
        :param filename (str): path to *.fbin file
        :param vecs (numpy.ndarray): array of float32 vectors to write
    """
    assert len(vecs.shape) == 2, "Input array must have 2 dimensions"
    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape
        f.write(struct.pack('<i', nvecs))
        f.write(struct.pack('<i', dim))
        vecs.astype('float32').flatten().tofile(f)
 
        
def write_deep_ibin(filename, vecs, dtype='int32'):
    """ Write an array of int32 or int64 vectors to *.ibin file
    Args:
        :param filename (str): path to *.ibin file
        :param vecs (numpy.ndarray): array of int32 vectors to write
    """
    assert len(vecs.shape) == 2, "Input array must have 2 dimensions"
    with open(filename, "wb") as f:
        nvecs, dim = vecs.shape
        f.write(struct.pack('<i', nvecs))
        f.write(struct.pack('<i', dim))
        if dtype == 'int32':
            vecs.astype('int32').flatten().tofile(f)
        elif dtype == 'uint32':
            vecs.astype('uint32').flatten().tofile(f)
        elif dtype == 'int64':
            vecs.astype('int64').flatten().tofile(f)
        else:
            print("Unsupported datatype ", dtype)
            raise ValueError