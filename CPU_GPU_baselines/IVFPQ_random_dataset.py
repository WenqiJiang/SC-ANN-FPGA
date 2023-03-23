import numpy as np
import time

# documents

d = 128                           # dimension
nb = int(1e6)                      # database size
nq = int(1e4)                       # nq of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 1000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 1000.

import faiss

nlist = 1024
nprob = 16
m = 8 # number of subvectors
nbits = 8
topk = 10
quantizer = faiss.IndexFlatL2(d)  # this remains the same
# IndexIVFPQ(quantizer, d, nlists, M, nbits)
index = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)


print("Start training.")
train_start = time.time()
index.train(xb)
index.add(xb)
D, I = index.search(xb[:5], topk) # sanity check
train_end = time.time()
print(I)
print(D)
print("Finish training. Takes {} seconds.".format(train_end - train_start))


index.nprobe = nprob

for i in range(5):
    nq_range = int(10 ** i)
    assert nq_range <= nq
    # print("Start querying.")

    start = time.time()
    D, I = index.search(xq[:nq_range], topk)     # search
    end = time.time()

    # print("End querying.")
    print("\n\n")
    print(I[-5:])

    print("Dataset size: {}\t".format(nb) + "Query num: {}\n".format(nq_range) + "TopK: {}\n".format(topk))
    print("Centroids: {}\tnprobe: {}\tscanned ratio: {}\n".format(nlist, nprob, nprob/nlist))
    print("Latency: {}sec\tQPS: {}\n".format(end - start, nq_range / (end - start)))