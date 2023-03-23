import os
import argparse 

""" Note! Make sure this script is run under conda environment with faiss, e.g., conda activate py37 """
# python train_gpu.py --dataset SIFT100M --index IVF --PQ 16 --OPQ 1 --ngpu 4 --startgpu 3

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="SIFT1000M", help="SIFT1M, SIFT10M, SIFT100M, SIFT1000M")
parser.add_argument('--index', type=str, default="IVF", help="IVF")
parser.add_argument('--PQ', type=int, default="16", help="8, 16")
parser.add_argument('--OPQ', type=int, default=0, help="0 -> Disable; 1 -> Enable")
parser.add_argument('--ngpu', type=int, default=1, help="the number of GPUs used for training")
parser.add_argument('--startgpu', type=int, default=0, help="the GPU id, e.g., ngpu=3,startgpu=2 means using GPU2,GPU3,GPU4")

FLAGS, unparsed = parser.parse_known_args()

# IVF: 1024 (2^10) to 262144 (2^18)
# IVF_range = [17, 18]
IVF_range = [10, 11, 12, 13, 14, 15, 16, 17, 18]
GPU_memory_use = int(4*1024*1024*1024) # in terms of bytes

if FLAGS.index == "IVF":
    if FLAGS.OPQ:
        for i in IVF_range:
            index_str = "OPQ{0},IVF{1},PQ{0}".format(FLAGS.PQ, 2 ** i)
            os.system("python bench_gpu_1bn.py -dbname {0} -index_key {1} -topK 100 -ngpu {2} -startgpu {3} -tempmem {4} -nprobe 1 -qbs 512".format(
                    FLAGS.dataset, index_str, FLAGS.ngpu, FLAGS.startgpu, GPU_memory_use))
    else:
        for i in IVF_range:
            index_str = "IVF{0},PQ{1}".format(2 ** i, FLAGS.PQ)
            os.system("python bench_gpu_1bn.py -dbname {0} -index_key {1} -topK 100 -ngpu {2} -startgpu {3} -tempmem {4} -nprobe 1 -qbs 512".format(
                    FLAGS.dataset, index_str, FLAGS.ngpu, FLAGS.startgpu, GPU_memory_use))
else:
    raise("Index error, this script only supports IVF")
