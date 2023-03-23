import os
import argparse 

""" Note! Make sure this script is run under conda environment with faiss, e.g., conda activate py37 """
# python performance_test_gpu.py --dataset SIFT100M --index IVF --PQ 16 --OPQ 1 --ngpu 1 --startgpu 0

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
GPU_memory_use = int(1536*1024*1024) # in terms of bytes

if FLAGS.index == "IVF":

    for i in IVF_range:

        # TODO: add user selected searching range in bench_gpu_1bn
        search_range = "-nprobe "
        nprobe_range = int(i / 2)
        for np in range(nprobe_range + 1):
            if np != 0:
                search_range += ','
            search_range += "{}".format(2 ** np)

        if FLAGS.OPQ:
                index_str = "OPQ{0},IVF{1},PQ{0}".format(FLAGS.PQ, 2 ** i)
                os.system("python bench_gpu_1bn.py {0} {1} -nnn 100 -ngpu {2} -startgpu {3} -tempmem {4} -qbs 512 {5} > gpu_performance_result/{0}_{1}".format(
                        FLAGS.dataset, index_str, FLAGS.ngpu, FLAGS.startgpu, GPU_memory_use, search_range))
        else:
                index_str = "IVF{0},PQ{1}".format(2 ** i, FLAGS.PQ)
                os.system("python bench_gpu_1bn.py {0} {1} -nnn 100 -ngpu {2} -startgpu {3} -tempmem {4} -qbs 512 {5} > gpu_performance_result/{0}_{1}".format(
                        FLAGS.dataset, index_str, FLAGS.ngpu, FLAGS.startgpu, GPU_memory_use, search_range))
else:
    raise("Index error, this script only supports IVF")