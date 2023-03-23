import os
import argparse 

""" Note! Make sure this script is run under conda environment with faiss, e.g., conda activate py37 """
# python performance_test_cpu.py --dataset SIFT100M --index IVF --PQ 16 --OPQ 0

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="SIFT1000M", help="SIFT1M, SIFT10M, SIFT100M, SIFT1000M")
parser.add_argument('--index', type=str, default="IVF", help="IVF, IMI")
parser.add_argument('--PQ', type=int, default="16", help="8, 16")
parser.add_argument('--OPQ', type=int, default=0, help="0 -> Disable; 1 -> Enable")

FLAGS, unparsed = parser.parse_known_args()

# IMI: 2x8 (256 cells per half) to 2x14 (16384 cells per half)
IMI_range = [8, 9, 10, 11, 12, 13, 14]

# IVF: 1024 (2^10) to 262144 (2^18)
# IVF_range = [17, 18]
IVF_range = [10, 11, 12, 13, 14, 15, 16, 17, 18]

if FLAGS.index == "IMI":
    for i in IMI_range:
        # set nprobe range, e.g., for IMI2x8, nprobe max is 2 ** (8 + 2) = 1024
        search_range = ""
        nprobe_range = int(i + 2)
        for np in range(nprobe_range + 1):
            search_range += "nprobe={} ".format(2 ** np)
        
        index_str = "IMI2x{},PQ{}".format(i, FLAGS.PQ)
        os.system("python bench_polysemous_1bn.py {0} {1} {2} > cpu_performance_result/{0}_{1}".format(
            FLAGS.dataset, index_str, search_range))

elif FLAGS.index == "IVF":

    for i in IVF_range:
        # set nprobe range, e.g., for IVF65536 (2 ^ 16), nprobe max is 2 ** (16 / 2) = 1024
        search_range = ""
        nprobe_range = int(i / 2)
        for np in range(nprobe_range + 1):
            search_range += "nprobe={} ".format(2 ** np)

        if FLAGS.OPQ:
            index_str = "OPQ{},IVF{},PQ{}".format(FLAGS.PQ, 2 ** i, FLAGS.PQ)
            os.system("python bench_polysemous_1bn.py {0} {1} {2} > cpu_performance_result/{0}_{1}".format(
                    FLAGS.dataset, index_str, search_range))
        else:
            index_str = "IVF{},PQ{}".format(2 ** i, FLAGS.PQ)
            os.system("python bench_polysemous_1bn.py {0} {1} {2} > cpu_performance_result/{0}_{1}".format(
                    FLAGS.dataset, index_str, search_range))
