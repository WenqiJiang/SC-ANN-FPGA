import os
import argparse 

""" Note! Make sure this script is run under conda environment with faiss, e.g., conda activate py37 """
# python train_cpu.py --dataset SIFT100M --index IVF --PQ 16 --OPQ 0

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default="SIFT1000M", help="SIFT1M, SIFT10M, SIFT100M, SIFT1000M")
parser.add_argument('--index', type=str, default="IVF", help="IVF, IMI")
parser.add_argument('--PQ', type=int, default="16", help="8, 16")
parser.add_argument('--OPQ', type=int, default=1, help="0 -> Disable; 1 -> Enable")

FLAGS, unparsed = parser.parse_known_args()

# IMI: 2x8 (256 cells per half) to 2x14 (16384 cells per half)
IMI_range = [8, 9, 10, 11, 12, 13, 14]

# IVF: 1024 (2^10) to 262144 (2^18)
# IVF_range = [17, 18]
IVF_range = [10, 11, 12, 13, 14, 15, 16, 17, 18]

if FLAGS.index == "IMI":
    if FLAGS.OPQ:
        for i in IMI_range:
            os.system("python bench_polysemous_1bn.py 0 {} OPQ{},IMI2x{},PQ{} nprobe=1".format(
                FLAGS.dataset, FLAGS.PQ, i, FLAGS.PQ))
    else:
        for i in IMI_range:
            os.system("python bench_polysemous_1bn.py 0 {} IMI2x{},PQ{} nprobe=1".format(
                FLAGS.dataset, i, FLAGS.PQ))
elif FLAGS.index == "IVF":
    if FLAGS.OPQ:
        for i in IVF_range:
            os.system("python bench_polysemous_1bn.py 0 {} OPQ{},IVF{},PQ{} nprobe=1".format(
                    FLAGS.dataset, FLAGS.PQ, 2 ** i, FLAGS.PQ))
    else:
        for i in IVF_range:
            os.system("python bench_polysemous_1bn.py 0 {} IVF{},PQ{} nprobe=1".format(
                    FLAGS.dataset, 2 ** i, FLAGS.PQ))
