# Example Usage:
#   python recall_print_CPU_GPU.py --dict_dir './cpu_recall_index_nprobe_pairs_SIFT100M.pkl' 
# Example Usage (specify topK and/or recall_goal)
#   python recall_print_CPU_GPU.py --dict_dir './cpu_recall_index_nprobe_pairs_SIFT100M.pkl' --topK 10 --recall_goal 0.8
import pickle
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--dict_dir', type=str, default='./cpu_recall_index_nprobe_pairs_SIFT100M.pkl', help="recall dictionary directory")

parser.add_argument('--topK', type=int, default=0, help="return topK most similar vector, related to recall, e.g., R@10=50perc or R@100=80perc")
parser.add_argument('--recall_goal', type=float, default=0, help="recall_goal, 0~1")

args = parser.parse_args()
topK = args.topK
recall_goal = args.recall_goal
assert recall_goal >= 0 and recall_goal <= 1

with open(args.dict_dir, 'rb') as f:
    d = pickle.load(f)
    print("\n\n======= Dictionary Name: {} =======\n".format(args.dict_dir))
    # if args.topK:
    #     print("topK = {}".format(args.topK))
    # if args.recall_goal:
    #     print("recall_goal = {}".format(args.recall_goal))
    for dbname in d:
        for index_key in d[dbname]:
            for k in d[dbname][index_key]:
                if topK and k != topK:
                    continue
                for r in d[dbname][index_key][k]:
                    if recall_goal and r != recall_goal:
                        continue
                    else:
                        print(index_key, k, r, ": nlist={}".format(d[dbname][index_key][k][r]))
        
    print("\n\n")
