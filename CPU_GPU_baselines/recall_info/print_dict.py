# Example Usage:
#   python print_dict.py --dict_dir './dict.pkl' 
import pickle
import argparse 
parser = argparse.ArgumentParser()
parser.add_argument('--dict_dir', type=str, default='./dict.pkl', help="recall dictionary directory")

args = parser.parse_args()

with open(args.dict_dir, 'rb') as f:
    d = pickle.load(f)
    print("\n\n======= Dictionary Name: {} =======\n".format(args.dict_dir))
    print(d)