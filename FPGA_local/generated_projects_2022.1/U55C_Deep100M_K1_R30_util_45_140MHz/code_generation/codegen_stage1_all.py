import argparse 
import os
import yaml
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default="./template_files/stage1_OPQ", help="template input directory")
parser.add_argument('--output_dir', type=str, default="./output_files", help="output directory")
args = parser.parse_args()

# Load YAML configurations
config_file = open("config.yaml", "r")
config = yaml.load(config_file)

if config["OPQ_ENABLE"]:
    if config["OPQ_UNROLL_FACTOR"] == 4:
        os.system("cp {} {}".format(
            os.path.join(args.input_dir, "OPQ_preprocessing_unroll_4.hpp"), 
            os.path.join(args.output_dir, "OPQ_preprocessing.hpp")))
    elif config["OPQ_UNROLL_FACTOR"] == 8:
        os.system("cp {} {}".format(
            os.path.join(args.input_dir, "OPQ_preprocessing_unroll_8.hpp"), 
            os.path.join(args.output_dir, "OPQ_preprocessing.hpp")))
    else:
        print("ERROR! OPQ_UNROLL_FACTOR wrong, should be 4 or 8 or 16")
        raise ValueError