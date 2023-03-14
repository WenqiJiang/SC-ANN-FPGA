import argparse 
import os
import yaml
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default="./template_files/stage4_LUT_construction", help="template input directory")
parser.add_argument('--output_dir', type=str, default="./output_files", help="output directory")
args = parser.parse_args()

# Load YAML configurations
config_file = open("config.yaml", "r")
config = yaml.load(config_file)

if config["PE_NUM_TABLE_CONSTRUCTION"] == 1:
    os.system("cp {} {}".format(os.path.join(args.input_dir, "LUT_construction_1_PE.hpp"), os.path.join(args.output_dir, "LUT_construction.hpp")))
else:
    os.system("cp {} {}".format(os.path.join(args.input_dir, "LUT_construction.hpp"), os.path.join(args.output_dir, "LUT_construction.hpp")))
