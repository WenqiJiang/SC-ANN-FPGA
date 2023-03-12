import argparse 
import os
import yaml
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default="./template_files/stage6_select_results", help="template input directory")
parser.add_argument('--output_dir', type=str, default="./output_files", help="output directory")
args = parser.parse_args()

# Load YAML configurations
config_file = open("config.yaml", "r")
config = yaml.load(config_file)

if config["SORT_GROUP_ENABLE"]:
    os.system("cp {} {}".format(os.path.join(args.input_dir, "sort_reduction_with_vecID.hpp"), args.output_dir))

os.system("cp {} {}".format(os.path.join(args.input_dir, "priority_queue_distance_results.hpp"), args.output_dir))
if config["STAGE_6_PRIORITY_QUEUE_LEVEL"] == 2:
    os.system("cp {} {}".format(os.path.join(args.input_dir, "priority_queue_distance_results_L2_wrapper.hpp"), 
        os.path.join(args.output_dir, "priority_queue_distance_results_wrapper.hpp")))
elif config["STAGE_6_PRIORITY_QUEUE_LEVEL"] == 3:
    assert config["STAGE5_COMP_PE_NUM"] * 2 == \
        (config["STAGE_6_PRIORITY_QUEUE_L2_NUM"] - 1) * config["STAGE_6_STREAM_PER_L2_QUEUE_LARGER"] + \
        config["STAGE_6_STREAM_PER_L2_QUEUE_SMALLER"],  "ERROR! 3-level priority group config numbers are wrong"
    os.system("cp {} {}".format(os.path.join(args.input_dir, "priority_queue_distance_results_L3_wrapper.hpp"), 
        os.path.join(args.output_dir, "priority_queue_distance_results_wrapper.hpp")))
else:
    print("ERROR! Invalid STAGE_6_PRIORITY_QUEUE_LEVEL number, should be 2 or 3")
    raise ValueError