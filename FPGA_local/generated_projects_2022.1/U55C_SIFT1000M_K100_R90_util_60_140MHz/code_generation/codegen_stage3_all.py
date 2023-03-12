import argparse 
import os
import yaml
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default="./template_files/stage3_select_Voronoi_cells", help="template input directory")
parser.add_argument('--output_dir', type=str, default="./output_files", help="output directory")
args = parser.parse_args()

# Load YAML configurations
config_file = open("config.yaml", "r")
config = yaml.load(config_file)

if config["STAGE_3_PRIORITY_QUEUE_LEVEL"] > 2:
    print("ERROR! STAGE_3_PRIORITY_QUEUE_LEVEL must be 1 or 2")
    raise ValueError
if config["STAGE_3_PRIORITY_QUEUE_LEVEL"] == 2 and config["STAGE_3_PRIORITY_QUEUE_L1_NUM"] >  2:
    print("ERROR! STAGE_3_PRIORITY_QUEUE_L1_NUM must be 2 when STAGE_3_PRIORITY_QUEUE_LEVEL=2")
    raise ValueError
os.system("cp {} {}".format(os.path.join(args.input_dir, "priority_queue_vector_quantizer.hpp"), args.output_dir))
os.system("cp {} {}".format(os.path.join(args.input_dir, "select_Voronoi_cell.hpp"), args.output_dir))
