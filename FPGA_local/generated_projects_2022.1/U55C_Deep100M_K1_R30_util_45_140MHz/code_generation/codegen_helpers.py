import argparse 
import os
import yaml
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default="./template_files", help="template input directory")
parser.add_argument('--output_dir', type=str, default="./output_files", help="output directory")
args = parser.parse_args()

# Load YAML configurations
config_file = open("config.yaml", "r")
config = yaml.load(config_file)

# Load template
template_dir = os.path.join(args.input_dir, "helpers.hpp")
template_str = None
with open(template_dir) as f:
    template_str = f.read()

# Fill template
template_fill_dict = dict()

if config['SORT_GROUP_ENABLE']:
    template_fill_dict["scan_controller_arg_s_scanned_entries_per_query_Sort_and_reduction"] = \
    "    hls::stream<int> &s_scanned_entries_per_query_Sort_and_reduction,"
    template_fill_dict["scan_controller_body_s_scanned_entries_per_query_Sort_and_reduction"] = \
    "        s_scanned_entries_per_query_Sort_and_reduction.write(accumulated_scanned_entries_per_query);"
else:
    template_fill_dict["scan_controller_arg_s_scanned_entries_per_query_Sort_and_reduction"] = ""
    template_fill_dict["scan_controller_body_s_scanned_entries_per_query_Sort_and_reduction"] = ""

for k in template_fill_dict:
    template_str = template_str.replace("<--{}-->".format(k), str(template_fill_dict[k]))
output_str = template_str

# Save generated file
output_dir = os.path.join(args.output_dir, "helpers.hpp")
with open(output_dir, "w+") as f:
    f.write(output_str)