import argparse 
import os
import yaml
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default="./template_files/stage5_distance_estimation_by_LUT", help="template input directory")
parser.add_argument('--output_dir', type=str, default="./output_files", help="output directory")
args = parser.parse_args()

# Load YAML configurations
config_file = open("config.yaml", "r")
config = yaml.load(config_file)

# Load template
template_dir = os.path.join(args.input_dir, "distance_estimation_by_LUT.hpp")
template_str = None
with open(template_dir) as f:
    template_str = f.read()

# Fill template
template_fill_dict = dict()

if config['SORT_GROUP_ENABLE']:
    template_fill_dict["PQ_lookup_computation_wrapper_arguments"] = \
    "    hls::stream<single_PQ_result> (&s_single_PQ_result)[SORT_GROUP_NUM][16]"
else:
    template_fill_dict["PQ_lookup_computation_wrapper_arguments"] = \
    "    hls::stream<single_PQ_result> (&s_single_PQ_result)[stream_num]"


if config['SORT_GROUP_ENABLE'] and (config["STAGE5_COMP_PE_NUM"] != config["SORT_GROUP_NUM"] * 16):
    """    
    If there's sort-reduction, then
      1. there's dummy signals (s_scanned_entries_every_cell_Dummy_replicated) in most cases (unless e.g., input stream 48 = 3 * 16, no remainder)
      2. there's 3rd element in replicate_s_scanned_entries_every_cell_PQ_lookup_computation, i.e., s_scanned_entries_every_cell_Dummy_replicated (unless no remainder)
    """
    template_fill_dict["PQ_lookup_computation_wrapper_replicate_signal"] = """
    hls::stream<int> s_scanned_entries_every_cell_Dummy_replicated[SORT_GROUP_NUM * 16 - stream_num]; // 32 - 3 * 10 = 2
#pragma HLS stream variable=s_scanned_entries_every_cell_Dummy_replicated depth=8
#pragma HLS array_partition variable=s_scanned_entries_every_cell_Dummy_replicated complete
// #pragma HLS RESOURCE variable=s_scanned_entries_every_cell_Dummy_replicated core=FIFO_SRL

    replicate_s_scanned_entries_every_cell_PQ_lookup_computation<stream_num>(
        query_num,
        nprobe, 
        s_scanned_entries_every_cell_PQ_lookup_computation, 
        s_scanned_entries_every_cell_PQ_lookup_computation_replicated,
        s_scanned_entries_every_cell_Dummy_replicated);
"""
    template_fill_dict["dummy_PQ_result_sender_sort_group_num_1"] = """
    for (int j = stream_num; j < 16; j++) {
#pragma HLS UNROLL
        dummy_PQ_result_sender(
            query_num,
            nprobe,
            s_scanned_entries_every_cell_Dummy_replicated[j - stream_num], 
            s_single_PQ_result[0][j]);
    }
    """
    template_fill_dict["dummy_PQ_result_sender_sort_group_num_2"] = """
    for (int j = stream_num - 16 * (SORT_GROUP_NUM - 1); j < 16; j++) {
#pragma HLS UNROLL
        dummy_PQ_result_sender(
            query_num,
            nprobe,
            s_scanned_entries_every_cell_Dummy_replicated[j - (stream_num - 16 * (SORT_GROUP_NUM - 1))], 
            s_single_PQ_result[SORT_GROUP_NUM - 1][j]);
    }
    """
    template_fill_dict["dummy_PQ_result_sender_sort_group_num_3"] = """
    for (int j = stream_num - 16 * (SORT_GROUP_NUM - 1); j < 16; j++) {
#pragma HLS UNROLL
        dummy_PQ_result_sender(
            query_num,
            nprobe,
            s_scanned_entries_every_cell_Dummy_replicated[j - (stream_num - 16 * (SORT_GROUP_NUM - 1))], 
            s_single_PQ_result[SORT_GROUP_NUM - 1][j]);
    }
    """

else:
    template_fill_dict["PQ_lookup_computation_wrapper_replicate_signal"] = """
    replicate_s_scanned_entries_every_cell_PQ_lookup_computation<stream_num>(
        query_num, 
        nprobe, 
        s_scanned_entries_every_cell_PQ_lookup_computation, 
        s_scanned_entries_every_cell_PQ_lookup_computation_replicated);
"""
    template_fill_dict["dummy_PQ_result_sender_sort_group_num_1"] = ""
    template_fill_dict["dummy_PQ_result_sender_sort_group_num_2"] = ""
    template_fill_dict["dummy_PQ_result_sender_sort_group_num_3"] = ""

for k in template_fill_dict:
    template_str = template_str.replace("<--{}-->".format(k), str(template_fill_dict[k]))
output_str = template_str

# Save generated file
output_dir = os.path.join(args.output_dir, "distance_estimation_by_LUT.hpp")
with open(output_dir, "w+") as f:
    f.write(output_str)