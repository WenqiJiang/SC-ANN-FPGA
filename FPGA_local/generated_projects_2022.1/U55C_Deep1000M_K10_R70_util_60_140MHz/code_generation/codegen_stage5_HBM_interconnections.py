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
template_dir = os.path.join(args.input_dir, "HBM_interconnections.hpp")
template_str = None
with open(template_dir) as f:
    template_str = f.read()

# Fill template
template_fill_dict = dict()

template_fill_dict["load_and_split_PQ_codes_wrapper_arguments"] = ""
for i in range(config["HBM_CHANNEL_NUM"]):
	template_fill_dict["load_and_split_PQ_codes_wrapper_arguments"] += \
        "    const ap_uint512_t* HBM_in{},\n".format(i)

template_fill_dict["load_and_split_PQ_codes_wrapper_func_body"] = ""

if config["HBM_CHANNEL_NUM"] > config["STAGE5_COMP_PE_NUM"]:
    """
    If merging contents from several channels, (1 HBM channel = e.g., 2 PQ code streams), then
        1. declare s_scanned_entries_every_cell_Merge_unit_replicated, s_single_PQ_per_channel 
        2. replicate_s_scanned_entries_every_cell has the 's_scanned_entries_every_cell_Merge_unit_replicated' argument
        3. load_and_split_PQ_codes's last argument is s_single_PQ_per_channel 
        4. has the merge_HBM_channel_PQ_codes functions (2 in 1, 3 in 1, 4 in 1)
    """
    template_fill_dict["load_and_split_PQ_codes_wrapper_func_body"] += """    hls::stream<int> s_scanned_entries_every_cell_Merge_unit_replicated[STAGE5_COMP_PE_NUM];
#pragma HLS stream variable=s_scanned_entries_every_cell_Merge_unit_replicated depth=8
#pragma HLS array_partition variable=s_scanned_entries_every_cell_Merge_unit_replicated complete
// #pragma HLS RESOURCE variable=s_scanned_entries_every_cell_Split_unit_replicated core=FIFO_SRL

    hls::stream<single_PQ> s_single_PQ_per_channel[HBM_CHANNEL_NUM];
#pragma HLS stream variable=s_single_PQ_per_channel depth=8
#pragma HLS array_partition variable=s_single_PQ_per_channel complete
// #pragma HLS RESOURCE variable=s_single_PQ_per_channel core=FIFO_SRL
"""
    template_fill_dict["load_and_split_PQ_codes_wrapper_func_body"] += """
    replicate_s_scanned_entries_every_cell(
        query_num,
        nprobe,
        s_scanned_entries_every_cell_Load_unit, 
        s_scanned_entries_every_cell_Load_unit_replicated,
        s_scanned_entries_every_cell_Split_unit_replicated,
        s_scanned_entries_every_cell_Merge_unit_replicated); 
"""
    for i in range(config["HBM_CHANNEL_NUM"]):
        template_fill_dict["load_and_split_PQ_codes_wrapper_func_body"] += \
        """
    load_and_split_PQ_codes(
        query_num,
        nprobe,
        HBM_in{i}, 
        s_start_addr_every_cell_replicated[{i}], 
        s_scanned_entries_every_cell_Load_unit_replicated[{i}], 
        s_scanned_entries_every_cell_Split_unit_replicated[{i}],
        s_single_PQ_per_channel[{i}]);""".format(i=i)

    if config["HBM_CHANNEL_NUM"] > config["STAGE5_COMP_PE_NUM"]:
        assert config["HBM_CHANNEL_NUM"] % config["STAGE5_COMP_PE_NUM"] == 0, \
            "when HBM Channel number > stage 5 PE number, HBM Channel number '%' stage 5 PE number must be 0"
    if config["HBM_CHANNEL_NUM"] / config["STAGE5_COMP_PE_NUM"] == 2:
        for i in range(config["STAGE5_COMP_PE_NUM"]):
            template_fill_dict["load_and_split_PQ_codes_wrapper_func_body"] += \
            """
    merge_HBM_channel_PQ_codes_2_in_1(
        query_num,
        nprobe,
        s_scanned_entries_every_cell_Merge_unit_replicated[{i}],
        s_single_PQ_per_channel[{i} * 2],
        s_single_PQ_per_channel[{i} * 2 + 1],
        s_single_PQ[{i}]);""".format(i=i)
    elif config["HBM_CHANNEL_NUM"] / config["STAGE5_COMP_PE_NUM"] == 3:
        for i in range(config["STAGE5_COMP_PE_NUM"]):
            template_fill_dict["load_and_split_PQ_codes_wrapper_func_body"] += \
            """
    merge_HBM_channel_PQ_codes_3_in_1(
        query_num,
        nprobe,
        s_scanned_entries_every_cell_Merge_unit_replicated[{i}],
        s_single_PQ_per_channel[{i} * 3],
        s_single_PQ_per_channel[{i} * 3 + 1],
        s_single_PQ_per_channel[{i} * 3 + 2],
        s_single_PQ[{i}]);""".format(i=i)
    elif config["HBM_CHANNEL_NUM"] / config["STAGE5_COMP_PE_NUM"] == 4:
        for i in range(config["STAGE5_COMP_PE_NUM"]):
            template_fill_dict["load_and_split_PQ_codes_wrapper_func_body"] += \
            """
    merge_HBM_channel_PQ_codes_4_in_1(
        query_num,
        nprobe,
        s_scanned_entries_every_cell_Merge_unit_replicated[{i}],
        s_single_PQ_per_channel[{i} * 4],
        s_single_PQ_per_channel[{i} * 4 + 1],
        s_single_PQ_per_channel[{i} * 4 + 2],
        s_single_PQ_per_channel[{i} * 4 + 3],
        s_single_PQ[{i}]);""".format(i=i)
    elif config["HBM_CHANNEL_NUM"] / config["STAGE5_COMP_PE_NUM"] == 5:
        for i in range(config["STAGE5_COMP_PE_NUM"]):
            template_fill_dict["load_and_split_PQ_codes_wrapper_func_body"] += \
            """
    merge_HBM_channel_PQ_codes_5_in_1(
        query_num,
        nprobe,
        s_scanned_entries_every_cell_Merge_unit_replicated[{i}],
        s_single_PQ_per_channel[{i} * 5],
        s_single_PQ_per_channel[{i} * 5 + 1],
        s_single_PQ_per_channel[{i} * 5 + 2],
        s_single_PQ_per_channel[{i} * 5 + 3],
        s_single_PQ_per_channel[{i} * 5 + 4],
        s_single_PQ[{i}]);""".format(i=i)
    elif config["HBM_CHANNEL_NUM"] / config["STAGE5_COMP_PE_NUM"] == 6:
        for i in range(config["STAGE5_COMP_PE_NUM"]):
            template_fill_dict["load_and_split_PQ_codes_wrapper_func_body"] += \
            """
    merge_HBM_channel_PQ_codes_6_in_1(
        query_num,
        nprobe,
        s_scanned_entries_every_cell_Merge_unit_replicated[{i}],
        s_single_PQ_per_channel[{i} * 6],
        s_single_PQ_per_channel[{i} * 6 + 1],
        s_single_PQ_per_channel[{i} * 6 + 2],
        s_single_PQ_per_channel[{i} * 6 + 3],
        s_single_PQ_per_channel[{i} * 6 + 4],
        s_single_PQ_per_channel[{i} * 6 + 5],
        s_single_PQ[{i}]);""".format(i=i)
    elif config["HBM_CHANNEL_NUM"] / config["STAGE5_COMP_PE_NUM"] == 7:
        for i in range(config["STAGE5_COMP_PE_NUM"]):
            template_fill_dict["load_and_split_PQ_codes_wrapper_func_body"] += \
            """
    merge_HBM_channel_PQ_codes_7_in_1(
        query_num,
        nprobe,
        s_scanned_entries_every_cell_Merge_unit_replicated[{i}],
        s_single_PQ_per_channel[{i} * 7],
        s_single_PQ_per_channel[{i} * 7 + 1],
        s_single_PQ_per_channel[{i} * 7 + 2],
        s_single_PQ_per_channel[{i} * 7 + 3],
        s_single_PQ_per_channel[{i} * 7 + 4],
        s_single_PQ_per_channel[{i} * 7 + 5],
        s_single_PQ_per_channel[{i} * 7 + 6],
        s_single_PQ[{i}]);""".format(i=i)
    elif config["HBM_CHANNEL_NUM"] / config["STAGE5_COMP_PE_NUM"] == 8:
        for i in range(config["STAGE5_COMP_PE_NUM"]):
            template_fill_dict["load_and_split_PQ_codes_wrapper_func_body"] += \
            """
    merge_HBM_channel_PQ_codes_8_in_1(
        query_num,
        nprobe,
        s_scanned_entries_every_cell_Merge_unit_replicated[{i}],
        s_single_PQ_per_channel[{i} * 8],
        s_single_PQ_per_channel[{i} * 8 + 1],
        s_single_PQ_per_channel[{i} * 8 + 2],
        s_single_PQ_per_channel[{i} * 8 + 3],
        s_single_PQ_per_channel[{i} * 8 + 4],
        s_single_PQ_per_channel[{i} * 8 + 5],
        s_single_PQ_per_channel[{i} * 8 + 6],
        s_single_PQ_per_channel[{i} * 8 + 7],
        s_single_PQ[{i}]);""".format(i=i)
    elif config["HBM_CHANNEL_NUM"] / config["STAGE5_COMP_PE_NUM"] == 9:
        for i in range(config["STAGE5_COMP_PE_NUM"]):
            template_fill_dict["load_and_split_PQ_codes_wrapper_func_body"] += \
            """
    merge_HBM_channel_PQ_codes_9_in_1(
        query_num,
        nprobe,
        s_scanned_entries_every_cell_Merge_unit_replicated[{i}],
        s_single_PQ_per_channel[{i} * 9],
        s_single_PQ_per_channel[{i} * 9 + 1],
        s_single_PQ_per_channel[{i} * 9 + 2],
        s_single_PQ_per_channel[{i} * 9 + 3],
        s_single_PQ_per_channel[{i} * 9 + 4],
        s_single_PQ_per_channel[{i} * 9 + 5],
        s_single_PQ_per_channel[{i} * 9 + 6],
        s_single_PQ_per_channel[{i} * 9 + 7],
        s_single_PQ_per_channel[{i} * 9 + 8],
        s_single_PQ[{i}]);""".format(i=i)
    elif config["HBM_CHANNEL_NUM"] / config["STAGE5_COMP_PE_NUM"] == 10:
        for i in range(config["STAGE5_COMP_PE_NUM"]):
            template_fill_dict["load_and_split_PQ_codes_wrapper_func_body"] += \
            """
    merge_HBM_channel_PQ_codes_10_in_1(
        query_num,
        nprobe,
        s_scanned_entries_every_cell_Merge_unit_replicated[{i}],
        s_single_PQ_per_channel[{i} * 10],
        s_single_PQ_per_channel[{i} * 10 + 1],
        s_single_PQ_per_channel[{i} * 10 + 2],
        s_single_PQ_per_channel[{i} * 10 + 3],
        s_single_PQ_per_channel[{i} * 10 + 4],
        s_single_PQ_per_channel[{i} * 10 + 5],
        s_single_PQ_per_channel[{i} * 10 + 6],
        s_single_PQ_per_channel[{i} * 10 + 7],
        s_single_PQ_per_channel[{i} * 10 + 8],
        s_single_PQ_per_channel[{i} * 10 + 9],
        s_single_PQ[{i}]);""".format(i=i)
    else:
        print("ERROR! This template does not support the case when STAGE5_COMP_PE_NUM / HBM_CHANNEL_NUM is not integer or > 4")
        raise ValueError
elif config["HBM_CHANNEL_NUM"] == config["STAGE5_COMP_PE_NUM"]:
    """
    If merging contents from 1 channel, (1 HBM channel = 1 PQ code stream), then
        1. no declarartion of s_scanned_entries_every_cell_Merge_unit_replicated, s_single_PQ_per_channel 
        2. replicate_s_scanned_entries_every_cell does not have s_scanned_entries_every_cell_Merge_unit_replicated argument
        3. load_and_split_PQ_codes's last argument is s_single_PQ 
        4. no merge_HBM_channel_PQ_codes_2_in_1 functions
    """
    template_fill_dict["load_and_split_PQ_codes_wrapper_func_body"] += """
    replicate_s_scanned_entries_every_cell(
        query_num,
        nprobe,
        s_scanned_entries_every_cell_Load_unit, 
        s_scanned_entries_every_cell_Load_unit_replicated,
        s_scanned_entries_every_cell_Split_unit_replicated); 
"""
    for i in range(config["HBM_CHANNEL_NUM"]):
        template_fill_dict["load_and_split_PQ_codes_wrapper_func_body"] += \
        """
    load_and_split_PQ_codes(
        query_num,
        nprobe,
        HBM_in{i}, 
        s_start_addr_every_cell_replicated[{i}], 
        s_scanned_entries_every_cell_Load_unit_replicated[{i}], 
        s_scanned_entries_every_cell_Split_unit_replicated[{i}],
        s_single_PQ[{i}]);""".format(i=i)

elif config["HBM_CHANNEL_NUM"] * 3 == config["STAGE5_COMP_PE_NUM"]:
    """
    If no merging at all (1 HBM channel = 3 PQ streams), then
        1. no declarartion of s_scanned_entries_every_cell_Merge_unit_replicated, s_single_PQ_per_channel 
        2. replicate_s_scanned_entries_every_cell does not have s_scanned_entries_every_cell_Merge_unit_replicated argument
        3. load_and_split_PQ_codes's last arguments are 3 s_single_PQ streams
        4. no merge_HBM_channel_PQ_codes_2_in_1 functions
    """
    template_fill_dict["load_and_split_PQ_codes_wrapper_func_body"] += """
    replicate_s_scanned_entries_every_cell(
        query_num,
        nprobe,
        s_scanned_entries_every_cell_Load_unit, 
        s_scanned_entries_every_cell_Load_unit_replicated,
        s_scanned_entries_every_cell_Split_unit_replicated); 
"""
    for i in range(config["HBM_CHANNEL_NUM"]):
        template_fill_dict["load_and_split_PQ_codes_wrapper_func_body"] += \
        """
    load_and_split_PQ_codes(
        query_num,
        nprobe,
        HBM_in{i}, 
        s_start_addr_every_cell_replicated[{i}], 
        s_scanned_entries_every_cell_Load_unit_replicated[{i}], 
        s_scanned_entries_every_cell_Split_unit_replicated[{i}],
        s_single_PQ[{i} * 3 + 0], 
        s_single_PQ[{i} * 3 + 1], 
        s_single_PQ[{i} * 3 + 2]);""".format(i=i)

else:
    print("ERROR! The relationship between STAGE5_COMP_PE_NUM and HBM_CHANNEL_NUM is wrong")
    raise ValueError
for k in template_fill_dict:
    template_str = template_str.replace("<--{}-->".format(k), str(template_fill_dict[k]))
output_str = template_str

# Save generated file
output_dir = os.path.join(args.output_dir, "HBM_interconnections.hpp")
with open(output_dir, "w+") as f:
    f.write(output_str)