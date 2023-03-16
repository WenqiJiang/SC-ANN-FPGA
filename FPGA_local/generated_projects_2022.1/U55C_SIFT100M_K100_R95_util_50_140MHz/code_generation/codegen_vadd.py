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
template_dir = os.path.join(args.input_dir, "vadd.cpp")
template_str = None
with open(template_dir) as f:
    template_str = f.read()

# Fill template
template_fill_dict = dict()

template_fill_dict["HBM_in_vadd_arg"] = ""
template_fill_dict["HBM_in_m_axi"] = ""
template_fill_dict["HBM_in_s_axilite"] = ""
template_fill_dict["load_and_split_PQ_codes_wrapper_arg"] = ""
for i in range(config["HBM_CHANNEL_NUM"]):
	template_fill_dict["HBM_in_vadd_arg"] += \
        "    const ap_uint512_t* HBM_in{i},\n".format(i=i)
	template_fill_dict["HBM_in_m_axi"] += \
        "#pragma HLS INTERFACE m_axi port=HBM_in{i} offset=slave bundle=gmem{i}\n".format(i=i)
	template_fill_dict["HBM_in_s_axilite"] += \
        "#pragma HLS INTERFACE s_axilite port=HBM_in{i}\n".format(i=i)
	template_fill_dict["load_and_split_PQ_codes_wrapper_arg"] += \
        "        HBM_in{i},\n".format(i=i)

if config["OPQ_ENABLE"]:
    template_fill_dict["OPQ_ENABLE"] = "true"
    template_fill_dict["stage_1_OPQ_preprocessing"] = """
    hls::stream<float> s_preprocessed_query_vectors;
#pragma HLS stream variable=s_preprocessed_query_vectors depth=512
// #pragma HLS resource variable=s_preprocessed_query_vectors core=FIFO_BRAM

    OPQ_preprocessing(
        query_num,
        s_OPQ_init,
        s_query_vectors,
        s_preprocessed_query_vectors);

    hls::stream<float> s_preprocessed_query_vectors_lookup_PE;
#pragma HLS stream variable=s_preprocessed_query_vectors_lookup_PE depth=512
// #pragma HLS resource variable=s_preprocessed_query_vectors_lookup_PE core=FIFO_BRAM

    hls::stream<float> s_preprocessed_query_vectors_distance_computation_PE;
#pragma HLS stream variable=s_preprocessed_query_vectors_distance_computation_PE depth=512
// #pragma HLS resource variable=s_preprocessed_query_vectors_distance_computation_PE core=FIFO_BRAM

    broadcast_preprocessed_query_vectors(
        query_num,
        s_preprocessed_query_vectors,
        s_preprocessed_query_vectors_distance_computation_PE,
        s_preprocessed_query_vectors_lookup_PE);"""
else:
    template_fill_dict["OPQ_ENABLE"] = "false"
    template_fill_dict["stage_1_OPQ_preprocessing"] = """
    hls::stream<float> s_preprocessed_query_vectors_lookup_PE;
#pragma HLS stream variable=s_preprocessed_query_vectors_lookup_PE depth=512
// #pragma HLS resource variable=s_preprocessed_query_vectors_lookup_PE core=FIFO_BRAM

    hls::stream<float> s_preprocessed_query_vectors_distance_computation_PE;
#pragma HLS stream variable=s_preprocessed_query_vectors_distance_computation_PE depth=512
// #pragma HLS resource variable=s_preprocessed_query_vectors_distance_computation_PE core=FIFO_BRAM

    broadcast_preprocessed_query_vectors(
        query_num,
        s_query_vectors,
        s_preprocessed_query_vectors_distance_computation_PE,
        s_preprocessed_query_vectors_lookup_PE);"""

# Stage 2 on-chip / off-chip replacement
template_fill_dict["stage2_vadd_arg"] = ""
template_fill_dict["stage2_m_axi"] = ""
template_fill_dict["stage2_s_axilite"] = ""
if config["STAGE2_ON_CHIP"] == True:
    if config["PE_NUM_CENTER_DIST_COMP"] > 1:
        template_fill_dict["stage_2_IVF_center_distance_computation"] = """
    compute_cell_distance_wrapper(
        query_num,
        centroids_per_partition_even, 
        centroids_per_partition_last_PE, 
        nlist,
        s_center_vectors_init_distance_computation_PE, 
        s_preprocessed_query_vectors_distance_computation_PE, 
        s_merged_cell_distance);"""
    elif config["PE_NUM_CENTER_DIST_COMP"] == 1:
        template_fill_dict["stage_2_IVF_center_distance_computation"] = """
    compute_cell_distance_wrapper(
        query_num,
        nlist,
        s_center_vectors_init_distance_computation_PE, 
        s_preprocessed_query_vectors_distance_computation_PE, 
        s_merged_cell_distance);"""
else:
    func_call_str = ""
    # 1 HBM channel per 2 PE
    HBM_channel_num_stage2 = int(np.ceil(config["PE_NUM_CENTER_DIST_COMP"] / 2.0))
    for i in range(HBM_channel_num_stage2):
        template_fill_dict["stage2_vadd_arg"] += \
            "    const ap_uint512_t* HBM_centroid_vectors_stage2_{i},\n".format(i=i)
        template_fill_dict["stage2_m_axi"] += \
            "#pragma HLS INTERFACE m_axi port=HBM_centroid_vectors_stage2_{i}  offset=slave bundle=gmemC{i}\n".format(i=i)
        template_fill_dict["stage2_s_axilite"] += \
            "#pragma HLS INTERFACE s_axilite port=HBM_centroid_vectors_stage2_{i}\n".format(i=i)
        func_call_str += "        HBM_centroid_vectors_stage2_{i},\n".format(i=i)
    if config["PE_NUM_CENTER_DIST_COMP"] > 1:
        template_fill_dict["stage_2_IVF_center_distance_computation"] = """
    compute_cell_distance_wrapper(
        query_num,
        centroids_per_partition_even, 
        centroids_per_partition_last_PE, 
        nlist,
{func_call_str}
        s_preprocessed_query_vectors_distance_computation_PE,
        s_merged_cell_distance);""".format(func_call_str=func_call_str)
    elif config["PE_NUM_CENTER_DIST_COMP"] == 1:
        template_fill_dict["stage_2_IVF_center_distance_computation"] = """
    compute_cell_distance_wrapper(
        query_num,
        nlist,
{func_call_str}
        s_preprocessed_query_vectors_distance_computation_PE,
        s_merged_cell_distance);""".format(func_call_str=func_call_str)

# stage 6
if config["SORT_GROUP_ENABLE"]:
    template_fill_dict["stage6_sort_reduction"] = """
        ////////////////////     Sort Results     ////////////////////    
    Sort_reduction<single_PQ_result, SORT_GROUP_NUM * 16, TOPK, Collect_smallest> sort_reduction_module;

    hls::stream<single_PQ_result> s_sorted_PQ_result[TOPK];
#pragma HLS stream variable=s_sorted_PQ_result depth=8
#pragma HLS array_partition variable=s_sorted_PQ_result complete
// #pragma HLS RESOURCE variable=s_sorted_PQ_result core=FIFO_SRL

    sort_reduction_module.sort_and_reduction(
        query_num,
        s_scanned_entries_per_query_Sort_and_reduction, 
        s_single_PQ_result, 
        s_sorted_PQ_result);"""
    template_fill_dict["stage6_priority_queue_group_L2_wrapper_arg"] = "        s_sorted_PQ_result," 
    template_fill_dict["stage6_priority_queue_group_L2_wrapper_stream_num"] = "TOPK"
else:
    template_fill_dict["stage6_sort_reduction"] = ""
    template_fill_dict["stage6_priority_queue_group_L2_wrapper_arg"] = "        s_single_PQ_result," 
    template_fill_dict["stage6_priority_queue_group_L2_wrapper_stream_num"] = "STAGE5_COMP_PE_NUM"

for k in template_fill_dict:
    template_str = template_str.replace("<--{}-->".format(k), str(template_fill_dict[k]))
output_str = template_str

# Save generated file
output_dir = os.path.join(args.output_dir, "vadd.cpp")
with open(output_dir, "w+") as f:
    f.write(output_str)