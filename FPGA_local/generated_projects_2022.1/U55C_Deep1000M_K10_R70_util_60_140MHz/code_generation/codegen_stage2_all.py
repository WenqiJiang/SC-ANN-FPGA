import argparse 
import os
import yaml
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default="./template_files/stage2_cluster_distance_computation", help="template input directory")
parser.add_argument('--output_dir', type=str, default="./output_files", help="output directory")
args = parser.parse_args()

# Load YAML configurations
config_file = open("config.yaml", "r")
config = yaml.load(config_file)

assert config["PE_NUM_CENTER_DIST_COMP"] <= 16, \
    "ERROR: Stage 2 PE number should be less than 16 (might be supported in the future), otherwise otherwise the computation will be faster than result forwarding. "
# select stage 2 on-chip / off-chip
if config["STAGE2_ON_CHIP"]:
    if config["PE_NUM_CENTER_DIST_COMP"] == 1:
        os.system("cp {i} {o}".format(
            i=os.path.join(args.input_dir, "on_chip_cluster_distance_computation_1_PE.hpp"),
            o=os.path.join(args.output_dir, "cluster_distance_computation.hpp")))
    else:
        os.system("cp {i} {o}".format(
            i=os.path.join(args.input_dir, "on_chip_cluster_distance_computation.hpp"),
            o=os.path.join(args.output_dir, "cluster_distance_computation.hpp")))
else:
    if config["PE_NUM_CENTER_DIST_COMP"] == 1:
        os.system("cp {i} {o}".format(
            i=os.path.join(args.input_dir, "off_chip_cluster_distance_computation_1_PE.hpp"),
            o=os.path.join(args.output_dir, "cluster_distance_computation.hpp")))
    else:
        template_dir = os.path.join(args.input_dir, "off_chip_cluster_distance_computation_with_HBM_FIFO_2_PE_per_HBM.hpp")
        template_str = None
        with open(template_dir) as f:
            template_str = f.read()

        # Fill template
        template_fill_dict = dict()

        # 1 HBM channel per 2 PE
        HBM_channel_num_stage2 = int(np.ceil(config["PE_NUM_CENTER_DIST_COMP"] / 2.0))
        template_fill_dict["compute_cell_distance_wrapper_arguments"] = "" 
        for i in range(HBM_channel_num_stage2):
            template_fill_dict["compute_cell_distance_wrapper_arguments"] += "    const ap_uint512_t* HBM_centroid_vectors_stage2_{i},\n".format(i=i)

        template_fill_dict["load_vector_quantizer_from_HBM"] = ""
        if config["PE_NUM_CENTER_DIST_COMP"] % 2 == 0:
            # centroids_per_partition_even + centroids_per_partition_even
            for i in range(HBM_channel_num_stage2 - 1):
                template_fill_dict["load_vector_quantizer_from_HBM"] += """
    int start_cell_ID_load_{i} = {i} * 2 * centroids_per_partition_even;
    load_vector_quantizer_from_HBM(
        query_num,
        start_cell_ID_load_{i},
        centroids_per_partition_even,
        centroids_per_partition_even,
        HBM_centroid_vectors_stage2_{i},
        s_HBM_centroid_vectors[{i} * 2],
        s_HBM_centroid_vectors[{i} * 2 + 1],
        s_cell_ID[{i} * 2],
        s_cell_ID[{i} * 2 + 1]);
""".format(i=i)
            # last PE: centroids_per_partition_even + centroids_per_partition_last_PE
            template_fill_dict["load_vector_quantizer_from_HBM"] += """
    int start_cell_ID_load_{i} = {i} * 2 * centroids_per_partition_even;
    load_vector_quantizer_from_HBM(
        query_num,
        start_cell_ID_load_{i},
        centroids_per_partition_even,
        centroids_per_partition_last_PE,
        HBM_centroid_vectors_stage2_{i},
        s_HBM_centroid_vectors[{i} * 2],
        s_HBM_centroid_vectors[{i} * 2 + 1],
        s_cell_ID[{i} * 2],
        s_cell_ID[{i} * 2 + 1]);
""".format(i=HBM_channel_num_stage2 - 1)
        else: # 2n * even + last
            for i in range(HBM_channel_num_stage2 - 1):
                template_fill_dict["load_vector_quantizer_from_HBM"] += """
    int start_cell_ID_load_{i} = {i} * 2 * centroids_per_partition_even;
    load_vector_quantizer_from_HBM(
        query_num,
        start_cell_ID_load_{i},
        centroids_per_partition_even,
        centroids_per_partition_even,
        HBM_centroid_vectors_stage2_{i},
        s_HBM_centroid_vectors[{i} * 2],
        s_HBM_centroid_vectors[{i} * 2 + 1],
        s_cell_ID[{i} * 2],
        s_cell_ID[{i} * 2 + 1]);
""".format(i=i)
            # last PE: centroids_per_partition_last_PE
            template_fill_dict["load_vector_quantizer_from_HBM"] += """
    int start_cell_ID_load_{i} = {i} * 2 * centroids_per_partition_even;
    load_vector_quantizer_from_HBM(
        query_num,
        start_cell_ID_load_{i},
        centroids_per_partition_last_PE,
        HBM_centroid_vectors_stage2_{i},
        s_HBM_centroid_vectors[{i} * 2],
        s_cell_ID[{i} * 2]);
""".format(i=HBM_channel_num_stage2 - 1)


        template_fill_dict["compute_cell_distance_wrapper_func_body"] = ""
        for i in range(1, config["PE_NUM_CENTER_DIST_COMP"] - 1):
            if i == 1:
                template_fill_dict["compute_cell_distance_wrapper_func_body"] += "    // middle"
            template_fill_dict["compute_cell_distance_wrapper_func_body"] += """
        compute_cell_distance_middle_PE(
            query_num,
            {i},
            centroids_per_partition, 
            total_centriods,
            s_HBM_centroid_vectors[{i}],
            s_cell_ID[{i}],
            s_query_vectors_forward[{i} - 1],
            s_query_vectors_forward[{i}],
            s_partial_cell_distance_forward[{i} - 1],
            s_partial_cell_distance_forward[{i}]);\n""".format(i=i)
        
        template_fill_dict["compute_cell_distance_wrapper_func_body"] += """    
        // tail
        compute_cell_distance_tail_PE(
            query_num,
            PE_NUM_CENTER_DIST_COMP_EVEN,
            centroids_per_partition_even, 
            centroids_per_partition_last_PE, 
            total_centriods,
            s_HBM_centroid_vectors[{i}],
            s_cell_ID[{i}],
            s_query_vectors_forward[PE_NUM_CENTER_DIST_COMP_EVEN - 1],
            s_partial_cell_distance_forward[PE_NUM_CENTER_DIST_COMP_EVEN - 1],
            s_cell_distance);""".format(i=config["PE_NUM_CENTER_DIST_COMP"] - 1)

        for k in template_fill_dict:
            template_str = template_str.replace("<--{}-->".format(k), str(template_fill_dict[k]))
        output_str = template_str

        # Save generated file
        output_dir = os.path.join(args.output_dir, "cluster_distance_computation.hpp")
        with open(output_dir, "w+") as f:
            f.write(output_str)
