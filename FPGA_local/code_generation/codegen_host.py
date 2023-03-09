import argparse
from multiprocessing.sharedctypes import Value 
import os
import yaml
import numpy as np

from get_channel_id import ChannelIterator

parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', type=str, default="./template_files", help="template input directory")
parser.add_argument('--output_dir', type=str, default="./output_files", help="output directory")
parser.add_argument('--vitis_version', type=str, default="2020.2", help="support 2020.2 and 2021.2")
args = parser.parse_args()

# Load YAML configurations
config_file = open("config.yaml", "r")
config = yaml.load(config_file)

if args.vitis_version == '2020.2':
    # Load template
    template_dir = os.path.join(args.input_dir, "host_2020.2.cpp")
    template_str = None
    with open(template_dir) as f:
        template_str = f.read()

    # Fill template
    template_fill_dict = dict()

    if config["DEVICE"] == "U280":

        # channel 30, 31 left unused; 3 = meta info + quantizer + output
        channel_in_use = config["HBM_CHANNEL_NUM"] + 3
        if not config["STAGE2_ON_CHIP"]:
            channel_in_use += int(np.ceil(config["PE_NUM_CENTER_DIST_COMP"] / 2.0))
        channel_iterator = ChannelIterator(total_channel_num=30, channel_in_use=channel_in_use)

        template_fill_dict["bank_topology"] = """
    /* for U280 specifically */
    const int bank[40] = {
        /* 0 ~ 31 HBM (256MB per channel) */
        BANK_NAME(0),  BANK_NAME(1),  BANK_NAME(2),  BANK_NAME(3),  BANK_NAME(4),
        BANK_NAME(5),  BANK_NAME(6),  BANK_NAME(7),  BANK_NAME(8),  BANK_NAME(9),
        BANK_NAME(10), BANK_NAME(11), BANK_NAME(12), BANK_NAME(13), BANK_NAME(14),
        BANK_NAME(15), BANK_NAME(16), BANK_NAME(17), BANK_NAME(18), BANK_NAME(19),
        BANK_NAME(20), BANK_NAME(21), BANK_NAME(22), BANK_NAME(23), BANK_NAME(24),
        BANK_NAME(25), BANK_NAME(26), BANK_NAME(27), BANK_NAME(28), BANK_NAME(29),
        BANK_NAME(30), BANK_NAME(31), 
        /* 32, 33 DDR (16GB per channel) */ 
        BANK_NAME(32), BANK_NAME(33), 
        /* 34 ~ 39 PLRAM */ 
        BANK_NAME(34), BANK_NAME(35), BANK_NAME(36), BANK_NAME(37), 
        BANK_NAME(38), BANK_NAME(39)};
        """
    elif config["DEVICE"] == "U250":

        channel_in_use = config["HBM_CHANNEL_NUM"]
        if not config["STAGE2_ON_CHIP"]:
            channel_in_use += int(np.ceil(config["PE_NUM_CENTER_DIST_COMP"] / 2.0))
        channel_iterator = ChannelIterator(total_channel_num=4, channel_in_use=channel_in_use)

        template_fill_dict["bank_topology"] = """
    /* for U250 specifically */
    const int bank[8] = {
        /* 0 ~ 3 DDR (16GB per channel) */
        BANK_NAME(0),  BANK_NAME(1),  BANK_NAME(2),  BANK_NAME(3), 
        /* 4 ~ 7 PLRAM */ 
        BANK_NAME(4), BANK_NAME(5), BANK_NAME(6), BANK_NAME(7)};
        """
    elif config["DEVICE"] == "U50":
        
        # channel 30, 31 left unused; 3 = meta info + quantizer + output
        channel_in_use = config["HBM_CHANNEL_NUM"] + 3
        if not config["STAGE2_ON_CHIP"]:
            channel_in_use += int(np.ceil(config["PE_NUM_CENTER_DIST_COMP"] / 2.0))
        channel_iterator = ChannelIterator(total_channel_num=30, channel_in_use=channel_in_use)

        template_fill_dict["bank_topology"] = """
    /* for U50 specifically */
    const int bank[36] = {
        /* 0 ~ 31 HBM (256MB per channel) */
        BANK_NAME(0),  BANK_NAME(1),  BANK_NAME(2),  BANK_NAME(3),  BANK_NAME(4),
        BANK_NAME(5),  BANK_NAME(6),  BANK_NAME(7),  BANK_NAME(8),  BANK_NAME(9),
        BANK_NAME(10), BANK_NAME(11), BANK_NAME(12), BANK_NAME(13), BANK_NAME(14),
        BANK_NAME(15), BANK_NAME(16), BANK_NAME(17), BANK_NAME(18), BANK_NAME(19),
        BANK_NAME(20), BANK_NAME(21), BANK_NAME(22), BANK_NAME(23), BANK_NAME(24),
        BANK_NAME(25), BANK_NAME(26), BANK_NAME(27), BANK_NAME(28), BANK_NAME(29),
        BANK_NAME(30), BANK_NAME(31), 
        /* 32 ~ 35 PLRAM */ 
        BANK_NAME(32), BANK_NAME(33), BANK_NAME(34), BANK_NAME(35)};
        """
    else:
        print("Unsupported device! Supported model: U280/U250/U50")
        raise ValueError


    template_fill_dict["QUERY_NUM"] = str(config["QUERY_NUM"])
    template_fill_dict["D"] = str(config["D"])
    template_fill_dict["M"] = str(config["M"])
    template_fill_dict["HBM_CHANNEL_NUM"] = str(config["HBM_CHANNEL_NUM"])

    template_fill_dict["HBM_embedding_len"] = ""
    template_fill_dict["HBM_embedding_size"] = ""
    template_fill_dict["HBM_embedding_allocate"] = ""
    template_fill_dict["HBM_embedding_char"] = ""
    template_fill_dict["HBM_embedding_fstream"] = ""
    template_fill_dict["HBM_embedding_fstream_read"] = ""
    template_fill_dict["HBM_embedding_memcpy"] = ""
    template_fill_dict["HBM_embedding_char_free"] = ""
    template_fill_dict["HBM_embeddingExt"] = ""
    template_fill_dict["HBM_embeddingExt_set"] = ""
    template_fill_dict["buffer_HBM_embedding"] = ""
    template_fill_dict["buffer_HBM_embedding_set_krnl_arg"] = ""
    template_fill_dict["buffer_HBM_embedding_enqueueMigrateMemObjects"] = ""
    for i in range(config["HBM_CHANNEL_NUM"]):
        template_fill_dict["HBM_embedding_size"] += '''
        HBM_embedding{i}_fstream.seekg(0, HBM_embedding{i}_fstream.end);
        size_t HBM_embedding{i}_size =  HBM_embedding{i}_fstream.tellg();
        if (!HBM_embedding{i}_size) std::cout << "HBM_embedding{i}_size is 0!";
        HBM_embedding{i}_fstream.seekg(0, HBM_embedding{i}_fstream.beg);'''.format(i=i)
        template_fill_dict["HBM_embedding_len"] += \
            "    size_t HBM_embedding{i}_len = (int) (HBM_embedding{i}_size / sizeof(ap_uint512_t));\n".format(i=i)
        template_fill_dict["HBM_embedding_allocate"] += \
            "    std::vector<ap_uint512_t, aligned_allocator<ap_uint512_t>> HBM_embedding{i}(HBM_embedding{i}_len, 0);\n".format(i=i)
        template_fill_dict["HBM_embedding_char"] += \
            "    char* HBM_embedding{i}_char = (char*) malloc(HBM_embedding{i}_size);\n".format(i=i)
        template_fill_dict["HBM_embedding_fstream"] += \
            '''    
        std::string HBM_embedding{i}_dir_suffix("HBM_bank_{i}_raw");
        std::string HBM_embedding{i}_dir = dir_concat(data_dir_prefix, HBM_embedding{i}_dir_suffix);
        std::ifstream HBM_embedding{i}_fstream(
            HBM_embedding{i}_dir, 
            std::ios::in | std::ios::binary);\n'''.format(i=i)
        template_fill_dict["HBM_embedding_fstream_read"] += \
            '''    HBM_embedding{i}_fstream.read(HBM_embedding{i}_char, HBM_embedding{i}_size);
        if (!HBM_embedding{i}_fstream) '''.format(i=i) + '{\n' + \
    '            std::cout << "error: only "' + ''' << HBM_embedding{i}_fstream.gcount() << " could be read";
            exit(1);\n '''.format(i=i) + '    }\n'
        template_fill_dict["HBM_embedding_memcpy"] += \
            "    memcpy(&HBM_embedding{i}[0], HBM_embedding{i}_char, HBM_embedding{i}_size);\n".format(i=i)
        template_fill_dict["HBM_embedding_char_free"] += \
            "    free(HBM_embedding{i}_char);\n".format(i=i)
        template_fill_dict["HBM_embeddingExt"] += \
            "        HBM_embedding{i}Ext,\n".format(i=i)
        template_fill_dict["HBM_embeddingExt_set"] += \
            '''    HBM_embedding{i}Ext.obj = HBM_embedding{i}.data();
        HBM_embedding{i}Ext.param = 0;
        HBM_embedding{i}Ext.flags = bank[{c}];\n'''.format(i=i, c=channel_iterator.get_next_channel_id())
        template_fill_dict["buffer_HBM_embedding"] += \
            '''    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding{i}(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
                HBM_embedding{i}_size, &HBM_embedding{i}Ext, &err));\n'''.format(i=i)
        template_fill_dict["buffer_HBM_embedding_set_krnl_arg"] += \
            "    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_HBM_embedding{i}));\n".format(i=i)
        template_fill_dict["buffer_HBM_embedding_enqueueMigrateMemObjects"] += \
            "        buffer_HBM_embedding{i},\n".format(i=i)

        template_fill_dict["HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_fstream"] = \
            '''  
        std::string HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_dir_suffix = 
            "HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_3_by_" + std::to_string(nlist) + "_raw";
        std::string HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_dir = 
            dir_concat(data_dir_prefix, HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_dir_suffix);
        std::ifstream HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_fstream(
            HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_dir, 
            std::ios::in | std::ios::binary);\n'''
        template_fill_dict["HBM_query_vector_fstream"] = \
            '''
        std::string HBM_query_vector_dir_suffix = "query_vectors_float32_{QUERY_NUM}_{D}_raw";
        std::string HBM_query_vector_path = dir_concat(data_dir_prefix, HBM_query_vector_dir_suffix);
        std::ifstream HBM_query_vector_fstream(
            HBM_query_vector_path,
            std::ios::in | std::ios::binary);\n'''.format(QUERY_NUM=config["QUERY_NUM"], D=config["D"])
        template_fill_dict["HBM_vector_quantizer_fstream"] = \
            '''    
        std::string HBM_vector_quantizer_dir_suffix = "vector_quantizer_float32_" + std::to_string(nlist) + "_{D}_raw";
        std::string HBM_vector_quantizer_dir = dir_concat(data_dir_prefix, HBM_vector_quantizer_dir_suffix);
        std::ifstream HBM_vector_quantizer_fstream(
            HBM_vector_quantizer_dir, 
            std::ios::in | std::ios::binary);\n'''.format(D=config["D"])
        template_fill_dict["HBM_product_quantizer_fstream"] = \
            '''    
        std::string HBM_product_quantizer_suffix_dir = "product_quantizer_float32_{M}_{K}_{PARTITION}_raw";
        std::string HBM_product_quantizer_dir = dir_concat(data_dir_prefix, HBM_product_quantizer_suffix_dir);
        std::ifstream HBM_product_quantizer_fstream(
            HBM_product_quantizer_dir,
            std::ios::in | std::ios::binary);\n'''.format(
                M=config["M"], K=config["K"],  PARTITION=int(config["D"]/config["M"]))
        template_fill_dict["HBM_OPQ_matrix_fstream"] = \
            '''    
        std::string HBM_OPQ_matrix_suffix_dir = "OPQ_matrix_float32_{D}_{D}_raw";
        std::string HBM_OPQ_matrix_dir = dir_concat(data_dir_prefix, HBM_OPQ_matrix_suffix_dir);
        std::ifstream HBM_OPQ_matrix_fstream(
            HBM_OPQ_matrix_dir,
            std::ios::in | std::ios::binary);\n'''.format(D=config["D"])
        template_fill_dict["raw_gt_vec_ID_fstream"] = '''
        std::string raw_gt_vec_ID_suffix_dir = "idx_{DB_SCALE}.ivecs";
        std::string raw_gt_vec_ID_dir = dir_concat(gnd_dir, raw_gt_vec_ID_suffix_dir);
        std::ifstream raw_gt_vec_ID_fstream(
            raw_gt_vec_ID_dir,
            std::ios::in | std::ios::binary);\n'''.format(DB_SCALE=config["DB_SCALE"]) + \
    '''    if (!raw_gt_vec_ID_fstream) {\n''' + \
    '''        std::cout << "error: only " << raw_gt_vec_ID_fstream.gcount() << " could be read";
            exit(1);\n''' + '}\n'

    bytes_float = 4
    bytes_ap512 = 64

    template_fill_dict["HBM_centroid_vectors_stage2_len"] = ""
    template_fill_dict["HBM_centroid_vectors_stage2_size"] = ""
    template_fill_dict["HBM_centroid_vectors_stage2_allocate"] = ""
    template_fill_dict["HBM_centroid_vectors_stage2_memcpy"] = ""
    template_fill_dict["HBM_centroid_vectorsExt"] = ""
    template_fill_dict["HBM_centroid_vectorsExt_set"] = ""
    template_fill_dict["HBM_metainfoExt_set"] = ""
    template_fill_dict["buffer_HBM_centroid_vectors"] = ""
    template_fill_dict["buffer_HBM_centroid_vectors_stage2_set_krnl_arg"] = ""
    template_fill_dict["buffer_HBM_centroid_vectors_stage2_enqueueMigrateMemObjects"] = ""
    if config["STAGE2_ON_CHIP"] == False:
        # 1 HBM channel per 2 PE
        HBM_channel_num_stage2 = int(np.ceil(config["PE_NUM_CENTER_DIST_COMP"] / 2.0))
        if config["PE_NUM_CENTER_DIST_COMP"] % 2 == 0:
            for i in range(HBM_channel_num_stage2 - 1):
                template_fill_dict["HBM_centroid_vectors_stage2_len"] += \
                    "    size_t HBM_centroid_vectors{i}_len = 2 * centroids_per_partition_even * D * sizeof(float) / sizeof(ap_uint512_t);\n".format(
                        i=i)
            template_fill_dict["HBM_centroid_vectors_stage2_len"] += \
                "    size_t HBM_centroid_vectors{i}_len = (centroids_per_partition_even + centroids_per_partition_last_PE) * D * sizeof(float) / sizeof(ap_uint512_t);\n".format(
                    i=HBM_channel_num_stage2 - 1)
        else: # % 2 == 1
            for i in range(HBM_channel_num_stage2 - 1):
                template_fill_dict["HBM_centroid_vectors_stage2_len"] += \
                    "    size_t HBM_centroid_vectors{i}_len = 2 * centroids_per_partition_even * D * sizeof(float) / sizeof(ap_uint512_t);\n".format(
                        i=i)
            template_fill_dict["HBM_centroid_vectors_stage2_len"] += \
                "    size_t HBM_centroid_vectors{i}_len = centroids_per_partition_last_PE * D * sizeof(float) / sizeof(ap_uint512_t);\n".format(
                    i=HBM_channel_num_stage2 - 1)
        for i in range(HBM_channel_num_stage2):
            template_fill_dict["HBM_centroid_vectors_stage2_size"] += \
                "    size_t HBM_centroid_vectors{i}_size =  HBM_centroid_vectors{i}_len * sizeof(ap_uint512_t);\n".format(i=i)
            template_fill_dict["HBM_centroid_vectors_stage2_allocate"] += \
                "    std::vector<ap_uint512_t, aligned_allocator<ap_uint512_t>> HBM_centroid_vectors{i}(HBM_centroid_vectors{i}_len, 0);\n".format(i=i)
            template_fill_dict["HBM_centroid_vectors_stage2_memcpy"] += """
        int HBM_centroid_vectors_stage2_start_addr_{i} = 2 * {i} * centroids_per_partition_even * D * sizeof(float);
        memcpy(&HBM_centroid_vectors{i}[0], HBM_vector_quantizer_char + HBM_centroid_vectors_stage2_start_addr_{i}, HBM_centroid_vectors{i}_size);\n""".format(i=i)
            template_fill_dict["HBM_centroid_vectorsExt"] += \
                "        HBM_centroid_vectors{i}Ext,\n".format(i=i)
            template_fill_dict["HBM_centroid_vectorsExt_set"] += \
                '''    HBM_centroid_vectors{i}Ext.obj = HBM_centroid_vectors{i}.data();
        HBM_centroid_vectors{i}Ext.param = 0;
        HBM_centroid_vectors{i}Ext.flags = bank[{c}];\n'''.format(i=i, c=channel_iterator.get_next_channel_id())
            template_fill_dict["buffer_HBM_centroid_vectors"] += \
                '''    OCL_CHECK(err, cl::Buffer buffer_HBM_centroid_vectors{i}(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY | CL_MEM_EXT_PTR_XILINX, 
                HBM_centroid_vectors{i}_size, &HBM_centroid_vectors{i}Ext, &err));\n'''.format(i=i)
            template_fill_dict["buffer_HBM_centroid_vectors_stage2_set_krnl_arg"] += \
                "    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_HBM_centroid_vectors{i}));\n".format(i=i)
            template_fill_dict["buffer_HBM_centroid_vectors_stage2_enqueueMigrateMemObjects"] += \
                "        buffer_HBM_centroid_vectors{i},\n".format(i=i)

    if config["DEVICE"] == "U280" or config["DEVICE"] == "U50":
        template_fill_dict["HBM_metainfoExt_set"] = """
        // HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_validExt.obj = 
        //     HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid.data();
        // HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_validExt.param = 0;
        // HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_validExt.flags = bank[21];

        // HBM_query_vectorExt.obj = HBM_query_vectors.data();
        // HBM_query_vectorExt.param = 0;
        // HBM_query_vectorExt.flags = bank[22];

        HBM_meta_infoExt.obj = HBM_meta_info.data();
        HBM_vector_quantizerExt.param = 0;
        HBM_vector_quantizerExt.flags = bank[{c0}];

        HBM_vector_quantizerExt.obj = HBM_vector_quantizer.data();
        HBM_vector_quantizerExt.param = 0;
        HBM_vector_quantizerExt.flags = bank[{c1}];

        // HBM_product_quantizerExt.obj = HBM_product_quantizer.data();
        // HBM_product_quantizerExt.param = 0;
        // HBM_product_quantizerExt.flags = bank[24];

    // #ifdef OPQ_ENABLE
        // HBM_OPQ_matrixExt.obj = HBM_OPQ_matrix.data();
        // HBM_OPQ_matrixExt.param = 0;
        // HBM_OPQ_matrixExt.flags = bank[25];
    // #endif

        HBM_outExt.obj = HBM_out.data();
        HBM_outExt.param = 0;
        HBM_outExt.flags = bank[{c2}];
        """.format(
            c0=channel_iterator.get_next_channel_id(),
            c1=channel_iterator.get_next_channel_id(),
            c2=channel_iterator.get_next_channel_id())
    elif config["DEVICE"] == "U250":
        template_fill_dict["HBM_metainfoExt_set"] = """
        HBM_meta_infoExt.obj = HBM_meta_info.data();
        HBM_vector_quantizerExt.param = 0;
        HBM_vector_quantizerExt.flags = bank[0];

        HBM_vector_quantizerExt.obj = HBM_vector_quantizer.data();
        HBM_vector_quantizerExt.param = 0;
        HBM_vector_quantizerExt.flags = bank[1];
        
        HBM_outExt.obj = HBM_out.data();
        HBM_outExt.param = 0;
        HBM_outExt.flags = bank[2];
        """

    if config["OPQ_ENABLE"]:
        template_fill_dict["OPQ_ENABLE"] = "true"
    else:
        template_fill_dict["OPQ_ENABLE"] = "false"

    for k in template_fill_dict:
        template_str = template_str.replace("<--{}-->".format(k), str(template_fill_dict[k]))
    output_str = template_str

    # Save generated file
    output_dir = os.path.join(args.output_dir, "host.cpp")
    with open(output_dir, "w+") as f:
        f.write(output_str)

elif args.vitis_version == '2021.2':
    # Load template
    template_dir = os.path.join(args.input_dir, "host_2021.2.cpp")
    template_str = None
    with open(template_dir) as f:
        template_str = f.read()

    # Fill template
    template_fill_dict = dict()

    if config["DB_NAME"].startswith('SIFT'):
        template_fill_dict["RAW_gt_vec_ID_len"] = "    size_t raw_gt_vec_ID_len = 10000 * 1001; "
        template_fill_dict["TOP1_ID"] = "        gt_vec_ID[i] = raw_gt_vec_ID[i * 1001 + 1];"
    elif config["DB_NAME"].startswith('Deep'):
        template_fill_dict["RAW_gt_vec_ID_len"] = "    size_t raw_gt_vec_ID_len = 10000 * 1000; "
        template_fill_dict["TOP1_ID"] = "        gt_vec_ID[i] = raw_gt_vec_ID[2 + i * 1000];"
    else:
        print("Unknown dataset")
        raise ValueError

    if config["DEVICE"] == "U280":

        # channel 30, 31 left unused; 3 = meta info + quantizer + output
        channel_in_use = config["HBM_CHANNEL_NUM"] + 3
        if not config["STAGE2_ON_CHIP"]:
            channel_in_use += int(np.ceil(config["PE_NUM_CENTER_DIST_COMP"] / 2.0))
        channel_iterator = ChannelIterator(total_channel_num=30, channel_in_use=channel_in_use)

    elif config["DEVICE"] == "U250":

        channel_in_use = config["HBM_CHANNEL_NUM"]
        if not config["STAGE2_ON_CHIP"]:
            channel_in_use += int(np.ceil(config["PE_NUM_CENTER_DIST_COMP"] / 2.0))
        channel_iterator = ChannelIterator(total_channel_num=4, channel_in_use=channel_in_use)

    elif config["DEVICE"] == "U50":
        
        # channel 30, 31 left unused; 3 = meta info + quantizer + output
        channel_in_use = config["HBM_CHANNEL_NUM"] + 3
        if not config["STAGE2_ON_CHIP"]:
            channel_in_use += int(np.ceil(config["PE_NUM_CENTER_DIST_COMP"] / 2.0))
        channel_iterator = ChannelIterator(total_channel_num=30, channel_in_use=channel_in_use)

    else:
        print("Unsupported device! Supported model: U280/U250/U50")
        raise ValueError


    template_fill_dict["QUERY_NUM"] = str(config["QUERY_NUM"])
    template_fill_dict["D"] = str(config["D"])
    template_fill_dict["M"] = str(config["M"])
    template_fill_dict["HBM_CHANNEL_NUM"] = str(config["HBM_CHANNEL_NUM"])

    template_fill_dict["HBM_embedding_len"] = ""
    template_fill_dict["HBM_embedding_size"] = ""
    template_fill_dict["HBM_embedding_allocate"] = ""
    template_fill_dict["HBM_embedding_char"] = ""
    template_fill_dict["HBM_embedding_fstream"] = ""
    template_fill_dict["HBM_embedding_fstream_read"] = ""
    template_fill_dict["HBM_embedding_memcpy"] = ""
    template_fill_dict["HBM_embeddingExt"] = ""
    template_fill_dict["HBM_embeddingExt_set"] = ""
    template_fill_dict["buffer_HBM_embedding"] = ""
    template_fill_dict["buffer_HBM_embedding_set_krnl_arg"] = ""
    template_fill_dict["buffer_HBM_embedding_enqueueMigrateMemObjects"] = ""
    for i in range(config["HBM_CHANNEL_NUM"]):
        template_fill_dict["HBM_embedding_size"] += '''
        HBM_embedding{i}_fstream.seekg(0, HBM_embedding{i}_fstream.end);
        size_t HBM_embedding{i}_size =  HBM_embedding{i}_fstream.tellg();
        if (!HBM_embedding{i}_size) std::cout << "HBM_embedding{i}_size is 0!";
        HBM_embedding{i}_fstream.seekg(0, HBM_embedding{i}_fstream.beg);'''.format(i=i)
        template_fill_dict["HBM_embedding_len"] += \
            "    size_t HBM_embedding{i}_len = (int) (HBM_embedding{i}_size / sizeof(uint32_t));\n".format(i=i)
        template_fill_dict["HBM_embedding_allocate"] += \
            "    std::vector<uint32_t, aligned_allocator<uint32_t>> HBM_embedding{i}(HBM_embedding{i}_len, 0);\n".format(i=i)
        template_fill_dict["HBM_embedding_char"] += \
            "    char* HBM_embedding{i}_char = (char*) malloc(HBM_embedding{i}_size);\n".format(i=i)
        template_fill_dict["HBM_embedding_fstream"] += \
            '''    
        std::string HBM_embedding{i}_dir_suffix("HBM_bank_{i}_raw");
        std::string HBM_embedding{i}_dir = dir_concat(data_dir_prefix, HBM_embedding{i}_dir_suffix);
        std::ifstream HBM_embedding{i}_fstream(
            HBM_embedding{i}_dir, 
            std::ios::in | std::ios::binary);\n'''.format(i=i)
        template_fill_dict["HBM_embedding_fstream_read"] += \
            '''    HBM_embedding{i}_fstream.read(HBM_embedding{i}_char, HBM_embedding{i}_size);
        if (!HBM_embedding{i}_fstream) '''.format(i=i) + '{\n' + \
    '            std::cout << "error: only "' + ''' << HBM_embedding{i}_fstream.gcount() << " could be read";
            exit(1);\n '''.format(i=i) + '    }\n'
        template_fill_dict["HBM_embedding_memcpy"] += \
            "    memcpy(&HBM_embedding{i}[0], HBM_embedding{i}_char, HBM_embedding{i}_size);\n".format(i=i)
        template_fill_dict["HBM_embeddingExt"] += \
            "        HBM_embedding{i}Ext,\n".format(i=i)
        template_fill_dict["HBM_embeddingExt_set"] += \
            '''    HBM_embedding{i}Ext.obj = HBM_embedding{i}.data();
        HBM_embedding{i}Ext.param = 0;
        HBM_embedding{i}Ext.flags = bank[{c}];\n'''.format(i=i, c=channel_iterator.get_next_channel_id())
        template_fill_dict["buffer_HBM_embedding"] += \
            '''    OCL_CHECK(err, cl::Buffer buffer_HBM_embedding{i}(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                HBM_embedding{i}_size, HBM_embedding{i}.data(), &err));\n'''.format(i=i)
        template_fill_dict["buffer_HBM_embedding_set_krnl_arg"] += \
            "    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_HBM_embedding{i}));\n".format(i=i)
        template_fill_dict["buffer_HBM_embedding_enqueueMigrateMemObjects"] += \
            "        buffer_HBM_embedding{i},\n".format(i=i)

        template_fill_dict["HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_fstream"] = \
            '''  
        std::string HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_dir_suffix = 
            "HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_3_by_" + std::to_string(nlist) + "_raw";
        std::string HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_dir = 
            dir_concat(data_dir_prefix, HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_dir_suffix);
        std::ifstream HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_fstream(
            HBM_info_start_addr_and_scanned_entries_every_cell_and_last_element_valid_dir, 
            std::ios::in | std::ios::binary);\n'''
        template_fill_dict["HBM_query_vector_fstream"] = \
            '''
        std::string HBM_query_vector_dir_suffix = "query_vectors_float32_{QUERY_NUM}_{D}_raw";
        std::string HBM_query_vector_path = dir_concat(data_dir_prefix, HBM_query_vector_dir_suffix);
        std::ifstream HBM_query_vector_fstream(
            HBM_query_vector_path,
            std::ios::in | std::ios::binary);\n'''.format(QUERY_NUM=config["QUERY_NUM"], D=config["D"])
        template_fill_dict["HBM_vector_quantizer_fstream"] = \
            '''    
        std::string HBM_vector_quantizer_dir_suffix = "vector_quantizer_float32_" + std::to_string(nlist) + "_{D}_raw";
        std::string HBM_vector_quantizer_dir = dir_concat(data_dir_prefix, HBM_vector_quantizer_dir_suffix);
        std::ifstream HBM_vector_quantizer_fstream(
            HBM_vector_quantizer_dir, 
            std::ios::in | std::ios::binary);\n'''.format(D=config["D"])
        template_fill_dict["HBM_product_quantizer_fstream"] = \
            '''    
        std::string HBM_product_quantizer_suffix_dir = "product_quantizer_float32_{M}_{K}_{PARTITION}_raw";
        std::string HBM_product_quantizer_dir = dir_concat(data_dir_prefix, HBM_product_quantizer_suffix_dir);
        std::ifstream HBM_product_quantizer_fstream(
            HBM_product_quantizer_dir,
            std::ios::in | std::ios::binary);\n'''.format(
                M=config["M"], K=config["K"],  PARTITION=int(config["D"]/config["M"]))
        template_fill_dict["HBM_OPQ_matrix_fstream"] = \
            '''    
        std::string HBM_OPQ_matrix_suffix_dir = "OPQ_matrix_float32_{D}_{D}_raw";
        std::string HBM_OPQ_matrix_dir = dir_concat(data_dir_prefix, HBM_OPQ_matrix_suffix_dir);
        std::ifstream HBM_OPQ_matrix_fstream(
            HBM_OPQ_matrix_dir,
            std::ios::in | std::ios::binary);\n'''.format(D=config["D"])
        template_fill_dict["raw_gt_vec_ID_fstream"] = '''
        std::string raw_gt_vec_ID_suffix_dir = "idx_{DB_SCALE}.ivecs";
        std::string raw_gt_vec_ID_dir = dir_concat(gnd_dir, raw_gt_vec_ID_suffix_dir);
        std::ifstream raw_gt_vec_ID_fstream(
            raw_gt_vec_ID_dir,
            std::ios::in | std::ios::binary);\n'''.format(DB_SCALE=config["DB_SCALE"]) + \
    '''    if (!raw_gt_vec_ID_fstream) {\n''' + \
    '''        std::cout << "error: only " << raw_gt_vec_ID_fstream.gcount() << " could be read";
            exit(1);\n''' + '}\n'

    bytes_float = 4
    bytes_ap512 = 64

    template_fill_dict["HBM_centroid_vectors_stage2_len"] = ""
    template_fill_dict["HBM_centroid_vectors_stage2_size"] = ""
    template_fill_dict["HBM_centroid_vectors_stage2_allocate"] = ""
    template_fill_dict["HBM_centroid_vectors_stage2_memcpy"] = ""
    template_fill_dict["HBM_centroid_vectorsExt"] = ""
    template_fill_dict["HBM_centroid_vectorsExt_set"] = ""
    template_fill_dict["HBM_metainfoExt_set"] = ""
    template_fill_dict["buffer_HBM_centroid_vectors"] = ""
    template_fill_dict["buffer_HBM_centroid_vectors_stage2_set_krnl_arg"] = ""
    template_fill_dict["buffer_HBM_centroid_vectors_stage2_enqueueMigrateMemObjects"] = ""
    if config["STAGE2_ON_CHIP"] == False:
        # 1 HBM channel per 2 PE
        HBM_channel_num_stage2 = int(np.ceil(config["PE_NUM_CENTER_DIST_COMP"] / 2.0))
        if config["PE_NUM_CENTER_DIST_COMP"] % 2 == 0:
            for i in range(HBM_channel_num_stage2 - 1):
                template_fill_dict["HBM_centroid_vectors_stage2_len"] += \
                    "    size_t HBM_centroid_vectors{i}_len = 2 * centroids_per_partition_even * D * sizeof(float) / sizeof(uint32_t);\n".format(
                        i=i)
            template_fill_dict["HBM_centroid_vectors_stage2_len"] += \
                "    size_t HBM_centroid_vectors{i}_len = (centroids_per_partition_even + centroids_per_partition_last_PE) * D * sizeof(float) / sizeof(uint32_t);\n".format(
                    i=HBM_channel_num_stage2 - 1)
        else: # % 2 == 1
            for i in range(HBM_channel_num_stage2 - 1):
                template_fill_dict["HBM_centroid_vectors_stage2_len"] += \
                    "    size_t HBM_centroid_vectors{i}_len = 2 * centroids_per_partition_even * D * sizeof(float) / sizeof(uint32_t);\n".format(
                        i=i)
            template_fill_dict["HBM_centroid_vectors_stage2_len"] += \
                "    size_t HBM_centroid_vectors{i}_len = centroids_per_partition_last_PE * D * sizeof(float) / sizeof(uint32_t);\n".format(
                    i=HBM_channel_num_stage2 - 1)
        for i in range(HBM_channel_num_stage2):
            template_fill_dict["HBM_centroid_vectors_stage2_size"] += \
                "    size_t HBM_centroid_vectors{i}_size =  HBM_centroid_vectors{i}_len * sizeof(uint32_t);\n".format(i=i)
            template_fill_dict["HBM_centroid_vectors_stage2_allocate"] += \
                "    std::vector<uint32_t, aligned_allocator<uint32_t>> HBM_centroid_vectors{i}(HBM_centroid_vectors{i}_len, 0);\n".format(i=i)
            template_fill_dict["HBM_centroid_vectors_stage2_memcpy"] += """
        int HBM_centroid_vectors_stage2_start_addr_{i} = 2 * {i} * centroids_per_partition_even * D * sizeof(float);
        memcpy(&HBM_centroid_vectors{i}[0], HBM_vector_quantizer_char + HBM_centroid_vectors_stage2_start_addr_{i}, HBM_centroid_vectors{i}_size);\n""".format(i=i)
            template_fill_dict["HBM_centroid_vectorsExt"] += \
                "        HBM_centroid_vectors{i}Ext,\n".format(i=i)
            template_fill_dict["HBM_centroid_vectorsExt_set"] += \
                '''    HBM_centroid_vectors{i}Ext.obj = HBM_centroid_vectors{i}.data();
        HBM_centroid_vectors{i}Ext.param = 0;
        HBM_centroid_vectors{i}Ext.flags = bank[{c}];\n'''.format(i=i, c=channel_iterator.get_next_channel_id())
            template_fill_dict["buffer_HBM_centroid_vectors"] += \
                '''    OCL_CHECK(err, cl::Buffer buffer_HBM_centroid_vectors{i}(context, CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, 
                HBM_centroid_vectors{i}_size, HBM_centroid_vectors{i}.data(), &err));\n'''.format(i=i)
            template_fill_dict["buffer_HBM_centroid_vectors_stage2_set_krnl_arg"] += \
                "    OCL_CHECK(err, err = krnl_vector_add.setArg(arg_counter++, buffer_HBM_centroid_vectors{i}));\n".format(i=i)
            template_fill_dict["buffer_HBM_centroid_vectors_stage2_enqueueMigrateMemObjects"] += \
                "        buffer_HBM_centroid_vectors{i},\n".format(i=i)

    if config["OPQ_ENABLE"]:
        template_fill_dict["OPQ_ENABLE"] = "true"
    else:
        template_fill_dict["OPQ_ENABLE"] = "false"

    for k in template_fill_dict:
        template_str = template_str.replace("<--{}-->".format(k), str(template_fill_dict[k]))
    output_str = template_str

    # Save generated file
    output_dir = os.path.join(args.output_dir, "host.cpp")
    with open(output_dir, "w+") as f:
        f.write(output_str)


    hpp_input_dir = os.path.join(args.input_dir, "host_2021.2.hpp")
    hpp_str = None
    with open(hpp_input_dir) as f:
        hpp_str = f.read()
    hpp_output_dir = os.path.join(args.output_dir, "host.hpp")
    with open(hpp_output_dir, "w+") as f:
        f.write(hpp_str)
else:
    print("Unsupported Vitis Version")
    raise ValueError