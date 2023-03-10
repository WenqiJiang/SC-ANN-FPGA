import argparse 
import datetime
import os
import yaml
import numpy as np

"""
Exmaple usage:
    python generate_entire_project.py --vitis_version 2022.1 --output_project_dir ../generated_projects_2022.1/ANNS_project_generated
"""
now = datetime.datetime.now()
time_str = now.strftime("%Y-%m-%d-%H-%M")

parser = argparse.ArgumentParser()
parser.add_argument('--output_project_dir', type=str, default="../generated_projects/ANNS_project_{}".format(time_str), help="output project directory")
parser.add_argument('--vitis_version', type=str, default="2021.2", help="support 2020.2 and 2021.2/2022.1 (same output)")
args = parser.parse_args()

# Load YAML configurations
config_file = open("config.yaml", "r")
config = yaml.load(config_file)

# Copy reference project
if os.path.exists(args.output_project_dir):
    print("ERROR! Output folder already exists.")
    raise ValueError
else:
    os.system("cp -r {i} {o}".format(
    i=os.path.join(os.getcwd(), "template_files/folder_structure"), 
    o=os.path.join(args.output_project_dir)))
os.system("cp config.yaml {o}".format(o=args.output_project_dir))
os.system("cp -r {i} {o}".format(i=os.getcwd(), o=os.path.join(args.output_project_dir)))
if args.vitis_version == '2020.2':
    os.system("cp template_files/Makefile_2020.2 {o}".format( o=os.path.join(args.output_project_dir, 'Makefile')))
elif args.vitis_version == '2021.2' or args.vitis_version == '2022.1':
    os.system("cp template_files/Makefile_2021.2 {o}".format( o=os.path.join(args.output_project_dir, 'Makefile')))
else:
    print("Unsupported Vitis version")

# Replace some files with code generated by templates
os.system("python codegen_stage1_all.py --input_dir template_files/stage1_OPQ --output_dir {}".format(
    os.path.join(args.output_project_dir, "src")))
os.system("python codegen_stage2_all.py --input_dir template_files/stage2_cluster_distance_computation --output_dir {}".format(
    os.path.join(args.output_project_dir, "src")))
os.system("python codegen_stage3_all.py --input_dir template_files/stage3_select_Voronoi_cells --output_dir {}".format(
    os.path.join(args.output_project_dir, "src")))
os.system("python codegen_stage4_all.py --input_dir template_files/stage4_LUT_construction --output_dir {}".format(
    os.path.join(args.output_project_dir, "src")))
os.system("python codegen_stage5_distance_estimation_by_LUT.py --input_dir template_files/stage5_distance_estimation_by_LUT --output_dir {}".format(
    os.path.join(args.output_project_dir, "src")))
os.system("python codegen_stage5_HBM_interconnections.py --input_dir template_files/stage5_distance_estimation_by_LUT --output_dir {}".format(
    os.path.join(args.output_project_dir, "src")))
os.system("python codegen_stage6_all.py --input_dir template_files/stage6_select_results --output_dir {}".format(
    os.path.join(args.output_project_dir, "src")))

os.system("python codegen_constants.py --input_dir template_files --output_dir {}".format(
    os.path.join(args.output_project_dir, "src")))
os.system("python codegen_helpers.py --input_dir template_files --output_dir {}".format(
    os.path.join(args.output_project_dir, "src")))
os.system("python codegen_host.py --input_dir template_files --output_dir {} --vitis_version {}".format(
    os.path.join(args.output_project_dir, "src"), args.vitis_version))
os.system("python codegen_types.py --input_dir template_files --output_dir {}".format(
    os.path.join(args.output_project_dir, "src")))
os.system("python codegen_vadd.py --input_dir template_files --output_dir {}".format(
    os.path.join(args.output_project_dir, "src")))

os.system("python codegen_design_cfg.py --input_dir template_files --output_dir {}".format(
    args.output_project_dir))