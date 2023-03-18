# cp the necessary file to the bitstream archive folder
# $1 -> dir of the bitstream archive folder, e.g. ~/bitstream_2020.2/folder_name
cp -r host build_dir.hw.xilinx_u280_xdma_201920_3/vadd.xclbin xrt.ini config.yaml perf_test.py $1
