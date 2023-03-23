# arg 1: FPGA num, arg 2: HBM num ber FPGA
# e.g., ./generate_FPGA_data_all_500M.sh 4 12
# all process will be run in backend simultaneously, thus either prepare a powerful server to do this, or remove the "&" symbol after commands to execute them sequentially

FPGANUM=$1
BANKNUM=$2

echo $BANKNUM

SUFFIX_F="_FPGA_"
SUFFIX_B="_banks"

PREFIX="/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT500M_IVF1024,PQ16_"
python extract_FPGA_required_data_multi_FPGA.py --dbname SIFT500M --index_key IVF1024,PQ16 --FPGA_num $FPGANUM --HBM_bank_num $BANKNUM --index_dir '../trained_CPU_indexes/bench_cpu_SIFT500M_IVF1024,PQ16' --output_dir "$PREFIX$FPGANUM$SUFFIX_F$BANKNUM$SUFFIX_B"&
PREFIX="/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT500M_IVF2048,PQ16_"
python extract_FPGA_required_data_multi_FPGA.py --dbname SIFT500M --index_key IVF2048,PQ16 --FPGA_num $FPGANUM --HBM_bank_num $BANKNUM --index_dir '../trained_CPU_indexes/bench_cpu_SIFT500M_IVF2048,PQ16' --output_dir "$PREFIX$FPGANUM$SUFFIX_F$BANKNUM$SUFFIX_B"&
PREFIX="/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT500M_IVF4096,PQ16_"
python extract_FPGA_required_data_multi_FPGA.py --dbname SIFT500M --index_key IVF4096,PQ16 --FPGA_num $FPGANUM --HBM_bank_num $BANKNUM --index_dir '../trained_CPU_indexes/bench_cpu_SIFT500M_IVF4096,PQ16' --output_dir "$PREFIX$FPGANUM$SUFFIX_F$BANKNUM$SUFFIX_B"&
PREFIX="/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT500M_IVF8192,PQ16_"
python extract_FPGA_required_data_multi_FPGA.py --dbname SIFT500M --index_key IVF8192,PQ16 --FPGA_num $FPGANUM --HBM_bank_num $BANKNUM --index_dir '../trained_CPU_indexes/bench_cpu_SIFT500M_IVF8192,PQ16' --output_dir "$PREFIX$FPGANUM$SUFFIX_F$BANKNUM$SUFFIX_B"&
PREFIX="/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT500M_IVF16384,PQ16_"
python extract_FPGA_required_data_multi_FPGA.py --dbname SIFT500M --index_key IVF16384,PQ16 --FPGA_num $FPGANUM --HBM_bank_num $BANKNUM --index_dir '../trained_CPU_indexes/bench_cpu_SIFT500M_IVF16384,PQ16' --output_dir "$PREFIX$FPGANUM$SUFFIX_F$BANKNUM$SUFFIX_B"&
PREFIX="/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT500M_IVF32768,PQ16_"
python extract_FPGA_required_data_multi_FPGA.py --dbname SIFT500M --index_key IVF32768,PQ16 --FPGA_num $FPGANUM --HBM_bank_num $BANKNUM --index_dir '../trained_CPU_indexes/bench_cpu_SIFT500M_IVF32768,PQ16' --output_dir "$PREFIX$FPGANUM$SUFFIX_F$BANKNUM$SUFFIX_B"&
PREFIX="/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT500M_IVF65536,PQ16_"
python extract_FPGA_required_data_multi_FPGA.py --dbname SIFT500M --index_key IVF65536,PQ16 --FPGA_num $FPGANUM --HBM_bank_num $BANKNUM --index_dir '../trained_CPU_indexes/bench_cpu_SIFT500M_IVF65536,PQ16' --output_dir "$PREFIX$FPGANUM$SUFFIX_F$BANKNUM$SUFFIX_B"&

PREFIX="/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT500M_OPQ16,IVF1024,PQ16_"
python extract_FPGA_required_data_multi_FPGA.py --dbname SIFT500M --index_key OPQ16,IVF1024,PQ16 --FPGA_num $FPGANUM --HBM_bank_num $BANKNUM --index_dir '../trained_CPU_indexes/bench_cpu_SIFT500M_OPQ16,IVF1024,PQ16' --output_dir "$PREFIX$FPGANUM$SUFFIX_F$BANKNUM$SUFFIX_B"&
PREFIX="/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT500M_OPQ16,IVF2048,PQ16_"
python extract_FPGA_required_data_multi_FPGA.py --dbname SIFT500M --index_key OPQ16,IVF2048,PQ16 --FPGA_num $FPGANUM --HBM_bank_num $BANKNUM --index_dir '../trained_CPU_indexes/bench_cpu_SIFT500M_OPQ16,IVF2048,PQ16' --output_dir "$PREFIX$FPGANUM$SUFFIX_F$BANKNUM$SUFFIX_B"&
PREFIX="/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT500M_OPQ16,IVF4096,PQ16_"
python extract_FPGA_required_data_multi_FPGA.py --dbname SIFT500M --index_key OPQ16,IVF4096,PQ16 --FPGA_num $FPGANUM --HBM_bank_num $BANKNUM --index_dir '../trained_CPU_indexes/bench_cpu_SIFT500M_OPQ16,IVF4096,PQ16' --output_dir "$PREFIX$FPGANUM$SUFFIX_F$BANKNUM$SUFFIX_B"&
PREFIX="/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT500M_OPQ16,IVF8192,PQ16_"
python extract_FPGA_required_data_multi_FPGA.py --dbname SIFT500M --index_key OPQ16,IVF8192,PQ16 --FPGA_num $FPGANUM --HBM_bank_num $BANKNUM --index_dir '../trained_CPU_indexes/bench_cpu_SIFT500M_OPQ16,IVF8192,PQ16' --output_dir "$PREFIX$FPGANUM$SUFFIX_F$BANKNUM$SUFFIX_B"&
PREFIX="/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT500M_OPQ16,IVF16384,PQ16_"
python extract_FPGA_required_data_multi_FPGA.py --dbname SIFT500M --index_key OPQ16,IVF16384,PQ16 --FPGA_num $FPGANUM --HBM_bank_num $BANKNUM --index_dir '../trained_CPU_indexes/bench_cpu_SIFT500M_OPQ16,IVF16384,PQ16' --output_dir "$PREFIX$FPGANUM$SUFFIX_F$BANKNUM$SUFFIX_B"&
PREFIX="/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT500M_OPQ16,IVF32768,PQ16_"
python extract_FPGA_required_data_multi_FPGA.py --dbname SIFT500M --index_key OPQ16,IVF32768,PQ16 --FPGA_num $FPGANUM --HBM_bank_num $BANKNUM --index_dir '../trained_CPU_indexes/bench_cpu_SIFT500M_OPQ16,IVF32768,PQ16' --output_dir "$PREFIX$FPGANUM$SUFFIX_F$BANKNUM$SUFFIX_B"&
PREFIX="/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT500M_OPQ16,IVF65536,PQ16_"
python extract_FPGA_required_data_multi_FPGA.py --dbname SIFT500M --index_key OPQ16,IVF65536,PQ16 --FPGA_num $FPGANUM --HBM_bank_num $BANKNUM --index_dir '../trained_CPU_indexes/bench_cpu_SIFT500M_OPQ16,IVF65536,PQ16' --output_dir "$PREFIX$FPGANUM$SUFFIX_F$BANKNUM$SUFFIX_B"&
