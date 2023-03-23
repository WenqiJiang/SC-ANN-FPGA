# arg 1: bank num
# e.g., ./generate_FPGA_data_all_100M.sh 16
# all process will be run in backend simultaneously, thus either prepare a powerful server to do this, or remove the "&" symbol after commands to execute them se
quentially

BANKNUM=$1

echo $BANKNUM

SUFFIX="_banks"
PREFIX="/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT100M_IVF1024,PQ16_"
python extract_FPGA_required_data.py --dbname SIFT100M --index_key IVF1024,PQ16 --HBM_bank_num $BANKNUM --index_dir '../trained_CPU_indexes/bench_cpu_SIFT100M_IVF1024,PQ16' --output_dir "$PREFIX$BANKNUM$SUFFIX"&
PREFIX="/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT100M_IVF2048,PQ16_"
python extract_FPGA_required_data.py --dbname SIFT100M --index_key IVF2048,PQ16 --HBM_bank_num $BANKNUM --index_dir '../trained_CPU_indexes/bench_cpu_SIFT100M_IVF2048,PQ16' --output_dir "$PREFIX$BANKNUM$SUFFIX"&
PREFIX="/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT100M_IVF4096,PQ16_"
python extract_FPGA_required_data.py --dbname SIFT100M --index_key IVF4096,PQ16 --HBM_bank_num $BANKNUM --index_dir '../trained_CPU_indexes/bench_cpu_SIFT100M_IVF4096,PQ16' --output_dir "$PREFIX$BANKNUM$SUFFIX"&
PREFIX="/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT100M_IVF8192,PQ16_"
python extract_FPGA_required_data.py --dbname SIFT100M --index_key IVF8192,PQ16 --HBM_bank_num $BANKNUM --index_dir '../trained_CPU_indexes/bench_cpu_SIFT100M_IVF8192,PQ16' --output_dir "$PREFIX$BANKNUM$SUFFIX"&
PREFIX="/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT100M_IVF16384,PQ16_"
python extract_FPGA_required_data.py --dbname SIFT100M --index_key IVF16384,PQ16 --HBM_bank_num $BANKNUM --index_dir '../trained_CPU_indexes/bench_cpu_SIFT100M_IVF16384,PQ16' --output_dir "$PREFIX$BANKNUM$SUFFIX"&
PREFIX="/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT100M_IVF32768,PQ16_"
python extract_FPGA_required_data.py --dbname SIFT100M --index_key IVF32768,PQ16 --HBM_bank_num $BANKNUM --index_dir '../trained_CPU_indexes/bench_cpu_SIFT100M_IVF32768,PQ16' --output_dir "$PREFIX$BANKNUM$SUFFIX"&
PREFIX="/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT100M_IVF65536,PQ16_"
python extract_FPGA_required_data.py --dbname SIFT100M --index_key IVF65536,PQ16 --HBM_bank_num $BANKNUM --index_dir '../trained_CPU_indexes/bench_cpu_SIFT100M_IVF65536,PQ16' --output_dir "$PREFIX$BANKNUM$SUFFIX"&

PREFIX="/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT100M_OPQ16,IVF1024,PQ16_"
python extract_FPGA_required_data.py --dbname SIFT100M --index_key OPQ16,IVF1024,PQ16 --HBM_bank_num $BANKNUM --index_dir '../trained_CPU_indexes/bench_cpu_SIFT100M_OPQ16,IVF1024,PQ16' --output_dir "$PREFIX$BANKNUM$SUFFIX"&
PREFIX="/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT100M_OPQ16,IVF2048,PQ16_"
python extract_FPGA_required_data.py --dbname SIFT100M --index_key OPQ16,IVF2048,PQ16 --HBM_bank_num $BANKNUM --index_dir '../trained_CPU_indexes/bench_cpu_SIFT100M_OPQ16,IVF2048,PQ16' --output_dir "$PREFIX$BANKNUM$SUFFIX"&
PREFIX="/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT100M_OPQ16,IVF4096,PQ16_"
python extract_FPGA_required_data.py --dbname SIFT100M --index_key OPQ16,IVF4096,PQ16 --HBM_bank_num $BANKNUM --index_dir '../trained_CPU_indexes/bench_cpu_SIFT100M_OPQ16,IVF4096,PQ16' --output_dir "$PREFIX$BANKNUM$SUFFIX"&
PREFIX="/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT100M_OPQ16,IVF8192,PQ16_"
python extract_FPGA_required_data.py --dbname SIFT100M --index_key OPQ16,IVF8192,PQ16 --HBM_bank_num $BANKNUM --index_dir '../trained_CPU_indexes/bench_cpu_SIFT100M_OPQ16,IVF8192,PQ16' --output_dir "$PREFIX$BANKNUM$SUFFIX"&
PREFIX="/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT100M_OPQ16,IVF16384,PQ16_"
python extract_FPGA_required_data.py --dbname SIFT100M --index_key OPQ16,IVF16384,PQ16 --HBM_bank_num $BANKNUM --index_dir '../trained_CPU_indexes/bench_cpu_SIFT100M_OPQ16,IVF16384,PQ16' --output_dir "$PREFIX$BANKNUM$SUFFIX"&
PREFIX="/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT100M_OPQ16,IVF32768,PQ16_"
python extract_FPGA_required_data.py --dbname SIFT100M --index_key OPQ16,IVF32768,PQ16 --HBM_bank_num $BANKNUM --index_dir '../trained_CPU_indexes/bench_cpu_SIFT100M_OPQ16,IVF32768,PQ16' --output_dir "$PREFIX$BANKNUM$SUFFIX"&
PREFIX="/mnt/scratch/wenqi/saved_npy_data/FPGA_data_SIFT100M_OPQ16,IVF65536,PQ16_"
python extract_FPGA_required_data.py --dbname SIFT100M --index_key OPQ16,IVF65536,PQ16 --HBM_bank_num $BANKNUM --index_dir '../trained_CPU_indexes/bench_cpu_SIFT100M_OPQ16,IVF65536,PQ16' --output_dir "$PREFIX$BANKNUM$SUFFIX"&
