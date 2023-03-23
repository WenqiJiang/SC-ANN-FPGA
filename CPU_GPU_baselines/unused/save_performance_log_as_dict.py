from os import listdir
from os.path import isfile, join
import pickle
import numpy as np
    
def save_obj(obj, dire, name):
    with open(dire + name + '.pkl', 'wb') as fl:
        pickle.dump(obj, fl, pickle.HIGHEST_PROTOCOL)

def load_CPU_performance_info_and_save_dict(log_dir):

    onlyfiles = [f for f in listdir(log_dir) if isfile(join(log_dir, f))] 
    onlyfiles = [f for f in onlyfiles if f[0] != '.'] # remove hidden files
    print(onlyfiles)

    for file_name in onlyfiles:
        with open(log_dir + file_name, 'r') as f:
            print(file_name)
            lines = f.readlines() 

            print('\nCPU, ' + file_name)
            all_elements = []
            # Strips the newline character 
            for line in lines: 
                # skip the title lines, e.g.,
                """
    Preparing dataset SIFT100M
    sizes: B (100000000, 128) Q (10000, 128) T (100000000, 128) gt (10000, 1000)
    loading ./trained_CPU_indexes/bench_cpu_SIFT100M_IMI2x9,PQ16/SIFT100M_IMI2x9,PQ16_populated.index
                R@1    R@10   R@100     time    %pass
    nprobe=1 	 0.1454 0.2455 0.2600    0.187    0.00
    nprobe=2 	 0.1989 0.3624 0.3876    0.314    0.00
    nprobe=4 	 0.2412 0.4724 0.5129    0.430    0.0
                """
                if "nprobe=" in line:
                    line = line.replace("\xa0", "").replace("\n", "").replace("\t", "").split(" ")
                    # print(line)
                    elements = [e for e in line if e != ""]
                    # print(elements)
                    assert len(elements) == 6
                    all_elements.append(elements)

            print("dictionary keys: nprobe, R1, R10, R100, time")
            nprobe = [int(e[0][7:]) for e in all_elements] # "nprobe=1024, remove the prefix"
            R1 = [float(e[1]) for e in all_elements]
            R10 = [float(e[2]) for e in all_elements]
            R100 = [float(e[3]) for e in all_elements]
            time = [float(e[4]) for e in all_elements]  # ms per query
            # the last column is the pass rate of polynomeous code, not needed
            # total_cells = 0
            # if "IMI2x" in file_name:
            #     total_cells = (2 ** int(file_name[5:7])) ** 2
            # else:
            #     seg = file_name.split(",")
            #     for s in seg:
            #         if "IVF" in s:
            #             total_cells = int(s[3:])
            # scanned_ratio = [np / total_cells for np in nprobe]

            # the final dictionary that contain all info of this set of experiments
            info = dict()
            info["nprobe"] = nprobe
            # info["total_cells"] = total_cells
            # info["scanned_ratio"] = scanned_ratio
            info["R1"] = R1
            info["R10"] = R10
            info["R100"] = R100
            info["Throughput"] = list(1 / (np.array(time) * 1e-3))
            print(info)
            save_obj(info, "{}dict/".format(log_dir), file_name)


def load_GPU_performance_info_and_save_dict(log_dir):

    onlyfiles = [f for f in listdir(log_dir) if isfile(join(log_dir, f))] 
    onlyfiles = [f for f in onlyfiles if f[0] != '.'] # remove hidden files
    print(onlyfiles)

    for file_name in onlyfiles:

        with open(log_dir + file_name, 'r') as f:
            print(file_name)
            lines = f.readlines() 

            print('\nGPU, ' + file_name)
            all_elements = []
            # Strips the newline character 
            for line in lines: 
                # skip the title lines, e.g.,
                """
move to GPU done in 5.455 s
search...
10000

0/10000 (0.010 s)      
9728/10000 (0.356 s)      probe=1  : 0.361 s 1-R@1: 0.1121 1-R@10: 0.3312 1-R@100: 0.5035 
0/10000 (0.021 s)    
9728/10000 (1.918 s)      probe=32 : 1.967 s 1-R@1: 0.1357 1-R@10: 0.4581 1-R@100: 0.8352 
                """
                if "probe=" in line:
                    line = line.replace("\xa0", "").replace("\n", "").replace("\t", "")\
                        .replace("1-R@1:", "").replace("1-R@10:", "").replace("1-R@100:", "")\
                        .replace(":", "").replace("s", "").split(" ")
                    elements = [e for e in line if e != ""]
                    # print(line) 
                    # print(elements)
                    assert len(elements) == 5
                    all_elements.append(elements)

            print("dictionary keys: nprobe, R1, R10, R100, time")
            nprobe = [int(e[0][6:]) for e in all_elements] # "nprobe=1024, remove the prefix"
            time = [float(e[1]) for e in all_elements] # time = total time in second to finish 10000 queries
            R1 = [float(e[2]) for e in all_elements]
            R10 = [float(e[3]) for e in all_elements]
            R100 = [float(e[4]) for e in all_elements]

            # the final dictionary that contain all info of this set of experiments
            info = dict()
            info["nprobe"] = nprobe
            info["R1"] = R1
            info["R10"] = R10
            info["R100"] = R100
            info["Throughput"] = list(10000 / np.array(time))
            print(info)
            save_obj(info, "{}dict/".format(log_dir), file_name)

if __name__ == "__main__":

    cpu_result_path="./cpu_performance_result/"
    load_CPU_performance_info_and_save_dict(cpu_result_path)

    gpu_result_path="./gpu_performance_result/"
    load_GPU_performance_info_and_save_dict(gpu_result_path)