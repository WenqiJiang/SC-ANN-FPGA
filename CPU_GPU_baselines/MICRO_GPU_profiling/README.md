# How to use?

There are a number of experiments exploring the effect of multiple paramaters.

## Faiss version

Make sure that the latest GPU Faiss is installed (see the guide in the root folder of this repo). Version should be at least 1.7.2

```
python
import faiss
print(faiss.__version__)
```

## Parameters

Typical parameters include:

(Start by the python argument name, followed by an example argument)

--dbname SIFT500M 

Dataset name.

--nprobe_dict_dir '../recall_info/gpu_recall_index_nprobe_pairs_SIFT500M.pkl' 

The dictionary that stores the nprobe (number of cells to be scanned) to achieve certain recall goals.

--topK 100 

The number of results to return.

--recall_goal 0.95 

The recall object, related to topK, the higher the topK, the higher the recall.

--ngpu 1 

Number of GPUs to execute the experiment.

--startgpu 0 

The GPU ID x. E.g., using 2 GPUs starting from GPU 0 == Use GPU 0 & 1

--qbs 10000 

Batch size

--nsys_enable 1

Do profiling with nsight. If disabled, only the throughput & recall are recorded.

## Example

Let's assume as want to evaluate the SIFT100M dataset, there are two arguments related to the dataset: dbname and nprobe_dict_dir. Make sure they are correct.

### Start with experiment 2

python experiment_2_algorithm_settings.py --dbname SIFT100M --nprobe_dict_dir '../recall_info/gpu_recall_index_nprobe_pairs_SIFT100M.pkl' --topK 100 --recall_goal 0.95 --ngpu 1 --startgpu 0 --qbs 10000 --nsys_enable 1

The output will be stored in the folder "result_experiment_2_algorithm_settings". For each algorithm setting experiment, there will be 4 files, e.g.:

```
nsys_report_SIFT100M_IVF1024,PQ16_R@10=0.8_nprobe_39_ngpu_1_batchsize_10000.sqlite

nsys_report_SIFT100M_IVF1024,PQ16_R@10=0.8_nprobe_39_ngpu_1_batchsize_10000.qdrep

nsys_report_SIFT100M_IVF1024,PQ16_R@10=0.8_nprobe_39_ngpu_1_batchsize_10000_gputrace.csv

nsys_report_SIFT100M_IVF1024,PQ16_R@10=0.8_nprobe_39_ngpu_1_batchsize_10000_gpukernsum.csv

```

There will also be a log file recording the performance, starting with "out", e.g.,

```
out_SIFT100M_R@100=0.95_ngpu_1
```

We will scan through this log file and find the index_key that can achieve the best performance, let's say, IVF65536,PQ16, and the nprobe can be found in the log. This setting will be used in the rest of the experiments.

### experiment 1

Let's go back to experiment 1. This experiment measures the relationship between batch sizes and throughput. Use the index_key and nprobe that we found to be optimal in experiment 2.

```
python experiment_1_batch_size_effect.py --dbname SIFT100M --index_key IVF65536,PQ16 --topK 100 --nprobe 46 --ngpu 1 --startgpu 0 --nsys_enable 1
```

Note that unfortunately nsys has memory related bug that leads to crash when small batch sizes are used (SIFT500M, V100 16GB version). So if the bug occurs, set nsys_enable to 0 


### experiment 3

fix nprobe, change nlist. Given the following numbers, fix OPQ_enable=1, nprobe=16, and change nlist would be interesting. -> observer performance & recall

This experiment is not related to the best setting, so just make sure the dbname is correct.

```
python experiment_3_nlist.py --dbname SIFT100M --topK 100 --nprobe 16 --ngpu 1 --startgpu 0 --qbs 10000 --nsys_enable 1
```

### experiment 4

given the best setting in experiment 2, chanage nprobe to observe recall and performance change.

```
python experiment_4_nprobe.py --dbname SIFT100M  --index_key IVF65536,PQ16 --topK 100 --min_nprobe 1 --max_nprobe 128 --ngpu 1 --startgpu 0 --qbs 10000 --nsys_enable 1
```

### experiment 5

given the best setting in experiment 2, chanage topK to observe recall and performance change.

```
python experiment_5_topK.py --dbname SIFT100M  --index_key IVF65536,PQ16 --nprobe 46 --ngpu 1 --startgpu 0 --qbs 10000 --nsys_enable 1
```

### collect the results

The results are stored in the folder starting with "result_experiment_X...". 

