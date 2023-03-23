# How to bench GPU performance?

## Recall

Use bench_all_gpu_recall.sh, which calls python bench_cpu_recall.py. When using it, remember to change two stuffs: (a) TOPK (b) the for loop, i.e., for RECALL in ...

An example of python bench_gpu_1bn.py: 

usage 3, recall test: find min nprobe to achieve certain recall 

```
python bench_gpu_1bn.py -dbname SIFT100M -index_key OPQ16,IVF262144,PQ16 -recall_goal 80 -topK 10 -ngpu 1 -startgpu 1 -qbs 512
```

It will store the recall information in a dictionary located in ./recall_info/

## Throughput & Response Time

bench_gpu_1bn.py

batch size can be set with the -qbs argument

usage 1, throughput test on a single parameter setting

```
python bench_gpu_1bn.py -dbname SIFT100M -index_key OPQ16,IVF262144,PQ16 -topK 100 -ngpu 1 -startgpu 1 -tempmem $[1536*1024*1024] -nprobe 1,32 -qbs 512
```

usage 2, (throughput and response time) test on a range of parameter settings (by loading a recall dictionary)

```
python bench_gpu_1bn.py -load_from_dict 1 -overwrite 0 -nprobe_dict_dir './recall_info/gpu_recall_index_nprobe_pairs_SIFT100M.pkl' -throughput_dict_dir './gpu_performance_result/gpu_throughput_SIFT100M.pkl' -response_time_dict_dir './gpu_performance_result/gpu_response_time_SIFT100M.pkl' -ngpu 1 -startgpu 0 -tempmem $[1536*1024*1024] -qbs 1
```
    
output dictionary format: d_throughput[dbname][index_key][topK][recall_goal][query_batch_size] = throughput
