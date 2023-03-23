# How to bench CPU performance?

## Recall

Use bench_all_cpu_recall.sh, which calls python bench_cpu_recall.py. When using it, remember to change two stuffs: (a) TOPK (b) the for loop, i.e., for RECALL in ...

An example of python bench_cpu_recall.py: 

```
python bench_cpu_recall.py --dbname SIFT100M --index_key IVF4096,PQ16 --recall_goal 80 --topK 10
```

It will store the recall information in a dictionary located in ./recall_info/

## Throughput & Response Time

bench_cpu_performance.py

For throughput test, we consider the best case that all query vectors are stored locally.

The throughput will be saved in ./cpu_performance_result/ as a dictionary when using the second option of the script.

Two ways to use the script

(1) Test the throughput of given DB & index & nprobe:

```
python bench_cpu_performance.py --on_disk 0 --qbs 10000 --dbname SIFT100M --index_key IVF4096,PQ16 --topK 10 --parametersets 'nprobe=1 nprobe=32'
```

(2) Load the dictionary that maps DB & index & topK & recall to nprobe, evaluate them all, then save the results

```
python bench_cpu_performance.py --on_disk 0 --qbs 10000 --load_from_dict 1 --overwrite 0 --nprobe_dict_dir './recall_info/cpu_recall_index_nprobe_pairs_SIFT100M_IVFPQ_only.pkl' --throughput_dict_dir './cpu_performance_result/cpu_throughput_SIFT100M.pkl' --response_time_dict_dir './cpu_performance_result/cpu_response_time_SIFT100M_qbs_10000.pkl' 
```

## (Archieved) Response time with network

In the "unused" folder

We consider a client sending query to an ANNS server, and measure the response time on the client side.

To conduct the experiments, first adjust IP and port used on the server in two scripts, i.e., bench_cpu_response_time_server.py and bench_cpu_response_time_client.py

The response time of every single query will be saved in ./cpu_performance_result/ as a dictionary when using the second option of the scripts.

### bench_cpu_response_time_client.py

There are 2 ways to use the script:

(1) Test the response time of given DB & index & nprobe:

```
python bench_cpu_response_time_client.py --dbname SIFT100M --index_key OPQ16,IVF4096,PQ16 --topK 10 --param 'nprobe=32' --HOST 127.0.0.1 --PORT 65432
```

(2) Load the dictionary that maps DB & index & topK & recall to nprobe, evaluate them all, then save the results

```
python bench_cpu_response_time_client.py --load_from_dict 1 --overwrite 0 --nprobe_dict_dir './recall_info/cpu_recall_index_nprobe_pairs_SIFT100M.pkl' --performance_dict_dir './cpu_performance_result/cpu_response_time_SIFT100M.pkl' --HOST 10.1.212.76 --PORT 65432
```

### bench_cpu_response_time_server.py

There are 2 ways to use the script:

(1) Test the response time of given DB & index & nprobe:

```
python bench_cpu_response_time_server.py --dbname SIFT100M --index_key OPQ16,IVF4096,PQ16 --topK 10 --param 'nprobe=32' --HOST 127.0.0.1 --PORT 65432
```

(2) Load the dictionary that maps DB & index & topK & recall to nprobe, evaluate them all, then save the results

```
python bench_cpu_response_time_server.py --load_from_dict 1 --overwrite 0 --nprobe_dict_dir './recall_info/cpu_recall_index_nprobe_pairs_SIFT100M.pkl' --performance_dict_dir './cpu_performance_result/cpu_response_time_SIFT100M.pkl' --HOST 10.1.212.76 --PORT 65432
```

### Measure network RTT

The measured time includes both network RTT and ANNS searching time. To measure the network RTT alone, on server side:

```
python network_RTT_server.py
```

on the client side:

```
python network_RTT_client.py
```

The client will save a networt RTT distribution named network_response_time.npy

## Unused scripts

In folder ./unused/, don't use them.

performance_test_cpu.py

performance_test_gpu.py
