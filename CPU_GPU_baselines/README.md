# Faiss_experiments

## Installation

install anaconda: https://docs.anaconda.com/anaconda/install/linux/

```
% faiss supports python3.7, not 3.8
conda create -n py37 python=3.7
conda activate py37
```

Either install cpu or gpu version (the gpu version already includes the cpu version, thus can skip the cpu installation step)

```
% cpu version 
conda install -c conda-forge openblas
conda install -c pytorch faiss-cpu==1.7.1
```

```
% install gpu version version >= 1.7.2
% install CUDA first if its not installed: https://docs.vmware.com/en/VMware-vSphere-Bitfusion/3.0/Example-Guide/GUID-ABB4A0B1-F26E-422E-85C5-BA9F2454363A.html
conda install faiss-gpu -c pytorch
```

Install pickle5 for recall dict load/save:
```
conda install -c conda-forge pickle5
```

Verify installation
```
python 
import faiss
```

Faiss's demo: 
https://github.com/facebookresearch/faiss/blob/master/tutorial/python/1-Flat.py

Faiss API: 
https://github.com/facebookresearch/faiss/wiki/Faiss-indexes

## Download dataset

SITF (bigann) dataset: http://corpus-texmex.irisa.fr/

```
mkdir bigann
cd bigann
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_base.bvecs.gz
gunzip bigann_base.bvecs.gz 
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_learn.bvecs.gz
gunzip bigann_learn.bvecs.gz
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_query.bvecs.gz
gunzip bigann_query.bvecs.gz
wget ftp://ftp.irisa.fr/local/texmex/corpus/bigann_gnd.tar.gz
tar xzvf bigann_gnd.tar.gz
rm bigann_gnd.tar.gz
```

## CPU

## Performance test

See [bench_cpu_performance.md](./bench_cpu_performance.md)

### bench_polysemous_1bn.py

Train / query a single index on CPU.

### train_cpu.py

Train a combination of indexes (IMI / IVF Cell number; PQ code length; with OPQ or not; dataset) on CPU.

To cover all indexes on SIFT100M dataset: 

```
python train_cpu.py --dataset SIFT100M --index IVF --PQ 16 --OPQ 1

python train_cpu.py --dataset SIFT100M --index IVF --PQ 16 --OPQ 0

python train_cpu.py --dataset SIFT100M --index IMI --PQ 16 --OPQ 1

python train_cpu.py --dataset SIFT100M --index IMI --PQ 16 --OPQ 0

python train_cpu.py --dataset SIFT10M --index IVF --PQ 16 --OPQ 1

python train_cpu.py --dataset SIFT10M --index IVF --PQ 16 --OPQ 0

python train_cpu.py --dataset SIFT10M --index IMI --PQ 16 --OPQ 1

python train_cpu.py --dataset SIFT10M --index IMI --PQ 16 --OPQ 0

python train_cpu.py --dataset SIFT1M --index IVF --PQ 16 --OPQ 1

python train_cpu.py --dataset SIFT1M --index IVF --PQ 16 --OPQ 0

python train_cpu.py --dataset SIFT1M --index IMI --PQ 16 --OPQ 1

python train_cpu.py --dataset SIFT1M --index IMI --PQ 16 --OPQ 0

```

## GPU

### performance test

See [bench_gpu_performance.md](./bench_gpu_performance.md)

### bench_gpu_1bn.py

(a) Train / query a single index on GPU. 

```
python bench_gpu_1bn.py SIFT1000M OPQ16,IVF262144,PQ16 -nnn 100 -ngpu 3 -startgpu 1 -tempmem $[1536*1024*1024] -qbs 512
```

(b) evaluate the throughput (of a single parameter setting / using the recall dictionary)

(c) evaluate the (nlist, nprobe) pair to achieve a certain recall


### train_gpu.py

Automatically train a set of indexes on GPU.

To cover all indexes on SIFT100M dataset:  

```
python train_gpu.py --dataset SIFT100M --index IVF --PQ 16 --OPQ 1 --ngpu 1 --startgpu 0

python train_gpu.py --dataset SIFT100M --index IVF --PQ 16 --OPQ 0 --ngpu 1 --startgpu 0

python train_gpu.py --dataset SIFT10M --index IVF --PQ 16 --OPQ 1 --ngpu 1 --startgpu 0

python train_gpu.py --dataset SIFT10M --index IVF --PQ 16 --OPQ 0 --ngpu 1 --startgpu 0

python train_gpu.py --dataset SIFT1M --index IVF --PQ 16 --OPQ 1 --ngpu 1 --startgpu 0

python train_gpu.py --dataset SIFT1M --index IVF --PQ 16 --OPQ 0 --ngpu 1 --startgpu 0
```

## CPU Experiments

### IVF

python bench_polysemous_1bn.py SIFT100M IVF8192,PQ16 nprobe=1 nprobe=2 nprobe=4 nprobe=8 nprobe=16 nprobe=32 nprobe=64 nprobe=128

### IMI

python bench_polysemous_1bn.py SIFT100M IMI2x14,PQ16 nprobe=1 nprobe=2 nprobe=4 nprobe=8 nprobe=16 nprobe=32 nprobe=64 nprobe=128 nprobe=256 nprobe=512 nprobe=1024 nprobe=2048 nprobe=4096 nprobe=8192 nprobe=16384 nprobe=32768 nprobe=65536 

## Recall Experiments

### Takeaways

* The smaller the dataset, the higher the recall (using the same index, e.g., IVF1024,PQ16)
  * e.g., on SIFT100M using IVF1024, R@1=32%, while for 1M R@1=46%
* The finer-grained the index, the higher the recall (e.g., IVF262144,PQ16 > IVF1024)
  * e.g., on SIFT1M using IVF1024, R@1=46%, using IVF262144, R@1=52%
* Set a single set of recall goals that is reasonable for all three scales and all indexes
  * R@1: 20%, 25%, 30%
  * R@10: 40%, 60%, 80%
  * R@100: 60%, 80%, 95%
  * use one index to show that the recall goal is reasonable (e.g., SIFT100M, IVF4096,PQ16)
```
loading ./trained_CPU_indexes/bench_cpu_SIFT100M_IVF4096,PQ16/SIFT100M_IVF4096,PQ16_populated.index
                 R@1    R@10   R@100     time    %pass
nprobe=1         0.2196 0.4128 0.4447    0.425    0.00
nprobe=2         0.2668 0.5526 0.6120    0.754    0.00
nprobe=4         0.2954 0.6606 0.7539    1.424    0.00
nprobe=8         0.3191 0.7420 0.8681    2.716    0.00
nprobe=16        0.3294 0.7860 0.9393    5.234    0.00
nprobe=32        0.3322 0.8029 0.9721   10.423    0.00
```

###  Experiments

on *SIFT 100M*, we evaluated 3 indexes to (IVF1024,PQ16 (lower bound); IMI2x8,PQ16 (lower bound); IVF242144,PQ16 (upper bound)) decide the recall target range:

```
loading ./trained_CPU_indexes/bench_cpu_SIFT100M_IVF1024,PQ16/SIFT100M_IVF1024,PQ16_populated.index
                 R@1    R@10   R@100     time    %pass
nprobe=1         0.2279 0.4674 0.5177    1.182    0.00
nprobe=2         0.2771 0.6107 0.6993    2.242    0.00
nprobe=4         0.3057 0.7105 0.8381    4.495    0.00
nprobe=8         0.3173 0.7630 0.9211    8.829    0.00
nprobe=16        0.3238 0.7857 0.9652   17.031    0.00
nprobe=32        0.3247 0.7926 0.9810   31.851    0.00

loading ./trained_CPU_indexes/bench_cpu_SIFT100M_IVF262144,PQ16/SIFT100M_IVF262144,PQ16_populated.index
                 R@1    R@10   R@100     time    %pass
nprobe=1         0.1582 0.2283 0.2305    0.656    0.00
nprobe=2         0.2142 0.3353 0.3427    0.641    0.00
nprobe=4         0.2689 0.4648 0.4814    0.688    0.00
nprobe=8         0.3139 0.5808 0.6121    0.757    0.00
nprobe=16        0.3507 0.6885 0.7376    0.871    0.00
nprobe=32        0.3718 0.7704 0.8400    1.021    0.00
nprobe=64        0.3827 0.8220 0.9119    1.387    0.00
nprobe=128       0.3885 0.8514 0.9562    2.103    0.00
nprobe=256       0.3909 0.8634 0.9791    3.569    0.00
nprobe=512       0.3918 0.8680 0.9903    6.334    0.00
nprobe=1024      0.3918 0.8701 0.9949   12.305    0.00

loading ./trained_CPU_indexes/bench_cpu_SIFT100M_IMI2x8,PQ16/SIFT100M_IMI2x8,PQ16_populated.index
                 R@1    R@10   R@100     time    %pass
nprobe=1         0.1715 0.2939 0.3175    0.322    0.00
nprobe=2         0.2187 0.4192 0.4588    0.502    0.00
nprobe=4         0.2634 0.5359 0.5966    0.787    0.00
nprobe=8         0.2934 0.6342 0.7190    1.253    0.00
nprobe=16        0.3154 0.7075 0.8143    2.010    0.00
nprobe=32        0.3263 0.7563 0.8894    3.087    0.00
nprobe=64        0.3330 0.7876 0.9392    4.935    0.00
nprobe=128       0.3366 0.8048 0.9678    8.015    0.00
nprobe=256       0.3381 0.8118 0.9811   13.160    0.00
nprobe=512       0.3389 0.8147 0.9873   21.813    0.00
nprobe=1024      0.3391 0.8158 0.9890   36.795    0.00
```

on *SIFT 10M*: 

```
loading ./trained_CPU_indexes/bench_cpu_SIFT10M_IVF1024,PQ16/SIFT10M_IVF1024,PQ16_populated.index
                 R@1    R@10   R@100     time    %pass
nprobe=1         0.2605 0.4656 0.4872    0.200    0.00
nprobe=2         0.3228 0.6186 0.6585    0.332    0.00
nprobe=4         0.3648 0.7362 0.8057    0.590    0.00
nprobe=8         0.3842 0.8093 0.9057    1.095    0.00
nprobe=16        0.3925 0.8478 0.9633    2.088    0.00
nprobe=32        0.3950 0.8634 0.9882    3.997    0.00

loading ./trained_CPU_indexes/bench_cpu_SIFT10M_IVF262144,PQ16/SIFT10M_IVF262144,PQ16_populated.index
                 R@1    R@10   R@100     time    %pass
nprobe=1         0.1394 0.1767 0.1770    1.257    0.00
nprobe=2         0.1986 0.2743 0.2756    1.245    0.00
nprobe=4         0.2585 0.3883 0.3911    1.203    0.00
nprobe=8         0.3205 0.5153 0.5215    1.243    0.00
nprobe=16        0.3697 0.6335 0.6467    1.383    0.00
nprobe=32        0.4098 0.7417 0.7643    1.463    0.00
nprobe=64        0.4313 0.8171 0.8534    1.759    0.00
nprobe=128       0.4445 0.8682 0.9189    2.358    0.00
nprobe=256       0.4500 0.8961 0.9599    3.629    0.00
nprobe=512       0.4528 0.9094 0.9813    6.008    0.00
nprobe=1024      0.4536 0.9154 0.9925   10.658    0.00

loading ./trained_CPU_indexes/bench_cpu_SIFT10M_IMI2x8,PQ16/SIFT10M_IMI2x8,PQ16_populated.index
                 R@1    R@10   R@100     time    %pass
nprobe=1         0.1606 0.2558 0.2650    0.079    0.00
nprobe=2         0.2202 0.3767 0.3932    0.103    0.00
nprobe=4         0.2776 0.5065 0.5308    0.140    0.00
nprobe=8         0.3189 0.6237 0.6578    0.198    0.00
nprobe=16        0.3556 0.7187 0.7686    0.286    0.00
nprobe=32        0.3758 0.7891 0.8569    0.447    0.00
nprobe=64        0.3870 0.8353 0.9179    0.709    0.00
nprobe=128       0.3937 0.8619 0.9582    1.147    0.00
nprobe=256       0.3968 0.8759 0.9795    1.922    0.00
nprobe=512       0.3983 0.8819 0.9909    3.242    0.00
nprobe=1024      0.3983 0.8840 0.9948    5.543    0.00
nprobe=2048      0.3982 0.8843 0.9953    9.631    0.00
```

on *SIFT 1M*: 

```
loading ./trained_CPU_indexes/bench_cpu_SIFT1M_IVF1024,PQ16/SIFT1M_IVF1024,PQ16_populated.index
                 R@1    R@10   R@100     time    %pass
nprobe=1         0.2774 0.4285 0.4349    0.063    0.00
nprobe=2         0.3539 0.5937 0.6058    0.077    0.00
nprobe=4         0.4071 0.7312 0.7544    0.108    0.00
nprobe=8         0.4412 0.8317 0.8700    0.164    0.00
nprobe=16        0.4576 0.8907 0.9446    0.275    0.00
nprobe=32        0.4633 0.9172 0.9799    0.489    0.00

loading ./trained_CPU_indexes/bench_cpu_SIFT1M_IVF262144,PQ16/SIFT1M_IVF262144,PQ16_populated.index
                 R@1    R@10   R@100     time    %pass
nprobe=1         0.1056 0.1194 0.1194    1.225    0.00
nprobe=2         0.1632 0.1935 0.1935    1.165    0.00
nprobe=4         0.2286 0.2869 0.2870    1.245    0.00
nprobe=8         0.3024 0.3997 0.4000    1.279    0.00
nprobe=16        0.3732 0.5314 0.5322    1.285    0.00
nprobe=32        0.4289 0.6534 0.6553    1.485    0.00
nprobe=64        0.4719 0.7638 0.7688    1.759    0.00
nprobe=128       0.4981 0.8485 0.8596    2.304    0.00
nprobe=256       0.5141 0.9044 0.9228    3.325    0.00
nprobe=512       0.5211 0.9386 0.9650    5.490    0.00
nprobe=1024      0.5247 0.9541 0.9870    9.860    0.00

loading ./trained_CPU_indexes/bench_cpu_SIFT1M_IMI2x8,PQ16/SIFT1M_IMI2x8,PQ16_populated.index
                 R@1    R@10   R@100     time    %pass
nprobe=1         0.1606 0.2205 0.2238    0.045    0.00
nprobe=2         0.2226 0.3243 0.3288    0.039    0.00
nprobe=4         0.2893 0.4420 0.4493    0.049    0.00
nprobe=8         0.3487 0.5629 0.5734    0.064    0.00
nprobe=16        0.4032 0.6764 0.6926    0.084    0.00
nprobe=32        0.4360 0.7682 0.7921    0.120    0.00
nprobe=64        0.4602 0.8370 0.8713    0.186    0.00
nprobe=128       0.4734 0.8862 0.9308    0.300    0.00
nprobe=256       0.4798 0.9152 0.9673    0.509    0.00
nprobe=512       0.4827 0.9296 0.9877    0.902    0.00
nprobe=1024      0.4837 0.9349 0.9962    1.651    0.00
nprobe=2048      0.4837 0.9361 0.9990    3.001    0.00
```
