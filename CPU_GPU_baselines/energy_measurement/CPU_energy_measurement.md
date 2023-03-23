# CPU Energy Measurement

Use bench_polysemous_1bn.py to run a bunch of workloads, use cpu-energy-meter to measure the energy at the mean time. Iterate on nprobe=32, such that both IVF index scan and PQ code scan will be measured.

Use m5.metal -> 2 sockets; 48 physical cores; 96 virtual cores; Intel(R) Xeon(R) Platinum 8259CL CPU @ 2.50GHz; 512 GB

## Summary

Energy consumption projected to m5.4xlarge: 56.144 ~ 61.216 W
## SIFT

python bench_polysemous_1bn.py --on_disk 0 --dbname SIFT1000M --index_key IVF32768,PQ32 --parametersets 'nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32'

```
+--------------------------------------+
| CPU Energy Meter            Socket 0 |
+--------------------------------------+
Duration                 24.952188 sec
Package                4842.551514 Joule
DRAM                    770.200256 Joule
+--------------------------------------+
| CPU Energy Meter            Socket 1 |
+--------------------------------------+
Duration                 24.952188 sec
Package                4323.901611 Joule
DRAM                    497.330830 Joule
```

Per virtual core energy = (4842 + 4323) / 24.952188 / 96 = 3.826 W

16 virtual cores on m5.4xlarge = 16 x 3.826 = 61.216 W

DRAM: (770 + 497) / 24.952188 = 50.77 W -> 0.1W / GB

## Deep

python bench_polysemous_1bn.py --on_disk 0 --dbname Deep1000M --index_key IVF32768,PQ32 --parametersets 'nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32'

```
+--------------------------------------+
| CPU Energy Meter            Socket 0 |
+--------------------------------------+
Duration                 21.711080 sec
Package                4005.537170 Joule
DRAM                    536.732446 Joule
+--------------------------------------+
| CPU Energy Meter            Socket 1 |
+--------------------------------------+
Duration                 21.711080 sec
Package                3990.011963 Joule
DRAM                    562.661233 Joule
```

Per virtual core energy = (4005 + 3990) / 21.711080 / 96 = 3.835 W

16 virtual cores on m5.4xlarge = 16 x 3.835 = 61.36 W


## GNN 

python bench_polysemous_1bn.py --on_disk 0 --dbname GNN1400M --index_key IVF32768,PQ64 --n_shards 2 --shard_id 0 --parametersets  'nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32'

```
+--------------------------------------+
| CPU Energy Meter            Socket 0 |
+--------------------------------------+
Duration                 90.320785 sec
Package               18127.345276 Joule
DRAM                   2328.632919 Joule
+--------------------------------------+
| CPU Energy Meter            Socket 1 |
+--------------------------------------+
Duration                 90.320785 sec
Package               17794.384460 Joule
DRAM                   2217.758638 Joule
```

Per virtual core energy = (18127 + 17794) / 90.320785 / 96 = 4.142 W

16 virtual cores on m5.4xlarge = 16 x 4.142 = 66.272 W


## SBERT

python bench_polysemous_1bn.py --on_disk 0 --dbname SBERT3000M --index_key IVF65536,PQ64 --n_shards 4 --shard_id 0 --parametersets 'nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32 nprobe=32'

```
+--------------------------------------+
| CPU Energy Meter            Socket 0 |
+--------------------------------------+
Duration                 21.252975 sec
Package                3694.679443 Joule
DRAM                    526.031029 Joule
+--------------------------------------+
| CPU Energy Meter            Socket 1 |
+--------------------------------------+
Duration                 21.252975 sec
Package                3466.197449 Joule
DRAM                    366.391533 Joule
```

Per virtual core energy = (3694 + 3466) /21.252975 / 96 = 3.509 W

16 virtual cores on m5.4xlarge = 16 x 3.509 = 56.144 W
