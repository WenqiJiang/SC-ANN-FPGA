import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import os

from classify_stages import classify_stages, get_percentage
from profiling_stages import draw_profiling_plot


x_labels = ['IVF1024', 'IVF1024\nw/ OPQ', \
    'IVF2048', 'IVF2048\nw/ OPQ', \
    'IVF4096', 'IVF4096\nw/ OPQ', \
    'IVF8192', 'IVF8192\nw/ OPQ', \
    'IVF16384', 'IVF16384\nw/ OPQ', \
    'IVF32768', 'IVF32768\nw/ OPQ', 
    'IVF65536', 'IVF65536\nw/ OPQ', \
    'IVF131072', 'IVF131072\nw/ OPQ', \
    'IVF262144', 'IVF262144\nw/ OPQ', \
    ]

file_prefixes = [ \
    'nsys_report_SIFT100M_IVF1024,PQ16_R@100=0.95_nprobe_11_ngpu_1_batchsize_10000', \
    'nsys_report_SIFT100M_OPQ16,IVF1024,PQ16_R@100=0.95_nprobe_11_ngpu_1_batchsize_10000', \
    'nsys_report_SIFT100M_IVF2048,PQ16_R@100=0.95_nprobe_14_ngpu_1_batchsize_10000', \
    'nsys_report_SIFT100M_OPQ16,IVF2048,PQ16_R@100=0.95_nprobe_14_ngpu_1_batchsize_10000', \
    'nsys_report_SIFT100M_IVF4096,PQ16_R@100=0.95_nprobe_19_ngpu_1_batchsize_10000', \
    'nsys_report_SIFT100M_OPQ16,IVF4096,PQ16_R@100=0.95_nprobe_18_ngpu_1_batchsize_10000', \
    'nsys_report_SIFT100M_IVF8192,PQ16_R@100=0.95_nprobe_26_ngpu_1_batchsize_10000', \
    'nsys_report_SIFT100M_OPQ16,IVF8192,PQ16_R@100=0.95_nprobe_24_ngpu_1_batchsize_10000', \
    'nsys_report_SIFT100M_IVF16384,PQ16_R@100=0.95_nprobe_31_ngpu_1_batchsize_10000', \
    'nsys_report_SIFT100M_OPQ16,IVF16384,PQ16_R@100=0.95_nprobe_30_ngpu_1_batchsize_10000', \
    'nsys_report_SIFT100M_IVF32768,PQ16_R@100=0.95_nprobe_48_ngpu_1_batchsize_10000', \
    'nsys_report_SIFT100M_OPQ16,IVF32768,PQ16_R@100=0.95_nprobe_46_ngpu_1_batchsize_10000', \
    'nsys_report_SIFT100M_IVF65536,PQ16_R@100=0.95_nprobe_63_ngpu_1_batchsize_10000', \
    'nsys_report_SIFT100M_OPQ16,IVF65536,PQ16_R@100=0.95_nprobe_60_ngpu_1_batchsize_10000', \
    'nsys_report_SIFT100M_IVF131072,PQ16_R@100=0.95_nprobe_88_ngpu_1_batchsize_10000', \
    'nsys_report_SIFT100M_OPQ16,IVF131072,PQ16_R@100=0.95_nprobe_86_ngpu_1_batchsize_10000', \
    'nsys_report_SIFT100M_IVF262144,PQ16_R@100=0.95_nprobe_121_ngpu_1_batchsize_10000', \
    'nsys_report_SIFT100M_OPQ16,IVF262144,PQ16_R@100=0.95_nprobe_118_ngpu_1_batchsize_10000', \
    ]
# file_prefixes = [ \
#     'nsys_report_SIFT500M_IVF1024,PQ16_R@100=0.95_nprobe_11_ngpu_1_batchsize_10000', \
#     'nsys_report_SIFT500M_OPQ16,IVF1024,PQ16_R@100=0.95_nprobe_10_ngpu_1_batchsize_10000', \
#     'nsys_report_SIFT500M_IVF2048,PQ16_R@100=0.95_nprobe_15_ngpu_1_batchsize_10000', \
#     'nsys_report_SIFT500M_OPQ16,IVF2048,PQ16_R@100=0.95_nprobe_13_ngpu_1_batchsize_10000', \
#     'nsys_report_SIFT500M_IVF4096,PQ16_R@100=0.95_nprobe_18_ngpu_1_batchsize_10000', \
#     'nsys_report_SIFT500M_OPQ16,IVF4096,PQ16_R@100=0.95_nprobe_17_ngpu_1_batchsize_10000', \
#     'nsys_report_SIFT500M_IVF8192,PQ16_R@100=0.95_nprobe_21_ngpu_1_batchsize_10000', \
#     'nsys_report_SIFT500M_OPQ16,IVF8192,PQ16_R@100=0.95_nprobe_21_ngpu_1_batchsize_10000', \
#     'nsys_report_SIFT500M_IVF16384,PQ16_R@100=0.95_nprobe_29_ngpu_1_batchsize_10000', \
#     'nsys_report_SIFT500M_OPQ16,IVF16384,PQ16_R@100=0.95_nprobe_28_ngpu_1_batchsize_10000', \
#     'nsys_report_SIFT500M_IVF32768,PQ16_R@100=0.95_nprobe_38_ngpu_1_batchsize_10000', \
#     'nsys_report_SIFT500M_OPQ16,IVF32768,PQ16_R@100=0.95_nprobe_36_ngpu_1_batchsize_10000', \
#     'nsys_report_SIFT500M_IVF65536,PQ16_R@100=0.95_nprobe_46_ngpu_1_batchsize_10000', \
#     'nsys_report_SIFT500M_OPQ16,IVF65536,PQ16_R@100=0.95_nprobe_47_ngpu_1_batchsize_10000']

assert len(x_labels) == len(file_prefixes)

path_prefixes = []
for p in file_prefixes:
    path_prefixes.append(os.path.join('result_experiment_2_algorithm_settings', p))

gputrace_csv_dirs = [p + '_gputrace.csv' for p in path_prefixes]
gpukernsum_csv_dirs = [p + '_gpukernsum.csv' for p in path_prefixes]

# Stage 1: OPQ
# Stage 2: vector quantizer
# Stage 3: select centroids
# Stage 4: construct distance LUT
# Stage 5: PQ code scan
# Stage 6: collect topK results

profile_perc_array = []
# example_profile_array = [
#     # 100M, 1
#     [8.606278140845747, 0.11607633274229297, 3.3378707089447355, 78.57136070072978, 9.368414116737446], \
#     # 100M, 10
#     [32.7008185883583, 0.5164703077320218, 4.674772663594282, 33.70847203114799, 28.399466409167403]
#     ]

for i in range(len(file_prefixes)):
    print(file_prefixes[i])
    t_1_2, t_3, t_4_5, t_6, t_other, t_transpose_4_5 = classify_stages(gputrace_csv_dirs[i], gpukernsum_csv_dirs[i])
    print(t_1_2, t_3, t_4_5, t_6, t_other, t_transpose_4_5)
    p_1_2, p_3, p_4_5, p_6, p_other, p_transpose_4_5 = get_percentage(t_1_2, t_3, t_4_5, t_6, t_other, t_transpose_4_5=t_transpose_4_5)
    profile_perc_array.append([p_1_2, p_3, p_4_5, p_6, p_other, p_transpose_4_5])

y_stage_1_2 = [r[0] for r in profile_perc_array]
y_stage_3 = [r[1] for r in profile_perc_array]
y_stage_4_5 = [r[2] for r in profile_perc_array]
y_stage_6 = [r[3] for r in profile_perc_array]
y_other = [r[4] for r in profile_perc_array]
y_transpose = [r[5] for r in profile_perc_array]

draw_profiling_plot(x_labels, y_stage_1_2, y_stage_3, y_stage_4_5, y_stage_6, y_other, 'profile_experiment_2_algorithm_settings', x_tick_rotation=70)
# draw_profiling_plot(x_labels, y_stage_1_2, y_stage_3, y_stage_4_5, y_stage_6, y_other, 'profile_experiment_2_algorithm_settings_distinct_y_transpose', x_tick_rotation=70, mark_transpose=True, y_transpose=y_transpose)
