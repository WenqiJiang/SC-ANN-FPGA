import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.style.use('ggplot')


# SIFT100M, K=100
y_no_OPQ = np.array([11270, 17459, 25600, 31852, 34667, 42813, 42926, 39325, 33100])
y_OPQ = np.array([10924, 14261, 19764, 18733, 17743, 21852, 17146, 18252, 23279, ])

# # SIFT500M, K=100
# y_no_OPQ = np.array([1466, 1742, 5024, 7857, 10827, 13970, 15918])
# y_OPQ = np.array([1751, 2280, 5390, 7670, 10971, 13902, 15299])

x_labels = ['nlist=1024', 'nlist=2048', 'nlist=4096',\
    'nlist=8192', 'nlist=16384', 'nlist=32768', 'nlist=65536', 'nlist=131072', 'nlist=262144']

x = np.arange(len(x_labels))  # the label locations
width = 0.3  # the width of the bars    

speedup_array = y_OPQ / y_no_OPQ
max_speedup = np.amax(speedup_array)
min_speedup = np.amin(speedup_array)
print("Speedup OPQ over no OPQ:\n{}\nmax: {:.2f}x\nmin: {:.2f}x".format(speedup_array, max_speedup, min_speedup))

fig, ax = plt.subplots(1, 1, figsize=(8, 2))
# 
rects1  = ax.bar(x - width / 2, y_no_OPQ, width)#, label='Men')
rects2   = ax.bar(x + width / 2, y_OPQ, width)#, label='Women')

label_font = 12
tick_font = 10
tick_label_font = 9
legend_font = 10
title_font = 14

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('QPS', fontsize=label_font)
# ax.set_xlabel('nlist', fontsize=label_font)
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontsize=tick_label_font)
plt.xticks(rotation=0)

legend_list = ['IVF-PQ with OPQ', 'IVF-PQ without OPQ']
ax.legend([rects1, rects2], legend_list, facecolor='white', framealpha=1, frameon=False, loc=(0.02, 0.65), fontsize=legend_font, ncol=1)

# ax.set_title('{} R@{}={}: {:.2f}x over CPU, {:.2f}x over GPU'.format(
#     dbname, topK, int(recall_goal*100), best_qps_fpga/best_qps_cpu, best_qps_fpga/best_qps_gpu), 
#     fontsize=label_font)

def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        if height > 0:
            notation = '{:.0f}'.format(height)
        else:   
            notation = 'N/A'
        ax.annotate(notation,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=tick_font, horizontalalignment='center', rotation=90)


autolabel(rects1)
autolabel(rects2)

# annotate speedup

def autolabel_speedup(rects, speedup_array):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for i, rect in enumerate(rects):
        height = rect.get_height()
        ax.annotate('{:.2f}x'.format(speedup_array[i]),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, -3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='top', fontsize=tick_font, horizontalalignment='center', rotation=90)

# autolabel_speedup(rects2, speedup_array)

ax.set(ylim=[0, np.amax(y_no_OPQ) * 1.5])

plt.rcParams.update({'figure.autolayout': True})

plt.savefig('./out_img/plot_throughput_experiment_2_algorithm_settings.png', transparent=False, dpi=200, bbox_inches="tight")
plt.show()
