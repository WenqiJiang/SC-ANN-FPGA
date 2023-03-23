import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

plt.style.use('ggplot')

# SIFT500M, K=100, IVF65536, nprobe=46
y_QPS = np.array([1014, 1947, 3451, 4094, 7424, 16496, 24987, 28258, 29626, \
    30835, 32286, 29570, 22537, 18500, 16817])

x_labels = ['B=1', 'B=2', 'B=4', 'B=8', 'B=16', 'B=32', 'B=64', 'B=128', \
    'B=256', 'B=512', 'B=1024', 'B=2048', 'B=4096', 'B=8192', 'B=10000']

x = np.arange(len(x_labels))  # the label locations
width = 0.3  # the width of the bars    
fig, ax = plt.subplots(1, 1, figsize=(8, 2))
# 
rects1  = ax.bar(x - width, y_QPS, width)#, label='Men')

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
plt.xticks(rotation=45)


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

ax.set(ylim=[0, np.amax(y_QPS) * 1.5])

plt.rcParams.update({'figure.autolayout': True})

plt.savefig('./out_img/plot_throughput_experiment_1_batch_size_effect.png', transparent=False, dpi=200, bbox_inches="tight")
plt.show()
