import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import pickle
from os import listdir
from os.path import isfile, join
from matplotlib.ticker import FuncFormatter

dict_dir = "../cpu_performance_result/dict/"
onlyfiles = [f for f in listdir(dict_dir) if isfile(join(dict_dir, f))] 
onlyfiles = [f for f in onlyfiles if f[0] != '.'] # remove hidden files
onlyfiles.sort()

def load_obj(dirc, name):
    with open(dirc + name, 'rb') as f:
        return pickle.load(f)

def get_min_time_95_recall(time, recall):
    """
    Given arrays time and recall, find return the minimum time that achieve 95% recall
    """
    assert len(time) == len(recall)
    for i in range(len(time)):
        if recall[i] > 0.95:
            return time[i]

if __name__ == "__main__":


    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    label_font = 16
    markersize = 8
    tick_font = 14
    # color_plot0 = "#008A00"
    # color_plot1 = "#1BA1E2"

    plots = []
    high_recall_filename = []
    min_time = []
    for file_name in onlyfiles:
        info = load_obj(dict_dir, file_name)

        data_x = info["time"]
        data_y = info["R100"]

        if np.max(data_y) > 0.95:
            plots.append(ax.plot(data_x, data_y))
            high_recall_filename.append(file_name)
            min_time.append(get_min_time_95_recall(data_x, data_y))
        # else:
        #     onlyfiles.remove(file_name)                     
    

    plt.xscale("log")
    ax.legend([plot[0] for plot in plots], [fname.replace(".pkl", "") for fname in high_recall_filename], loc=(1.1, 0), fontsize=label_font)
    ax.tick_params(top=False, bottom=True, left=True, right=False, labelleft=True, labelsize=tick_font)
    ax.get_xaxis().set_visible(True)
    ax.set_xlabel('Time per query / ms', fontsize=label_font)
    ax.set_ylabel('Recall R@100', fontsize=label_font)

    ax.hlines(0.95, 0, 300, color='#3F3F3F', linestyles='dashed')
    ax.text(0.2,0.96, "95% Recall", fontsize=16)

    ax.arrow(0.2, 0.1, 300, 0, width=0.01, shape="full", length_includes_head=True, head_length=50, head_width=0.03)
    ax.text(0.5,0.12, "Increase search space", fontsize=16)

    min_95_time = np.min(min_time)
    ax.annotate("{} ms".format(min_95_time), xy=(min_95_time, 0.95), xytext=(20, 0.8), arrowprops={"arrowstyle": '-|>', 'color': '#1f1f1f', 'linewidth': 2}, fontsize=16)
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)

    plt.rcParams.update({'figure.autolayout': True})

    plt.savefig('../images/cpu_bench_PQ16_indexes.png', transparent=False, dpi=200, bbox_inches="tight")
    # plt.show()