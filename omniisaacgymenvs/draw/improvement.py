import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize
import seaborn
import os
import xlrd
from itertools import cycle
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import statsmodels.api as sm




def draw(res_l, algo_dict:dict, metric_l, color_l, base_line, annotates):
    plt.style.use('seaborn-v0_8-white')
    # plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['pdf.fonttype'] = 42
    # plt.tick_params(axis='both', labelsize=50)
    params = {'legend.fontsize': 15,
          'legend.handlelength': 2}
    plt.rcParams.update(params)
    res_array = np.array(res_l)
    annotates_array = np.array(annotates)
    labels = metric_l
    del algo_dict[base_line]
    algo_list = list(algo_dict.keys())
    
    num = len(algo_dict.keys())
    x = np.arange(len(labels))*1.5  # 标签位置

    width = 0.2  # 柱状图的宽度，可以根据自己的需求和审美来改
    fig, ax = plt.subplots(figsize=(18, 6), dpi=100)
    rects_l = []
    for i in range(0, num):
        _rects = ax.bar(x + (i - (num-1)/2)*(width+0.05), res_array[:, i], width, color=color_l[i],label=algo_list[i])
        rects_l.append(_rects)



    # 为y轴、标题和x轴等添加一些文本。
    # ax.set_ylabel('Y轴', fontsize=16)
    # ax.set_xlabel('X轴', fontsize=16)
    ax.set_title('Improvement', fontsize=20)
    ax.set_xticks(x)
    plt.yticks([])
    ax.set_xticklabels(labels, fontsize=15)
    ax.legend( loc='upper left', 
        bbox_to_anchor=(0.3, 1.0),
        ncol=1)
    # ax.legend()
    for ax in fig.get_axes():
        ax.grid(True)
    ax.set_ylim([-0.02,0.3])



    def autolabel(rects, anno):
        """在*rects*中的每个柱状条上方附加一个文本标签，显示其高度"""
        for rect, text in zip(rects, anno):
            height = rect.get_height()
            ax.annotate('{:.3}'.format(text),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=15)

    for i in range(0, num):
        autolabel(rects_l[i], annotates_array[:,i])

    fig.tight_layout()

    # plt.show()
    path = os.path.dirname(__file__)
    fig.savefig('{}.pdf'.format(path + '/' + 'improve'), bbox_inches='tight')


def caculate(data_list, metric_l, scaling, base_line):
    res_l = []
    annotates = []
    for data, metric, _scal in zip(data_list, metric_l, scaling):
        data:dict
        d_base = data[base_line]
        del data[base_line]
        val = (np.array(list(data.values()))-d_base)/d_base
        annotates.append(val*np.sign(_scal))
        res = _scal*val
        res_l.append(res)
    return res_l, annotates

'''=========================================================Main drawing code=========================================================='''
if __name__ == '__main__':
    ### zero_shot performance
    #from o1 to o10
    metric_l = ["Training time", "Test: makespan", "ZeroShot: makespan", "ZeroShot: success rate"]
    color_dict = {'D3QN': 'crimson', 'EDQN1': 'orange', 'EDQN2': 'forestgreen', 'EBQ-G': 'dodgerblue', 'EBQ-N': 'palevioletred', 'EBQ-GN':'blueviolet', "NoSp": 'silver'}
    scaling_factor = np.array([0.09,-1,-1,1])
    algo_dict = {"D3QN":"test_rainbownoe_ep_8000.pth_2024-12-09_21-42-46", 
                 "EDQN1":"test_edqn_ep_10500.pth_2024-12-11_13-29-49", 
                 "EDQN2":"test_no_dueling_ep_24900.pth_2024-12-10_13-06-23", 
                 "NoSp":"test_no_spatial_rainbowmini_ep_20100.pth_2024-12-23_18-12-29",
                 "EBQ-G":"test_rainbowepsilon_ep_19500.pth_2024-12-08_17-36-58", 
                 "EBQ-N":"test_rainbowmini_ep_24000.pth_2024-12-08_15-44-10", 
                 "EBQ-GN":"test_epsilon_noisy_ep_5700.pth_2024-12-09_14-31-02"}
    
    Training_time = {"D3QN": 0.264, 
                "EDQN1": 1.0, 
                "EDQN2":1.0, 
                "NoSp":1.0,
                "EBQ-G":1.0, 
                "EBQ-N":1.0, 
                "EBQ-GN":1.0}
    
    Test_time ={"D3QN": 874.43, 
            "EDQN1": 834.92, 
            "EDQN2":808.04, 
            "NoSp":888.23,
            "EBQ-G":806.00, 
            "EBQ-N":805.45, 
            "EBQ-GN":865.19}

    zero_timespan = {"D3QN": 1225.94, 
            "EDQN1": 992.98, 
            "EDQN2":966.30, 
            "NoSp":966.32,
            "EBQ-G":962.57, 
            "EBQ-N":947.70, 
            "EBQ-GN":996.86}
    zero_succ = {"D3QN": 0.843, 
        "EDQN1": 0.969, 
        "EDQN2":0.987,
        "NoSp":0.994, 
        "EBQ-G":0.989, 
        "EBQ-N":1.0, 
        "EBQ-GN":0.980}
    data = [Training_time, Test_time, zero_timespan, zero_succ]
    base_line = 'D3QN'
    del color_dict[base_line]
    res_l, annotates = caculate(data, metric_l, scaling_factor, base_line)
    draw(res_l, algo_dict, metric_l, list(color_dict.values()), base_line, annotates)






