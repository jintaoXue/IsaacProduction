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

def draw(ax, data_p, color_dict, name_dict, alpha, x_lable, title, y_tick, y_tick_f,y_label):
    '''loss curve plot'''
    df = pd.read_csv(data_p, header=None)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(x_lable, fontsize=15)
    ax.set_ylabel(y_label, fontsize=15)
    ax.tick_params(axis='both', which='both', labelsize=15)
    # if log_x:
    #     ax.set_xscale('log')
    if y_tick_f:
        # ax.set_yscale('symlog')
        # ax.set_yticks(y_tick)
        # ax.set_yscale('log')  # or 'logit'
        ax.set_yscale('function', functions=(forward, inverse))
        ax.set_ylim(y_tick)
        # ax.yaxis.set_inverted(True)
    # ax.tick_params(axis='both', size=12)
    algo_dict_rev = {v: k for k, v in name_dict.items()}
    x = np.array(df[0][1:].to_list(), dtype=np.float32)
    data_names = df.loc[0]
    data_dict = {}
    for data_name, i in zip(data_names, range(len(data_names))):
        if "step" in data_name or "MIN" in data_name or "MAX" in data_name:
            pass
        else:
            data_dict[data_name.split(' ')[0]] = df[i][1:]
    #one data
    for name in name_dict.values():
        label = algo_dict_rev[name]
        raw_y, color = data_dict[name], color_dict[label]
        # raw_y, color = np.array(data_dict[name], dtype=np.float32), color_dict[label]
        smoothed_y = smooth_line(raw_y, alpha)
        ax.plot(x, smoothed_y, '-', color=color, label=label, ms=5, linewidth=1.5)
    # ax.legend(
    #     fontsize="x-large",
    #     handlelength=5.0)
        # handleheight=3)
    # get the legend object
    leg = ax.legend()
    # change the line width for the legend
    for line in leg.get_lines():
        line.set_linewidth(6.0)

    return 

def smooth_line(data, alpha=0.005):
    smoothed_d = data.ewm(alpha=alpha,adjust=False).mean()
    np_data = np.array(smoothed_d.to_list(), dtype=np.float32)
    return np_data

# Function x**(1/2)
def forward(x):
    return x**(4)


def inverse(x):
    return x**2

# # Function Mercator transform
# def forward(a):
#     a = np.deg2rad(a)
#     return np.rad2deg(np.log(np.abs(np.tan(a) + 1.0 / np.cos(a))))


# def inverse(a):
#     a = np.deg2rad(a)
#     return np.rad2deg(np.arctan(np.sinh(a)))

'''=========================================================Main drawing code=========================================================='''
if __name__ == '__main__':
    ### training metric
    p1 = os.path.dirname(__file__) + "/metric4" + "/learning_rate.csv"
    p2 = os.path.dirname(__file__) + "/metric4" + "/repeat.csv"
    p3 = os.path.dirname(__file__) + "/metric4" + "/batch_size.csv"
    # t_buffer = os.path.dirname(__file__) + "/metric1" + "/t_buffer.csv"
    data_path = [p1,p3,p2]
    lr_dict = {"1e-3":"le-3_rainbowepsilon_2024-12-18_13-26-06", 
                 "1e-4":"epsilon_FactoryTaskAllocationMiC_2024-12-08_17-36-58", 
                 "1e-5":"1e-5_rainbowepsilon_2024-12-19_12-45-13"}
    batch_size = {
                "64":"b64_rainbowepsilon_2024-12-16_00-29-47", 
                "128":"b128_rainbowepsilon_2024-12-16_18-07-34", 
                "256":"b256_rainbowepsilon_2024-12-17_13-08-41",
                "512":"epsilon_FactoryTaskAllocationMiC_2024-12-08_17-36-58",
                }
    repeat_dict = {"2":"2_rainbowepsilon_2024-12-19_00-27-24", 
                "5":"epsilon_FactoryTaskAllocationMiC_2024-12-08_17-36-58", 
                "10":"10_rainbowepsilon_2024-12-15_02-47-52",
                "15":"15_rainbowepsilon_2024-12-16_00-15-51"
                }
    name_dict_list = [lr_dict, batch_size, repeat_dict]
    color_dict_1 = {'1e-3': 'crimson', '1e-4': 'orange', '1e-5': 'forestgreen'}
    color_dict_2 = {'64': 'forestgreen', '128': 'dodgerblue', '256': 'palevioletred', '512':'blueviolet'}
    color_dict_3 = {'2': 'crimson', '5': 'orange', '10': 'forestgreen', '15': 'dodgerblue'}
    color_dict_list = [color_dict_1, color_dict_2, color_dict_3]
    # data_list = [e_return, e_len, e_succ]
    titles = ["Learning rate", "Batch size","Duplicate number"]
    x_lables = ["Episode", "Episode", "Episode"]
    log_x = [False, False, False]
    y_tick_f = [True, True, True]
    y_tick = [[0.4, 1.01], [0.5, 1.0], [0.5, 1.0]]
    # https://stackoverflow.com/questions/74001262/how-to-change-x-axis-to-not-be-uniform-scale
    smooth_alphas = [0.01, 0.01, 0.01]
    # y_lables = ["Loss", "Return", "Timespan", "Progress"]
    # draw_metric_from_csv_res(data_list, algo_dict, titles, x_lables, color_dict, log_x, y_tick_f, y_tick, smooth_alphas, 'metric_2')
    

    fig = plt.figure(figsize=(15,4), dpi=100)
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132)
    ax3 = plt.subplot(133)
    # plt.gca()
    axs = [ax1,ax2,ax3]
    plt.style.use('seaborn-v0_8-white')
    plt.rcParams['pdf.fonttype'] = 42
    params = {'legend.fontsize': 15,
            'legend.handlelength': 2}
    plt.rcParams.update(params)
    for i in range(0,3):
        draw(axs[i], data_path[i], color_dict_list[i], name_dict_list[i], smooth_alphas[i], x_lables[i], titles[i], y_tick[i], y_tick_f[i], y_label="Progress")


    plt.tight_layout()
    # plt.show()
    path = os.path.dirname(__file__)
    fig.savefig('{}.pdf'.format(path + '/' + 'ablation'), bbox_inches='tight')





