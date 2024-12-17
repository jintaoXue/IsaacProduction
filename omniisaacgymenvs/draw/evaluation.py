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

def draw_metric_from_csv_res(data_list, algo_dict, titles, x_lables, color_dict, log_x, y_tick_f, y_tick, smooth_alphas, save_name):
    
    import matplotlib.pyplot as plt 
    import matplotlib.colors as mcolors 
    
    # https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
    styles_dict = plt.style.available
    plt.style.use('seaborn-v0_8-white')
    # plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['pdf.fonttype'] = 42
    # plt.tick_params(axis='both', labelsize=50)
    params = {'legend.fontsize': 15,
          'legend.handlelength': 2}
    plt.rcParams.update(params)
    # plt.rcParams['xtick.labelsize'] = 14  # x轴刻度标签字体大小
    # plt.rcParams['ytick.labelsize'] = 14  # y轴刻度标签字体大小
    fig = plt.figure(figsize=(15,10), dpi=100)
    # gs = gridspec(1,4, )
    # gs = fig.add_gridspec(1,4) 
    ax_1 = plt.subplot(221)
    ax_2 = plt.subplot(212)
    ax_3 = plt.subplot(222)

    for ax in fig.get_axes():
        ax.grid(True)
    # plt.tick_params(axis='both', labelsize=50)
    for i, ax in zip(range(0, len(data_list)), [ax_1, ax_2, ax_3]):
        draw_one_sub_pic(ax, data_list[i], titles[i], x_lables[i], algo_dict, color_dict, log_x[i], y_tick_f[i], y_tick[i], smooth_alphas[i])

    plt.show(block=False)
    fig.tight_layout()
    path = os.path.dirname(__file__)
    fig.savefig('{}.pdf'.format(path + '/' + save_name), bbox_inches='tight')
    return

def draw_one_sub_pic(ax, data, title, x_lable, algo_dict, color_dict, log_x, y_tick_f, y_tick, alpha):
    '''loss curve plot'''
    df = pd.read_csv(data, header=None)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(x_lable, fontsize=15)
    ax.tick_params(axis='both', which='both', labelsize=15)
    if log_x:
        ax.set_xscale('log')
    if y_tick_f:
        # ax.set_yscale('symlog')
        # ax.set_yticks(y_tick)
        # ax.set_yscale('log')  # or 'logit'
        ax.set_yscale('function', functions=(forward, inverse))
        ax.set_ylim(y_tick)
        # ax.yaxis.set_inverted(True)
    # ax.tick_params(axis='both', size=12)
    algo_dict_rev = {v: k for k, v in algo_dict.items()}
    x = np.array(df[0][1:].to_list(), dtype=np.float32)
    data_names = df.loc[0]
    data_dict = {}
    for data_name, i in zip(data_names, range(len(data_names))):
        if "step" in data_name or "MIN" in data_name or "MAX" in data_name:
            pass
        else:
            data_dict[data_name.split(' ')[0]] = df[i][1:]
    #one data
    for name in algo_dict.values():
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
    e_return = os.path.dirname(__file__) + "/metric2" + "/e_return.csv"
    e_len = os.path.dirname(__file__) + "/metric2" + "/e_len.csv"
    e_succ = os.path.dirname(__file__) + "/metric2" + "/e_success.csv"
    # t_buffer = os.path.dirname(__file__) + "/metric1" + "/t_buffer.csv"
    algo_dict = {"D3QN":"noe_FactoryTaskAllocationMiC_2024-12-09_21-42-46", 
                 "EDQN1":"edqn_2024-12-11_13-29-49", 
                 "EDQN2":"no_dueling_2024-12-10_13-06-23", 
                 "EQX-G":"epsilon_FactoryTaskAllocationMiC_2024-12-08_17-36-58", 
                 "EQX-N":"FactoryTaskAllocationMiC_2024-12-08_15-44-10", 
                 "EQX-GN":"epsilon_nosiy_FactoryTaskAllocationMiC_2024-12-09_14-31-02"}
    color_dict = {'D3QN': 'crimson', 'EDQN1': 'orange', 'EDQN2': 'forestgreen', 'EQX-G': 'dodgerblue', 'EQX-N': 'palevioletred', 'EQX-GN':'blueviolet'}
    
    data_list = [e_return, e_len, e_succ]
    titles = ["Return", "Makespan", "Progress"]
    x_lables = ["Episode", "Episode", "Episode"]
    log_x = [False, False, False]
    y_tick_f = [True, False, True]
    y_tick = [[2.7, 2.92], [], [0.8, 1.01]]
    # https://stackoverflow.com/questions/74001262/how-to-change-x-axis-to-not-be-uniform-scale
    smooth_alphas = [0.01, 0.01, 0.01]
    # y_lables = ["Loss", "Return", "Timespan", "Progress"]
    draw_metric_from_csv_res(data_list, algo_dict, titles, x_lables, color_dict, log_x, y_tick_f, y_tick, smooth_alphas, 'metric_2')
    







