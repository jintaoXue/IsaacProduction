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

flatui = ["#9b59b6", "#3498db", "#95a5a6", "#e74c3c", "#34495e", "#2ecc71"]
    
def draw_train_metric_from_csv_res(loss_path, hit1_path, hit3_path, hit6_path):
    import matplotlib.pyplot as plt 
    import matplotlib.colors as mcolors 
    def get_cmap(n, name='hsv'):
        '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
        RGB color; the keyword argument name must be a standard mpl colormap name.'''
        return plt.cm.get_cmap(name, n)
        # return plt.colormaps.get_cmap(name, n)
    cmap = get_cmap(22)
    color_inter_num = 6

    color_dict = {0: 'darkorange', 4: 'orange', 2: 'forestgreen', 3: 'dodgerblue', 1: 'palevioletred', 5:'blueviolet'}
    # https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html
    styles_dict = plt.style.available
    

    # for i in range(0, self._num_action):
    #     x,y,yaw = _sdc_trj[i,:,0], _sdc_trj[i,:,1], _sdc_trj[i,:,2]
    #     plt.plot(x, y, '-', color=cmap(i), ms=5, linewidth=2)
    #     for j in range(0, horizon):
    #         plt.arrow(x[j], y[j], torch.cos(yaw[j])/5, torch.sin(yaw[j])/5, width=0.01, head_width=0.03, head_length=0.02, fc=cmap(i),ec=cmap(i))

    # plt.style.use('fast')
    plt.style.use('seaborn-v0_8-white')
    # plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['pdf.fonttype'] = 42
    fig = plt.figure(figsize=(10,10), dpi=100)
    # gs = gridspec(1,4, )
    # gs = fig.add_gridspec(1,4) 
    ax_1 = plt.subplot(241)
    ax_2 = plt.subplot(245)
    ax_3 = plt.subplot(242)
    ax_4 = plt.subplot(246)
    ax_5 = plt.subplot(243)
    ax_6 = plt.subplot(247)
    ax_7 = plt.subplot(244)
    ax_8 = plt.subplot(248)
    for ax in fig.get_axes():
        ax.grid(True)
    interval = 500
    '''loss curve plot'''
    df = pd.read_csv(loss_path, header=None)
    # all_items = df.iloc[0]
    #one column is one data
    one_data_len = len(df) - 1
    x_data = np.arange(0, one_data_len)
    ax_1.set_title('Loss\Without Spatial Information', fontsize = 12)
    # ax_1.set_xlabel('Number Agents', fontsize=15)
    ax_1.set_ylabel('Training loss', fontsize=15)
    ax_2.set_title('Loss\With Spatial Information', fontsize = 12)
    ax_2.set_xlabel('Steps', fontsize=15)
    ax_2.set_ylabel('Training loss', fontsize=15)
    for i in np.arange(0, len(df.columns)):
        one_data = df[i]
        data_name = one_data[0]
        data = np.array(one_data[1:]).astype(float)
        suffix = data_name.split('__')[-1]
        if data_name == "Step" or suffix == 'MIN' or suffix == 'MAX':
            continue 
        if "no_spatial" in data_name:
            if 'simple_hgn' in data_name or 'hgn-mc' in data_name:
                ax_1.plot(x_data, data, '-', color=color_dict[0], label='HGN-MC',ms=5, linewidth=2)
            if 'rgcn' in data_name and 'rgcn_lstm' not in data_name:
                ax_1.plot(x_data, data, '-', color=color_dict[1], label='RGCN',ms=5, linewidth=2)
            if 'lstm' in data_name and 'rgcn_lstm' not in data_name:
                ax_1.plot(x_data, data, '-', color=color_dict[2], label='BiLSTM',ms=5, linewidth=2)
            if 'rgcn_lstm' in data_name:
                ax_1.plot(x_data, data, '-', color=color_dict[3], label='RGCN-BiLSTM',ms=5, linewidth=2)
        else:
            if 'simple_hgn' in data_name or 'hgn-mc' in data_name:
                ax_2.plot(x_data, data, '-', color=color_dict[0], label='HGN-MC',ms=5, linewidth=2)
            if 'rgcn' in data_name and 'rgcn_lstm' not in data_name:
                ax_2.plot(x_data, data, '-', color=color_dict[1], label='RGCN',ms=5, linewidth=2)
            if 'lstm' in data_name and 'rgcn_lstm' not in data_name:
                ax_2.plot(x_data, data, '-', color=color_dict[2], label='BiLSTM',ms=5, linewidth=2)
            if 'rgcn_lstm' in data_name:
                ax_2.plot(x_data, data, '-', color=color_dict[3], label='RGCN-BiLSTM',ms=5, linewidth=2)
    '''hit1 curve plot'''
    df = pd.read_csv(hit1_path, header=None)
    # all_items = df.iloc[0]
    #one column is one data
    one_data_len = len(df) - 1
    x_data = np.arange(0, one_data_len)
    ax_3.set_title('Hit@1\Without Spatial Information', fontsize = 12)
    # ax_1.set_xlabel('Number Agents', fontsize=15)
    ax_3.set_ylabel('Hit@1', fontsize=15)
    ax_4.set_title('Hit@1\With Spatial Information', fontsize = 12)
    ax_4.set_xlabel('Steps', fontsize=15)
    ax_4.set_ylabel('Hit@1', fontsize=15)
    for i in np.arange(0, len(df.columns)):
        one_data = df[i]
        data_name = one_data[0]
        data = np.array(one_data[1:]).astype(float)
        suffix = data_name.split('__')[-1]
        if data_name == "Step" or suffix == 'MIN' or suffix == 'MIN':
            continue 
        if "no_spatial" in data_name:
            if 'simple_hgn' in data_name or 'hgn-mc' in data_name:
                ax_3.plot(x_data, data, '-', color=color_dict[0], label='HGN-MC',ms=5, linewidth=2)
            if 'rgcn' in data_name and 'rgcn_lstm' not in data_name:
                ax_3.plot(x_data, data, '-', color=color_dict[1], label='RGCN',ms=5, linewidth=2)
            if 'lstm' in data_name and 'rgcn_lstm' not in data_name:
                ax_3.plot(x_data, data, '-', color=color_dict[2], label='BiLSTM',ms=5, linewidth=2)
            if 'rgcn_lstm' in data_name:
                ax_3.plot(x_data, data, '-', color=color_dict[3], label='RGCN-BiLSTM',ms=5, linewidth=2)
        else:
            if 'simple_hgn' in data_name or 'hgn-mc' in data_name:
                ax_4.plot(x_data, data, '-', color=color_dict[0], label='HGN-MC',ms=5, linewidth=2)
            if 'rgcn' in data_name and 'rgcn_lstm' not in data_name:
                ax_4.plot(x_data, data, '-', color=color_dict[1], label='RGCN',ms=5, linewidth=2)
            if 'lstm' in data_name and 'rgcn_lstm' not in data_name:
                ax_4.plot(x_data, data, '-', color=color_dict[2], label='BiLSTM',ms=5, linewidth=2)
            if 'rgcn_lstm' in data_name:
                ax_4.plot(x_data, data, '-', color=color_dict[3], label='RGCN-BiLSTM',ms=5, linewidth=2)
    '''hit3 curve plot'''
    df = pd.read_csv(hit3_path, header=None)
    # all_items = df.iloc[0]
    #one column is one data
    one_data_len = len(df) - 1
    x_data = np.arange(0, one_data_len)
    ax_5.set_title('Hit@3\Without Spatial Information', fontsize = 12)
    # ax_1.set_xlabel('Number Agents', fontsize=15)
    ax_5.set_ylabel('Hit@3', fontsize=15)
    ax_6.set_title('Hit@3\With Spatial Information', fontsize = 12)
    ax_6.set_xlabel('Steps', fontsize=15)
    ax_6.set_ylabel('Hit@3', fontsize=15)
    for i in np.arange(0, len(df.columns)):
        one_data = df[i]
        data_name = one_data[0]
        data = np.array(one_data[1:]).astype(float)
        suffix = data_name.split('__')[-1]
        if data_name == "Step" or suffix == 'MIN' or suffix == 'MIN':
            continue 
        if "no_spatial" in data_name:
            if 'simple_hgn' in data_name or 'hgn-mc' in data_name:
                ax_5.plot(x_data, data, '-', color=color_dict[0], label='HGN-MC',ms=5, linewidth=2)
            if 'rgcn' in data_name and 'rgcn_lstm' not in data_name:
                ax_5.plot(x_data, data, '-', color=color_dict[1], label='RGCN',ms=5, linewidth=2)
            if 'lstm' in data_name and 'rgcn_lstm' not in data_name:
                ax_5.plot(x_data, data, '-', color=color_dict[2], label='BiLSTM',ms=5, linewidth=2)
            if 'rgcn_lstm' in data_name:
                ax_5.plot(x_data, data, '-', color=color_dict[3], label='RGCN-BiLSTM',ms=5, linewidth=2)
        else:
            if 'simple_hgn' in data_name or 'hgn-mc' in data_name:
                ax_6.plot(x_data, data, '-', color=color_dict[0], label='HGN-MC',ms=5, linewidth=2)
            if 'rgcn' in data_name and 'rgcn_lstm' not in data_name:
                ax_6.plot(x_data, data, '-', color=color_dict[1], label='RGCN',ms=5, linewidth=2)
            if 'lstm' in data_name and 'rgcn_lstm' not in data_name:
                ax_6.plot(x_data, data, '-', color=color_dict[2], label='BiLSTM',ms=5, linewidth=2)
            if 'rgcn_lstm' in data_name:
                ax_6.plot(x_data, data, '-', color=color_dict[3], label='RGCN-BiLSTM',ms=5, linewidth=2)
    '''hit6 curve plot'''
    df = pd.read_csv(hit6_path, header=None)
    # all_items = df.iloc[0]
    #one column is one data
    one_data_len = len(df) - 1
    x_data = np.arange(0, one_data_len)
    ax_7.set_title('Hit@6\Without Spatial Information', fontsize = 12)
    # ax_1.set_xlabel('Number Agents', fontsize=15)
    ax_7.set_ylabel('Hit@6', fontsize=15)
    ax_8.set_title('Hit@6\With Spatial Information', fontsize = 12)
    ax_8.set_xlabel('Steps', fontsize=15)
    ax_8.set_ylabel('Hit@6', fontsize=15)
    for i in np.arange(0, len(df.columns)):
        one_data = df[i]
        data_name = one_data[0]
        data = np.array(one_data[1:]).astype(float)
        suffix = data_name.split('__')[-1]
        if data_name == "Step" or suffix == 'MIN' or suffix == 'MIN':
            continue 
        if "no_spatial" in data_name:
            if 'simple_hgn' in data_name or 'hgn-mc' in data_name:
                ax_7.plot(x_data, data, '-', color=color_dict[0], label='HGN-MC',ms=5, linewidth=2)
            if 'rgcn' in data_name and 'rgcn_lstm' not in data_name:
                ax_7.plot(x_data, data, '-', color=color_dict[1], label='RGCN',ms=5, linewidth=2)
            if 'lstm' in data_name and 'rgcn_lstm' not in data_name:
                ax_7.plot(x_data, data, '-', color=color_dict[2], label='BiLSTM',ms=5, linewidth=2)
            if 'rgcn_lstm' in data_name:
                ax_7.plot(x_data, data, '-', color=color_dict[3], label='RGCN-BiLSTM',ms=5, linewidth=2)
        else:
            if 'simple_hgn' in data_name or 'hgn-mc' in data_name:
                ax_8.plot(x_data, data, '-', color=color_dict[0], label='HGN-MC',ms=5, linewidth=2)
            if 'rgcn' in data_name and 'rgcn_lstm' not in data_name:
                ax_8.plot(x_data, data, '-', color=color_dict[1], label='RGCN',ms=5, linewidth=2)
            if 'lstm' in data_name and 'rgcn_lstm' not in data_name:
                ax_8.plot(x_data, data, '-', color=color_dict[2], label='BiLSTM',ms=5, linewidth=2)
            if 'rgcn_lstm' in data_name:
                ax_8.plot(x_data, data, '-', color=color_dict[3], label='RGCN-BiLSTM',ms=5, linewidth=2)
    
    # c=[1,2,3,4]
    # labels = ['HGN-MC', 'RGCN', 'BiLSTM', 'RGCN-BiLSTM']
    # cmap = mcolors.ListedColormap(['darkorange','palevioletred','forestgreen','dodgerblue'])
    # norm = mcolors.BoundaryNorm([1,2,3,4,5],4)
    # sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    # # cbar=plt.colorbar(sm, ticks=c, orientation='horizontal')
    # cbar = plt.colorbar(sm,ax=ax_8, orientation='horizontal', ticks=c)
    # cbar.set_ticklabels(labels)

    # pos = ax_4.get_position()
    # ax_4.set_position([pos.x0, pos.y0, pos.width, pos.height * 0.85])
    ax_1.legend(
        loc='upper right', 
        bbox_to_anchor=(1.0, 0.9),
        ncol=1, 
        fontsize="12"
    )
    plt.show(block=False)
    fig.tight_layout()
    fig.savefig('{}.pdf'.format('./' + 'training_curves'), bbox_inches='tight')

    return



def draw_metric_from_csv_res(data_list, algo_dict, titles, x_lables, color_dict, log_x, y_tick_f, y_tick, smooth_alphas):
    
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
    ax_2 = plt.subplot(222)
    ax_3 = plt.subplot(223)
    ax_4 = plt.subplot(224)
    for ax in fig.get_axes():
        ax.grid(True)
    # plt.tick_params(axis='both', labelsize=50)
    for i, ax in zip(range(0, len(data_list)), [ax_1, ax_2, ax_3, ax_4]):
        draw_one_sub_pic(ax, data_list[i], titles[i], x_lables[i], algo_dict, color_dict, log_x[i], y_tick_f[i], y_tick[i], smooth_alphas[i])

    # ax_1.legend(
    #     loc='upper right', 
    #     bbox_to_anchor=(1.0, 0.6),
    #     ncol=1, 
    #     fontsize="12"
    # )
    # ax_2.legend(
    #     loc='upper right', 
    #     bbox_to_anchor=(1.0, 0.6),
    #     ncol=1, 
    #     fontsize="12"
    # )
    # plt.show(block=True)
    plt.show(block=False)
    fig.tight_layout()
    path = os.path.dirname(__file__)
    fig.savefig('{}.pdf'.format(path + '/' + 'metric_1'), bbox_inches='tight')
    return

def draw_one_sub_pic(ax, data, title, x_lable, algo_dict, color_dict, log_x, y_tick_f, y_tick, alpha):
    '''loss curve plot'''
    df = pd.read_csv(data, header=None)
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(x_lable, fontsize=15)
    ax.tick_params(axis='both', which='both', labelsize=15)
    if log_x:
        ax.set_xscale('log')
    if y_tick_f is "1":
        # ax.set_yscale('symlog')
        # ax.set_yticks(y_tick)
        # ax.set_yscale('log')  # or 'logit'
        ax.set_yscale('function', functions=(forward, inverse))
        ax.set_ylim(y_tick)
    elif y_tick_f is "2":
        ax.set_yscale('function', functions=(inverse, forward))
        ax.set_ylim(y_tick)
    elif y_tick_f is "3":
        # ax.set_yscale('function', functions=(inverse, forward))
        ax.set_ylim(y_tick)
    # ax.tick_params(axis='both', size=12)
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
        ax.plot(x, smoothed_y, '-', color=color, label=label, ms=5, linewidth=2)
    # ax.legend(
    #     fontsize="x-large",
    #     handlelength=5.0)
        # handleheight=3)
    # get the legend object
    leg = ax.legend(ncol=2)
    # change the line width for the legend
    for line in leg.get_lines():
        line.set_linewidth(6.0)

    return 

def smooth_line(data, alpha=0.005):
    # # 执行指数平滑
    # model = ExponentialSmoothing(pd.to_numeric(data), trend='add', seasonal='add', seasonal_periods=1e4)
    # results = model.fit()

    # # 生成平滑后的数据和预测
    # smoothed = results.fittedvalues
    
    # # 执行Loess平滑
    # _y = np.array(y, dtype=np.float32)
    # lowess = sm.nonparametric.lowess(_y, x, frac=0.1)  # frac参数控制平滑带宽，可以调整以获得不同的平滑度

    # # 获取平滑后的数据
    # x_smooth, y_smooth = lowess.T

    # 定义平滑参数（通常称为平滑因子）
    # alpha = 0.005

    # # 计算EMA
    # ema = [data[0]]  # 初始EMA值等于第一个数据点
    # for i in range(1, len(data)):
    #     ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
    smoothed_d = data.ewm(alpha=alpha,adjust=False).mean()
    np_data = np.array(smoothed_d.to_list(), dtype=np.float32)
    return np_data

# Function x**(1/2)
def forward(x):
    return x**(2)


def inverse(x):
    return x**(1/2)

'''=========================================================Main drawing code=========================================================='''
if __name__ == '__main__':
    ### training metric
    t_loss = os.path.dirname(__file__) + "/metric1" + "/t_loss.csv"
    t_return = os.path.dirname(__file__) + "/metric1" + "/t_return.csv"
    t_len = os.path.dirname(__file__) + "/metric1" + "/t_len.csv"
    t_succ = os.path.dirname(__file__) + "/metric1" + "/t_success.csv"
    # t_buffer = os.path.dirname(__file__) + "/metric1" + "/t_buffer.csv"
    algo_dict = {"D3QN":"noe_FactoryTaskAllocationMiC_2024-12-09_21-42-46", 
                 "EDQN1":"edqn_2024-12-11_13-29-49", 
                 "EDQN2":"no_dueling_2024-12-10_13-06-23", 
                 "NoSp":"no_spatial_rainbowmini_2024-12-23_18-12-29",
                 "EBQ-G":"epsilon_FactoryTaskAllocationMiC_2024-12-08_17-36-58", 
                 "EBQ-N":"FactoryTaskAllocationMiC_2024-12-08_15-44-10", 
                 "EBQ-GN":"epsilon_nosiy_FactoryTaskAllocationMiC_2024-12-09_14-31-02",
                 }
    color_dict = {'D3QN': 'crimson', 'EDQN1': 'orange', 'EDQN2': 'forestgreen', 'EBQ-G': 'dodgerblue', 'EBQ-N': 'palevioletred', 'EBQ-GN':'blueviolet', "NoSp": 'silver'}
    
    data_list = [t_loss, t_return, t_len, t_succ]
    titles = ["Loss", "Return", "Makespan", "Progress"]
    x_lables = ["Step", "Episode", "Episode", "Episode"]
    log_x = [True, False, False, False]
    y_tick_f = ["None", "None", "2", "1"]
    y_tick = [[2.7, 2.92], [0, 8], [800, 1200], [0.2, 1.05]]
    smooth_alphas = [0.01, 0.005, 0.001, 0.005]
    # y_lables = ["Loss", "Return", "Timespan", "Progress"]
    draw_metric_from_csv_res(data_list, algo_dict, titles, x_lables, color_dict, log_x, y_tick_f, y_tick, smooth_alphas)
    


    # draw_roc(os.path.join(os.getcwd(), 'results/tfr_df.csv'), 
    #          os.path.join(os.getcwd(), 'results/roc_df.csv'), 
    #          os.path.join(os.getcwd(), 'roc'))
    
    # draw_prcurve(os.path.join(os.getcwd(), 'results/pr_df.csv'),  
    #          os.path.join(os.getcwd(), 'prg'))
        
    # plot mapping performance
    # result_pth = os.path.join(os.path.join(os.getcwd(),'results'), 'record res')
    
    #bert_pth = os.path.join(result_pth, 'BERT_res.xls')
    #bert_df, bert_df_detail = process_xls_sheets(bert_pth, 'BERT') #bert_sp_df, bert_nosp_df, bert_sp_detail_df, bert_nosp_detail_df
    #
    #lstm_pth = os.path.join(result_pth, 'LSTM_res.xls')
    #lstm_df, lstm_df_detail = process_xls_sheets(lstm_pth, 'BiLSTM')
    #
    #dnn_pth = os.path.join(result_pth, 'DNN_res.xls')
    #dnn_df, dnn_df_detail = process_xls_sheets(dnn_pth, 'DNN')
    #
    # total_df = pd.concat([lstm_df, dnn_df, bert_df])
    #total_df_detail = pd.concat([lstm_df_detail, dnn_df_detail, bert_df_detail])
    #
    # total_pth = os.path.join(result_pth, 'total.png')
    #total_detail_pth = os.path.join(result_pth, 'total_detail.png')
    
    #box_plot_figure(total_df, total_pth)
    #box_plot_figure(total_df_detail, total_detail_pth, detail=True)
    
    # plot loss curves
    #loss_pth = os.path.join(result_pth, 'Loss.xls')
    #xls_file = xlrd.open_workbook(loss_pth)
    #loss_plot(xls_file, 0, save_name='lstm-g.png')
    #loss_plot(xls_file, 1, save_name='lstm-d.png')
    #loss_plot(xls_file, 2, save_name='other-g.png')
    #loss_plot(xls_file, 3, save_name='other-d.png')
    
    # tx0 = 0
    # tx1 = 25
    # ty0 = 0.055
    # ty1 = 0.10
    #
    # sx = [tx0, tx1, tx1, tx0, tx0]
    # sy = [ty0, ty0, ty1, ty1, ty0]
    # plt.plot(sx, sy,'purple', linewidth=2)
    #
    # axins = inset_axes(ax1, width=2, height=1.5, loc='center right')
    # axins.plot(steps, loss_df['LSTM-SP'], color=flatui[0], ls='-', linewidth=1.2)
    # axins.plot(steps, loss_df['LSTM-NSP'], color=flatui[3], ls='-', linewidth=1.2)
    # axins.axis([tx0, tx1, ty0, ty1])







