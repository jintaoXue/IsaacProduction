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

def convert(hr_dict):
    max_o = 10
    data_array = [] 
    for key, val in hr_dict.items():
        one_data = [float(str) for str in val.split('&')][:max_o]
        data_array.append(one_data)
    return np.array(data_array)
        # assert len(one_data)==10, "warning"

def draw(data, data_dict, color_dict):
    plt.style.use('seaborn-v0_8-white')
    # plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['pdf.fonttype'] = 42
    # plt.tick_params(axis='both', labelsize=50)
    params = {'legend.fontsize': 15,
          'legend.handlelength': 2}
    plt.rcParams.update(params)
    # plt.rcParams['xtick.labelsize'] = 14  # x轴刻度标签字体大小
    # plt.rcParams['ytick.labelsize'] = 14  # y轴刻度标签字体大小
    fig = plt.figure(figsize=(20,5), dpi=100)
    # gs = gridspec(1,4, )
    # gs = fig.add_gridspec(1,4) 
    plt.subplot(111)
    for ax, i in zip(fig.get_axes(), range(0,6)):
        ax.grid(True)
        ax.set_xticks(range(1,11))
        draw_helper(ax, "Zero shot: makespan", 'Number order', data, color_dict, list(data_dict.keys()))
    plt.tight_layout()
    # plt.show()
    path = os.path.dirname(__file__)
    fig.savefig('{}.pdf'.format(path + '/' + 'polyline2'), bbox_inches='tight')

def draw_helper(ax, title, x_lable, data, color, line_labels):
    ax.set_title(title, fontsize=20)
    ax.set_xlabel(x_lable, fontsize=15)
    ax.tick_params(axis='both', which='both', labelsize=15)
    for _data, line_label in zip(data, line_labels):
        x = np.arange(1, len(_data)+1)
        ax.plot(x, _data, '-', color=color[line_label], label=line_label, ms=5, linewidth=2, marker='o')
    ax.legend()


'''=========================================================Main drawing code=========================================================='''
if __name__ == '__main__':
    ###human robot number peformance
    # t_buffer = os.path.dirname(__file__) + "/metric1" + "/t_buffer.csv"
    o_dict = {
        "D3QN":"1336.32 & 817.98  & 1051.68 & 1138.52& 879.73& 1032.91& 1322.52 & 1403.76&1555.34&1720.59", 
        "EDQN1":"354.78 & 745.46 & 577.41& 846.89& 839.91& 1033.08 & 1160.78&1329.76&1463.38& 1578.36", 
        "EDQN2":"365.21 & 616.91 & 584.10& 717.67& 819.80&998.68 &1155.13&1308.30&1496.31& 1600.91", 
        "EQX-G":"312.91 & 454.68 & 576.31& 850.27& 810.13& 989.74&1156.67&1350.73&1511.72&1612.56", 
        "EQX-N":"316.42 & 442.59 & 561.81 & 733.99& 810.63& 984.82 & 1147.13&1343.44&1499.92& 1636.28", 
        "EQX-GN":"324.00 & 451.73 & 665.94 & 818.73& 864.74& 1023.90&1190.40&1320.18&1653.77&1655.14"
    }
    color_dict = {'D3QN': 'crimson', 'EDQN1': 'orange', 'EDQN2': 'forestgreen', 'EQX-G': 'dodgerblue', 'EQX-N': 'palevioletred', 'EQX-GN':'blueviolet'}
    
    o_data = convert(o_dict)
    draw(o_data, o_dict, color_dict)
    


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







