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
    max_h = 3
    max_r = 3
    data_array = [[],[],[]] 
    for key, val in hr_dict.items():
        one_data = [float(str) for str in val.split('&')][:max_h*max_r]
        _array = np.array(one_data).reshape(max_h, max_r)
        for i in range(0, max_h):
            data_array[i].append(_array[i])
    return np.array(data_array)
        # assert len(one_data)==10, "warning"

def draw(data, data_dict, color_dict, legend_fs):
    plt.style.use('seaborn-v0_8-white')
    # plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['pdf.fonttype'] = 42
    # plt.tick_params(axis='both', labelsize=50)
    params = {'legend.fontsize': 15,
          'legend.handlelength': 2}
    plt.rcParams.update(params)
    # plt.rcParams['xtick.labelsize'] = 14  # x轴刻度标签字体大小
    # plt.rcParams['ytick.labelsize'] = 14  # y轴刻度标签字体大小
    fig = plt.figure(figsize=(20,10), dpi=100)
    # gs = gridspec(1,4, )
    # gs = fig.add_gridspec(1,4) 
    ax_1 = plt.subplot(231)
    ax_2 = plt.subplot(232)
    ax_3 = plt.subplot(233)
    plt.subplot(234)
    plt.subplot(235)
    plt.subplot(236)
    for ax, i, legend_f in zip(fig.get_axes(), range(0,6), legend_fs):
        ax.grid(True)
        ax.set_xticks([1,2,3])
        if i%3 ==0:
            ax.set_ylabel('Makespan', fontsize=15)
        if i in range(0,3):
            draw_helper(ax, "Human={}".format(i+1), 'Number of robots', data[i], color_dict, list(data_dict.keys()))
        if i in range(3,6):
            j = i%3
            draw_helper(ax, "Robot={}".format(j+1), 'Number of humans', data[:,:,j].transpose(), color_dict, list(data_dict.keys()))
        if legend_f == False:
            ax.get_legend().remove()
    plt.tight_layout()
    # plt.show()
    path = os.path.dirname(__file__)
    fig.savefig('{}.pdf'.format(path + '/' + 'polyline'), bbox_inches='tight')

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
    hr_dict = {
        "D3QN":"1013.93 & 983.03  & 961.05& 837.25& 855.96& 746.61&871.47&865.49&737.82 & 874.73", 
        "EDQN1":"1032.35& 1069.95& 1019.08& 755.75& 751.14& 752.90& 750.01&692.63&690.46&834.92", 
        "EDQN2":"991.35& 927.71&922.94& 758.83& 745.86& 760.86& 755.80&702.56&706.45&808.04", 
        "EBQ-G":"945.35 & 958.63& 973.70& 757.06&741.82 &746.26 &749.71&692.60&688.77&806.00", 
        "EBQ-N":"934.35& 954.64&924.31 &758.69 & 764.78& 773.74&749.71&692.89&695.95&805.45", 
        "EBQ-GN":"1137.67& 1020.94& 1003.99& 782.51& 820.10& 750.87&832.52 &714.48&723.65&865.19",
        "NoSp": "983.35& 1181.39&995.71&794.91&798.23&839.59&803.93&812.29&785"
    }
    color_dict = {'D3QN': 'crimson', 'EDQN1': 'orange', 'EDQN2': 'forestgreen', 'EBQ-G': 'dodgerblue', 'EBQ-N': 'palevioletred', 'EBQ-GN':'blueviolet', "NoSp": 'silver'}
    legend = [False, False, False, True, True, True]
    hr_data = convert(hr_dict)
    draw(hr_data, hr_dict, color_dict, legend)
    

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







