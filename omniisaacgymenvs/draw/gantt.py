import matplotlib.pyplot as plt
import os 
import pickle
import numpy as np
def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
    # return plt.colormaps.get_cmap(name, n)
cmap = get_cmap(11)
color_dict_inv = {0: 'none', 1: 'hoop_preparing', 2:'bending_tube_preparing', 3:'hoop_loading_inner', 4:'bending_tube_loading_inner', 5:'hoop_loading_outer', 
    6:'bending_tube_loading_outer', 7:'cutting_cube', 8:'collect_product', 9:'placing_product'}
color_dict =  {v: k for k, v in color_dict_inv.items()}
y_pos_dic = {""}
y_pos_dic = {'robot1':(10,5),'robot0':(20,5), 'human1':(30,5), 'human0':(40,5), 'action':(50,5)}

y_ticklabels = list(y_pos_dic.keys())
y_ticks = [15-2.5, 25-2.5, 35-2.5,45-2.5, 55-2.5]
def draw(ax, data_list, title, vis_flag):
    labeled_color_dict = {}
    for data in data_list:
        for y_label, color, s_time, diff_time in data:
            if y_label == 'action':
                ax.arrow(s_time, sum(y_pos_dic[y_label]), 0, -2, color=cmap(color), width=1.0, shape='full', head_starts_at_zero=False)
                continue
            if color not in labeled_color_dict.keys():
                labeled_color_dict[color] = 'labeled'
                ax.broken_barh([(s_time, diff_time)], y_pos_dic[y_label], facecolors=cmap(color),edgecolor="black", label="task_{}".format(color),zorder=2)
            else:
                ax.broken_barh([(s_time, diff_time)], y_pos_dic[y_label], facecolors=cmap(color),edgecolor="black",zorder=2)


    ax.set_ylim(5, 60)
    ax.set_xlim(-5, 900)
    for spine in ["top","left","right"]:
        ax.spines[spine].set_color('none')
    if vis_flag:
        ax.set_xlabel('Timespan',fontsize=15)
        # ax.legend()
        # reordering the labels 
        handles, labels = ax.get_legend_handles_labels() 
        idx = [ int(label.split('_')[-1])  for label in labels]
        order = np.argsort(idx) 
        # pass handle & labels lists along with order as below 
        ax.legend([handles[i] for i in order], [labels[i] for i in order], frameon=True,ncol=10,loc = "lower center",bbox_to_anchor=(0.5, -.25)) 
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(direction="out",left=False)
    ax.tick_params()
        
    # ax.set_ylabel('Project Name ',fontsize=15)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_ticklabels, fontsize=15)
    ax.grid(linestyle="-",linewidth=.5,color="gray",alpha=.6)
    # ax.set_title(title,pad=10,fontsize=20,fontweight="bold")
    ax.set_title(title,fontsize=20)
    
    # ax.text(.85,.05,'Visualization by DataCharm',transform = ax.transAxes,
    #             ha='center', va='center',fontsize = 9,color='black')


def proprocess(data : dict):
    last_time = data['charc'][-1][0][0]
    actions = data['action']
    a_time = data['time']
    data_list = []

    a_time.append(last_time)
    diff_time = np.array(a_time[1:]) - np.array(a_time[:-1])
    _color = [ color_dict[act] for act in actions]
    _list = [ ('action',i,j,1) for i,j,k in zip(_color, a_time[:-1], diff_time)]
    #(labelm, )
    data_list.append(_list)

    data_charc = np.array(data['charc'])
    data_list.append(process_helper(data_charc[:,0], 'human0'))
    data_list.append(process_helper(data_charc[:,1], 'human1'))

    data_robot = np.array(data['agv'])
    data_list.append(process_helper(data_robot[:,0], 'robot0'))
    data_list.append(process_helper(data_robot[:,1], 'robot1'))

    return data_list

def process_helper(data, label):
    list = []
    temp_list = []
    for _d in data:
        if _d[-1] != 'free':
            if len(temp_list) == 0:
                temp_list.append(_d)
            elif temp_list[-1][-1] == _d[-1]:
                temp_list.append(_d)
            else:
                get_temp_data(list, temp_list, label)
                temp_list = []
        else:
            if len(temp_list) == 0:
                continue
            else:
                get_temp_data(list, temp_list, label)
                temp_list = []
    if len(temp_list)>0:
        get_temp_data(list, temp_list, label)
    return list

def get_temp_data(list, temp_list, label):
    color = color_dict[temp_list[-1][-1]] 
    s_time = int(temp_list[0][0])
    diff_time = int(temp_list[-1][0])-int(temp_list[0][0]) + 1 
    list.append((label,color,s_time, diff_time))
    return

if __name__ == '__main__':
    noe_path = os.getcwd() + '/omniisaacgymenvs/draw/gantt' + '/' + 'noe_data'
    n_path = os.getcwd() + '/omniisaacgymenvs/draw/gantt' + '/' + 'n_data'
    # color_dict = {'D3QN': 'crimson', 'EDQN1': 'orange', 'EDQN2': 'forestgreen', 'EQX-G': 'dodgerblue', 'EQX-N': 'palevioletred', 'EQX-GN':'blueviolet'}
    data_noe = {}
    data_n = {}
    with open(noe_path, 'rb') as f:
        data_noe = pickle.load(f)
    with open(n_path, 'rb') as f:
        data_n = pickle.load(f)
    
    pic_name = ['EQX-N', 'D3QN'] 
    n_date = proprocess(data_n)
    noe_date = proprocess(data_noe)


    fig = plt.figure(figsize=(20,12), dpi=100)
    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)
    plt.style.use('seaborn-v0_8-white')
    plt.rcParams['pdf.fonttype'] = 42
    params = {'legend.fontsize': 15,
            'legend.handlelength': 2}
    plt.rcParams.update(params)

    draw(ax1, n_date, 'EQX-N', False)
    draw(ax2, noe_date, 'D3QN', True)

    plt.tight_layout()
    # plt.show()
    path = os.path.dirname(__file__)
    fig.savefig('{}.pdf'.format(path + '/' + 'gantt'), bbox_inches='tight')
