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

def cacu_env_len(data_path, algo_dict):
    res_len = {}
    for i in range(0, len(data_path)):
        m_len = cacu_env_len_helper(data_path=data_path[i], algo_dict=algo_dict)
        id  = (data_path[i].split('/')[-1]).split('_')[0]
        res_len[id+'_len'] = m_len
    return res_len

def cacu_env_len_helper(data_path, algo_dict):
    df = pd.read_csv(data_path, header=None)
    data_names = df.loc[0]
    _dict = {}
    algo_dict_rev = {v: k for k, v in algo_dict.items()}
    for data_name, i in zip(data_names, range(len(data_names))):
        if "step" in data_name or "MIN" in data_name or "MAX" in data_name:
            pass
        else:
            alog_name = data_name.split(' ')[0]
            alog_name = algo_dict_rev[alog_name]
            _dict[alog_name] = df[i][1:].astype(float).mean()
    return _dict 


def cacu_success_progress(data_path, algo_dict):
    res_succ, res_prog = {}, {}
    for i in range(0, len(data_path)):
        succ, prog = cacu_success_progress_helper(data_path=data_path[i], algo_dict=algo_dict)
        id  = (data_path[i].split('/')[-1]).split('_')[0]
        res_succ[id+'_succ'] = succ
        res_prog[id+'_prog'] = prog
    return res_succ, res_prog

def cacu_success_progress_helper(data_path, algo_dict):
    df = pd.read_csv(data_path, header=None)
    data_names = df.loc[0]
    data_dict_prog = {}
    data_dict_succ = {}
    algo_dict_rev = {v: k for k, v in algo_dict.items()}
    for data_name, i in zip(data_names, range(len(data_names))):
        if "step" in data_name or "MIN" in data_name or "MAX" in data_name:
            pass
        else:
            alog_name = data_name.split(' ')[0]
            alog_name = algo_dict_rev[alog_name]

            data_dict_prog[alog_name] = df[i][1:].astype(float).mean()

            array = df[i][1:].astype(float).to_numpy()
            data_dict_succ[alog_name] = (np.where(array<1.0, 0, 1.0)).mean()
    return data_dict_prog, data_dict_succ 


def mean_res(data):
    m_res =  {}
    num = len(data.keys())
    for key, val in data.items():
        for k, v in val.items():
            if k not in m_res.keys():
                m_res[k]=v/num
            else:
                m_res[k]+=v/num    
    return m_res
'''=========================================================Main drawing code=========================================================='''
if __name__ == '__main__':
    ### zero_shot performance
    #from o1 to o10
    datas_len = []
    datas_succ = []
    for i in range(1, 11):
        _len = os.path.dirname(__file__) + "/metric3/zero_shot" + "/{}_len.csv".format(i)
        _succ = os.path.dirname(__file__) + "/metric3/zero_shot" + "/{}_succ.csv".format(i)
        datas_len.append(_len)
        datas_succ.append(_succ)
    algo_dict = {"D3QN":"test_rainbownoe_ep_8000.pth_2024-12-09_21-42-46", 
                 "EDQN1":"test_edqn_ep_10500.pth_2024-12-11_13-29-49", 
                 "EDQN2":"test_no_dueling_ep_24900.pth_2024-12-10_13-06-23", 
                 "EQX-G":"test_rainbowepsilon_ep_19500.pth_2024-12-08_17-36-58", 
                 "EQX-N":"test_rainbowmini_ep_24000.pth_2024-12-08_15-44-10", 
                 "EQX-GN":"test_epsilon_noisy_ep_5700.pth_2024-12-09_14-31-02"}
    
    res_succ, res_prog = cacu_success_progress(datas_succ, algo_dict)
    res_len = cacu_env_len(datas_len, algo_dict)
    m_succ = mean_res(res_succ)
    m_len = mean_res(res_len)
    a=1
    
    







