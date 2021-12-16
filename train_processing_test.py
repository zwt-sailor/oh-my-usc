#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
import pdb
import zipfile
import csv
from io import StringIO
import matplotlib
warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
import multiprocessing as mp
import pickle
import random
import re
import os
import pandas as pd
import numpy as np


# In[2]:


data_train = pd.read_csv("/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train.csv")         
print('The shape of train data:', data_train.shape)
data_train.head()


# In[3]:


DataFrame1 = data_train.copy() # 建立数据副本，以便多次修改

DataFrame1


# In[11]:


data1 = DataFrame1[DataFrame1.type.isin(['围网'])]  

data1


# In[12]:


data2 = DataFrame1[DataFrame1.type.isin(['拖网'])]  

data2


# In[14]:


data3 = DataFrame1[DataFrame1.type.isin(['刺网'])]  

data3


# In[10]:





# In[5]:


label_dic1 = {'拖网': 0, '围网': 1, '刺网': 2}
label_dic2 = {0: '拖网', 1: '围网', 2: '刺网'}
name_dic = {'渔船ID': 'id', '速度': 'v', '方向': 'direct', 'type': 'label'}

# 修改列名
data_train.rename(columns=name_dic, inplace=True)

# 将标签映射为数值
data_train['label'] = data_train['label'].map(label_dic1)

# # 转换数据格式
# cols = ['lat', 'lon', 'v']
# for col in cols:
#     data_train[col] = data_train[col].astype('float')
# data_train['direct'] = data_train['direct'].astype('int')

# 将时间数据转换成datetime格式，并抽取出相关的时间特征
data_train['time'] = pd.to_datetime(data_train['time'], format='%Y-%m-%d %H:%M:%S')
data_train['date'] = data_train['time'].dt.date
data_train['hour'] = data_train['time'].dt.hour
data_train['month'] = data_train['time'].dt.month
data_train['weekday'] = data_train['time'].dt.weekday

data_train.head()


# In[6]:


data_train['day_night'] = 0
data_train.loc[(data_train['hour']>5) & (data_train['hour']<20), 'day_night'] = 1
data_train.head()


# In[7]:


data_train['quarter'] = 0
data_train.loc[(data_train['month'].isin([1, 2, 3])), 'quarter'] = 1
data_train.loc[(data_train['month'].isin([4, 5, 6])), 'quarter'] = 2
data_train.loc[(data_train['month'].isin([7, 8, 9])), 'quarter'] = 3
data_train.loc[(data_train['month'].isin([10, 11, 12])), 'quarter'] = 4
data_train.head()


# In[8]:


tmp_df = data_train.copy()
tmp_df.rename(columns={'id': 'ship', 'direct': 'd'}, inplace=True)

# 将速度划分等级
def vlevel(v):
    if v < 0.1:
        return 0
    elif v < 0.5:
        return 1
    elif v < 1:
        return 2
    elif v < 2.5:
        return 3
    elif v < 5:
        return 4
    elif v < 10:
        return 5
    elif v < 20:
        return 6
    else:
        return 7

# 统计每条渔船各个速度等级的数目
def get_vlevel_cnt(df):
    df['vlevel'] = df['v'].apply(lambda x: vlevel(x))
    tmp = df.groupby(['ship', 'vlevel'], as_index=False)['vlevel'].agg({'vlevel_cnt': 'count'})
    tmp = tmp.pivot(index='ship', columns='vlevel', values='vlevel_cnt')
    
    new_col_name = ['vlevel_' + str(col) for col in tmp.columns.tolist()]
    tmp.columns = new_col_name
    tmp = tmp.reset_index()
    
    return tmp

c1 = get_vlevel_cnt(tmp_df)
c1.head()


# In[7]:


# 将方向分为16等分
def direct_level(df):
    df['d16'] = df['d'].apply(lambda x: int((x/22.5)+0.5)%16 if not np.isnan(x) else np.nan)
    return df

def get_direct_level_cnt(df):
    df = direct_level(df)
    tmp = df.groupby(['ship', 'd16'], as_index=False)['d16'].agg({'d16_cnt': 'count'})
    tmp = tmp.pivot(index='ship', columns='d16', values='d16_cnt')
    
    new_col_name = ['d16_' + str(col) for col in tmp.columns.tolist()]
    tmp.columns = new_col_name
    tmp = tmp.reset_index()
    
    return tmp

c2 = get_direct_level_cnt(tmp_df)
c2.head()


# In[9]:


name_dic = {'lat': 'x', 'lon': 'y'}

# 修改列名
data_train.rename(columns=name_dic, inplace=True)
data_train.head()


# In[11]:


# 对速度进行200分位数分箱，例如将前1/200分入一个桶
data_train['v_bin'] = pd.qcut(data_train['v'], 200, duplicates='drop') 
# 分箱后进行映射编码
data_train['v_bin'] = data_train['v_bin'].map(dict(zip(data_train['v_bin'].unique(), range(data_train['v_bin'].nunique()))))


# In[15]:


import math

def traj_to_bin(traj, x_min, x_max, y_min, y_max, row_bins, col_bins):
    # row_bins = (x_max - x_min) / 700，即将x坐标每700划分为一个区域
    # col_bins = (y_max - y_min) / 3000，即将y坐标每3000划分为一个区域
    x_bins = np.linspace(x_min, x_max, endpoint=True, num=row_bins+1)  # 注意row_bins+1是包括两端点的划分点个数
    y_bins = np.linspace(y_min, y_max, endpoint=True, num=col_bins+1)
    
    # 确定每个lat坐标属于哪个区域，对x坐标进行分箱编码
    traj.sort_values(by='x', inplace=True)
    x_res = np.zeros((len(traj), ))
    j = 0
    for i in range(1, row_bins+1):
        low, high = x_bins[i-1], x_bins[i]
        while j < len(traj):
            # low-0.001是为了保证数值稳定性，因为linspace得到的划分点精度可能与给出的坐标精度不同
            if (traj['x'].iloc[j] <= high) & (traj['x'].iloc[j] > low-0.001):
                x_res[j] = i
                j += 1
            else:
                break
    traj['x_grid'] = x_res
    traj['x_grid'] = traj['x_grid'].astype('int')
    traj['x_grid'] = traj['x_grid'].apply(str)
    
    # 确定每个lon坐标属于哪个区域，对y坐标进行分箱编码
    traj.sort_values(by='y', inplace=True)
    y_res = np.zeros((len(traj), ))
    j = 0
    for i in range(1, col_bins+1):
        low, high = y_bins[i-1], y_bins[i]
        while j < len(traj):
            # low-0.001是为了保证数值稳定性，因为linspace得到的划分点精度可能与给出的坐标精度不同
            if (traj['y'].iloc[j] <= high) & (traj['y'].iloc[j] > low-0.001):
                y_res[j] = i
                j += 1
            else:
                break
    traj['y_grid'] = y_res
    traj['y_grid'] = traj['y_grid'].astype('int')
    traj['y_grid'] = traj['y_grid'].apply(str)
    
    # 确定每个lat、lon坐标对属于哪个区域，lat坐标编码与lon坐标编码组合成区域编码
    traj['x_y_grid'] = [i + '_' + j for i, j in zip(traj['x_grid'].values.tolist(), traj['y_grid'].values.tolist())]
    
    traj.sort_values(by='time', inplace=True)
    
    return traj

x_min, x_max = data_train['x'].min(), data_train['x'].max()
y_min, y_max = data_train['y'].min(), data_train['y'].max()
row_bins = math.ceil((x_max - x_min) / 700)  # 向上取整
col_bins = math.ceil((y_max - y_min) / 3000)

data_train = traj_to_bin(data_train, x_min, x_max, y_min, y_max, row_bins, col_bins)

data_train.head()


# In[16]:


df = data_train.copy()
name_dic = {'id':'ID','x': 'lat', 'y': 'lon','v':'speed'}

# 修改列名
df.rename(columns=name_dic, inplace=True)
df.head()


# In[ ]:


# 本行与下一行的经纬度、速度、分钟的差值
df['lat_diff'] = df.groupby('ID')['lat'].diff(1)
df['lon_diff'] = df.groupby('ID')['lon'].diff(1)
df['speed_diff'] = df.groupby('ID')['speed'].diff(1)
df['diff_minutes'] = df.groupby('ID')['time'].diff(1).dt.seconds // 60

# 找出锚点
df['anchor'] = df.apply(lambda x: 1 if x['lat_diff'] < 0.01 and x['lon_diff'] < 0.01
                        and x['speed'] < 0.1 and x['diff_minutes'] <= 10 else 0, axis=1)

# 选取出经纬度均不为0 以及速度不为0
lat_lon_neq_zero = df[(df['lat_diff'] != 0) & (df['lon_diff'] != 0)]
speed_neg_zero = df[df['speed_diff'] != 0]


# In[2]:


data_train


# In[23]:


raw_cols = data_train.columns.tolist()

# 获得序列的起始值
def start(x):
    try:
        return x[0]
    except:
        return None

# 获得序列的结束值
def end(x):
    try:
        return x[-1]
    except:
        return None

# 获得序列的众数    
def mode(x):
    try:
        return pd.Series(x).value_counts().index[0]
    except:
        return None

# 获取每条渔船的dist_prev_move_bin和v_bin序列
for f in ['dist_prev_move_bin', 'v_bin']:
    train_df[f + '_seq'] = train_df['id'].map(train_df.groupby('id')[f].agg(lambda x: ','.join(x.astype(str))))
    
# 对序列构造一些基本的统计特征，其中np.ptp用于计算最大最小值之间的差值
tmp_df = data_train.groupby('id').agg({
    'id': ['count'], 'x_bin1': [mode], 'y_bin1': [mode], 'x_bin2': [mode], 'y_bin2': [mode], 'x_y_bin1': [mode],
    'x': ['mean', 'max', 'min', 'std', np.ptp, start, end],
    'y': ['mean', 'max', 'min', 'std', np.ptp, start, end],
    'v': ['mean', 'max', 'min', 'std', np.ptp],
    'direct': ['mean'], 'x_bin1_cnt': ['mean', 'max', 'min'], 
    'y_bin1_cnt': ['mean', 'max', 'min'],
    'x_bin2_cnt': ['mean', 'max', 'min'], 
    'y_bin2_cnt': ['mean', 'max', 'min'], 
    'x_y_bin1_cnt': ['mean', 'max', 'min'],
    'dist_prev_move': ['mean', 'max', 'min', 'std', 'sum'],
    'y_xbin1_ymax_diff': ['mean', 'min'], 
    'y_xbin1_ymin_diff': ['mean', 'min'],
    'x_ybin1_xmax_diff': ['mean', 'min'],
    'x_ybin1_xmin_diff': ['mean', 'min']
}).reset_index()

tmp_df.columns = ['_'.join(col).strip()for col in tmp_df.columns]
tmp_df.rename(columns={'id_': 'id'}, inplace=True)
cols = [f for f in tmp_df.keys() if f !='id']


# In[11]:


import warnings
import pdb
import zipfile
import csv
from collections import Counter
from io import StringIO
import matplotlib
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings('ignore')
from matplotlib import pyplot as plt
import seaborn as sns
from tqdm import tqdm
import multiprocessing as mp
import pickle
import random
import re
import os
import pandas as pd
import numpy as np

data_train = pd.read_csv("/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train.csv")
label_dic1 = {'拖网': 0, '围网': 1, '刺网': 2}
label_dic2 = {0: '拖网', 1: '围网', 2: '刺网'}
name_dic = {'渔船ID': 'id', '速度': 'v', '方向': 'direct', 'type': 'label'}

# 修改列名
data_train.rename(columns=name_dic, inplace=True)

# 将标签映射为数值
data_train['label'] = data_train['label'].map(label_dic1)

# # 转换数据格式
# cols = ['lat', 'lon', 'v']
# for col in cols:
#     data_train[col] = data_train[col].astype('float')
# data_train['direct'] = data_train['direct'].astype('int')

# 将时间数据转换成datetime格式，并抽取出相关的时间特征
data_train['time'] = pd.to_datetime(data_train['time'], format='%Y-%m-%d %H:%M:%S')
data_train['date'] = data_train['time'].dt.date
data_train['hour'] = data_train['time'].dt.hour
data_train['month'] = data_train['time'].dt.month
data_train['weekday'] = data_train['time'].dt.weekday

data_train.head()

class nmf_list(object):
    def __init__(self,data,by_name,to_list,nmf_n,top_n):
        self.data = data
        self.by_name = by_name
        self.to_list = to_list
        self.nmf_n = nmf_n
        self.top_n = top_n

    def run(self,tf_n):
        df_all = self.data.groupby(self.by_name)[self.to_list].apply(lambda x :'|'.join(x)).reset_index()
        self.data =df_all.copy()

        print('bulid word_fre')
        # 词频的构建
        def word_fre(x):
            word_dict = []
            x = x.split('|')
            docs = []
            for doc in x:
                doc = doc.split()
                docs.append(doc)
                word_dict.extend(doc)
            word_dict = Counter(word_dict)
            new_word_dict = {}
            for key,value in word_dict.items():
                new_word_dict[key] = [value,0]
            del word_dict
            del x
            for doc in docs:
                doc = Counter(doc)
                for word in doc.keys():
                    new_word_dict[word][1] += 1
            return new_word_dict
        self.data['word_fre'] = self.data[self.to_list].apply(word_fre)

        print('bulid top_' + str(self.top_n))
        # 设定100个高频词
        def top_100(word_dict):
            return sorted(word_dict.items(),key = lambda x:(x[1][1],x[1][0]),reverse = True)[:self.top_n]
        self.data['top_'+str(self.top_n)] = self.data['word_fre'].apply(top_100)
        def top_100_word(word_list):
            words = []
            for i in word_list:
                i = list(i)
                words.append(i[0])
            return words
        self.data['top_'+str(self.top_n)+'_word'] = self.data['top_' + str(self.top_n)].apply(top_100_word)
        # print('top_'+str(self.top_n)+'_word的shape')
        print(self.data.shape)

        word_list = []
        for i in self.data['top_'+str(self.top_n)+'_word'].values:
            word_list.extend(i)
        word_list = Counter(word_list)
        word_list = sorted(word_list.items(),key = lambda x:x[1],reverse = True)
        user_fre = []
        for i in word_list:
            i = list(i)
            user_fre.append(i[1]/self.data[self.by_name].nunique())
        stop_words = []
        for i,j in zip(word_list,user_fre):
            if j>0.5:
                i = list(i)
                stop_words.append(i[0])

        print('start title_feature')
        # 讲融合后的taglist当作一句话进行文本处理
        self.data['title_feature'] = self.data[self.to_list].apply(lambda x: x.split('|'))
        self.data['title_feature'] = self.data['title_feature'].apply(lambda line: [w for w in line if w not in stop_words])
        self.data['title_feature'] = self.data['title_feature'].apply(lambda x: ' '.join(x))

        print('start NMF')
        # 使用tfidf对元素进行处理
        tfidf_vectorizer = TfidfVectorizer(ngram_range=(tf_n,tf_n))
        tfidf = tfidf_vectorizer.fit_transform(self.data['title_feature'].values)
        #使用nmf算法，提取文本的主题分布
        text_nmf = NMF(n_components=self.nmf_n).fit_transform(tfidf)


        # 整理并输出文件
        name = [str(tf_n) + self.to_list + '_' +str(x) for x in range(1,self.nmf_n+1)]
        tag_list = pd.DataFrame(text_nmf)
        print(tag_list.shape)
        tag_list.columns = name
        tag_list[self.by_name] = self.data[self.by_name]
        column_name = [self.by_name] + name
        tag_list = tag_list[column_name]
        return tag_list


data = data_train.copy()
data.rename(columns={'v':'speed','id':'ship'},inplace=True)
for j in range(1,4):
    print('********* {} *******'.format(j))
    for i in ['speed','x','y']:
        data[i + '_str'] = data[i].astype(str)
        nmf = nmf_list(data,'ship',i + '_str',8,2)
        nmf_a = nmf.run(j)
        nmf_a.rename(columns={'ship':'id'},inplace=True)
        data_label = data_label.merge(nmf_a,on = 'id',how = 'left')

new_cols = [i for i in data_label.columns if i not in data.columns]
data = data.merge(data_label[new_cols+['id']],on='id',how='left')

data[new_cols].head()


# In[1]:


import numpy as np
import pandas as pd
import lightgbm as lgb
import math
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import os
import random
np.random.seed(78)
random.seed(78)

features = []

def dis_lat_lon(lat1, lon1, lat2, lon2):
    R = 6373.0
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = R * c
    return distance

def produce_feature(df, train):
    # 造统计数量的特征,train为时间段总数，df为某时间段中某航速段
    nums = len(df)
    mean = np.mean(df['速度'])
    ratio = len(df) / len(train) if len(train) != 0 else 0
    std = np.std(df['速度'])
    v_ = df['速度'].quantile(0.75)

    return nums, mean, ratio, std, v_


def angle(a, b, c):
    # 计算空间中，连续三点所形成的角度
    ab = [aa - bb for aa, bb in zip(a, b)]
    bc = [bb - cc for cc, bb in zip(c, b)]

    nab = np.sqrt(sum((x ** 2.0 for x in ab)))
    ab = [x / nab for x in ab]

    nbc = np.sqrt(sum((x ** 2.0 for x in bc)))
    bc = [x / nbc for x in bc]
    scal = sum((aa * bb for aa, bb in zip(ab, bc)))
    if scal > 1:
        scal = 1
    elif scal < -1:
        scal = -1
    angle = int(math.acos(scal) * 180 / math.pi)
    angle = 180 - angle
    return angle


def produce_feature_v_xy(df):
    # 造统计各时间段坐标信息
    k = df['y'] / df['x']
    k_min = k.min()
    k_max = k.max()
    # k_mean = k.mean()
    # x_50_ = df['x'].quantile(0.5)
    x_min_ = df['x'].min()
    x_max_ = df['x'].max()
    y_min_ = df['y'].min()
    y_max_ = df['y'].max()
    x_max_y_min_ = df['x'].max() - df['y'].min()
    y_max_x_min_ = df['y'].max() - df['x'].min()
    x_25_ = df['x'].quantile(0.25)
    y_75_ = df['y'].quantile(0.75)
    if len(df) <= 1:
        xy_cov_ = 0
    else:
        xy_cov_ = df['x'].cov(df['y'])
    df['time'] = pd.to_datetime(df['time'], format='%m%d %H:%M:%S')
    t_diff = df['time'].diff().iloc[1:].dt.total_seconds()
    x_diff = df['x'].diff().iloc[1:].abs()
    y_diff = df['y'].diff().iloc[1:].abs()
    x_a_mean = (x_diff / t_diff).mean()
    y_a_mean = (y_diff / t_diff).mean()
    xy_a_ = np.sqrt(x_a_mean ** 2 + y_a_mean ** 2)
    return k_min, k_max, x_min_, x_max_, y_min_, y_max_, x_max_y_min_, y_max_x_min_, x_25_, y_75_, xy_cov_, xy_a_


def produce_feature_ang_ext(df):
    # 构造角度，距离等特征###
    df['time'] = pd.to_datetime(df['time'], format='%Y-%m-%d %H:%M:%S')
    df = df.sort_values(by='time')
    df['hour'] = df['time'].dt.hour
    df_tortuosity = df[['x', 'y', '方向', '速度', 'hour']].values.tolist()
    if len(df_tortuosity) > 1:
        ang_list = [0]
        dis_list = [0]
        for i in range(1, len(df_tortuosity) - 1):
            a = [df_tortuosity[i - 1][0], df_tortuosity[i - 1][1]]
            b = [df_tortuosity[i][0], df_tortuosity[i][1]]
            c = [df_tortuosity[i + 1][0], df_tortuosity[i + 1][1]]
            #             dis = np.sqrt((float((a[0] - b[0]) ** 2) + float((a[1] - b[1]) ** 2)))
            dis = dis_lat_lon(a[0], a[1], b[0], b[1])
            dis_list.append(dis)
            if a == b or b == c or a == c:
                ang_list.append(0)
            else:
                res = angle(a, b, c)
                ang_list.append(int(res))

        #         dis_list.append(np.sqrt((float((df_tortuosity[-1][0] - df_tortuosity[-2][0]) ** 2) + float(
        #             (df_tortuosity[-1][1] - df_tortuosity[-2][1]) ** 2))))
        last_dis = dis_lat_lon(df_tortuosity[-1][0], df_tortuosity[-1][1], df_tortuosity[-2][0], df_tortuosity[-2][1])
        dis_list.append(last_dis)
        ang_list.append(int(ang_list[-1]))
        num_ang_all = len(ang_list)
        num_ang_0_100 = len([x for x in ang_list if x <= 100])
        ratio_ang_0_100 = num_ang_0_100 / num_ang_all

        num_ang_10_150 = len([x for x in ang_list if x > 10 and x < 150])
        ratio_ang_10_150 = num_ang_10_150 / num_ang_all

        num_ang_100_165 = len([x for x in ang_list if x > 100 and x < 165])
        ratio_ang_100_165 = num_ang_100_165 / num_ang_all

        df['est_d'] = ang_list
        df['est_dis'] = dis_list

        t_diff = df['time'].diff().iloc[1:].dt.total_seconds()

        t = [0]
        t.extend(t_diff.values.tolist())
        df['est_t'] = [x / 3600 for x in t]
        df['est_v'] = df['est_d'] / df['est_t']
        beg_end = dis_lat_lon(df_tortuosity[0][0], df_tortuosity[0][1], df_tortuosity[-1][0], df_tortuosity[-1][1])
    elif len(df_tortuosity) == 1:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0  # , 0, 0, 0
    else:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0  # , 0, 0, 0
    return df['est_v'].mean(), df['est_v'].std(), df['est_v'].quantile(0.75), df[
        'est_d'].mean(), num_ang_0_100, ratio_ang_0_100, num_ang_10_150, ratio_ang_10_150, num_ang_100_165, ratio_ang_100_165, beg_end


def x_y_area_count(df, all_df):
    num_all = len(all_df)
    num_ = len(df)
    num_ratio_ = num_ / num_all
    v_mean_c_ = df['速度'].mean()
    v_std_c_ = df['速度'].std()
    d_mean_c_ = df['方向'].mean()
    # x_mean_c = df['x'].mean()
    # x_max_ =
    return [num_, num_ratio_, v_mean_c_, v_std_c_, d_mean_c_]


def read_information(path, know_type=True):
    df = pd.read_csv(path)
    #     print(path)
    if know_type:
        df.columns = ['ship', 'x', 'y', '速度', '方向', 'time', 'type']
    else:
        df.columns = ['ship', 'x', 'y', '速度', '方向', 'time']
    # 构造角度，距离等特征###
    df['time'] = pd.to_datetime(df['time'], format='%m%d %H:%M:%S')
    df = df.sort_values(by='time')
    df['hour'] = df['time'].dt.hour

    df_tortuosity = df[['x', 'y', '速度', '方向', 'hour']].values.tolist()
    ang_list = [0]
    dis_list = [0]
    for i in range(1, len(df_tortuosity) - 1):
        a = [df_tortuosity[i - 1][0], df_tortuosity[i - 1][1]]
        b = [df_tortuosity[i][0], df_tortuosity[i][1]]
        c = [df_tortuosity[i + 1][0], df_tortuosity[i + 1][1]]
        #         dis = np.sqrt((float((a[0] - b[0]) ** 2) + float((a[1] - b[1]) ** 2)))
        dis = dis_lat_lon(a[0], a[1], b[0], b[1])
        dis_list.append(dis)
        if a == b or b == c or a == c:
            ang_list.append(0)
        else:
            res = angle(a, b, c)
            ang_list.append(int(res))
    #     dis_list.append(np.sqrt((float((df_tortuosity[-1][0] - df_tortuosity[-2][0]) ** 2) + float((df_tortuosity[-1][1] - df_tortuosity[-2][1]) ** 2))))
    dis_list.append(dis_lat_lon(df_tortuosity[-1][0], df_tortuosity[-1][1], df_tortuosity[-2][0], df_tortuosity[-2][1]))
    ang_list.append(int(ang_list[-1]))
    num_ang_all = len(ang_list)
    num_ang_0_100 = len([x for x in ang_list if x <= 100])
    ratio_ang_0_100 = num_ang_0_100 / num_ang_all

    num_ang_10_150 = len([x for x in ang_list if x > 10 and x < 150])
    ratio_ang_10_150 = num_ang_10_150 / num_ang_all

    num_ang_100_165 = len([x for x in ang_list if x > 100 and x < 165])
    ratio_ang_100_165 = num_ang_100_165 / num_ang_all

    df['est_d'] = ang_list
    df['d_diff'] = df['est_d'] - df['方向']
    df['est_dis'] = dis_list

    t_diff = df['time'].diff().iloc[1:].dt.total_seconds()

    t = [0]
    t.extend(t_diff.values.tolist())

    df['est_t'] = [x / 3600 for x in t]

    df['est_v_dis'] = df['est_dis'] / df['est_t']  # 这个是est_v
    df['est_v_d'] = df['est_d'] / df['est_t']  # 这个是角速度
    df['v_diff'] = df['速度'] - df['est_v_dis']
    #     beg_end = np.sqrt((float((df_tortuosity[0][0] - df_tortuosity[-1][0]) ** 2) + float(
    #         (df_tortuosity[0][1] - df_tortuosity[-1][1]) ** 2)))
    beg_end = dis_lat_lon(df_tortuosity[0][0], df_tortuosity[0][1], df_tortuosity[-1][0], df_tortuosity[-1][1])
    features.append(int(df['ship'].unique()))
    features.append(df['est_v_d'].mean())
    features.append(df['est_v_d'].std())
    features.append(df['est_v_d'].quantile(0.75))

    features.append(df['v_diff'].mean())
    # features.append(df['v_diff'].max())
    # features.append(df['v_diff'].std())
    # features.append(df['v_diff'].quantile(0.75))

    features.append(df['d_diff'].mean())
    features.append(df['d_diff'].max())
    features.append(df['d_diff'].min())
    # features.append(df['d_diff'].std())
    # features.append(df['d_diff'].quantile(0.75))

    features.append(df['est_d'].mean())
    features.append(num_ang_0_100)
    features.append(ratio_ang_0_100)
    features.append(num_ang_10_150)
    features.append(ratio_ang_10_150)
    features.append(num_ang_100_165)
    features.append(ratio_ang_100_165)
    features.append(beg_end)
    night1 = df[19 <= df['hour']]
    night1 = night1[night1['hour'] < 23]
    night2_1 = df[23 <= df['hour']]
    night2_2 = df[df['hour'] <= 3]
    night2 = pd.concat([night2_1, night2_2], axis=0)
    night = pd.concat([night1, night2_1, night2_2], axis=0)

    day1 = df[3 < df['hour']]
    day1 = day1[day1['hour'] < 10]
    day2 = df[10 <= df['hour']]
    day2 = day2[day2['hour'] < 16]
    day3 = df[16 <= df['hour']]
    day3 = day3[day3['hour'] < 19]
    day = pd.concat([day1, day2, day3], axis=0)

    # 根据时间段划分后再统计
    k_min_1, k_max_1, x_min_n_1, x_max_n_1, y_min_n_1, y_max_n_1, x_max_y_min_n_1, y_max_x_min_n_1, x_25_n_1, y_75_n_1, xy_cov_n_1, xy_a_n_1 = produce_feature_v_xy(
        night1)
    k_min_2, k_max_2, x_min_n_2, x_max_n_2, y_min_n_2, y_max_n_2, x_max_y_min_n_2, y_max_x_min_n_2, x_25_n_2, y_75_n_2, xy_cov_n_2, xy_a_n_2 = produce_feature_v_xy(
        night2)
    k_min_3, k_max_3, x_min_d_1, x_max_d_1, y_min_d_1, y_max_d_1, x_max_y_min_d_1, y_max_x_min_d_1, x_25_d_1, y_75_d_1, xy_cov_d_1, xy_a_d_1 = produce_feature_v_xy(
        day1)
    k_min_4, k_max_4, x_min_d_2, x_max_d_2, y_min_d_2, y_max_d_2, x_max_y_min_d_2, y_max_x_min_d_2, x_25_d_2, y_75_d_2, xy_cov_d_2, xy_a_d_2 = produce_feature_v_xy(
        day2)
    k_min_5, k_max_5, x_min_d_3, x_max_d_3, y_min_d_3, y_max_d_3, x_max_y_min_d_3, y_max_x_min_d_3, x_25_d_3, y_75_d_3, xy_cov_d_3, xy_a_d_3 = produce_feature_v_xy(
        day3)
    features.extend(
        [k_min_1, k_max_1, x_min_n_1, x_max_n_1, y_min_n_1, y_max_n_1, x_max_y_min_n_1, y_max_x_min_n_1, x_25_n_1,
         y_75_n_1, xy_cov_n_1, xy_a_n_1])
    features.extend(
        [k_min_2, k_max_2, x_min_n_2, x_max_n_2, y_min_n_2, y_max_n_2, x_max_y_min_n_2, y_max_x_min_n_2, x_25_n_2,
         y_75_n_2, xy_cov_n_2,
         xy_a_n_2])
    features.extend(
        [k_min_3, k_max_3, x_min_d_1, x_max_d_1, y_min_d_1, y_max_d_1, x_max_y_min_d_1, y_max_x_min_d_1, x_25_d_1,
         y_75_d_1, xy_cov_d_1,
         xy_a_d_1])
    features.extend(
        [k_min_4, k_max_4, x_min_d_2, x_max_d_2, y_min_d_2, y_max_d_2, x_max_y_min_d_2, y_max_x_min_d_2, x_25_d_2,
         y_75_d_2, xy_cov_d_2,
         xy_a_d_2])
    features.extend(
        [k_min_5, k_max_5, x_min_d_3, x_max_d_3, y_min_d_3, y_max_d_3, x_max_y_min_d_3, y_max_x_min_d_3, x_25_d_3,
         y_75_d_3, xy_cov_d_3,
         xy_a_d_3])

    k_min_n, k_max_n, x_min_n_, x_max_n_, y_min_n_, y_max_n_, x_max_y_min_n_, y_max_x_min_n_, x_25_n_, y_75_n_, xy_cov_n_, xy_a_n_ = produce_feature_v_xy(
        night)
    k_min_d, k_max_d, x_min_d_, x_max_d_, y_min_d_, y_max_d_, x_max_y_min_d_, y_max_x_min_d_, x_25_d_, y_75_d_, xy_cov_d_, xy_a_d_ = produce_feature_v_xy(
        day)
    features.extend(
        [k_min_n, k_max_n, x_min_n_, x_max_n_, y_min_n_, y_max_n_, x_max_y_min_n_, y_max_x_min_n_, x_25_n_, y_75_n_,
         xy_cov_n_, xy_a_n_])
    features.extend(
        [k_min_d, k_max_d, x_min_d_, x_max_d_, y_min_d_, y_max_d_, x_max_y_min_d_, y_max_x_min_d_, x_25_d_, y_75_d_,
         xy_cov_d_, xy_a_d_])

    ###细分角度等特征###
    est_v_m_1, est_v_s_1, est_v_75_1, est_d_1, num_ang_0_100_1, ratio_ang_0_100_1, num_ang_10_150_1, ratio_ang_10_150_1, num_ang_100_165_1, ratio_ang_100_165_1, beg_end_1 = produce_feature_ang_ext(
        night)
    est_v_m_2, est_v_s_2, est_v_75_2, est_d_2, num_ang_0_100_2, ratio_ang_0_100_2, num_ang_10_150_2, ratio_ang_10_150_2, num_ang_100_165_2, ratio_ang_100_165_2, beg_end_2 = produce_feature_ang_ext(
        day)

    features.extend([est_v_m_1, est_v_s_1, est_v_75_1, est_d_1, num_ang_0_100_1, ratio_ang_0_100_1, num_ang_10_150_1,
                     ratio_ang_10_150_1, num_ang_100_165_1, ratio_ang_100_165_1, beg_end_1])
    features.extend([est_v_m_2, est_v_s_2, est_v_75_2, est_d_2, num_ang_0_100_2, ratio_ang_0_100_2, num_ang_10_150_2,
                     ratio_ang_10_150_2, num_ang_100_165_2, ratio_ang_100_165_2, beg_end_2])

    # 全局统计特征
    features.append(df['x'].min())
    features.append(df['x'].max())
    features.append(df['x'].mean())
    features.append(df['x'].quantile(0.25))

    features.append(df['y'].min())
    features.append(df['y'].max())
    features.append(df['y'].mean())
    features.append(df['y'].quantile(0.75))

    features.append(df['x'].cov(df['y']))

    df['time'] = pd.to_datetime(df['time'], format='%m%d %H:%M:%S')
    t_diff = df['time'].diff().iloc[1:].dt.total_seconds()
    x_diff = df['x'].diff().iloc[1:].abs()
    y_diff = df['y'].diff().iloc[1:].abs()
    dis = sum(np.sqrt(x_diff ** 2 + y_diff ** 2))
    x_a_mean = (x_diff / t_diff).mean()
    y_a_mean = (y_diff / t_diff).mean()
    features.append(np.sqrt(x_a_mean ** 2 + y_a_mean ** 2))

    features.append(df['速度'].mean())
    features.append(df['速度'].std())
    features.append(df['速度'].quantile(0.75))

    features.append(df['方向'].mean())

    if (know_type):
        if (df["type"].iloc[0] == '拖网'):   #有变动
            features.append(0)
        if (df["type"].iloc[0] == '刺网'):
            features.append(2)
        if (df["type"].iloc[0] == '围网'):
            features.append(1)

train_path = r"/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/"
train_files = os.listdir(train_path)
train_files = list(np.sort(train_files))
length_tr = len(train_files)
for files in train_files:
    path = train_path + str(files)
    read_information(path, know_type=True)
train_data = pd.DataFrame(np.array(features).reshape(length_tr, int(len(features) / length_tr)))
train_data.columns = ['ship', 'est_v_m', 'est_v_s', 'est_v_75', 'v_diff_mean',
                      'd_diff_mean', 'd_diff_max', 'd_diff_min',
                      'est_d', 'num_ang_0_100', 'ratio_ang_0_100', 'num_ang_10_150',
                      'ratio_ang_10_150', 'num_ang_100_165', 'ratio_ang_100_165', 'beg_end',
                      'k_min_1', 'k_max_1', 'x_min_n_1', 'x_max_n_1', 'y_min_n_1', 'y_max_n_1', 'x_max_y_min_n_1',
                      'y_max_x_min_n_1',
                      'x_25_n_1', 'y_75_n_1', 'xy_cov_n_1', 'xy_a_n_1',
                      'k_min_2', 'k_max_2', 'x_min_n_2', 'x_max_n_2', 'y_min_n_2', 'y_max_n_2', 'x_max_y_min_n_2',
                      'y_max_x_min_n_2',
                      'x_25_n_2', 'y_75_n_2', 'xy_cov_n_2', 'xy_a_n_2',
                      'k_min_3', 'k_max_3', 'x_min_d_1', 'x_max_d_1', 'y_min_d_1', 'y_max_d_1', 'x_max_y_min_d_1',
                      'y_max_x_min_d_1',
                      'x_25_d_1', 'y_75_d_1', 'xy_cov_d_1', 'xy_a_d_1',
                      'k_min_4', 'k_max_4', 'x_min_d_2', 'x_max_d_2', 'y_min_d_2', 'y_max_d_2', 'x_max_y_min_d_2',
                      'y_max_x_min_d_2',
                      'x_25_d_2', 'y_75_d_2', 'xy_cov_d_2', 'xy_a_d_2',
                      'k_min_5', 'k_max_5', 'x_min_d_3', 'x_max_d_3', 'y_min_d_3', 'y_max_d_3', 'x_max_y_min_d_3',
                      'y_max_x_min_d_3',
                      'x_25_d_3', 'y_75_d_3', 'xy_cov_d_3', 'xy_a_d_3',
                      'k_min_n', 'k_max_n', 'x_min_n_', 'x_max_n_', 'y_min_n_', 'y_max_n_', 'x_max_y_min_n_',
                      'y_max_x_min_n_',
                      'x_25_n_', 'y_75_n_', 'xy_cov_n_', 'xy_a_n_',
                      'k_min_d', 'k_max_d', 'x_min_d_', 'x_max_d_', 'y_min_d_', 'y_max_d_', 'x_max_y_min_d_',
                      'y_max_x_min_d_',
                      'x_25_d_', 'y_75_d_', 'xy_cov_d_', 'xy_a_d_',
                      'est_v_m_1', 'est_v_s_1', 'est_v_75_1', 'est_d_1', 'num_ang_0_100_1', 'ratio_ang_0_100_1',
                      'num_ang_10_150_1', 'ratio_ang_10_150_1', 'num_ang_100_165_1', 'ratio_ang_100_165_1', 'beg_end_1',
                      'est_v_m_2', 'est_v_s_2', 'est_v_75_2', 'est_d_2', 'num_ang_0_100_2', 'ratio_ang_0_100_2',
                      'num_ang_10_150_2', 'ratio_ang_10_150_2', 'num_ang_100_165_2', 'ratio_ang_100_165_2', 'beg_end_2',
                      'x_min', 'x_max', 'x_mean', 'x_1/4', 'y_min', 'y_max', 'y_mean', 'y_3/4', 'xy_cov', 'a',
                      'v_mean', 'v_std', 'v_3/4', 'd_mean', 'type']
train_data.fillna(0, inplace=True)


features = []
test_path = r'/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/test_dataset/'
test_files = os.listdir(test_path)
test_files = list(np.sort(test_files))
length_te = len(test_files)
for files in test_files:
    path = test_path + str(files)
    read_information(path, know_type=False)
test_data = pd.DataFrame(np.array(features).reshape(length_te, int(len(features) / length_te)))
test_data.columns = ['ship', 'est_v_m', 'est_v_s', 'est_v_75', 'v_diff_mean',
                     'd_diff_mean', 'd_diff_max', 'd_diff_min',
                     'est_d', 'num_ang_0_100', 'ratio_ang_0_100', 'num_ang_10_150',
                     'ratio_ang_10_150', 'num_ang_100_165', 'ratio_ang_100_165', 'beg_end',
                     'k_min_1', 'k_max_1', 'x_min_n_1', 'x_max_n_1', 'y_min_n_1', 'y_max_n_1', 'x_max_y_min_n_1',
                     'y_max_x_min_n_1',
                     'x_25_n_1', 'y_75_n_1', 'xy_cov_n_1', 'xy_a_n_1',
                     'k_min_2', 'k_max_2', 'x_min_n_2', 'x_max_n_2', 'y_min_n_2', 'y_max_n_2', 'x_max_y_min_n_2',
                     'y_max_x_min_n_2',
                     'x_25_n_2', 'y_75_n_2', 'xy_cov_n_2', 'xy_a_n_2',
                     'k_min_3', 'k_max_3', 'x_min_d_1', 'x_max_d_1', 'y_min_d_1', 'y_max_d_1', 'x_max_y_min_d_1',
                     'y_max_x_min_d_1',
                     'x_25_d_1', 'y_75_d_1', 'xy_cov_d_1', 'xy_a_d_1',
                     'k_min_4', 'k_max_4', 'x_min_d_2', 'x_max_d_2', 'y_min_d_2', 'y_max_d_2', 'x_max_y_min_d_2',
                     'y_max_x_min_d_2',
                     'x_25_d_2', 'y_75_d_2', 'xy_cov_d_2', 'xy_a_d_2',
                     'k_min_5', 'k_max_5', 'x_min_d_3', 'x_max_d_3', 'y_min_d_3', 'y_max_d_3', 'x_max_y_min_d_3',
                     'y_max_x_min_d_3',
                     'x_25_d_3', 'y_75_d_3', 'xy_cov_d_3', 'xy_a_d_3',
                     'k_min_n', 'k_max_n', 'x_min_n_', 'x_max_n_', 'y_min_n_', 'y_max_n_', 'x_max_y_min_n_',
                     'y_max_x_min_n_',
                     'x_25_n_', 'y_75_n_', 'xy_cov_n_', 'xy_a_n_',
                     'k_min_d', 'k_max_d', 'x_min_d_', 'x_max_d_', 'y_min_d_', 'y_max_d_', 'x_max_y_min_d_',
                     'y_max_x_min_d_',
                     'x_25_d_', 'y_75_d_', 'xy_cov_d_', 'xy_a_d_',
                     'est_v_m_1', 'est_v_s_1', 'est_v_75_1', 'est_d_1', 'num_ang_0_100_1', 'ratio_ang_0_100_1',
                     'num_ang_10_150_1', 'ratio_ang_10_150_1', 'num_ang_100_165_1', 'ratio_ang_100_165_1', 'beg_end_1',
                     'est_v_m_2', 'est_v_s_2', 'est_v_75_2', 'est_d_2', 'num_ang_0_100_2', 'ratio_ang_0_100_2',
                     'num_ang_10_150_2', 'ratio_ang_10_150_2', 'num_ang_100_165_2', 'ratio_ang_100_165_2', 'beg_end_2',
                     'x_min', 'x_max', 'x_mean', 'x_1/4', 'y_min', 'y_max', 'y_mean', 'y_3/4', 'xy_cov', 'a',
                     'v_mean', 'v_std', 'v_3/4', 'd_mean']
test_data.fillna(0, inplace=True)


kind = train_data.type
train_data = train_data.drop('type', axis=1)

features = [x for x in train_data.columns]
train_data = train_data[features]
# test_data = test_data[features]


x_train, x_test, y_train, y_test = train_test_split(train_data, kind, test_size=0.1, random_state=78)


params = {
    'learning_rate': 0.2036,
    'max_depth': 6,  # .6787,
    'boosting': 'gbdt',
    'objective': 'multiclass',
    'n_estimators': 5561,
    'num_class': 3,
    'feature_fraction': .5242,
    'bagging_fraction': .3624,
    'class_weight': {0: 3, 1: 5, 2: 2.5},
    'seed': 78
    # 'early_stopping_rounds': 100
}

llf = lgb.LGBMClassifier(**params)
llf.fit(x_train, y_train)
weight_lgb = f1_score(y_test, llf.predict(x_test), average='macro')

details = []
answers = []
scores = []
sk = StratifiedKFold(n_splits=20, shuffle=True, random_state=2020)
for train, test in sk.split(train_data, kind):
    x_train = train_data.iloc[train]
    y_train = kind.iloc[train]
    x_test = train_data.iloc[test]
    y_test = kind.iloc[test]

    llf.fit(x_train, y_train)
    pred_llf = llf.predict(x_test)
    weight_lgb = f1_score(y_test, pred_llf, average='macro')

    prob_lgb = llf.predict_proba(x_test)
    prob_end = prob_lgb
    score = f1_score(y_test, np.argmax(prob_end, axis=1), average='macro')
    scores.append(score)

    details.append(score)
    details.append(weight_lgb)

    # answers.append(llf.predict(test_data))
    print('score: ', score)
print(np.mean(details))


# print(answers)
# 使用贝叶斯优化调参
from sklearn import metrics

params = {
    'learning_rate': 0.2036,
    'max_depth': 6,  # .6787,
    'boosting': 'gbdt',
    'objective': 'multiclass',
    'n_estimators': 5561,
    'num_class': 3,
    'feature_fraction': .5242,
    'bagging_fraction': .3624,
    'early_stopping_rounds': 100
}
fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

X = train_data.copy()
y = kind
models = []
# pred = np.zeros((len(test_data),3))
oof = np.zeros((len(X), 3))
for index, (train_idx, val_idx) in enumerate(fold.split(X, y)):
    train_set = lgb.Dataset(X.iloc[train_idx], y.iloc[train_idx])
    val_set = lgb.Dataset(X.iloc[val_idx], y.iloc[val_idx])
    # print(y.iloc[train_idx])

    model = lgb.train(params, train_set, valid_sets=[train_set, val_set], verbose_eval=100)
    models.append(model)
    # print(X.iloc[val_idx][0:5])
    val_pred = model.predict(X.iloc[val_idx])
    oof[val_idx] = val_pred
    # print('val_pred',val_pred[0:5])
    val_y = y.iloc[val_idx]
    val_pred = np.argmax(val_pred, axis=1)
    # print('val_y',val_y[0:5])
    # print('val_pred',val_pred[0:5])
    print(index, 'val f1', metrics.f1_score(val_y, val_pred, average='macro'))
    # 0.8695539641133697
    # 0.8866211724839532

    # test_pred = model.predict(test_data)
    # pred += test_pred/5
oof = np.argmax(oof, axis=1)
print('oof f1', metrics.f1_score(oof, y, average='macro'))
# 0.8701544575329372


# In[8]:


x_train, x_test, y_train, y_test = train_test_split(train_data, kind, test_size=0.1, random_state=78)
params = {
    'learning_rate': 0.2036,
    'max_depth': 6,  # .6787,
    'boosting': 'gbdt',
    'objective': 'multiclass',
    'n_estimators': 5561,
    'num_class': 3,
    'feature_fraction': .5242,
    'bagging_fraction': .3624,
    'class_weight': {0: 3, 1: 5, 2: 2.5},
    'seed': 78
    # 'early_stopping_rounds': 100
}

llf = lgb.LGBMClassifier(**params)
llf.fit(x_train, y_train)
weight_lgb = f1_score(y_test, llf.predict(x_test), average='macro')
print(weight_lgb)

