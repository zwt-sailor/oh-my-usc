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


data_train.describe([0.01, 0.025, 0.05, 0.5, 0.75, 0.9, 0.99])


# In[4]:


# 查看训练集缺失值
print(f'There are {data_train.isnull().any().sum()} columns in train dataset with missing values.')


# In[5]:


data_test = pd.read_csv("/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/test.csv")         
print('The shape of test data:', data_test.shape)
data_test.head()


# In[6]:


# 查看测试集缺失值
print(f'There are {data_test.isnull().any().sum()} columns in test dataset with missing values.')


# In[7]:


data_test.describe([0.01, 0.025, 0.05, 0.5, 0.75, 0.9, 0.99])


# In[8]:


data1 = data_train.loc[data_train['type'] == "拖网"]
data1
with open("/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/tuowang_data.pkl", "wb") as f:
    pickle.dump(data1, f)


# In[9]:


data2 = data_train.loc[data_train['type'] == "围网"]
data2
with open("/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/weiwang_data.pkl", "wb") as f:
    pickle.dump(data2, f)


# In[10]:


data3 = data_train.loc[data_train['type'] == "刺网"]
data3
with open("/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/ciwang_data.pkl", "wb") as f:
    pickle.dump(data3, f)


# In[9]:


# 直接从三个网中分别选取三条渔船，进行挨个数据分析（分别从三类数据文件中，随机读取三条渔船的数据）
#由于不是随机去选取，可能会导致数据的不严谨，有误差，这里暂时只是作为前期的数据观察

# #weiwang1
df = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/1.csv', usecols=[1],
                       names=None)  # 读取项目名称和行业领域两列，并不要列名
df_xweiwang1 = df.values.tolist()
print(df_xweiwang1)  
df = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/1.csv', usecols=[2],
                       names=None)  # 读取项目名称和行业领域两列，并不要列名
df_yweiwang1 = df.values.tolist()
print(df_yweiwang1)  


# #weiwang2
df = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/4.csv', usecols=[1],
                       names=None)  # 读取项目名称和行业领域两列，并不要列名
df_xweiwang2 = df.values.tolist()
print(df_xweiwang2)  
df = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/4.csv', usecols=[2],
                       names=None)  # 读取项目名称和行业领域两列，并不要列名
df_yweiwang2 = df.values.tolist()
print(df_yweiwang2)  

# #weiwang3
df = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/8651.csv', usecols=[1],
                       names=None)  # 读取项目名称和行业领域两列，并不要列名
df_xweiwang3 = df.values.tolist()
print(df_xweiwang3)  
df = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/8651.csv', usecols=[2],
                       names=None)  # 读取项目名称和行业领域两列，并不要列名
df_yweiwang3 = df.values.tolist()
print(df_yweiwang3)  



# In[10]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
plt.subplots_adjust(wspace=0.2, hspace=0.2)
 
# 画第1个图：weiwang1折线图
x=np.arange(1,100)
x = [i for i in df_xweiwang1]
y = [i for i in df_yweiwang1]
# print(x)
# print(y)
axes[0].plot(x, y,label="weiwang1")
axes[0].scatter(x[0], y[0], label='start', c='red', s=10, marker='8')
axes[0].scatter(x[len(x)-1], y[len(y)-1], label='end', c='green', s=10, marker='v')

axes[0].set_xlabel('lat')   # 为x轴添加标签
axes[0].set_ylabel('lon')   # 为y轴添加标签
axes[0].legend(loc='lower right')   # 设置图表图例在左上角


# 画第2个图：weiwang2折线图
x=np.arange(1,100)
x = [i for i in df_xweiwang2]
y = [i for i in df_yweiwang2]
# print(x)
# print(y)
axes[1].plot(x, y,label="weiwang2")
axes[1].scatter(x[0], y[0], label='start', c='red', s=10, marker='8')
axes[1].scatter(x[len(x)-1], y[len(y)-1], label='end', c='green', s=10, marker='v')

axes[1].set_xlabel('lat')   # 为x轴添加标签
axes[1].set_ylabel('lon')   # 为y轴添加标签
axes[1].legend(loc='lower right')   # 设置图表图例在左上角



# 画第3个图：weiwang3折线图
x=np.arange(1,100)
x = [i for i in df_xweiwang3]
y = [i for i in df_yweiwang3]
# print(x)
# print(y)
axes[2].plot(x, y,label="weiwang3")
axes[2].scatter(x[0], y[0], label='start', c='red', s=10, marker='8')
axes[2].scatter(x[len(x)-1], y[len(y)-1], label='end', c='green', s=10, marker='v')

axes[2].set_xlabel('lat')   # 为x轴添加标签
axes[2].set_ylabel('lon')   # 为y轴添加标签
axes[2].legend(loc='lower right')   # 设置图表图例在左上角



plt.show()


# In[11]:


plt.title("loss function")
plt.xlabel("lat")
plt.ylabel("lon")
x = [i for i in df_x]
y = [i for i in df_y]
# print(x)
# print(y)
plt.plot(x, y,label="weiwang1")
plt.scatter(x[0], y[0], label='start', c='red', s=10, marker='8')
plt.scatter(x[len(x)-1], y[len(y)-1], label='end', c='green', s=10, marker='v')
plt.legend()
plt.show()


# In[12]:


# 直接从三个网中分别选取三条渔船，进行挨个数据分析（分别从三类数据文件中，随机读取三条渔船的数据）
#由于不是随机去选取，可能会导致数据的不严谨，有误差，这里暂时只是作为前期的数据观察

# #tuowang1
df = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/16050.csv', usecols=[1],
                       names=None)  # 读取项目名称和行业领域两列，并不要列名
df_xtuowang1 = df.values.tolist()
print(df_xtuowang1)  
df = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/16050.csv', usecols=[2],
                       names=None)  # 读取项目名称和行业领域两列，并不要列名
df_ytuowang1 = df.values.tolist()
print(df_ytuowang1)  


# #tuowang2
df = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/3228.csv', usecols=[1],
                       names=None)  # 读取项目名称和行业领域两列，并不要列名
df_xtuowang2 = df.values.tolist()
print(df_xtuowang2)  
df = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/3228.csv', usecols=[2],
                       names=None)  # 读取项目名称和行业领域两列，并不要列名
df_ytuowang2 = df.values.tolist()
print(df_ytuowang2)  

# #tuowang3
df = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/6284.csv', usecols=[1],
                       names=None)  # 读取项目名称和行业领域两列，并不要列名
df_xtuowang3 = df.values.tolist()
print(df_xtuowang3)  
df = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/6284.csv', usecols=[2],
                       names=None)  # 读取项目名称和行业领域两列，并不要列名
df_ytuowang3 = df.values.tolist()
print(df_ytuowang3)  




# In[13]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
plt.subplots_adjust(wspace=0.2, hspace=0.2)
 
# 画第1个图：tuowang1折线图
x=np.arange(1,100)
x = [i for i in df_xtuowang1]
y = [i for i in df_ytuowang1]
# print(x)
# print(y)
axes[0].plot(x, y,label="tuowang1")
axes[0].scatter(x[0], y[0], label='start', c='red', s=10, marker='8')
axes[0].scatter(x[len(x)-1], y[len(y)-1], label='end', c='green', s=10, marker='v')

axes[0].set_xlabel('lat')   # 为x轴添加标签
axes[0].set_ylabel('lon')   # 为y轴添加标签
axes[0].legend(loc='lower right')   # 设置图表图例在左上角


# 画第2个图：tuowang2折线图
x=np.arange(1,100)
x = [i for i in df_xtuowang2]
y = [i for i in df_ytuowang2]
# print(x)
# print(y)
axes[1].plot(x, y,label="tuowang2")
axes[1].scatter(x[0], y[0], label='start', c='red', s=10, marker='8')
axes[1].scatter(x[len(x)-1], y[len(y)-1], label='end', c='green', s=10, marker='v')

axes[1].set_xlabel('lat')   # 为x轴添加标签
axes[1].set_ylabel('lon')   # 为y轴添加标签
axes[1].legend(loc='lower right')   # 设置图表图例在左上角



# 画第3个图：tuowang3折线图
x=np.arange(1,100)
x = [i for i in df_xtuowang3]
y = [i for i in df_ytuowang3]
# print(x)
# print(y)
axes[2].plot(x, y,label="tuowang3")
axes[2].scatter(x[0], y[0], label='start', c='red', s=10, marker='8')
axes[2].scatter(x[len(x)-1], y[len(y)-1], label='end', c='green', s=10, marker='v')

axes[2].set_xlabel('lat')   # 为x轴添加标签
axes[2].set_ylabel('lon')   # 为y轴添加标签
axes[2].legend(loc='lower right')   # 设置图表图例在左上角



plt.show()


# In[14]:


# 直接从三个网中分别选取三条渔船，进行挨个数据分析（分别从三类数据文件中，随机读取三条渔船的数据）
#由于不是随机去选取，可能会导致数据的不严谨，有误差，这里暂时只是作为前期的数据观察

# #ciwang1
df = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/8863.csv', usecols=[1],
                       names=None)  # 读取项目名称和行业领域两列，并不要列名
df_xciwang1 = df.values.tolist()
print(df_xciwang1)  
df = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/8863.csv', usecols=[2],
                       names=None)  # 读取项目名称和行业领域两列，并不要列名
df_yciwang1 = df.values.tolist()
print(df_yciwang1)  


# #ciwang2
df = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/15998.csv', usecols=[1],
                       names=None)  # 读取项目名称和行业领域两列，并不要列名
df_xciwang2 = df.values.tolist()
print(df_xciwang2)  
df = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/15998.csv', usecols=[2],
                       names=None)  # 读取项目名称和行业领域两列，并不要列名
df_yciwang2 = df.values.tolist()
print(df_yciwang2)  

# #ciwang3
df = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/18287.csv', usecols=[1],
                       names=None)  # 读取项目名称和行业领域两列，并不要列名
df_xciwang3 = df.values.tolist()
print(df_xciwang3)  
df = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/18287.csv', usecols=[2],
                       names=None)  # 读取项目名称和行业领域两列，并不要列名
df_yciwang3 = df.values.tolist()
print(df_yciwang3)  


# In[15]:


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 5))
plt.subplots_adjust(wspace=0.2, hspace=0.2)
 
# 画第1个图：ciwang1折线图
x=np.arange(1,100)
x = [i for i in df_xciwang1]
y = [i for i in df_yciwang1]
# print(x)
# print(y)
axes[0].plot(x, y,label="ciwang1")
axes[0].scatter(x[0], y[0], label='start', c='red', s=10, marker='8')
axes[0].scatter(x[len(x)-1], y[len(y)-1], label='end', c='green', s=10, marker='v')

axes[0].set_xlabel('lat')   # 为x轴添加标签
axes[0].set_ylabel('lon')   # 为y轴添加标签
axes[0].legend(loc='lower right')   # 设置图表图例在左上角


# 画第2个图：ciwang2折线图
x=np.arange(1,100)
x = [i for i in df_xciwang2]
y = [i for i in df_yciwang2]
# print(x)
# print(y)
axes[1].plot(x, y,label="ciwang2")
axes[1].scatter(x[0], y[0], label='start', c='red', s=10, marker='8')
axes[1].scatter(x[len(x)-1], y[len(y)-1], label='end', c='green', s=10, marker='v')

axes[1].set_xlabel('lat')   # 为x轴添加标签
axes[1].set_ylabel('lon')   # 为y轴添加标签
axes[1].legend(loc='lower right')   # 设置图表图例在左上角



# 画第3个图：ciwang3折线图
x=np.arange(1,100)
x = [i for i in df_xciwang3]
y = [i for i in df_yciwang3]
# print(x)
# print(y)
axes[2].plot(x, y,label="ciwang3")
axes[2].scatter(x[0], y[0], label='start', c='red', s=10, marker='8')
axes[2].scatter(x[len(x)-1], y[len(y)-1], label='end', c='green', s=10, marker='v')

axes[2].set_xlabel('lat')   # 为x轴添加标签
axes[2].set_ylabel('lon')   # 为y轴添加标签
axes[2].legend(loc='lower right')   # 设置图表图例在左上角

plt.show()


# In[16]:


# 绘制训练数据的速度和方向分布图
plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
plt.subplots_adjust(wspace=0.3, hspace=0.5)
types = ['拖网', '围网', '刺网']
labels = ['target==tw', 'target==ww', 'target==cw']
colors = ['red', 'green', 'blue']
for i, t in enumerate(types):
    type_df = data_train[data_train['type']==t]
    plt.subplot(1, 2, 1)
    ax1 = sns.kdeplot(type_df['速度'].values, color=colors[i], shade=True)
    plt.subplot(1, 2, 2)
    ax2 = sns.kdeplot(type_df['方向'].values, color=colors[i], shade=True)
    ax1.legend(labels)
    ax1.set_xlabel('Speed')
    ax2.legend(labels)
    ax2.set_xlabel('Direction')
plt.show()


# In[17]:


# 随机读取某类数据中一条渔船的数据
# 可视化x、y变化
def visualize_one_traj_x_y():
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    
    # 获取围网数据中某条渔船的数据
    data = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/1.csv')
    x = data['lat'].loc[-1:]
    x /= 1000
    
    y = data['lon'].loc[-1:]
    y /= 1000
    
    arr1 = np.arange(len(x))
    arr2 = np.arange(len(y))
    
    axes[0].plot(arr1, x, label='lat')
    axes[1].plot(arr2, y, label='lon')
    axes[0].grid(alpha=0.5)
    axes[0].legend(loc='best')
    axes[1].grid(alpha=0.5)
    axes[1].legend(loc='best')
    plt.show()

visualize_one_traj_x_y()


# In[18]:


# 随机读取某类数据中一条渔船的数据
# 可视化x、y变化
def visualize_one_traj_x_y():
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))
    plt.subplots_adjust(wspace=0.5, hspace=0.5)
    
    # 获取围网数据中某条渔船的数据
    data = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/8651.csv')
    x = data['lat'].loc[-1:]
    x /= 1000
    
    y = data['lon'].loc[-1:]
    y /= 1000
    
    arr1 = np.arange(len(x))
    arr2 = np.arange(len(y))
    
    axes[0].plot(arr1, x, label='lat')
    axes[1].plot(arr2, y, label='lon')
    axes[0].grid(alpha=0.5)
    axes[0].legend(loc='best')
    axes[1].grid(alpha=0.5)
    axes[1].legend(loc='best')
    plt.show()

visualize_one_traj_x_y()


# In[20]:


######随机选取三种渔船的数据，并对其的速度和方向进行分析
# #weiwang
df = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/8651.csv', usecols=[3],
                       names=None)  # 读取项目名称和行业领域两列，并不要列名
df_weiwang_speed = df.values.tolist()
print(df_weiwang_speed)  

df = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/8651.csv', usecols=[4],
                       names=None)  # 读取项目名称和行业领域两列，并不要列名
df_weiwang_direction = df.values.tolist()
print(df_weiwang_direction) 


# #tuowang
df = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/6284.csv', usecols=[3],
                       names=None)  # 读取项目名称和行业领域两列，并不要列名
df_tuowang_speed = df.values.tolist()
print(df_tuowang_speed)  

df = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/6284.csv', usecols=[4],
                       names=None)  # 读取项目名称和行业领域两列，并不要列名
df_tuowang_direction = df.values.tolist()
print(df_tuowang_direction)  

# #ciwang
df = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/15998.csv', usecols=[3],
                       names=None)  # 读取项目名称和行业领域两列，并不要列名
df_ciwang_speed = df.values.tolist()
print(df_ciwang_speed)  

df = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/15998.csv', usecols=[4],
                       names=None)  # 读取项目名称和行业领域两列，并不要列名
df_ciwang_direction = df.values.tolist()
print(df_ciwang_direction) 


# In[21]:


fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(20, 15))
plt.subplots_adjust(wspace=0.3, hspace=0.3)
 
# 画第1个图：weiwang_speed折线图
x=np.arange(1,100)
# x = [i for i in df_xciwang1]
y = [i for i in df_weiwang_speed]
# print(x)
# print(y)
axes[0][0].plot( y,c='forestgreen',label="weiwang_speed")
axes[0][0].legend(loc='upper right')   # 设置图表图例在左上角


# 画第1个图：weiwang_direction折线图
x=np.arange(1,100)
# x = [i for i in df_xciwang1]
y = [i for i in df_weiwang_direction]
# print(x)
# print(y)
axes[0][1].plot( y,c='forestgreen',label="weiwang_direction")
axes[0][1].legend(loc='upper right')   # 设置图表图例在左上角

# 画第1个图：tuowang_speed折线图
x=np.arange(1,100)
# x = [i for i in df_xciwang1]
y = [i for i in df_tuowang_speed]
# print(x)
# print(y)
axes[1][0].plot( y,c='pink',label="tuowang_speed")
axes[1][0].legend(loc='upper right')   # 设置图表图例在左上角

# 画第1个图：tuowang_direction折线图
x=np.arange(1,100)
# x = [i for i in df_xciwang1]
y = [i for i in df_tuowang_direction]
# print(x)
# print(y)
axes[1][1].plot( y,c='pink',label="tuowang_direction")
axes[1][1].legend(loc='upper right')   # 设置图表图例在左上角

# 画第1个图：ciwang_speed折线图
x=np.arange(1,100)
# x = [i for i in df_xciwang1]
y = [i for i in df_ciwang_speed]
# print(x)
# print(y)
axes[2][0].plot( y,c='lightblue',label="ciwang_speed")
axes[2][0].legend(loc='upper right')   # 设置图表图例在左上角

# 画第1个图：ciwang_direction折线图
x=np.arange(1,100)
# x = [i for i in df_xciwang1]
y = [i for i in df_ciwang_direction]
# print(x)
# print(y)
axes[2][1].plot( y,c='lightblue',label="ciwang_direction")
axes[2][1].legend(loc='upper right')   # 设置图表图例在左上角


# In[22]:


# 使用箱线图进行可视化
def plot_speed_direction2_ditribution():
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    colors = ['pink', 'lightblue', 'lightgreen']
    
    df0 = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/8651.csv')
    df1 = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/6284.csv')
    df2 = pd.read_csv('/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train/15998.csv')
                      
    bplot1 = axes[0].boxplot([df0.速度, df1.速度, df2.速度],
                             vert=True, patch_artist=True, showmeans = True,
                             flierprops = {'marker':'o','markerfacecolor':'red','color':'black'},
                             meanprops = {'marker':'D','markerfacecolor':'indianred'},
                             medianprops = {'linestyle':'--','color':'orange'},
                             labels=['tw', 'ww', 'cw'])
    bplot2 = axes[1].boxplot([df0.方向, df1.方向, df2.方向],
                             vert=True, patch_artist=True,showmeans = True,
                             flierprops = {'marker':'o','markerfacecolor':'red','color':'black'},
                             meanprops = {'marker':'D','markerfacecolor':'indianred'},
                             medianprops = {'linestyle':'--','color':'orange'},
                             labels=['tw', 'ww', 'cw'])
    
    for bplot in (bplot1, bplot2):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    
    axes[0].set_title('Speed')
    axes[1].set_title('Direction')
    plt.show()

plot_speed_direction2_ditribution()


# In[23]:


# 使用箱线图进行可视化
def plot_speed_direction2_ditribution():
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    colors = ['pink', 'lightblue', 'lightgreen']
    
    types = ['拖网', '围网', '刺网']
    
    type_df0 = data_train[data_train['type']=="拖网"]
    type_df1 = data_train[data_train['type']=="围网"]
    type_df2 = data_train[data_train['type']=="刺网"]
    
#     plt.subplot(1, 2, 1)
#     ax1 = sns.kdeplot([type_df0['速度'].values,type_df1['速度'].values,type_df2['速度'].values],
#                       color=colors[i], shade=True)  
#     plt.subplot(1, 2, 2)
#     ax2 = sns.kdeplot(type_df0['方向'].values, color=colors[i], shade=True)                    
    bplot1 = axes[0].boxplot([type_df0['速度'].values,type_df1['速度'].values,type_df2['速度'].values],
                             vert=True, patch_artist=True, showmeans = True,
                             flierprops = {'marker':'o','markerfacecolor':'red','color':'black'},
                             meanprops = {'marker':'D','markerfacecolor':'indianred'},
                             medianprops = {'linestyle':'--','color':'orange'},
                             labels=['tw', 'ww', 'cw'])
    bplot2 = axes[1].boxplot([type_df0['方向'].values,type_df1['方向'].values,type_df2['方向'].values],
                             vert=True, patch_artist=True,showmeans = True,
                             flierprops = {'marker':'o','markerfacecolor':'red','color':'black'},
                             meanprops = {'marker':'D','markerfacecolor':'indianred'},
                             medianprops = {'linestyle':'--','color':'orange'},
                             labels=['tw', 'ww', 'cw'])
    
    for bplot in (bplot1, bplot2):
        for patch, color in zip(bplot['boxes'], colors):
            patch.set_facecolor(color)
    
    axes[0].set_title('Speed')
    axes[1].set_title('Direction')
    plt.show()

plot_speed_direction2_ditribution()


# In[25]:


# # 3-sigma算法判定异常值
# def three_sigma(data):
#     data_mean = np.mean(data)
#     data_std = np.std(data)
#     low = data_mean - 3 * data_std
#     high = data_mean + 3 * data_std
#     judge = []
#     for d in data:
#         if d < low or d > high:
#             judge.append(True)
#         else:
#             judge.append(False)
#     return judge
#################################使用
def boxfilter(data, proportion:float):
    allbox = data.boxplot(return_type='dict')
    zipped = zip(allbox['fliers'], data.columns)
    i = 0
    for flier, index_name in zipped:
        # 获取一个列的异常y值
        y = flier.get_ydata()
        # 获取数据长度
        datalenth = data.shape[0]
        # 如果有异常值才处理
        if(len(y) >= 1):
            #对于每一个异常值做处理
            for i in y:          
                #因为你不知道异常值是高于还是低于，因此这里做两次判断(异常值毕竟是少数)
                #如果留下来的数据规模是大于以前数据规模*proportion的，则进行处理，这个取决于你来定
                if(data[data[index_name] < i].shape[0] >=datalenth*proportion):
                    data = data[data[index_name] < i]
                elif(data[data[index_name] > i].shape[0] >= datalenth*proportion):
                    data = data[data[index_name] > i] 
    return data

train_ids = list(data_train['渔船ID'].unique())
for ID in tqdm(train_ids):
    id_df = data_train[data_train['渔船ID'] == ID]
    new_data_train = boxfilter(id_df,0.7)

    new_data_train.boxplot(return_type='dict')




# # 将异常值置为空值
# def assign_traj_anomaly_points_nan(df):
#     # 速度异常值用空值填充
#     is_speed_anomaly = three_sigma(df['速度'])
#     df['速度'][is_speed_anomaly] = np.nan
    
#     # 方向异常值用空值填充
#     is_direction_anomaly = three_sigma(df['方向'])
#     df['方向'][is_direction_anomaly] = np.nan
    
#     # 纬度和经度异常值直接剔除
#     is_lat_anomaly = three_sigma(df['lat'])
#     is_lon_anomaly = three_sigma(df['lon'])
#     lat_lon_anomaly = np.array([a | b for a, b in zip(is_lat_anomaly, is_lon_anomaly)])
#     df = df[~lat_lon_anomaly].reset_index(drop=True)
    
#     # 统计异常值个数
#     anomaly_cnt = len(is_speed_anomaly) - len(df)
    
#     return df, anomaly_cnt

# # 判断每个渔船ID轨迹中的异常值，采用多项式插值来填充
# train_ids = list(data_train['渔船ID'].unique())
# train_new = []
# train_anomaly_cnts = []

# for ID in tqdm(train_ids):
#     id_df = data_train[data_train['渔船ID'] == ID]
#     id_df, anomaly_cnt = assign_traj_anomaly_points_nan(id_df)
#     # 对速度和方向异常值进行二阶插值
#     id_df = id_df.interpolate(method='polynomial', axis=0, order=2)
#     # 填充剩下的空值
#     id_df = id_df.fillna(method='bfill')
#     id_df = id_df.fillna(method='ffill')
#     # 设置阈值过滤掉不合理的值
#     id_df['速度'] = id_df['速度'].clip(0, 7.2)
    
#     id_df['方向'] = id_df['方向'].clip(0, 247)
#     # 统计每个id的异常值个数
#     train_anomaly_cnts.append(anomaly_cnt)
#     train_new.append(id_df)
    
# new_data_train = pd.concat(train_new)


# In[26]:


# 绘制训练数据的速度和方向分布图
plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
plt.subplots_adjust(wspace=0.3, hspace=0.5)
types = ['拖网', '围网', '刺网']
labels = ['target==tw', 'target==ww', 'target==cw']
colors = ['red', 'green', 'blue']

for i, t in enumerate(types):
    type_df = data_train[data_train['type']==t]
    plt.subplot(1, 2, 1)
    ax1 = sns.kdeplot(type_df['lat'].values, color=colors[i], shade=True)
    plt.subplot(1, 2, 2)
    ax2 = sns.kdeplot(type_df['lon'].values, color=colors[i], shade=True)
    ax1.legend(labels)
    ax1.set_xlabel('lat')
    ax2.legend(labels)
    ax2.set_xlabel('lon')
plt.show()


# In[ ]:


# #1.将训练集数据全部合成一份csv文件便于后续分析  /Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train
# data_list = []  #定义一个空表
# file_in = "/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train"
# file_out = "/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train.csv"
#
# for info in os.listdir(file_in):
#     domain = os.path.abspath(file_in)    #获取文件夹的路径
#     info = os.path.join(domain,info)    #将路径与文件名结合起来就是每个文件的完整路径
#     data = pd.read_csv(info)
#     data_list.append(data)
#
# all_data = pd.concat(data_list)
# #print(all_data)
# all_data.to_csv(file_out,index=False,sep=',')
# print("===============================================================================")

# #2.将测试集数据全部合成一份csv文件便于后续分析  /Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/test_dataset
# data_list = []  #定义一个空表
# file_in = "/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/test_dataset"
# file_out = "/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/text.csv"
#
# for info in os.listdir(file_in):
#     domain = os.path.abspath(file_in)    #获取文件夹的路径
#     info = os.path.join(domain,info)    #将路径与文件名结合起来就是每个文件的完整路径
#     data = pd.read_csv(info)
#     data_list.append(data)
#
# all_data = pd.concat(data_list)
# #print(all_data)
# all_data.to_csv(file_out,index=False,sep=',')
#print("===============================================================================")

#读取文件
# data_train = pd.read_csv("/Users/zhaowentao/Downloads/u 机器学习/z 渔船作业方式/train.csv")           	#读取csv文件
# print('The shape of train data:', data_train.shape)
# data_train.head()
# #查看训练集的各个特征的基本统计值
# data_train.describe([0.01, 0.025, 0.05, 0.5, 0.75, 0.9, 0.99])

