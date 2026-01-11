from effective_dimension import Model, EffectiveDimension, ClassicalNeuralNetwork
from effective_dimension import WaveCell, WaveGeometryFreeForm, WaveIntensityProbe, WaveRNN, WaveSource
from effective_dimension import set_dtype, accuracy_onehot, normalize_power
import numpy as np
from math import pi
from scipy.special import logsumexp
import torch
import matplotlib.pyplot as plt
import pandas as pd
import sys

# This is an example file to create a classical model and compute its effective dimension

# create ranges for the number of data, n
n = [5000, 8000, 10000, 40000, 60000, 100000, 150000, 200000, 500000, 1000000, 3000000, 5000000, 7000000, 10000000]
_data={
      'n_class': 10,
      'sr': 10000,          #Sampling rate
      'window_size': 1000,
      'gender': 'both', #women, men, or both
      'vowels':['0','1','2','3','4','5','6','7','8','9']#[ae, eh, ih, oo, ah, ei, iy, uh, aw, er, oa, uw]
      }
_geom={
      'use_design_region': True,#位于src和探针之间的5网格单元缓冲区的设计区域
      'init': 'half',#设计区域的初始化，有三种选择'rand', 'half', or 'blank'
    #   'Nx': 150,
    #   'Ny': 150,
      'Nx': 50,
      'Ny': 50,
      'dt': 1.0,
      'h': 1.4283556979968262,#空间网格步长
      'c0': 1.0,#波速背景值(例如在PML和非设计区域)
      'c1': 0.5,#二值化过程中与c0一起使用的波速值
      'px':None,#探测器x坐标
      'py':None,#探测器y坐标
      'pd':None,#探测器间距
      'src_x':None,#源x坐标
      'src_y':None,
      'blur_radius': 1,
      'blur_N': 1,
      'pml':{
              'N':20,#PML厚度（网格单元数)
              'p': 4.0,   # PML polynomial order 多项式阶
              'max': 3.0, # PML max dampening factor 最大阻尼系数
           },
      'nonlinearity':{
              'cnl': 0.0,  # Kerr-like nonlinear wavespeed term  类克尔非线性波速项
              'b0': 0.0,   # Saturable abs. strength   强度
              'uth': 1.0,  # Saturable abs. threshold  阈值
          },
      'binarization':{
              'beta': 1000,   # 参数化二值投影函数
              'eta': 0.5,
          }  #二值化的参数
      }
_training={
      'prefix': 'ex',       #保存模型文件名
      'N_epochs': 20,       #训练次数
      'lr': 0.0004,         #优化学习率
      'batch_size': 64,
      'max_samples':None,     #样本的最大数量，默认12
      'N_folds': 7,         #(1/N_folds) * (the total number of samples) is the size of the test dataset
      'cross_validation': False
    }
cfg = {'seed': 10,              #random seed for shuffle
       'dtype': 'float64',      #张量的数据类型
       'geom':_geom,
       'data':_data,
       'training':_training,
       }
set_dtype(cfg['dtype'])
if cfg['seed'] is not None:
    torch.manual_seed(cfg['seed'])
N_classes = len(cfg['data']['vowels'])
# probes = [WaveIntensityProbe(110, 40), WaveIntensityProbe(110, 60), WaveIntensityProbe(110, 80)]
# source = [WaveSource(40, 60), WaveSource(40, 70), WaveSource(40, 80),WaveSource(40, 90)]
probes = [WaveIntensityProbe(40, 15), WaveIntensityProbe(40, 25), WaveIntensityProbe(40, 35)]
source = [WaveSource(10, 10), WaveSource(10, 20), WaveSource(10, 30),WaveSource(10, 40)]
# design_region = torch.zeros(cfg['geom']['Nx'], cfg['geom']['Ny'], dtype=torch.uint8)
design_region = torch.rand(cfg['geom']['Nx'], cfg['geom']['Ny'])
design_region[source[0].x.item() + 1:probes[0].x.item() - 1] = 1
# design_region[source[0].x.item() + 5:probes[0].x.item() - 5] = 1
# design_region = torch.rand(cfg['geom']['Nx'], cfg['geom']['Ny'])
# # design_region[source[0].x.item() + 5:probes[0].x.item() - 5] = 1
# design_region[0:source[0].x.item() + 5] = 0
# design_region[probes[0].x.item() - 5:] = 0
geom = WaveGeometryFreeForm((cfg['geom']['Nx'], cfg['geom']['Ny']), cfg['geom']['h'],
                                c0=cfg['geom']['c0'],
                                c1=cfg['geom']['c1'],
                                eta=cfg['geom']['binarization']['eta'],
                                beta=cfg['geom']['binarization']['beta'],
                                abs_sig=cfg['geom']['pml']['max'],
                                abs_N=cfg['geom']['pml']['N'],
                                abs_p=cfg['geom']['pml']['p'],
                                rho=cfg['geom']['init'],
                                blur_radius=cfg['geom']['blur_radius'],
                                blur_N=cfg['geom']['blur_N'],
                                design_region=design_region
                                )  # 逆向设计优化区域定义
cell = WaveCell(cfg['geom']['dt'], geom,
                satdamp_b0=cfg['geom']['nonlinearity']['b0'],
                satdamp_uth=cfg['geom']['nonlinearity']['uth'],
                c_nl=cfg['geom']['nonlinearity']['cnl']
                )
model = WaveRNN(cell, source, probes)
# x = torch.tensor([[10,1,3,1]])
# 生成随机输入数据


# 修改为生成大量符合统计规律的输入数据
num_thetas = 10
num_inputs = 10
input_size = 4

x = np.random.normal(0, 1, size=(num_inputs, input_size))
X = np.tile(x, (num_thetas, 1))

# 直接使用PyTorch生成随机数
x = torch.rand(num_inputs, input_size)  # 生成标准正态分布随机数
X = x.repeat(num_thetas, 1)  # 在PyTorch中使用repeat替代numpy的tile
# x = torch.randn(1, 4)  # 生成1个样本,每个样本4个特征的随机张量


d = cfg['geom']['Nx'] * cfg['geom']['Ny']

# %%
## get gradient
############################## 当前是只获取一个输入的梯度，需要修改为获取多个输入的梯度 ##########################
# gradvectors = []
# seed = 0

# output = normalize_power(model(x))
# print(output)
# logoutput = torch.log(output)  # get the output values to calculate the jacobian
# # 雅可比矩阵包括每个输出相对于每个输入的偏导数
# grad = []
# for i in range(len(output[0])):
#     model.zero_grad()
#     # print('logoutput:',logoutput)
#     logoutput[0][i].backward(retain_graph=True)
#     grads = []
#     for param in model.parameters():
#         grads.append(param.grad.view(-1))
#     gr = torch.cat(grads)
#     grad.append(gr*torch.sqrt(output[0][i]))   # 为什么要把梯度乘输出的开方项加入grad？
# jacobian = torch.cat(grad)  # 长度为80的torch
# print(jacobian.shape)
# # 相当于存储了每个输出对于所有权重参数的梯度值
# # jacobian = torch.reshape(jacobian, (len(output[0]), 22500))   # reshape成2*40的大小
# jacobian = torch.reshape(jacobian, (len(output[0]), d))   # reshape成2*40的大小
# gradvectors.append(jacobian.detach().numpy())
# # gradvectors is the grads

#################################### 修改为获取多个输入的梯度 #################################################
gradvectors = []
seed = 0
for m in range(num_inputs*num_thetas):
    if m % num_inputs == 0:  # num x's = 100!  num_data默认100
        seed += 1
    torch.manual_seed(seed)
    model.cell.geom.design_region = torch.rand(cfg['geom']['Nx'], cfg['geom']['Ny'])
    output = normalize_power(model(X[m].unsqueeze(0)))
    logoutput = torch.log(output)  # get the output values to calculate the jacobian
    # 雅可比矩阵包括每个输出相对于每个输入的偏导数
    grad = []
    for i in range(len(output[0])):
        model.zero_grad()
        logoutput[0][i].backward(retain_graph=True)
        grads = []
        for param in model.parameters():
            grads.append(param.grad.view(-1))
        gr = torch.cat(grads)
        grad.append(gr*torch.sqrt(output[0][i]))   # 为什么要把梯度乘输出的开方项加入grad？
    jacobian = torch.cat(grad)  # 长度为80的torch
    # print(jacobian.shape)
    # 相当于存储了每个输出对于所有权重参数的梯度值
    # jacobian = torch.reshape(jacobian, (len(output[0]), 22500))   # reshape成2*40的大小
    jacobian = torch.reshape(jacobian, (len(output[0]), d))   # reshape成2*40的大小
    gradvectors.append(jacobian.detach().numpy())

#%%
## get fisher
def get_fisher(gradients, model_output, d):
    """
    Computes average gradients over outputs.
    :param gradients: numpy array containing gradients
    :param model_output: remove?
    :return: numpy array, average jacobian of size (len(x), d)
    """
    fishers = np.zeros((len(gradients), d, d))
    for i in range(len(gradients)):
        grads = gradients[i]
        temp_sum = np.zeros((len(model_output), d, d))
        for j in range(len(model_output)):
            temp_sum[j] += np.array(np.outer(grads[j], np.transpose(grads[j])))
            # 计算外积再将他们进行累加
        fishers[i] += np.sum(temp_sum, axis=0)
    return fishers
# fishers = get_fisher(gradvectors, output[0], 22500)
fishers = get_fisher(gradvectors, output[0], d)
print('get_fisher!')
#%% get fhat
fisher_trace = np.trace(np.average(fishers, axis=0))
# fisher = np.average(np.reshape(fishers, (1, 1, 22500, 22500)), axis=1)
# f_hat = 22500 * fisher / fisher_trace
fisher = np.average(np.reshape(fishers, (num_thetas, num_inputs, d, d)), axis=1)
f_hat = d * fisher / fisher_trace
print("get fhat!")
#%% 
# get eff_dim
def eff_dim(f_hat, n):
    """
    Compute the effective dimension.
    :param f_hat: ndarray
    :param n: list, used to represent number of data samples available as per the effective dimension calc
    :return: list, effective dimension for each n
    """
    effective_dim = []
    for ns in n:
        Fhat = f_hat * ns / (2 * pi * np.log(ns))
        # one_plus_F = np.eye(22500) + Fhat
        one_plus_F = np.eye(d) + Fhat
        det = np.linalg.slogdet(one_plus_F)[1]  # log det because of overflow
        r = det / 2  # divide by 2 because of sqrt
        effective_dim.append(2 * (logsumexp(r) - np.log(1)) / np.log(ns / (2 * pi * np.log(ns))))
    return effective_dim
effdim = eff_dim(f_hat, n)
# # plot the normalised effective dimension
# plt.plot(n, np.array(effdim)/d)
# plt.xlabel('number of data')
# plt.ylabel('normalised effective dimension')
# plt.show()
# 绘制未归一化的effective dimension
plt.figure()
plt.plot(n, np.array(effdim))
plt.xlabel('number of data')
plt.ylabel('effective dimension')
plt.show()
print("parameter num:", d)
print("effective dimension:", effdim)
