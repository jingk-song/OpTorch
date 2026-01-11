from effective_dimension import Model, EffectiveDimension, ClassicalNeuralNetwork
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

# This is an example file to create a classical model and compute its effective dimension

# create ranges for the number of data, n
# ?
n = [5000, 8000, 10000, 40000, 60000, 100000, 150000, 200000, 500000, 1000000, 3000000, 5000000, 7000000, 10000000]

# specify the size of your neural network
nnsize = [4, 50, 40, 3]


# specify number of data samples and parameter sets to estimate the effective dimension
num_inputs = 10
num_thetas = 10

# create the model
cnet = ClassicalNeuralNetwork(nnsize)

# compute the effective dimension
ed = EffectiveDimension(cnet, num_thetas=num_thetas, num_inputs=num_inputs)
f, trace, grads, output, fishers = ed.get_fhat()
# f (100, 40, 40) 
# grads 长度为1k的列表，其中每个元素是(2, 40)的np.array
# output torch.Size([10000, 2]) 且这两个元素相加为1
# fishers (10000, 40, 40)
effdim = ed.eff_dim(f, n)

# true dimension of the model
d = cnet.d

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

# rep_range = np.tile(np.array([10]), 5) 
# params = np.random.uniform(0, 1, size=(5, 2))
# nparams = np.repeat(params, repeats=rep_range, axis=0)
# print(rep_range, params, nparams)
