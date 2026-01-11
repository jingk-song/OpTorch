import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from . import Model
# from model_base import Model

# This class creates a classical fully connected feedforward neural network model with size \
# = [input size, neurons in layer 1, neurons in layer 2 ..., output size]


class ClassicalNeuralNetwork(Model, nn.Module):
    def __init__(self, size, num_data=100):
        """
        :param size: list, the size must contain [input_size, #neurons in 1st hidden layer, ...,
        #neurons in nth hidden layer, output_size]
        """
        Model.__init__(self)
        nn.Module.__init__(self)
        self.size = size
        self.inputsize = size[0]
        self.outputsize = size[-1]
        self.layers = nn.ModuleList(
            [nn.Linear(self.size[i - 1], self.size[i], bias=False) for i in range(1, len(self.size))]).double()
        self.d = sum(size[i] * size[i + 1] for i in range(len(size) - 1))       # 注意这个self.d，计算了每两层参数连接关系，4*4+4*4+4*2=40
        # 注意self.d也就是全部的权重数量 等会儿要用这个来计算梯度数量
        self.num_data = num_data        # 注意num_data的含义

    def forward(self, x, params):
        """
        Computes the output of the neural network with tanh activation adam and log_softmax of last layer.
        :param x: ndarray, data inputs for the model (can be one or multiple)
        :param params: for now, random params are used, need to add functionality for using passed params
        :现阶段使用的随机参数param
        :return: torch tensor, model output of size (len(x), output_size)
        """
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        for i in range(len(self.size) - 2):
            x = F.leaky_relu(self.layers[i](x))     # 前面的隐藏层完成计算后进行ReLU操作
        x = F.softmax(torch.tanh(self.layers[-1](x)), dim=-1)   # 最后一层output在输出前进行tanh和softmax操作
        return x

    def get_gradient(self, x, params):
        """
        Computes the gradients of every parameter using each input x, wrt every output.
        :param x: ndarray, data inputs for the model (can be one or multiple)
        :param params: for now, random params are used, need to add functionality for using passed params
        :return: numpy array, gradients of size (len(x), output_size, d)
        # 返回的是对每一个输入项计算的所有权重参数的梯度
        """
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
            x.requires_grad_(False)
        gradvectors = []
        seed = 0
        for m in range(len(x)):
            if m % self.num_data == 0:  # num x's = 100!  num_data默认100
                seed += 1
            torch.manual_seed(seed)
            net = ClassicalNeuralNetwork(self.size)
            net.apply(self._create_rand_params)  # needs to change 进行权重的初始化 这里直接生成随机权重参数
            output = net(x[m], params)
            logoutput = torch.log(output)  # get the output values to calculate the jacobian
            # 雅可比矩阵包括每个输出相对于每个输入的偏导数
            grad = []
            for i in range(self.outputsize):
                net.zero_grad()
                logoutput[i].backward(retain_graph=True)
                grads = []
                for param in net.parameters():
                    grads.append(param.grad.view(-1))
                gr = torch.cat(grads)
                grad.append(gr*torch.sqrt(output[i]))   # 为什么要把梯度乘输出的开方项加入grad？
            jacobian = torch.cat(grad)  # 长度为80的torch
            # 相当于存储了每个输出对于所有权重参数的梯度值
            jacobian = torch.reshape(jacobian, (self.outputsize, self.d))   # reshape成2*40的大小
            gradvectors.append(jacobian.detach().numpy())
        return gradvectors

    def get_fisher(self, gradients, model_output):
        """
        Computes average gradients over outputs.
        :param gradients: numpy array containing gradients
        :param model_output: remove?
        :return: numpy array, average jacobian of size (len(x), d)
        """
        fishers = np.zeros((len(gradients), self.d, self.d))
        for i in range(len(gradients)):
            grads = gradients[i]
            temp_sum = np.zeros((self.outputsize, self.d, self.d))
            for j in range(self.outputsize):
                temp_sum[j] += np.array(np.outer(grads[j], np.transpose(grads[j])))
                # 计算外积再将他们进行累加
            fishers[i] += np.sum(temp_sum, axis=0)
        return fishers

    def _create_rand_params(self, h):
        if type(h) == nn.Linear:
            h.weight.data.uniform_(self.thetamin, self.thetamax)



# nnsize = [4, 4, 4, 2]
# cnet = ClassicalNeuralNetwork(nnsize)
# cgrad = cnet.get_gradient()
# print(cgrad)