# -*- coding: utf-8 -*-
# 9.14修改：量化改成1bit
# 待完成的修改：输出训练后的介电常数分布矩阵
# 所有数据点下的输出探头结果
# 待完成的修改：在训练完成后将数据集的所有数据便利一遍，并记录所有输出结果  
# 待完成的修改：现在每次完成后都要输出图片，然后程序会卡在这里，已修改
# 待完成的修改：一个pixel的尺寸改成CUMAC的特征尺寸150nm    已修改
# 硅和二氧化硅的折射率改成实际的折射率 二氧化硅1.44 硅3.47 已修改
# 选择达到稳态后某一时刻的输出作为结果！
"""
Created on Sat Dec 18 00:36:26 2021
交叉熵
@author: Lenovo
"""
import os
import csv
import Sim as oNN
import torch,math
import numpy as np
import Sim as oNN
import matplotlib.pyplot as plt
from scipy.signal import hilbert 
import torch
from torch.autograd import Variable
import torch.optim as optim
from sklearn import datasets
from sklearn.preprocessing import MinMaxScaler
import torch.nn.functional as F
from tqdm import tqdm

WAVELENGTH = 1550e-9
SPEED_LIGHT: float = 299_792_458.0  # [m/s] 光速
sys_path = 'D:/研究生工作/Photonic-Computing/code/OpTorch_9_13/'
exp_name = 'exp_3'
folder_path = sys_path + exp_name
# 检查文件夹是否存在，如果不存在则创建
if not os.path.exists(folder_path):
    os.makedirs(folder_path)
    print(f"文件夹 '{folder_path}' 已创建")
else:
    print(f"文件夹 '{folder_path}' 已存在")


def Init(p1,p2,p3,p4):
    oNN.set_backend("torch.float32") # torch.cuda.float64
    #设置整体几何区域
    # grid = oNN.Grid(shape=(100, 100, 1), grid_spacing=0.05*WAVELENGTH,permittivity = 1)
    grid = oNN.Grid(shape=(100, 100, 1), grid_spacing=150e-9,permittivity = 1.44)
    #设置波源
    grid[20, 33:37, 0] = oNN.LineSource(period= WAVELENGTH / (SPEED_LIGHT),
                                                phase_shift=p1*torch.pi,name="source1") # , pulse = True, cycle = 10, hanning_dt=15)
    grid[20, 43:47, 0] = oNN.LineSource(period= WAVELENGTH / (SPEED_LIGHT),
                                                phase_shift=p2*torch.pi,name="source2") # , pulse = True, cycle = 10, hanning_dt=15)
    grid[20, 53:57, 0] = oNN.LineSource(period= WAVELENGTH / (SPEED_LIGHT),
                                                phase_shift=p3*torch.pi,name="source3") # , pulse = True, cycle = 10, hanning_dt=15)
    grid[20, 63:67, 0] = oNN.LineSource(period= WAVELENGTH / (SPEED_LIGHT),
                                                phase_shift=p4*torch.pi,name="source4") # , pulse = True, cycle = 10, hanning_dt=15)#,
                                                # pulse=True, cycle=3, hanning_dt=4e-15)
    # grid[20, 35, 0] = oNN.PointSource(period=WAVELENGTH / (SPEED_LIGHT),
    #                                             phase_shift=p1*torch.pi,name="source1")
    # grid[20, 45, 0] = oNN.PointSource(period=WAVELENGTH / (SPEED_LIGHT),
    #                                             phase_shift=p2*torch.pi,name="source2")
    # grid[20, 55, 0] = oNN.PointSource(period=WAVELENGTH / (SPEED_LIGHT),
    #                                             phase_shift=p3*torch.pi,name="source3")
    # grid[20, 65, 0] = oNN.PointSource(period=WAVELENGTH / (SPEED_LIGHT),
    #                                             phase_shift=p4*torch.pi,name="source4")#,
    #边界条件
    # x 边界
    grid[0:10, :, :] = oNN.PML(name="pml_xlow")
    grid[-10:, :, :] = oNN.PML(name="pml_xhigh")
    # y 边界
    grid[:, 0:10, :] = oNN.PML(name="pml_ylow")
    grid[:, -10:, :] = oNN.PML(name="pml_yhigh")

    grid[0:20, 33:37, 0] = oNN.Object(permittivity=torch.ones([20, 4, 1], device=device)*2.03 + 1.44, name="wg1")  # 这里设置硅波导的折射率为3.47
    grid[0:20, 43:47, 0] = oNN.Object(permittivity=torch.ones([20, 4, 1], device=device)*2.03 + 1.44, name="wg2")
    grid[0:20, 53:57, 0] = oNN.Object(permittivity=torch.ones([20, 4, 1], device=device)*2.03 + 1.44, name="wg3")
    grid[0:20, 63:67, 0] = oNN.Object(permittivity=torch.ones([20, 4, 1], device=device)*2.03 + 1.44, name="wg4")

    #探测器位置
    for i in range(0, 10):
        grid[80, 31 + 4 * i : 33 + 4 * i, 0] = oNN.LineDetector(name="detector" + str(i))
        grid[80:100, 31 + 4 * i : 33 + 4 * i, 0] = oNN.Object(permittivity=torch.ones([20, 2, 1], device=device)*2.03 + 1.44, name="op" + str(i))
    # 放置探测器查看光源的波形
    # grid[21, 33 : 37, 0] = oNN.LineDetector(name="detector" + str(10))
    return grid

def heaviside(x, beta):
    # 由于涉及到梯度，需要确保操作是可微分的
    return 1 / (1 + torch.exp(-beta * x))

# 定义量化函数，根据位数和范围对参数进行量化
def quantize(x, beta, bits):
    # 生成阈值，这里x的范围假设为0到x的最大值，根据n_bits自动生成阈值
    thresholds = torch.linspace(x.min().item(), x.max().item(), 2**bits + 1)[1:-1]  # 生成2^n等级，但是去掉了两端的0和1，保留2^n-1个阈值
    # 初始化y为零张量，与x形状相同
    y = torch.zeros_like(x)
    for t in thresholds:
        y += heaviside(x - t, beta)
    y = y / len(thresholds)
    # 调整y的范围以匹配x的范围
    # y = y * (x.max() - x.min()) + x.min()
    return y

# 定义量化感知模块，根据位数和范围对参数进行量化，并记录量化误差
class QuantAware(torch.nn.Module):
    def __init__(self, beta, bits):
        super(QuantAware, self).__init__()
        self.bits = bits
        self.beta = beta

    def forward(self, x):
        # Quantize the input
        x_q = quantize(x, self.beta, self.bits)
        return x_q

def custom_blur_2d(image, radius):
    # Create a circular kernel with specified radius
    kernel_size = 2 * radius + 1
    kernel = torch.zeros((kernel_size, kernel_size), dtype=torch.float64)
    center = (radius, radius)
    for i in range(kernel_size):
        for j in range(kernel_size):
            # Convert i and j to tensors
            i_tensor = torch.tensor([i], dtype=torch.float64)
            j_tensor = torch.tensor([j], dtype=torch.float64)
            # Perform the distance calculation using tensors
            dist = torch.sqrt((i_tensor - center[0])**2 + (j_tensor - center[1])**2)
            kernel[i, j] = max(radius - dist, torch.tensor(0.0, dtype=torch.float64)).item()

    # Normalize the kernel
    kernel /= torch.sum(kernel)
    # Reshape kernel for conv2d (out_channels, in_channels, height, width)
    kernel = kernel.reshape(1, 1, kernel_size, kernel_size).double()

    # Apply convolution
    # image needs to be of shape (batch_size, channels, height, width)
    # We add two singleton dimensions to fit this requirement: one for batch size, one for channels
    image = image.unsqueeze(0).unsqueeze(0).double()
    blurred_image = F.conv2d(image, kernel, padding=radius)###注意！！！
    blurred_image = blurred_image.squeeze(0).squeeze(0)
    return blurred_image

# import time as tm
class OpTorch(torch.nn.Module):
    def __init__(self, x_min, x_max, y_min, y_max, beta, bits, radius):
        super(OpTorch, self).__init__()
        self.x_geo=int(x_max-x_min)
        self.y_geo=int(y_max-y_min)
        self.Permittivity = torch.nn.Parameter(1*torch.ones([self.x_geo, self.y_geo,1], device=device))#0.5*
        # 定义量化感知模块，对自定义的参数进行量化，假设范围为[0, 1]
        self.quant = QuantAware(beta, bits)
        self.radius = radius
        self.x_min=x_min
        self.x_max=x_max
        self.y_min=y_min
        self.y_max=y_max

    def forward(self, time, x_in, blur=False, Quant=False):
        x1 = torch.tensor(x_in[0])
        x2 = torch.tensor(x_in[1])
        x3 = torch.tensor(x_in[2])
        x4 = torch.tensor(x_in[3])
        grid=Init(x1, x2, x3, x4)
        if blur:
            self.Permittivity.data = custom_blur_2d(self.Permittivity.squeeze(), self.radius).unsqueeze(2)
        # 对自定义的参数进行量化
        if Quant:
            self.Permittivity.data = self.quant(self.Permittivity)
        grid[self.x_min:self.x_max, self.y_min:self.y_max, 0] = oNN.Object(permittivity=self.Permittivity*2.03+1.44, name="object")
        # start = tm.time() # 记录循环开始的时间
        for i in range(time):
            grid.step()
        # end = tm.time() # 记录循环结束的时间
        # print(f"The loop took {end-start} seconds.")
        ###预测值为第i个监视器的电场
        # m=(grid.E**2)#.requires_grad_(True)
        # print(m)
        return grid

# 加载iris数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 数据归一化
scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)  # 其中元素类型为<class 'numpy.ndarray'>
# 打乱样本和标签
indices = np.arange(X_scaled.shape[0])
np.random.shuffle(indices)
X_shuffled = X_scaled[indices]
y_shuffled = y[indices]

from sklearn.model_selection import train_test_split
from matplotlib.backends.backend_pdf import PdfPages

X_train, X_test, y_train, y_test = train_test_split(X_shuffled, y_shuffled, test_size=0.3, random_state=42)


device = torch.device("cpu")#torch.device("cuda:0") # 选择第一个GPU
# device = torch.device("cuda:0")
# torch.cuda.set_device(device)
x_min=35
x_max=75
y_min=30
y_max=70

# Set the desired radius for smoothing
radius = 2 # 实际距离为radius*grid_space
beta=5
bits = 1
model = OpTorch(x_min,x_max,y_min,y_max,beta,bits,radius).to(device)

# loss_fn = torch.nn.MSELoss()
loss_fn = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)#1.5e-3

loss_iter = []
blur = False
Quant = False
# filename = 'greyBlur'
filename = exp_name
time_steps = 300
EPOCH_NUM = 24
batch_size = 12
for t in range(EPOCH_NUM):
    # if t >=10:
    #     blur = True
    #     # Quant = True
    if t%5 == 4:
        Quant = True
        beta = 5*beta
    else:
        Quant = False
    # 从第20个epoch之后，一直开启量化操作
    if t == 20:
        Quant = True
    # if t >=5: #第10次后开启工艺约束或者量化
    #     # blur = True
    #     Quant = True
    # 按batch_size循环取出数据并送入model()
    num_samples = X_train.shape[0]
    loss_avg = []
    for i in tqdm(range(0, num_samples, batch_size)):
        # 取出一个batch的样本和标签
        X_batch = X_train[i:i+batch_size]
        y_batch = y_train[i:i+batch_size]
        losses = []
        for j in range(0, X_batch.shape[0]):
            sim = model(time_steps,X_batch[j,],blur=blur,Quant=Quant)
            output = torch.tensor([], device=device)
            for det_index in [1,5,9]: # 选择哪几个探测器作为输出，把150个timestep求和了
                output=torch.cat([output,sum([(x**2).mean(dim=0)[-1] for x in sim.detectors[det_index].detector_values()["E"]]).unsqueeze(dim=0)],dim=0)# 获取Ez
            loss = loss_fn(output/torch.sum(output).reshape(1,-1), torch.tensor(y_batch[j], dtype=torch.long,device = output.device).unsqueeze(0))
            losses.append(loss)
        # if i % 2 == 0:
        # sim.visualize(z=0, index=i,)# animate=True, save=True, folder=simfolder)
        # tmp = np.ones((100, 100, 1))
        # tmp[x_min:x_max, y_min:y_max, :] = model.Permittivity.detach().cpu().numpy()*1.8+1
        # plt.imshow(tmp,)#.squeeze()) 'viridis'
        # plt.colorbar()
        # plt.title(f"batch_idx:{i:3.0f}")
        # plt.show()

        # model.zero_grad()
        # 计算平均损失
        batch_loss = sum(losses) / len(losses)
        num_gpu = 8
        if t!=0:
            batch_loss = batch_loss * math.sqrt(num_gpu/t)
        else:
            batch_loss = batch_loss * math.sqrt(num_gpu)
        # 执行反向传播
        batch_loss.backward()
        # 更新模型参数
        optimizer.step()
        model.Permittivity.data.clamp_(0, 1)
        # 清空梯度
        optimizer.zero_grad()
        loss_avg.append(batch_loss)
    loss = sum(loss_avg) / len(loss_avg)
    # 初始化准确率计数器
    correct_predictions = 0

    # 现在修改成每训练一个epoch都输出一个介电常数分布
    # 保存训练好的介电常数分布
    dielec_dist = model.Permittivity.detach().cpu().numpy()
    array_2d = np.squeeze(dielec_dist)
    np.savetxt(folder_path + '/dielec_dist_epoch' + str(t) + '.txt', array_2d, delimiter=' ', fmt='%.6f')
    np.savetxt(folder_path + '/dielec_dist_epoch' + str(t) + '.csv', array_2d, delimiter=',', fmt='%.8f')

    # 遍历测试集
    for i in range(X_test.shape[0]):
        # 取出一个测试样本
        X_sample = X_test[i:i + 1]  # 确保是一个二维数组
        y_sample = y_test[i]

        # 使用模型进行预测
        # 注意：这里需要根据您的模型调整，确保model(t, X_sample)能够输出一个预测值
        with torch.no_grad():
            sim = model(time_steps, X_sample[0],blur=blur,Quant=Quant)
        output = torch.tensor([], device=device)
        for det_index in [1, 5, 9]:  # 选择哪几个探测器作为输出
            output = torch.cat([output, sum([(x ** 2).mean(dim=0)[-1] for x in
                                             sim.detectors[det_index].detector_values()["E"]]).unsqueeze(dim=0)], dim=0)
        output = output / torch.sum(output).reshape(1, -1)
        # 获取预测结果
        # 假设output是一个包含预测概率的向量，我们需要找到概率最高的类别
        prediction = torch.argmax(output, dim=1)

        # 更新准确率计数器
        if prediction == y_sample:
            correct_predictions += 1
    # with PdfPages(str(t)+'.pdf') as pdf:
    plt.figure()
    sim.visualize(z=0)
    # plt.imshow(sim.E[:,:,0,2].detach().cpu(), cmap=plt.cm.RdBu)
    tmp = np.ones((100, 100, 1))
    tmp[x_min:x_max, y_min:y_max, :] = model.Permittivity.detach().cpu().numpy() * 2.03 + 1.44
    plt.imshow(tmp,)  # .squeeze()) 'viridis'  alpha=0.5
    plt.colorbar()
    plt.savefig('D:/研究生工作/Photonic-Computing/code/OpTorch_9_13/'+filename+'/'+str(t)+'.svg',format='svg',bbox_inches='tight', dpi=330)
    # plt.show()
    plt.close()
    
    # 计算准确率
    accuracy = correct_predictions / X_test.shape[0]
    # print(f"Test accuracy: {accuracy:.2f}")
    print("Epoch: {} -- Loss: {} -- Test accuracy: {}".format(t, loss, accuracy))
    loss_iter.append(loss.item())
    if (t+1) % 10 == 0:
        print(t+1, loss.item())
plt.figure()
plt.plot(loss_iter,'b', label='Training loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.savefig('./'+filename+'/'+'loss.svg',format='svg',bbox_inches='tight', dpi=330)
#plt.show()
plt.close()
# 保存模型参数
# model_path = 'model_gpu_phase_MMI_batch_size'+str(batch_size)+'.pth'  # 您可以更改这个路径来保存模型到您想要的位置
# torch.save(model.state_dict(), model_path)
# # 如果您还想保存模型的结构以便以后重新实例化模型，可以这样做：
# # torch.save(model, 'model_complete.pth')
#
# # 加载模型参数
# loaded_state_dict = torch.load(model_path)
# # 如果您有模型的实例，可以直接加载参数
# model.load_state_dict(loaded_state_dict)
# 或者如果您没有模型的实例，您需要先实例化模型，然后加载参数
# model = OpTorch(x_min, x_max, y_min, y_max)
# model.load_state_dict(loaded_state_dict)


'''结果测试'''
# 保存训练好的介电常数分布
dielec_dist = model.Permittivity.detach().cpu().numpy()
# print('dielec_dist', dielec_dist.shape)
array_2d = np.squeeze(dielec_dist)
np.savetxt(folder_path + '/dielec_dist.txt', array_2d, delimiter=' ', fmt='%.6f')
np.savetxt(folder_path + '/dielec_dist.csv', array_2d, delimiter=',', fmt='%.8f')

# 初始化准确率计数器
correct_predictions = 0

# 遍历测试集
for i in tqdm(range(X_test.shape[0])):
    # 取出一个测试样本
    X_sample = X_test[i:i+1]  # 确保是一个二维数组
    y_sample = y_test[i]   

    # 使用模型进行预测
    # 注意：这里需要根据您的模型调整，确保model(t, X_sample)能够输出一个预测值
    with torch.no_grad():
        sim = model(time_steps, X_sample[0],Quant=Quant)
    output = torch.tensor([], device=device)
    for det_index in [1,5,9]: # 选择哪几个探测器作为输出
        output=torch.cat([output,sum([(x**2).mean(dim=0)[-1] for x in sim.detectors[det_index].detector_values()["E"]]).unsqueeze(dim=0)],dim=0)
    output = output/torch.sum(output).reshape(1,-1)
    # 获取预测结果
    # 假设output是一个包含预测概率的向量，我们需要找到概率最高的类别
    prediction = torch.argmax(output, dim=1)
    
    # 更新准确率计数器
    if prediction == y_sample:
        correct_predictions += 1


# 计算准确率
accuracy = correct_predictions / X_test.shape[0]
print(f"Test accuracy: {accuracy:.2f}")
plt.figure()
tmp = 1.44 * np.ones((100, 100, 1))
tmp[x_min:x_max, y_min:y_max, :] = model.Permittivity.detach().cpu().numpy() * 2.03 + 1.44
plt.imshow(tmp, )  # .squeeze()) 'viridis'
plt.colorbar()
plt.savefig('./'+filename+'/'+'epsilon.svg',format='svg',bbox_inches='tight', dpi=330)
# plt.show()
plt.close()
plt.figure()
sim.visualize(z=0)
# plt.imshow(sim.E[:,:,0,2].detach().cpu(), cmap=plt.cm.RdBu)
plt.savefig('./'+filename+'/'+'field.svg',format='svg',bbox_inches='tight', dpi=330)
# plt.show()
plt.close()
plt.figure()
plt.bar(range(3), output[0].cpu(), width=0.5)
plt.savefig('./'+filename+'/'+'output.svg',format='svg',bbox_inches='tight', dpi=330)
# plt.show()
plt.close()

# 遍历所有数据集，获取探头电场数据并保存 使用X_scaled和Y数据，不打乱顺序，便于检验
# 初始化准确率计数器
all_correct_predictions = 0
result = []
# 遍历测试集
for i in tqdm(range(X_scaled.shape[0])):
    E = np.zeros((6, time_steps))
    # 取出一个测试样本
    X_sample = X_scaled[i:i+1]  # 确保是一个二维数组
    y_sample = y[i]
    # print('X_sample', X_sample)
    # print('Y_sample', y_sample)
    # 使用模型进行预测
    # 注意：这里需要根据您的模型调整，确保model(t, X_sample)能够输出一个预测值
    with torch.no_grad():
        sim = model(time_steps, X_sample[0],Quant=Quant)
    output = torch.tensor([], device=device)
    for det_index in [1,5,9]: # 选择哪几个探测器作为输出
        flag = 0
        for x in sim.detectors[det_index].detector_values()["E"]:
            E[int(det_index/2), flag] = x[0,2].item()
            E[int(det_index/2)+1, flag] = x[1,2].item()
            flag = flag + 1
        output=torch.cat([output,sum([(x**2).mean(dim=0)[-1] for x in sim.detectors[det_index].detector_values()["E"]]).unsqueeze(dim=0)],dim=0)
    output = output/torch.sum(output).reshape(1,-1)
    # 获取预测结果
    # 假设output是一个包含预测概率的向量，我们需要找到概率最高的类别
    prediction = torch.argmax(output, dim=1)
    np.savetxt(folder_path + '/E' + str(i) + '.csv', E, delimiter=',', fmt='%.12f')

    
    # 更新准确率计数器
    if prediction == y_sample:
        all_correct_predictions += 1
        result.append(1)
    else:
        result.append(0)
# 将结果写入csv文件
# 写入 CSV 文件
with open(folder_path + '/result.txt', mode='w') as file:
    for item in result:
        file.write(f"{item}\n")  # 每个元素写在一行，\n 表示换行

# 计算准确率
accuracy = all_correct_predictions / X_scaled.shape[0]
print(f"all dataset Test accuracy: {accuracy:.2f}")
plt.figure()
tmp = 1.44 * np.ones((100, 100, 1))
tmp[x_min:x_max, y_min:y_max, :] = model.Permittivity.detach().cpu().numpy() * 2.03 + 1.44
# plt.imshow(tmp, )  # .squeeze()) 'viridis'
plt.colorbar()
plt.savefig('./'+filename+'/'+'epsilon.svg',format='svg',bbox_inches='tight', dpi=330)
# plt.show()
plt.close()
plt.figure()
sim.visualize(z=0)
# plt.imshow(sim.E[:,:,0,2].detach().cpu(), cmap=plt.cm.RdBu)
plt.savefig('./'+filename+'/'+'field.svg',format='svg',bbox_inches='tight', dpi=330)
# plt.show()
plt.close()
plt.figure()
plt.bar(range(3), output[0].cpu(), width=0.5)
plt.savefig('./'+filename+'/'+'output.svg',format='svg',bbox_inches='tight', dpi=330)
# plt.show()
plt.close()



# with torch.no_grad():
#     model(EPOCH_NUM)

# plt.figure()
# plt.imshow(E[:,:,0,2].detach().numpy())#.squeeze())
# plt.colorbar()
# # loss_iter=[x.detach().numpy() for x in loss_iter]
# plt.figure()
# plt.plot(loss_iter)

    
#     def closure():
#         optimizer.zero_grad()
#         # tmp=model(X)
#         u = wavetorch.utils.normalize_power(model(X).sum(dim=1))
#         loss = criterion(u, torch.tensor([2]))
#         loss.backward()
#         return loss

#     loss = optimizer.step(closure)
#     model.cell.geom.constrain_to_design_region()
#     print("Epoch: {} -- Loss: {}".format(i, loss))
#     loss_iter.append(loss.item())

# plt.imshow(E[:,:,0,2].detach().numpy())#.squeeze())
# plt.colorbar()
   
# tmp=np.array([x.numpy() for x in grid.detectors[0].detector_values()["E"]])

# hilbertPlot = abs(
#     hilbert([x[0][2] for x in tmp])
# )
# plt.figure()
# plt.plot(hilbertPlot)
# plt.plot(tmp[:,0,2])
# E=grid.E.numpy()
# e=E**2
# plt.figure()
# plt.imshow(E[:,:,0,2])
# plt.colorbar()
# plt.figure()
# plt.imshow(e[:,:,0,2])
# plt.colorbar()

# plt.figure()
# m=grid.E**2
# plt.imshow(m[:,:,0,2])
# plt.colorbar()