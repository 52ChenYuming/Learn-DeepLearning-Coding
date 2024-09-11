# Learn-DeepLearning-Coding

## 经卷积过后的矩阵尺寸大小的计算公式

`N = (W - F + 2P) / S + 1` 
N为小数则向下取整

W：输出图片大小 W * W
F：卷积核大小
P：padding
S：步长

## kaiming_normal 参数初始化
核心思想是根据前一层的节点数（或称为特征数量）来调整权重的方差，以保持在网络的每一层，梯度的方差大致相同。这有助于避免在训练过程中出现梯度消失或梯度爆炸的问题。
>来自论文：Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification


## 感受野大小
`f(i) = (f(i+1) - 1) * s + k`

## 网络深度越来越大的问题

1. 梯度消失，梯度爆炸：每次求偏导都乘一个小于1或者大于1的系数，久而久之梯度消失或者爆炸
2. 退化问题：网络越深错误率越高

## 残差神经网络贡献点：
1. 超深的网络（突破1000）
2. 提出residual模型
3. 使用Batch Normalization加速训练（丢弃dropout） 

## Batch Normalization

使用BN不需要使用偏置

