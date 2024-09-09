# Learn-DeepLearning-Coding

## 经卷积过后的矩阵尺寸大小的计算公式

`N = (W - F + 2P) / S + 1`

W：输出图片大小 W * W
F：卷积核大小
P：padding
S：步长

## kaiming_normal 参数初始化
核心思想是根据前一层的节点数（或称为特征数量）来调整权重的方差，以保持在网络的每一层，梯度的方差大致相同。这有助于避免在训练过程中出现梯度消失或梯度爆炸的问题。
>来自论文：Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification
