# TL_on_MI_BCI : Transfer Learning on Motor Imagery based on Brain-computer Interface

> 基于Pytorch深度学习平台展开的迁移学习在基于脑机接口的运动想象识别研究
>
> Research on Motor Imagery classification based on Brain-computer Interface based on Pytorch deep learning paltform


## 仓库结构
  - DAonSession : 领域自适应方法在跨Session运动想象识别任务上的研究
  - DGonSession : 领域泛化方法在跨Session运动想象识别任务上的研究
  - DGonSub : 领域泛化方法在跨被试运动想象识别任务上的研究
  - KFold : 在单一Session数据上进行K折交叉验证实验，对模型性能进行验证
  - attention_models : 一些已经实现或改进的注意力机制方法
  - loss_funcs : 实验中所用到的损失函数及其相关文件
  - data : 用于存放数据集及DataLoader、Datasets文件
  
## 环境依赖
  - python == 3.10.14
  - mne == 1.7.0
  - torchvision == 0.18.0
