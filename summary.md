# Experimental Summary

## 6.2

### Work Description
- 初步搭建训练框架，复现Koopman_AE模型、生成lorenz63数据集、并进行结果可视化以及训练策略优化。
    - 目前复现的Koopman_AE模型在逐点预测中表现出较为一致的性能，但是zero-shot中并不能很好预测。
- 阅读相关文献，优化之前的idea


## 6.3

### Work Description
- 学习了重构核希尔伯特空间理论，包括reproducing kernel Hilbert Space来源、定义、构造（Mercer定理）。
- 通过对昨天模型的隐空间可视化发现，模型隐空间并不能很好的学习出线性过程，降维后的隐空间呈现出复杂形状。
    - 尝试调优但是失败了，准备汇总问题向学长汇报。

my idea:
- 结合重构核希尔伯特空间理论，控制方程未知的或部分已知（做一个物理先验信息的接口，如果传入先验信息则考虑先验信息进行Kernel Learning，如果没传入则完全从数据进行Kernel learning。），不规则时间序列的，超分辨率，全时序物理场重建以及预测。
- 从数据以及先验物理信息中学习Mercer kernel，使用SINDy确保该空间的具有解析性质并满足重构核希尔伯特空间性质（验证该方式恢复的latent space的动力学特性是否是线性的），再在该空间中构建可学习的Koopman算子进行推理等。
    - from GPT：若加入物理先验，例如保守量、PDE形式的软约束，可形成 Physics-Informed Kernel Learning（已有文献尝试，将 GP 与 PINN 相结合）。


## 6.4

### Work Description
- 跟学长汇报了这两天的工作内容，使用Koopman_AE模型对lorenz63数据的模拟效果基本达标。
- 尝试寻找了komologrov flow数据集，只能产生出小雷诺数。大雷诺数的数据集暂时没有找到。
- 明天工作内容；
    - 细化我的模型的思路，产生流程图。

## 6.5

### Work Description

- 对小雷诺数的komologrov flow数据集进行了框架搭建以及训练。但是训练结果显示，自回归只会产生出相同的结果，可能原因在于训练数据本身变化很小，此外多步预测损失函数可能也存在问题。
- 完成了我的模型的流程图绘制