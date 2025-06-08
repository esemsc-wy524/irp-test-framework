# Dataset Overview

## 1. Lorenz 63 模型（Lorenz System）

### 背景：

* 最早由 **Edward Lorenz** 在 1963 年提出，用于简化对流（大气热对流）模型。
* 是最著名的**混沌系统**之一。
* 仅有 **3个状态变量**，但系统行为呈现出高度非线性和混沌特性。

### 方程：

$$
\begin{cases}
\frac{dx}{dt} = \sigma (y - x) \\
\frac{dy}{dt} = x(\rho - z) - y \\
\frac{dz}{dt} = xy - \beta z
\end{cases}
$$

其中：

* $\sigma = 10$（Prandtl数）
* $\rho = 28$
* $\beta = 8/3$

### 特点：

* 系统在某些参数下产生“蝴蝶效应”（极度敏感）。
* 经典的“Lorenz吸引子”图像呈现蝴蝶状。

---

## 2. Lorenz 96 模型（Lorenz 1996）

### 背景：

* 由 Lorenz 在 1996 年提出，用于测试数值天气预报方法的稳定性。
* 是一个**高维、周期边界**的动力系统模型。
* 常被用作 **多变量时间序列建模**、**高维系统降维建模**的测试平台。

### 方程：

对于 $N$ 个状态变量 $x_i$：

$$
\frac{dx_i}{dt} = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F
$$

* 所有索引都是 mod $N$，表示**周期边界**。
* $F$ 是强迫项，常取 $F=8$ 时系统呈现混沌。

### 特点：

* 可调维度（一般用 $N=40$）
* 较适合**多维 latent 表征学习**。
* 动力学复杂但控制良好，适合用于 Koopman/GNN 等建模。

---

## 3. Kolmogorov 流（Kolmogorov Flow）

### 背景：

* 源于对二维 Navier-Stokes 方程（不可压缩流体动力学方程）在周期边界下的简化。
* 是**强非线性流体系统**的典型模型，能产生湍流行为。
* 多用于深度学习对流场预测能力的检验（如 FNO、Koopman AE、Diffusion models）。

### 方程（二维不可压缩Navier-Stokes）：

$$
\frac{\partial \mathbf{u}}{\partial t} + \mathbf{u} \cdot \nabla \mathbf{u} = -\nabla p + \nu \Delta \mathbf{u} + f
$$

$$
\nabla \cdot \mathbf{u} = 0
$$

其中 $f = F \sin(k y)$ 是周期性外力。

### 特点：

* 表现出丰富的**涡旋结构**和湍流行为。
* 是物理场建模、生成建模、Koopman嵌入等任务中的**高级 benchmark**。
* 数据通常来自高分辨率数值模拟（如 DNS）。

