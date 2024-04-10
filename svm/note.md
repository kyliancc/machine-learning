# SVM (支持向量机) 笔记

### 介绍

SVM (支持向量机, Support Vector Machine) 算法是一个在样本空间中找到一个**划分超平面**，将正类样本与父类样本分开，完成分类任务的算法。

我们举一个在二维平面中的例子，如图：

![](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Svm_separating_hyperplanes_%28SVG%29.svg/1920px-Svm_separating_hyperplanes_%28SVG%29.svg.png)
*图片来源：wikipedia.org*

凭借直觉，我们可以很容易得出 $H_3$ 是效果最好的划分超平面。

对于 $H_1$，我们发现它不能正确地将样本分开；对于 $H_2$，它虽然可以将样本正确分类，但是它的泛化性不足；对于 $H_3$，我们可以看到它既能正确分类，而且泛化性不错。

我们的目标是，找到一个类似于上图 $H_3$ 的划分超平面，使其能够正确分类，并且保持最好的泛化性。

### 间隔与支持向量

在样本空间中，我们可使用一个方程描述划分超平面：
$$
\bm{w}^T\bm{x}+b=0.
$$

其中 $\bm{w}$ 为超平面法向量，决定超平面的方向；$b$ 为位移项，决定超平面与原点间的距离。我们可以使用 $(\bm{w},b)$ 来表示超平面。

我们假设划分超平面能将所有样本正确分类，我们规定：
$$
\left\{
    \begin{aligned}
    & \bm{w}^T\bm{x}_i+b \ge +1, \qquad y_i=+1; \\
    & \bm{w}^T\bm{x}_i+b \le -1, \qquad y_i=-1.
    \end{aligned}
\right.
$$

如下图：

![](https://upload.wikimedia.org/wikipedia/commons/2/2a/Svm_max_sep_hyperplane_with_margin.png)
*图片来源：wikipedia.org*

可以发现，距离超平面最近的几个样本正好位于直线上，也就是使上面不等式的等号成立。这些样本被称为**支持向量** (support vector)。

将两直线间距离公式扩展到高维可以得到**间隔** (margin)：
$$
\gamma = \frac{2}{||\bm{w}||}
$$

**为了使泛化性能最佳，我们需要使间隔最大。**

因此我们需要解决以下问题：
$$
\begin{aligned}
& \max_{\bm{w},b} \frac{2}{||\bm{w}||} \\
& \text{s.t.} \quad y_i(\bm{w}^T\bm{x}_i+b) \ge 1, \qquad i=1,2,...,m.
\end{aligned}
$$

为了运算方便，我们将 $\max ||\bm{w}||^{-1}$ 转化为等价的 $\min ||\bm{w}||^2$，并且我们将约束转化为标准形式。所以问题转化为：
$$
\begin{aligned}
& \min_{\bm{w},b} \frac{1}{2}||\bm{w}||^2 \\
& \text{s.t.} \quad 1-y_i(\bm{w}^T\bm{x}_i+b) \le 0, \qquad i=1,2,...,m.
\end{aligned}
$$

该算式就是 SVM 的**基本型**。

### 对偶问题

为了求解这个问题，我们将使用拉格朗日乘子法得到其对偶问题。

对基本型添加拉格朗日乘子 $\alpha \ge 0$，则拉格朗日函数为：
$$
L(\bm{w},b,\bm{\alpha})=\frac{1}{2}||\bm{w}||^2 + \sum_{i=1}^m \alpha_i(1-y_i(\bm{w}^T\bm{x}_i+b)).
$$

其中 $\bm{\alpha}=[\alpha_1;\alpha_2;...;\alpha_m]$。

令 $L(\bm{w},b,\bm{\alpha})$ 对 $\bm{w},b$ 的偏导为零：
$$
\left\{
    \begin{aligned}
    \bm{w} & = \sum_{i=1}^m \alpha_iy_i\bm{x}_i \\
    0 & = \alpha_iy_i
    \end{aligned}
\right.
$$

将 $\bm{w}$ 代入基本型，并且结合约束条件可得对偶问题：
$$
\begin{aligned}
& \max_{\bm{\alpha}} \sum_{i=1}^m\alpha_i - \frac{1}{2} \sum_{i=1}^m\sum_{j=1}^m \alpha_i\alpha_jy_iy_j\bm{x}^T_i\bm{x}_j \\
& \text{s.t.} \quad \sum_{i=1}^m\alpha_iy_i=0, \qquad \alpha_i \ge 0, \quad i=1,2,...,m.
\end{aligned}
$$

且上式需要满足 KKT (Karush-Kuhn-Tucker) 条件：
$$
\left\{
    \begin{aligned}
    & \alpha_i \ge 0; \\
    & y_if(\bm{x}_i)-1 \ge 0; \\
    & \alpha_i(y_if(\bm{x}_i)-1)=0.
    \end{aligned}
\right.
$$

将 $\bm{w}$ 代入划分超平面方程，得到模型：
$$
\begin{aligned}
f(\bm{x}) = & \bm{w}^T\bm{x}+b \\
= & \sum_{i=1}^m \alpha_iy_i\bm{x}^T_i\bm{x}+b
\end{aligned}
$$

这说明，由对偶问题解出的 $\bm{\alpha}$ 代入模型即可求出超平面方程的 $\bm{w}$。

由 KKT 条件可以看出，对于任何样本都有 $\alpha_i=0$ 或 $y_if(\bm{x}_i)=1$。若 $\alpha_i=0$，则对 $f(\bm{x})$ 没有影响，说明不是支持向量。相反，若 $\alpha_i>0$，必有 $y_if(\bm{x}_i)=1$，所对应的样本是一个支持向量。这说明，**训练完成后，大部分训练样本都不重要，最终模型仅与支持向量有关。**

因此，在接下来求解 $\bm{\alpha}$ 的过程中，我们只需要考虑约束 $\sum_{i=1}^m\alpha_iy_i=0$ 和条件 $\alpha_i \ge 0$。

### SMO 算法

为了方便查看，我再贴一次需要解决的对偶问题：
$$
\begin{aligned}
& \max_{\bm{\alpha}} \sum_{i=1}^m\alpha_i - \frac{1}{2} \sum_{i=1}^m\sum_{j=1}^m \alpha_i\alpha_jy_iy_j\bm{x}^T_i\bm{x}_j \\
& \text{s.t.} \quad \sum_{i=1}^m\alpha_iy_i=0, \qquad \alpha_i \ge 0, \quad i=1,2,...,m.
\end{aligned}
$$

为了求解上面的对偶问题，SMO (Sequential Minimal Optimization) 算法是一个不错的选择。它相比于通用的二次规划算法更加高效。

SMO 算法的基本思路是先选取一个 $\alpha_i$，然后固定其他参数。但是考虑到约束 $\sum_{i=1}^m\alpha_iy_i=0$，在更新 $\alpha_i$ 的同时，需要同时更新另外一个 $\alpha_j$，才能满足约束。也就是在更新的过程中需要满足：
$$
c = -\sum_{k \neq i,j} \alpha_ky_k \\
\alpha_iy_i+\alpha_jy_j=c, \quad \alpha_i\ge0, \quad \alpha_j\ge0
$$

因此算法的思路大概为：
- 选取一对需要更新的变量 $\alpha_i$ 和 $\alpha_j$；
- 固定除 $\alpha_i$ 和 $\alpha_j$ 以外的参数，求解基本型获得更新后的 $\alpha_i$ 和 $\alpha_j$。

用 $\alpha_iy_i+\alpha_jy_j=c$ 消去基本型的 $\alpha_j$，可以得到一个关于 $\alpha_i$ 的单变量二次规划问题，唯一的约束为 $\alpha_i\ge0$。我们可以根据二次函数的性质轻易算出更新后的 $\alpha_i$，而不需要优化算法。

另外 SMO 采用了一种启发式：使选取的两个变量对应的样本之间的间隔最大。对它们进行更新会带给目标函数值最大的变化。

### 核函数

在之前，我们假设训练样本是线性可分的。但是在现实任务中，仅仅一个超平面或许并不能很好地将样本正确分类。我们可以将样本**映射到高维空间**，然后再进行分类。

![](https://upload.wikimedia.org/wikipedia/commons/1/1b/Kernel_Machine.png)
*图片来源：wikipedia.org*

我们可以使用一个函数 $\phi(\bm{x})$ 对样本进行映射。代入到基本型可以得到：
$$
\max_{\bm{\alpha}} \sum_{i=1}^m\alpha_i - \frac{1}{2} \sum_{i=1}^m\sum_{j=1}^m \alpha_i\alpha_jy_iy_j\phi(\bm{x}_i)^T\phi(\bm{x}_j),
$$
约束略。

由于 $\bm{x}$ 再映射到高维后维数可能会很高，不方便计算内积，所以我们可以设想一个新函数：
$$
\kappa(\bm{x}_i,\bm{x}_j) = \phi(\bm{x}_i)^T\phi(\bm{x}_j).
$$

对偶问题变为：
$$
\max_{\bm{\alpha}} \sum_{i=1}^m\alpha_i - \frac{1}{2} \sum_{i=1}^m\sum_{j=1}^m \alpha_i\alpha_jy_iy_j\kappa(\bm{x}_i,\bm{x}_j),
$$
约束略。

求解后可以得到：
$$
\begin{aligned}
f(\bm{x}) & =\bm{w}^T\phi(\bm{x})+b \\
& =\sum_{i=1}^m \alpha_iy_i\kappa(\bm{x}_i,\bm{x}_j)+b.
\end{aligned}
$$

那么，什么样的函数能做核函数呢？

**只要一个矩阵所对应的核矩阵半正定，它就能作为核函数使用。**

下面列举几种常用核函数：

| 名称       | 表达式                                                   | 参数                      |
| ---------- | ------------------------------------------------------- | ------------------------ |
| 线性核     | $\bm{x}_i^T\bm{x}_j$                                     |                          |
| 多项式核   | $(\bm{x}_i^T\bm{x}_j)^d$                                 | $d\ge1$ 为多项式的次数    |
| 高斯核     | $\exp(-\frac{\Vert\bm{x}_i-\bm{x}_j\Vert^2}{2\sigma^2})$ | $\sigma>0$ 为高斯核的带宽 |
| 拉普拉斯核  | $\exp(-\frac{\Vert\bm{x}_i-\bm{x}_j\Vert}{\sigma})$     | $\sigma>0$               |
| Sigmoid 核 | $\tanh(\beta\bm{x}_i^T\bm{x}_j+\theta)$                 | $\beta>0,\theta<0$       |

此外还可以通过函数组合：
- 核函数 $\kappa_1,\kappa_2$ 的线性组合 $\gamma_1\kappa_1 + \gamma_2\kappa_2, \quad (\gamma_1,\gamma_2>0)$ 也是核函数；
- 核函数 $\kappa_1,\kappa_2$ 的直积 $\kappa_1 \bigotimes \kappa_2$ 也是核函数；
- 对于任意函数 $g(\bm{x})$，$g(\bm{x})\kappa(\bm{x},\bm{z})g(\bm{z})$ 也是核函数。

