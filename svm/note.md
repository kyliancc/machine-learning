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

### SMO 算法

