需要修正此前的结论范围：**测试矩阵NMF给出的是该矩阵上非负因子最优误差的一个可行上界，既不是总体正特征误差下界，也不是可泛化feature map。此前的指数节点构造属于FAVOR/DERF方法族；它没有完成从Schmidt最优截断到近最优正特征的推导。**

下面从真实分布的积分算子出发，给出可验证的构造与误差分解。相关数值验证见REPORT.zh.md。

**总体对象与适用条件**

固定模型、attention head、输入文档分布及token位置采样规则，它们诱导边缘分布P_Q和P_K。目标是

$$\kappa(q,k)=e^{q^\top k/\sqrt d},\qquad
(Tf)(q)=\int\kappa(q,k)f(k)\,dP_K(k).$$

以下分析使用乘积分布P_Q×P_K。实际同文档、因果attention配对的联合分布一般不是这个乘积，不能直接套用同一个L2谱尾结论。

如果H=Eκ²<∞，T是Hilbert–Schmidt算子，存在Schmidt展开

$$\kappa(q,k)=\sum_{j\ge1}\sigma_j u_j(q)v_j(k),\qquad
E_m^\star=\inf_{\mathrm{rank}(\hat\kappa)\le m}\|\kappa-\hat\kappa\|_2^2
=\sum_{j>m}\sigma_j^2.$$

这里允许两侧独立、带符号的平方可积特征。非负特征的最优误差E_m^+满足E_m^+≥E_m^⋆，但不保证相等。

严格地说，总体非负问题是

$$E_m^+(P_Q,P_K)=\inf_{f_r,g_r\ge0}\mathbb E_{P_Q\times P_K}
\left(\kappa(Q,K)-\sum_{r=1}^m f_r(Q)g_r(K)\right)^2.$$

它需要一组函数同时作用于整个分布；对不同测试子矩阵分别NMF，可以为每个子矩阵重新选择因子，因而不是同一个问题。本轮没有求出这个非负总体最优值。任一合法总体谱下界L与任一固定非负函数的真实风险R，只能给出L≤E_m^⋆≤E_m^+≤R；若L和R用经验测度算出，这个关系也只针对该经验测度。

对固定权重的Qwen类pre-RMSNorm模型，归一化后向量的范数有界，有限线性投影仍有界，RoPE保持范数，因此相应Q/K有界，H有限。由此一般Schmidt框架适用；不需要Q/K服从Gaussian。Gaussian替代模型的H发散，不等于真实模型的H发散。

总体积分算子谱与经验谱之间的统计近似是额外问题；标准谱学习理论会显式处理采样和集中条件。[Rosasco等，On Learning with Integral Operators](https://jmlr.org/papers/v11/rosasco10a.html)

**为什么covariance通常不足以决定谱**

定义真实K分布的矩母函数M_K(t)=E exp(tᵀK)。由定义直接得到

$$
(TT^*)(q,q')=\mathbb E_K[\kappa(q,K)\kappa(q',K)]
=M_K\!\left(\frac{q+q'}{\sqrt d}\right).
$$

同理，T* T的kernel为M_Q((k+k')/√d)。这两个算子的非零特征值是σ_j²。

因此，真正决定谱的是完整矩母函数及另一侧的分布；covariance只给出了log M在原点的二阶导数。只有在Gaussian等额外模型成立时，才能把完整矩母函数压缩成均值和covariance。

一个有界的一维反例：取d=1，a=0.2，令Q和K独立同分布，且

$$P_\varepsilon(X=0)=1-\varepsilon,\qquad
P_\varepsilon\left(X=\pm\sqrt{a/\varepsilon}\right)=\varepsilon/2.$$

所有ε都有均值0、方差a，但

$$\mathbb E e^{2QK}=1-\varepsilon^2+\varepsilon^2\cosh(2a/\varepsilon).$$

| ε | Var(Q)=Var(K) | Eκ² | 最优rank-1相对误差 |
|---|---:|---:|---:|
| 1 | 0.2 | 1.0811 | 0.03750 |
| 0.1 | 0.2 | 1.2631 | 0.14970 |
| 0.01 | 0.2 | 1.1769×10¹³ | 约0.5 |

同covariance的Gaussian surrogate满足a<1/2，且预测Eκ²约1.0911，却无法区分上述分布。这一反例已用有限支持分布的精确加权算子SVD核验。它否定的是不加分布假设的covariance-only预测，不是否定Gaussian谱公式在其假设下的正确性。

**由Schmidt框架构造非负kernel：单侧条件期望**

把Q空间分为m个可测集合C₁,…,C_m，设p_r=P_Q(C_r)>0，定义

$$g_r(k)=\mathbb E[\kappa(Q,k)\mid Q\in C_r]
=M_{Q\mid C_r}(k/\sqrt d).$$

构造

$$\boxed{\kappa_{\mathcal C}^{+}(q,k)=
\sum_{r=1}^{m}\mathbf1_{C_r}(q)g_r(k).}$$

它是rank≤m的非负特征kernel：φ_Q(q)_r=1_{C_r}(q)，φ_K(k)_r=g_r(k)>0。节点积分使用的是实际条件分布Q|C_r，而不是FAVOR的高斯随机特征表示。

令Π_C表示对Q分区的条件期望，则κ_C^+=(Π_C⊗I)κ。条件期望是L2正交投影，所以它在“对q分区内常数、对k任意”的函数类中是唯一的L2最优解。[条件期望的正交投影性质](https://adembo.su.domains/stat-310b/lnotes.pdf)

代入Schmidt展开，并利用v_j在L2(P_K)上的正交性：

$$R_{\mathcal C}:=\|\kappa-\kappa_{\mathcal C}^{+}\|_2^2
=\sum_j\sigma_j^2\|(I-\Pi_{\mathcal C})u_j\|_2^2.$$

因此

$$\boxed{E_m^\star\le R_{\mathcal C}
\le E_m^\star+D_Q^{(m)}(\mathcal C),}$$

$$D_Q^{(m)}(\mathcal C)=\sum_{j=1}^{m}\sigma_j^2
\mathbb E\operatorname{Var}(u_j(Q)\mid\mathcal C).$$

D_Q正是加权谱坐标ζ_Q(q)=(σ₁u₁(q),…,σ_m u_m(q))的分区量化误差。**只有当D_Q相对于谱尾小，才能推出该正kernel接近最优rank-m误差。** 低rank本身并不自动保证存在这样的m个分区。

这给出了明确构造原则：从训练算子估计谱坐标，寻找低量化误差的分区，再估计条件矩母函数。谱嵌入上的聚类和Nyström近似已有研究基础，不能把这个构造本身直接宣称为新的顶会贡献。[Nyström近似与kernel k-means](https://jmlr.org/beta/papers/v20/17-517.html)

**估计误差能单独分离**

固定分区，以独立训练数据估计g_r为ĝ_r。由于估计误差仍在同一个分区子空间内，正交投影给出精确分解：

$$\boxed{\|\kappa-\hat\kappa_{\mathcal C}\|_2^2
=R_{\mathcal C}+\sum_r p_r\|\hat g_r-g_r\|_{L^2(P_K)}^2.}$$

如果每个cell有n_r个独立条件样本，并直接平均κ(Q_i,k)，对训练采样再取期望：

$$\mathbb E_{\rm train}\|\kappa-\hat\kappa_{\mathcal C}\|_2^2
=R_{\mathcal C}+\sum_r\frac{p_r}{n_r}
\mathbb E_{Q\mid C_r}\|\kappa(Q,\cdot)-g_r\|_{L^2(P_K)}^2.$$

独立样本条件很重要：同一文档中的token不能不加分析地当作独立条件样本。上述采样公式没有被直接套到本轮相关token数据上。

**本轮实际实现：两侧谱分区的条件均值kernel**

为避免推理时对大量Q参考点求和，再把K空间分为D₁,…,D_m。定义

$$b_{rs}=\mathbb E[\kappa(Q,K)\mid Q\in C_r,K\in D_s],$$

$$\boxed{\kappa_{\mathcal C,\mathcal D}^{+}(q,k)
=\sum_{r,s=1}^{m}b_{rs}\mathbf1_{C_r}(q)\mathbf1_{D_s}(k).}$$

它仍只需要m维特征，而非m²维：

$$\phi_Q(q)=e_{c(q)},\qquad
\phi_K(k)=B e_{d(k)},\qquad
\kappa_{\mathcal C,\mathcal D}^{+}=\phi_Q^\top\phi_K.$$

B有m²个标量参数，特征维度为m。Q侧特征非负且稀疏，K侧为正。若要求两侧逐分量严格为正，可把e_c换为(1−ε)e_c+ε1/m；当ε→0时恢复原构造。本轮报告默认ε=0。

记Π_Q、Π_K为两个分区投影。两个误差项正交：

$$R_{CD}=\|(I-\Pi_Q)T\|_{HS}^2+
\|\Pi_Q T(I-\Pi_K)\|_{HS}^2.$$

利用投影收缩性，得到

$$E_m^\star\le R_{CD}
\le 2E_m^\star+D_Q^{(m)}(\mathcal C)+D_K^{(m)}(\mathcal D).$$

设q_s=P_K(D_s)，训练估计矩阵为B̂，则

$$\boxed{R(\hat\kappa_{CD})
=R_{CD}+\underbrace{\sum_{r,s}p_rq_s(\hat b_{rs}-b_{rs})^2}_{\Delta_{\rm estimation}}.}$$

因此，在谱量化误差和条件均值估计误差小于εE_m^⋆的附加条件下，可以获得(2+ε)倍的误差保证；当前实验没有验证这些“小误差”条件。

这个构造还能处理此前的损失不一致问题。在固定cell中，广义KL的最优常数满足

$$\frac{\partial}{\partial b}\mathbb E[D_I(\kappa\|b)\mid C_r,D_s]
=1-\frac{\mathbb E[\kappa\mid C_r,D_s]}{b}=0.$$

所以最优值也是条件均值b_rs。**在这个受限函数类内，非MSE的广义KL与L2投影具有相同最优解。** 本轮直接计算训练均值，没有MSE神经训练或多epoch梯度拟合。

实际谱坐标来自训练锚点的cross-Nyström延拓，采用前64个近似加权谱坐标。分区使用贪心轴向切分，m=16/32/64/128，系数在与分区校准文档不重叠的训练文档上估计。近似谱坐标和贪心划分都不是已知总体最优解，不能把理想保证中的小D_Q、D_K当作已满足。

**用算子压缩给出上下界，而不是把NMF误差当下界**

取训练确定的有限维函数空间，令A(q)、B(k)分别是它们在真实P_Q、P_K下的正交基。压缩矩阵为

$$C=\mathbb E[\kappa(Q,K)A(Q)B(K)^\top].$$

记C的奇异值为s₁≥…≥s_r，则压缩算子的奇异值满足s_j≤σ_j。另一方面，C的rank-m截断对应一个实际rank-m近似，其误差为H−Σ_{j≤m}s_j²。因此

$$\boxed{\sum_{j>m}^{r}s_j^2\le E_m^\star
\le H-\sum_{j\le m}s_j^2.}$$

上下界宽度正是压缩未捕获的能量H−||C||_F²。宽度大时，不能用区间端点或压缩谱尾冒充精确最优误差。

本轮在完整留出经验边缘分布下精确累积H、Gram矩阵和C，并按经验Gram正交化。因此数值区间适用于对应的完整**经验乘积分布**。这些held-out压缩系数只用于诊断；实际可泛化kernel的参数来自训练数据。

若要获得真正总体的置信区间，需要额外证明|H−Ĥ|≤ε_H以及||C−Ĉ||op≤ε_C（包括Gram估计和求逆误差）。此时才有

$$\sum_{j>m}(\hat s_j-\epsilon_C)_+^2\le E_m^\star
\le\hat H+\epsilon_H-\sum_{j\le m}(\hat s_j-\epsilon_C)_+^2.$$

本轮没有建立足够紧的ε_H、ε_C，文档bootstrap也不能替代这个证明。经验积分检查通过，不能据此宣称总体谱估计已被确认。

**如何接回KAN**

KAN可以用来近似非负分区权重或条件均值函数。例如用两侧非负、和为1的权重a(q)、b(k)构造a(q)ᵀB b(k)，特征可取a(q)与B b(k)，目标仍是未归一化κ。这时需要增加谱坐标估计、分区软化和KAN逼近的误差项。当前理论没有给出KAN优于等参数MLP的结论，两层mulKAN是否更有效仍需独立证明和公平实验。

当前成立的是一般算子分解、条件期望投影、误差分离和压缩谱区间的数学关系。尚未成立的实证结论是：Gaussian covariance公式能预测真实LLM总体谱、当前正kernel接近E_m^⋆、以及KAN具有独占优势。

**用于实验的有符号训练算子对照**

给定仅由校准数据确定的函数字典F(q)、G(k)，在独立的moment-fit训练边缘分布下正交化，得到A(q)=F(q)W_Q、B(k)=G(k)W_K。然后只在该训练乘积分布下计算C_train=E[κ A Bᵀ]，并做C_train=U S Vᵀ。冻结所有参数后构造

$$\hat\kappa_m^{G}(q,k)=
\left[A(q)U_mS_m^{1/2}\right]
\left[B(k)V_mS_m^{1/2}\right]^\top.$$

这是Schmidt截断在训练函数空间中的直接实例化，保证的是该训练压缩空间内的最优性，不是未知总体或所有rank-m函数中的最优性。它与测试时重新正交化、重新计算C的谱诊断分开。代码中所有κ先除以一个固定head常数e^{c_h}计算，预测时恢复e^{c_h}，相对误差不变。

**与FAVOR的关系和创新性范围**

FAVOR的基础恒等式是，令x=q/d^{1/4}、y=k/d^{1/4}，则

$$e^{x^\top y}=e^{-(\|x\|^2+\|y\|^2)/2}
\mathbb E_{\omega\sim N(0,I)}e^{\omega^\top x}e^{\omega^\top y}.$$

改变该辅助高斯积分的节点、权重或指数特征参数，仍属于相关随机特征/求积方法族；此前的构造正是这种路线。[FAVOR# / DERF原始论文](https://proceedings.neurips.cc/paper_files/paper/2023/file/02dec8877fb7c6aa9a79f81661baca7c-Paper-Conference.pdf)

这里的条件期望则对真实Q|C_r分布积分，Q侧为分区指示函数；实际两侧版本是分区查表，既没有辅助高斯随机变量，也没有将上述恒等式离散化。它当然仍是有限维内积kernel，谱坐标估计也使用kernel锚点，但不能因此把它等同于FAVOR求积。另一方面，条件期望、分区投影与Nyström都是经典工具：这一差异本身不构成方法新颖性证明，也不足以支持顶会贡献主张。
