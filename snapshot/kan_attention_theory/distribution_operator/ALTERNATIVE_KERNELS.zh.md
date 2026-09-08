本文给出当前总体算子框架下的其他 kernel 构造及证明，不包含新增 LLM 实验，也不把这些构造称为已经验证的新方法。此前的双侧硬分区是受限实现选择；Schmidt 理论不要求分区。

**共同对象与边界。** 固定真实边缘分布 P_Q、P_K，目标始终为

\[
\kappa(q,k)=e^{q^\top k/\sqrt d},\quad
(Tf)(q)=\mathbb E_K[\kappa(q,K)f(K)],\quad
R(\hat\kappa)=\|\kappa-\hat\kappa\|_{L^2(P_Q\times P_K)}^2.
\]

假设 κ∈L²，写 κ=Σ_j σ_j u_j v_j，u_j、v_j 分别在两侧总体 L² 空间正交归一。允许独立两侧特征，不额外要求同一 feature map 或对称 PSD。以下“非负特征”指每个特征分量非负，比“最终 kernel 数值非负”更强。

所有 m 维可分离构造的无约束下界为 E_m*=Σ_{j>m}σ_j²。该谱尾不保证存在达到它的 m 维非负因子。仅知道 σ_j 也不足以构造 feature map；还需要奇异函数及其正性几何。未知总体的函数和期望不可直接从有限样本当成已知。

**构造一：直接谱截断与连续 Galerkin。**

\[
\hat\kappa_m^S(q,k)=\sum_{j=1}^m\sigma_j u_j(q)v_j(k),\quad
\phi_Q(q)_j=\sqrt{\sigma_j}u_j(q),\quad
\phi_K(k)_j=\sqrt{\sigma_j}v_j(k).
\]

理想总体误差恰为 E_m*，没有硬类别限制；在有界 Q/K 支持、指数 kernel 的情形可以选取连续的非零奇异函数。截断后可能出现负 kernel 值，非负奇异值不意味着非负函数或逐点非负 kernel。

一般 Galerkin 用任意有限函数空间近似上述构造。基函数可以连续，不必是分区指示函数。现有实现的训练算子 Galerkin 属于这一类，保证限于其训练压缩空间。

神经网络直接参数化算子的奇异函数已有先例，不能作为这里的新颖性主张。[Operator SVD with Neural Networks via Nested Low-Rank Approximation，ICML 2024](https://arxiv.org/abs/2402.03655)

**构造二：连续非负基函数与非负锥投影。**

选取 m 个非负函数 b_s∈L²(P_K)，例如连续 spline 或非负神经特征，不要求和为 1，也不要求互斥。记 b(k)=(b_1(k),…,b_m(k))ᵀ。假设

\[
G=\mathbb E_K[b(K)b(K)^\top]\succ0,
\qquad h(q)=\mathbb E_K[\kappa(q,K)b(K)].
\]

固定 q，使用 c(q)ᵀb(k) 近似 κ(q,k)。展开总体风险：

\[
\mathbb E_K(\kappa(q,K)-c^\top b(K))^2
=\mathbb E_K\kappa(q,K)^2-2h(q)^\top c+c^\top Gc.
\]

无约束系数为 c_0(q)=G^{-1}h(q)，一般带符号。约束系数非负时，定义总体理论解

\[
c_+(q)=\arg\min_{c\ge0}
\|c-G^{-1}h(q)\|_G^2,\qquad \|x\|_G^2=x^\top Gx.
\]

得到

\[
\boxed{\hat\kappa_b^+(q,k)=c_+(q)^\top b(k).}
\]

它是 m 维非负特征 kernel。每个 query 可以使用任意连续的非负组合，不局限于 m 种固定模板。若 h 连续，则正定 G 下的闭凸锥投影连续，c_+ 也连续；有界 Q/K 支持和连续指数 kernel 足以保证 h 连续。

令 V=span{b_s}，Π_V 是 L²(P_K) 正交投影。无约束的 c_0ᵀb 正是 κ 对 key 函数空间 V 的正交投影。投影残差和该空间内的系数修正正交，因此

\[
\boxed{
R(\hat\kappa_b^+)
=\underbrace{\sum_j\sigma_j^2\|(I-\Pi_V)v_j\|_2^2}_{R_V}
+\underbrace{\mathbb E_Q\min_{c\ge0}
\|c-G^{-1}h(Q)\|_G^2}_{\Delta_+(b)}.
}
\]

相对于无约束 rank-m 理论下界，

\[
R(\hat\kappa_b^+)-E_m^*=(R_V-E_m^*)+\Delta_+(b),
\]

两项均非负。第一项测量基函数空间没有对齐有效谱子空间的代价；第二项测量在这组基函数中要求系数非负的代价。Δ_+(b) 依赖基函数选取，不是所有正特征的不可避免总体差距。

若部署非负系数网络 c_θ(q)，精确关系仍是

\[
R(c_\theta^\top b)=R_V+
\mathbb E_Q\|c_\theta(Q)-G^{-1}h(Q)\|_G^2.
\]

可进一步把最后一项写成 Δ_+(b) 加上非负的系数逼近/优化超额风险。不能直接把 ||c_θ−c_+||_G² 当作全部超额风险，因为凸锥投影的边界可能产生交叉项。

硬 key 分区 b_s=1_{D_s} 是这一构造的特殊情况：G=diag(P_K(D_s))，G^{-1}h 的每个分量就是条件均值，本来就非负，故 Δ_+=0。软基函数重叠时 G 通常非对角，G^{-1}h 不再自动非负。这解释了为什么“把 one-hot 改成 softmax”不能直接继承硬分区的最优性。

上述平方风险优化只是与 Schmidt 下界一致的理论 oracle，不是此次新增的 MSE 训练实验。实际两层 KAN/mulKAN 或 MLP 可取 b_η(k)=softplus(F_η(k))、c_θ(q)=softplus(G_θ(q))，使用用户要求的非 MSE 目标训练；这时需重新测量谱风险，不能宣称非 MSE 训练达到了这里的平方风险 oracle。函数化 NMF 也已有文献，连续非负函数因子本身不是全新概念。[Nonnegative Matrix Factorization over Continuous Signals using Parametrizable Functions](https://doi.org/10.1016/j.neucom.2019.11.109)

计算上，G 可以在训练阶段估计，理论 c_+ 需要函数积分和约束求解，不能直接假定廉价。神经参数化之后在线代价取决于 feature 网络；若继续使用大量锚点计算 h，仍会保留现有高特征开销。

**构造三：连续权重的正积分算子，给出显式特征。**

选取连续 b_s(k)≥0、Σ_s b_s(k)=1，p_s=E_K b_s(K)>0。定义

\[
g_s(q)=\frac{\mathbb E_{K'}[\kappa(q,K')b_s(K')]}{p_s},\qquad
\boxed{\hat\kappa_b^{\mathrm{avg}}(q,k)=\sum_{s=1}^m g_s(q)b_s(k).}
\]

因为 κ>0，每个 g_s>0；和为 1 的 b 保证 kernel 严格为正。两侧都可连续，q 和 k 均不必归入唯一类别。对 b 的内部归一化只是基函数约束，目标仍是 raw κ，attention 的归一化仍交给分母。

定义作用于 L²(P_K) 的有限秩算子

\[
(P_bf)(k)=\sum_s b_s(k)\frac{\langle b_s,f\rangle}{p_s}.
\]

P_b 自伴、保正且 P_b1=1。加权 Cauchy–Schwarz 给出

\[
0\le\langle f,P_bf\rangle
=\sum_s\frac{\langle b_s,f\rangle^2}{p_s}
\le\sum_s\mathbb E[b_s(K)f(K)^2]=\|f\|_2^2.
\]

故 0≼P_b≼I，||I−P_b||op≤1。对应 kernel 的算子为 TP_b，于是

\[
\boxed{
R(\hat\kappa_b^{\mathrm{avg}})
=\sum_j\sigma_j^2\|(I-P_b)v_j\|_2^2
\le E_m^*+\sum_{j=1}^m\sigma_j^2\|(I-P_b)v_j\|_2^2.
}
\]

加上 rank-m 下界，得到 E_m*≤R≤E_m*+D_b。若 D_b≤εE_m*，才有 (1+ε) 倍保证；目前没有在真实 Q/K 上验证该条件。P_b 通常不是正交投影，不能因为 b 平滑就假设误差更小。

这是构造二固定基函数下的一个显式非负可行解，因此理论上 R(κ_b^+)≤R(κ_b^avg)。它可以免去 G^{-1} 和每个 q 的非负约束求解，却仍需廉价近似 g_s。这里积分的是实际 K 分布，不是 FAVOR 的辅助 Gaussian 随机特征积分；这一区别不构成新颖性证明。

**构造四：利用第一正谱模态，将高阶模态成对正化。**

这是从 Schmidt 展开直接作代数构造的候选，下面给出完整条件与风险，不声称文献首创或已经有实测优势。

保留 r 个模态，假设 σ_1>0，并选取 u_1>0、v_1>0。对严格正 kernel 可选择第一奇异函数对为正。进一步假设对 j=2,…,r 存在有限常数 a_j,b_j，使得在各自总体分布下几乎处处

\[
|u_j(q)|\le a_j u_1(q),\qquad
|v_j(k)|\le b_j v_1(k).
\]

这些是总体包络，不能用训练样本最大值冒充。在 κ 上下均有严格正常数界的有界 Q/K 支持上，可以保证有限包络存在，但数值可能极大；对一般无界分布不保证存在。

令

\[
C_r=\sum_{j=2}^r\sigma_j a_jb_j,\quad
\delta_r=(C_r-\sigma_1)_+,\quad
c_0=\sigma_1+\delta_r-C_r\ge0.
\]

定义两组非负特征

\[
f_0(q)=\sqrt{c_0}u_1(q),\quad
g_0(k)=\sqrt{c_0}v_1(k),
\]

\[
f_{j,\pm}(q)=\sqrt{\sigma_j/2}[a_j u_1(q)\pm u_j(q)],\quad
g_{j,\pm}(k)=\sqrt{\sigma_j/2}[b_j v_1(k)\pm v_j(k)].
\]

每个模态对满足

\[
f_{j,+}g_{j,+}+f_{j,-}g_{j,-}
=\sigma_j a_jb_j u_1v_1+\sigma_j u_jv_j.
\]

所以

\[
\boxed{
\hat\kappa_r^{\mathrm{pair}}(q,k)
=f_0(q)g_0(k)+\sum_{j=2}^r\sum_{\pm}f_{j,\pm}(q)g_{j,\pm}(k)
=\sum_{j=1}^r\sigma_j u_j(q)v_j(k)+\delta_r u_1(q)v_1(k).
}
\]

该 kernel 不分区、不采用辅助 Gaussian 求积，两侧特征非负，特征数至多 2r−1；若 c_0=0 可以省去该分量。虽然非负特征数可能接近 2r，普通算子秩仍至多 r。由奇异函数正交归一，谱尾与 u_1v_1 正交，因此

\[
\boxed{R(\hat\kappa_r^{\mathrm{pair}})=E_r^*+\delta_r^2.}
\]

若 C_r≤σ_1，就能用至多 2r−1 个非负特征精确实现 rank-r Schmidt 截断，误差 E_r*。若 C_r>σ_1，第一模态需要额外增加 δ_r，其平方正是额外误差。这个增量算进原始 kernel 风险，不被当作 attention 归一化可忽略的误差。

如果要求所有 q,k 的 kernel 严格正，可以给 c_0 再加 ε>0，即 kernel 再加 εu_1v_1；风险改为 E_r*+(δ_r+ε)²。

固定部署特征预算 m，保守选择 r≤floor((m+1)/2)。相对于 E_m*，精确差距是

\[
R(\hat\kappa_r^{\mathrm{pair}})-E_m^*
=\sum_{j=r+1}^{m}\sigma_j^2+\delta_r^2.
\]

因此不能拿约 2r 个特征的方法与 r 维 FAVOR 比较后宣称同预算优势。m=64 时保守只保留 r=32 个谱模态，使用至多 63 个分量；条件允许省略零分量时可单独核算预算。

主要风险是包络常数很大，从而 δ_r² 主导；这需要真实分布下的证据，不能从快谱衰减直接推断。高阶模态在尾部相对第一模态的峰值尤其重要。

当前只能估计谱函数。若最终实现 kernel 和理想构造的 L² 差为 ε_fun，三角不等式仅给出

\[
\sqrt{R(\hat\kappa_{\rm impl})}
\le\sqrt{E_r^*+\delta_r^2}+\epsilon_{\rm fun}.
\]

网络拟合还必须重新保证非负包络；真实谱函数上的包络不自动对拟合函数成立。SVD 构造非负因子的相关工作已有长期研究，如 NNSVD-LRC；上述代数推导不等于已经完成新颖性审查。[相关原论文](https://arxiv.org/abs/1807.04020)

**训练目标、线性状态与研究判断。**

上述平方风险用于用户设定的 Schmidt 理论指标。神经实现仍可对未归一化 κ 使用广义 KL：D_I(x||y)=x log(x/y)−x+y。它的最优解一般与 L² 最优解不同，不能把 E_m*直接称为 KL 理论下界。若另外有真实的 0<ℓ≤x,y≤M，则

\[
\frac{(x-y)^2}{2M}\le D_I(x\|y)\le\frac{(x-y)^2}{2\ell}.
\]

该条件下可以关联两个风险，但极大的 M/ℓ 会使保证无实际用途。这里没有通过截断或归一化目标去制造这个条件。

对任意 κ̂=φ_Qᵀφ_K，统一状态仍为 S_t=Σ_{j≤t}φ_K(k_j)v_jᵀ、z_t=Σ_{j≤t}φ_K(k_j)，输出 φ_Q(q_t)ᵀS_t/[φ_Q(q_t)ᵀz_t]。m 维状态的缓存 O(md_v)，一般稠密特征的更新与读取 O(md_v)，还要加上特征计算；硬 key 分区才具有单桶 O(d_v) 更新。去掉硬分区不会破坏对序列长度的线性复杂度，但可能失去稀疏更新便利。不得把“固定状态”当成已经获得端到端速度优势。

不能直接对带符号 kernel 总分作 softplus 或逐点截断，然后继续声称是同一个 m 维线性 kernel；这样的非线性通常破坏有限可分离秩。若对两侧特征分别 softplus，秩仍≤m，但 kernel 已改变，原 Schmidt 最优误差不再成立。

建议把构造二作为接回两层 KAN/mulKAN 的主要理论参数化：明确分离谱空间损失和非负约束损失，并与相同预算 MLP 比较。构造三提供显式正的积分参考；构造四提供更直接的谱正化参考及可证伪的包络条件。当前没有证据确认任何一种在真实 LLM 上优于已测方法，也没有证明 KAN 独有优势。
