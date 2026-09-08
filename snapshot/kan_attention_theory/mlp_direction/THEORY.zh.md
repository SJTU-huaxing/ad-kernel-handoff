本轮把原有谱框架与非 MSE 的原始 kernel 拟合连接起来。所有恒等式、信息论工具与本轮实验发现分别标明；不把经典工具重新命名成新定理。

**目标与不同的测度。** 记 κ(q,k)=exp(qᵀk/√d)，κ̂=φ(q)ᵀψ(k)，φ,ψ≥0。两层 MLP 是函数实现，推理只存 Σψ(k)vᵀ 和 Σψ(k)。固定 head 常数缩放只用于数值计算，评价 raw kernel 时恢复。

原来的 E_m*=Σ_{r>m}σ_r² 对应 L²(P_Q×P_K) 下自由有符号秩 m 的最优值。实际一个文档中的合法 Q/K 矩形对应同文档条件分布；全因果位置还存在依赖 query 位置的 mask。原始谱尾不直接等于这些不同测度下的最优值。正特征和固定 MLP 预算的最优误差也可能更高。

**质量与方向的精确分解。** 对一个固定查询及其合法 key 集合，设 a_j=κ(q,k_j)>0，b_j=κ̂(q,k_j)>0，Z=Σa_j，Ẑ=Σb_j，p=a/Z，p̂=b/Ẑ。令 d_I(a,b)=a log(a/b)−a+b，则

$$\frac{\sum_jd_I(a_j,b_j)}{Z}
=D_{KL}(p\|\hat p)+\left[\frac{\hat Z}{Z}-1-\log\frac{\hat Z}{Z}\right].$$

右侧第一项衡量注意力权重方向，第二项校准原始 kernel 的行质量。两项同时为零才有 b=a；仅拟合归一化权重无法识别 query 标度。该关系是标准 KL 质量分解，不主张新颖性。

本轮 balanced 直接使用左侧作为损失。raw 相当于再乘 Z/J；half 再乘 sqrt(Z/J)，J 为 key 数。三者保留同一个无约束原始目标，有限容量下的最优解一般不同。无需假装它们与无权 L² 谱尾相同。

令 y=Σp_jv_j，ŷ=Σp̂_jv_j，R=max_j||v_j−y||。由 Pinsker 不等式：

$$\|\hat y-y\|^2\le2R^2D_{KL}(p\|\hat p)\le2R^2L_{balanced}.$$

这是逐查询条件界；若 R 随文档变化，取期望时不能擅自把 E[R²L] 写成 E[R²]E[L]。因子 R 大时界可能很松。

**保留原始目标的 value 加权构造。** 对 w_j≥||v_j−y|| 且 w_j>0，定义

$$A_w=\sum_jw_ja_j,\quad B_w=\sum_jw_jb_j,\quad
D_w=\sum_jw_jd_I(a_j,b_j),\quad L_w=D_w/A_w.$$

因 d_I(w a,w b)=w d_I(a,b)，D_w 仍逐点以 b_j=a_j 为唯一最优目标。主方法没有将训练标签替换成归一化注意力分数。

由 Σa_j(v_j−y)=0，有精确恒等式

$$\hat y-y=\hat Z^{-1}\sum_j(b_j-a_j)(v_j-y).$$

于是 ||ŷ−y||≤Ẑ⁻¹Σw_j|b_j−a_j|。对任意非负数组 u,v，令 U=Σu、V=Σv，利用 d_I(u,v)≥(√u−√v)² 以及 Cauchy–Schwarz：

$$\left(\sum_j|u_j-v_j|\right)^2
\le2(U+V)\sum_jd_I(u_j,v_j).$$

代入 u=w a、v=w b 得到

$$\boxed{\|\hat y-y\|^2\le
\frac{2(A_w+B_w)}{\hat Z^2}D_w
=\frac{2A_w(A_w+B_w)}{\hat Z^2}L_w.}$$

这是真实原始 kernel 误差到 value 输出误差的逐查询控制，不要求 Gaussian 或 L² Hilbert–Schmidt 条件。它也不是关于 LLM 最终困惑度的定理。

实装取 w_j=||v_j−y||+.05Σp_l||v_l−y||+10⁻⁶。V 与 y 只用于离线权重；φ、ψ 的推理结构和参数数目不变。上述界前的系数依赖预测，最小化 L_w 本身并不等价于精确最小化整个上界。是否改善泛化和 PPL 必须由独立实验决定。

该构造是在初始四损失验证后添加的探索性候选，使用固定 .05 权重底限，没有在测试集上调系数。另加入相同 MLP、数据、更新次数的标准 attention-weight KL 蒸馏作为对照。后者属于已发表路线的目标对照，不声称复现了 Hedgehog 的全部结构、初始化或 LoLCATs 的 LoRA 流程。

**进一步区分 Q 的幅度与方向。** 标准 KL 对照在 validation 上显著优于同时校准质量的 softplus MLP，这促使我们追加同参数的输出坐标分解。Q 网络仍输出 m 个标量，其中前 m−1 个作为方向 logits z，最后一个作为 log-amplitude s；K 网络保持原来的 softplus 映射：

$$\phi_Q(q)=\sqrt m\,e^{s(q)}\operatorname{softmax}([z(q),0]),\qquad
\phi_K(k)=\frac{\operatorname{softplus}(g(k))}{\log2\sqrt m}.$$

两侧网络仍为两层，隐藏宽度 192，m=64 时每 head 73856 个活跃参数。固定一个方向 logit 为零去掉 softmax 冗余自由度，并把这个输出坐标用于幅度；没有额外增加网络参数。训练仍优化 balanced 原始 I-divergence，raw 评价完整保留 e^s，在线 attention 的分子分母可以同时约去这个 query-only 幅度，不改变输出或线性状态。

同时预先固定两项第二轮消融：factorized_both 在 K 侧也用同样的 m−1 方向坐标加一个幅度坐标，φ_K=√m e^{t(k)}softmax([z_K(k),0])；其 K 幅度必须进入线性状态，不能像 Q 幅度一样约去。exp_control 则使用两侧普通 exp(MLP)/√m 特征和标准 attention-weight KL 训练，作为激活函数与蒸馏目标的匹配对照。三者均为同参数、同数据、单遍、三个种子，在第二轮确认之前固定。

对固定方向与 K 特征，方向 KL 对 s 的导数严格为零，质量项的导数是 Ẑ/Z−1。这是输出坐标的分离；共享的第一层及 K 参数仍然耦合，不能宣称两个训练子问题完全独立。普通 softmax 的尺度不变性和幅度/方向参数化不是全新的数学原理。

一个可严格证明的表达能力例子是：K 集中在固定 k₀，目标为 exp(qᵀk₀/√d)。标准有限权重的两层 SiLU MLP 加 softplus 输出沿任意输入射线最多线性增长；固定 k₀ 后，其有限维正特征内积仍最多线性增长。因此沿 q=tk₀,t→∞，对目标的相对误差趋于 1。相反，上面的幅度分支可精确表示线性的 s(q)=qᵀk₀/√d 加常数，因为 SiLU(x)−SiLU(−x)=x；方向与 K 特征取常数即可精确拟合这个 rank-1 原始 kernel。

然而，在该例中所有 key 相同，注意力权重本来就是均匀的。也就是说，原始 kernel 的幅度表达困难可以很大，实际注意力方向却完全没有困难。该例刻画了当前具体激活函数族的区别，不是所有 MLP 的不可能性，也不是有界真实 Q/K 分布上的误差率定理。

追加候选发生于首批确认的自动计算之后；这些旧测试对它只算探索性评估。另固定第二轮全新 128 篇 Wikipedia 和 96 篇 SWDE 文档，先完成训练与配置，再提取和评价，作为它的独立确认。不得把第二轮的成绩回填为首轮预先假设。

**一个与正特征直接相关、无需 L² 的下界。** 在乘积参考测度 μ_Q⊗μ_K 下，设 M=∫κ，P 的联合密度为 κ/M。有限 m 的非负可分离 κ̂ 经全局归一化后是至多 m 个乘积分布的混合。引入隐变量 Z∈{1,…,m}，满足 Q⊥K|Z。

对目标 P 和任意这种混合 R，按 R(Z|Q,K) 提升 P 为 P′，KL 链式法则给出

$$D_{KL}(P\|R)=D_{KL}(P'\|R_{QKZ})
\ge I_{P'}(Q;K\mid Z)
\ge I_P(Q;K)-H_{P'}(Z)
\ge I_P(Q;K)-\log m.$$

再使用前面的质量分解：

$$\boxed{\frac{\int d_I(\kappa,\hat\kappa)\,d\mu_Qd\mu_K}{M}
\ge [I_P(Q;K)-\log m]_+.}$$

类似地，对于固定 key 参考测度下的 query 平均 balanced 损失，以 P(dq,dk)=μ_Q(dq)κ(q,k)μ_K(dk)/Z(q) 定义互信息，也得到同样下界。它约束非负特征维度，而非自由有符号 Schmidt 秩；这是经典混合模型信息论的应用，不能单独声称新的顶会定理。

可验证的紧性例子：n 个离散正交 q_i、k_i，使原始指数 kernel 对角为 e^t、非对角为 1。均分为 m 个组，在每组内用条件均值近似，得到秩 m 非负原始 kernel。t→∞ 且 m 整除 n 时，其质量精确匹配、损失趋于 log(n/m)，与下界一致。本轮数值核验 n=128,m=16,t=30。

适用边界：真实数据上的经验互信息不能冒充总体互信息的置信下界；若 I≤log m，结果就是零。不能把这条 rank-m 混合界直接套到整张因果 mask 后的 attention 矩阵，因为三角 mask 可以提高矩阵秩。它适用于未归一化 kernel 的乘积测度，或固定 key 集合的矩形诊断。

独立 Gaussian 参考分布下，令 A=Σ_Q^(1/2)Σ_K^(1/2)/√d，其奇异值为 a_i。当 max a_i<1 时，kernel 倾斜后的联合 Gaussian 可归一化，互信息为 −½Σ_i log(1−a_i²)。该条件比 L² 谱要求的 max a_i<½ 宽，但不意味着近似的 Gaussian 互信息等于真实 LLM 互信息。在此前 16 个 head 的 covariance 上，13 个满足此条件，且 m64 信息下界仍全为零；本轮真实 4096×4096 训练经验乘积也全部为零。它目前没有提供实用的 rank64 近最优证书。

**与已有工作的关系。** [Hedgehog](https://arxiv.org/abs/2402.04347) 已采用学习型映射和 attention-weight 蒸馏；[LoLCATs](https://arxiv.org/abs/2410.10254) 使用输出匹配与 LoRA；[Degrees of Freedom for Linear Attention，NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c98ef086dc70d528e1c1aa1e66893365-Abstract-Conference.html) 已分析分布相关复杂度和特征维度分配；[Value-aware Approximate Attention，EMNLP 2021](https://aclanthology.org/2021.emnlp-main.753/) 已指出忽略 V 的局限。

因此，本轮若有研究价值，应来自“保持原始指数 kernel 校准的非 MSE 目标，如何在相同可部署正特征预算下改变跨文档和跨域表现”的可复现证据与进一步理论，而非 MLP、容量分配或 value-aware 这几个概念本身。当前尚无近总体最优、全 head 推理加速或完整 LLM 质量保证。

**最终构造：在整个参考分布上校准原始 kernel，同时保持 attention 输出。**

令已有正特征 kernel 为 g(q,k)=f_Q(q)ᵀf_K(k)。固定参考 P_K，假定以下正数有限：μ_r=E_K f_{K,r}(K)，G(q)=f_Q(q)ᵀμ，Z(q)=E_K κ(q,K)。此处 Z 是原始指数 kernel 的分布积分；不是用当前测试上下文拟合出来的一行标签。

定义逐元素除法和乘法下的特征：

$$\psi_K(k)=f_K(k)\oslash\mu,\qquad
\pi_Q(q)=\frac{f_Q(q)\odot\mu}{G(q)},\qquad
\phi_Q(q)=e^{\ell_\theta(q)}\pi_Q(q).$$

由此得到本轮最终实际构造：

$$\boxed{\hat\kappa_\theta(q,k)=
e^{\ell_\theta(q)}\pi_Q(q)^\top\psi_K(k)
=\frac{e^{\ell_\theta(q)}}{G(q)}g(q,k).}$$

它是连续、正、至多 rank-m 的可分离 kernel，没有固定分区。π_Q 落在 simplex 上表示连续混合系数，不是把 q 离散成 m 类；ψ_K 连续取值。μ 来自固定训练 key bank，ℓ 由训练查询学习；测试时不存在 q 与训练 keys 的逐项比较。E_K ψ_K=1，因而 E_K κ̂=e^ℓ。原始目标仍是 κ；若只丢掉 e^ℓ 就不再是当前原始 kernel，因此 raw 指标必须完整保留它。

对任意实际 key 集合、任意 causal prefix 及任意 V，系数 e^ℓ/G 只依赖当前 q，故

$$\frac{\sum_{j\le t}\hat\kappa(q,k_j)v_j}{\sum_{j\le t}\hat\kappa(q,k_j)}
=\frac{\sum_{j\le t}g(q,k_j)v_j}{\sum_{j\le t}g(q,k_j)}.$$

这是代数恒等式，不依赖实际上下文的 key 分布恰好等于参考 P_K。特别地，单独改善该标度没有能力改善 PPL：在实数计算下，整个网络输出也保持原样。实际 FP32/BF16 舍入可能带来极小变化。

分布风险 R(h)=E_{P_Q×P_K}d_I(κ,h)。在积分有限的条件下，对任何正查询标度 a(q)，逐 q 展开对数并使用 ∫κ=Z、∫g=G，可得

$$\boxed{R(a g)=R((Z/G)g)+E_Q d_I(Z,aG).}$$

因此在保持 g 的注意力方向不变的整类查询标度中，a*=Z/G 是原始 I-divergence 的最优校准；对 balanced 风险，只需把右侧逐查询各项除以 Z。对当前构造特别有

$$R(\hat\kappa_\theta)-\inf_{a(q)>0}R(a g)=E_Qd_I(Z,e^{\ell_\theta}).$$

这是固定方向等价类内的精确投影与误差分解，**不是**全部正 rank-m kernel 的总体最优性，也不等于原始 L² Schmidt 下界。在改变 P_K 后，原来学习的 ℓ 一般不再最优，需通过跨域测试检验。

本轮实际 ℓ 使用已训练 exp-MLP 的固定 SiLU 隐藏特征 h(q)，仅优化线性读出 ℓ=uᵀh+b。经验 balanced 标度损失为 exp(ℓ−log Z)−1−(ℓ−log Z)，加上正二次正则；它对 u,b 是凸函数。采用一遍 Adam 更新，未声称解到精确凸最优值。μ 的估计误差、训练 key bank 的 Z 近似误差、有限读出的表达误差与优化误差均未被消除，尚无总体有限样本保证。

推理参数保持不变：把原来 m 个 Q 输出改写为 m−1 个相对方向 logits 与 1 个 log-amplitude 坐标。相对 logits 和 μ 的平移折叠入原有最后一层；K 偏置减 log μ。m64 时每 head 仍 73856 个参数，在线 attention 可直接省略幅度，不增加线性状态。校准额外优化每 head 193 个已存在的读出参数，未增加一个新的网络。

校准使用训练 4096 文档中每篇新取的 64 个 Q，和固定的 16384 个跨文档 K bank，合计每 head 4,294,967,296 次原始 kernel 计算。新 Q 位置与方向预训练的 64 个 Q 不重合；两阶段共用文档但不重复选定 Q/K 配对。方向预训练本身使用标准 KL 控制的检查点；因此此方案是“方向 KL 预训练 + 原始 kernel 校准”的两阶段方法，不能描述成从始至终只训练原始 kernel 的单阶段方法。factorized_both 才是本轮全程使用原始 balanced 损失的主候选。推理预算相同不代表训练计算相同。

公平标度对照使用相同校准样本：全局常数 c_bal=1/E_Q[G/Z] 与 c_raw=E_QZ/E_QG，分别为 balanced/raw I 损失的最优 head 常数。不得仅用没有确定 raw 标度的 KL 检查点作 raw 指标对照来夸大收益。

行缩放不变性本身是基本归一化性质，也在 [Geometric Attention (2026), §2.5](https://arxiv.org/html/2601.11618v1) 中明确讨论。本轮推导是经典 I-projection 思路在当前预算与目标下的具体应用，没有确立其数学新颖性。[Taylor-Calibrate (2026)](https://arxiv.org/abs/2606.16429) 也已研究基于教师统计的线性模型校准，但其目标是 Gated DeltaNet 初始化与输出匹配，与这里保持 attention 输出的原始 kernel 校准不相同。“校准”这个概念本身同样不能作为新颖性主张。

部署补充：第一次直接执行上述μ折叠特征时，FP64不变性成立，但不同的FP32乘加/缩放顺序在BF16模型中产生舍入传播。最终部署利用κ̂=a(q)g，在归一化attention中直接执行父kernel g 的原特征；只保留一套同预算参数与同尺寸状态。raw诊断仍执行完整校准网络，二者在实数运算上是同一个attention。第三轮六个模型/数据组合复测PPL与对应父kernel完全一致；初次折叠实现的结果另存，不用于声称质量改善。
