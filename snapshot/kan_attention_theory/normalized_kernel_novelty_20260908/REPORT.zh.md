截至2026-09-08，当前纯删除AD的数学形式创新偏弱，有限网络参数化及固定状态预算下的效果仍有研究价值。仅凭“幅度×方向、指数压低key、KL蒸馏、查询尺度消去”不足以主张新的正kernel范式。检索没有确认某篇论文逐项采用完全相同的代码配方，但这不构成首创证明。

本次评议只针对当前方案：

\[
g(q,k)=e^{s_K(k)}\pi_Q(q)^\top\pi_K(k),\qquad
\pi_X(x)=\operatorname{softmax}([z_X(x),0]),
\]

以实际因果上下文的分母归一化，用教师attention KL拟合。Q网络128→192→63，K网络128→192→64（63方向+1幅度），隐藏SiLU，m64，73536参数/head。独立Q/K映射只保证非负可分离相似度，不自动保证对称Mercer正定核。此前原始kernel I风险及SVD理论不能直接作为本目标的逼近证书。

**为什么函数族层面的创新较弱**

对于任意严格正m维feature maps f_Q,f_K，定义a_X=Σ_r f_{X,r}，π_X=f_X/a_X，则f_Q(q)^T f_K(k)=a_Q(q)a_K(k)π_Q(q)^Tπ_K(k)。归一化attention中的a_Q精确消去，令s_K=log a_K，便得到当前AD形式。因此它是一般严格正可分离attention的一种规范化表示。有限两层网络的容量和优化仍可能不同；此恒等式不意味着相同参数预算下任意MLP均能精确实现另一种参数化。

对当前AD本身还可进一步严格改写。以下z_Q,z_K均表示补入末尾0后的m维向量。令

\[
\widetilde z_K(k)=z_K(k)+[s_K(k)-\operatorname{LSE}(z_K(k))]\mathbf 1.
\]

则

\[
g(q,k)=\frac{\exp(z_Q(q))^\top\exp(\widetilde z_K(k))}{\sum_r e^{z_{Q,r}(q)}}.
\]

其中指数按元素作用，分子为两个正向量内积。末尾查询分母在attention归一化中也消去。因此归一化AD等价于Q端指数特征、K端带共享log-sum-exp校正的指数特征。这不是说AD等于一层Hedgehog；差别在网络深度、参数约束及幅度校正方式。它也说明Q端softmax归一化属于可以约去的计算，不能将这类约消包装为独有推理理论。这里仅核对恒等式，未修改推理代码或新增速度主张。

**已有工作的重合与区别**

| 研究 | 已有内容 | 与当前AD的关系 |
| --- | --- | --- |
| [Hedgehog，ICLR2024](https://arxiv.org/html/2402.04347v1#S4.SS2) | 指数正特征、可学习投影、用attention交叉熵/KL蒸馏教师；关注尖锐性 | 当前最直接的训练/feature-map近邻。AD采用两层网络、单纯形方向与独立K幅度参数化，不是首次学习尖锐正特征 |
| [Norm×Direction / NaLaFormer，arXiv v3](https://arxiv.org/html/2506.21137v3#S3) | 显式分解范数与方向，将query范数用于幂映射，结合三角方向特征 | “幅度—方向”已被直接研究；其query范数进入方向映射内部，不是可约掉的外部标量。AD则学习K幅度，且不强制保留输入范数 |
| [STILL / NP-Map，2026](https://arxiv.org/html/2602.02180v1#S3.SS2) | u=f(x)/||f(x)||·||x||，再拼接softmax(u)、softmax(−u) | LLM线性化中已有方向/幅度分离。其幅度进入softmax内部，AD的K幅度乘在外部；STILL的整体系统还含稀疏softmax路由 |
| [MALA，ICCV2025](https://arxiv.org/html/2507.00698v3#S3.SS2) | 显式指出query特征范数被分母消去，并修改计算以引入其影响 | 查询幅度消去的观察已存在；MALA的上下文相关缩放/平移不等于当前固定正可分离kernel |
| [ReGLA，NAACL2025](https://aclanthology.org/anthology-files/anthology-files/pdf/naacl/2025.naacl-long.147.pdf) | 归一化指数特征、稳定性与门控设计 | 指数映射、非负性及尺度控制已有系统研究；其状态门控不等于AD的静态key权重 |
| [Degrees of Freedom for Linear Attention，NeurIPS2025](https://papers.nips.cc/paper_files/paper/2025/hash/c98ef086dc70d528e1c1aa1e66893365-Abstract-Conference.html) | 分布相关有效维度、逼近界、分层feature预算与非线性特征蒸馏 | 分布理论指导m和可学习feature不是空白；其原文§3.3也比较raw L2与softmax交叉熵 |
| [LoLCATs，ICLR2025](https://arxiv.org/html/2410.10254v1#S3) | 冻结模型的attention输出拟合，再以LoRA恢复质量 | “关注V加权后的输出而不仅是attention系数”已有直接先例；当前KL与PPL排序反转的具体量化仍可研究 |
| [The Key to Going Linear，2026-07预印本](https://arxiv.org/html/2607.07706v1#S3) | 严格冻结backbone，对比kernel、门控与delta更新，并以分析指导线性化 | 冻结真实LLM、研究机制/误差也已有近期直接工作。其主要机制是state update，与当前正feature参数化不同 |

HF的NaLaFormer markdown缓存的是旧标题/版本，本次以arXiv v3公式为准。上述表只表示相关性，不背书每篇论文所有证明或报告的性能；预印本不标作已录用论文。

删除query特征的共同标量不等于删除原始q的范数信息：当前MLP接收完整q，输入范数仍可以改变方向logits，进而影响归一化attention。NaLaFormer/NP-Map将输入norm注入非线性方向映射内部，与该标量约消并不矛盾。

**与FAVOR的关系**

当前AD没有采用随机高斯积分的蒙特卡洛估计，也没有相应无偏性保证，更接近可学习的确定性正feature map。[FAVOR+](https://arxiv.org/abs/2009.14794)使用正交正随机特征。[FAVOR++](https://research.google/pubs/chefs-random-tables-non-trigonometric-random-features/)及[FAVOR#](https://arxiv.org/abs/2302.00787)已经研究可利用统计量优化的正随机特征及方差界。因此AD与基础FAVOR+有机制区别，但超过固定m64 FAVOR+不足以证明新的分布适配原理。

**目前可以保留的研究贡献候选**

一是有限状态下的参数使用效率。两层AD用m64状态，在约相同训练参数下取得略低于m576 Hedgehog适配版的PPL和较低参考耗时。差异可能来自网络深度、K端参数化或更适合当前数据的归纳偏置，尚未被拆分。参数化即使不扩展抽象函数类，也可能构成方法贡献，但必须证明它在受限计算预算下有稳定收益。

最关键且无需扩大主线的对照是：保持相同Q方向网络，K端使用同一个128→192→64网络，直接输出exp(u_K)；与AD的exp(s_K)softmax([z_K,0])比较。二者均73536参数/head、m64、同深度、相同数据/损失，可直接检验K幅度/方向参数化的额外收益。另可测试K端softplus(u_K)。不能仅用两层AD对一层加宽HH的结果解释幅度分支的贡献。这些对照本次尚未训练。

二是固定m正attention的长上下文概率校准与value几何。当前8k出现AD的KL更差但TV、W_O后误差和PPL更好；我们已经定位到三个heads及被低估一万倍的概率质量。KL与输出不单调以及误差δVVᵀδᵀ本身是基本事实，LoLCATs也已采用输出拟合。真正有价值的新结果需要定量预测何时出现这种分歧、什么有限状态结构导致分歧，以及针对性改动能否跨模型改善它，不能把单一现象称为首发现。

三是理论必须围绕实际归一化目标。固定上下文C，令b_r=Σ_{j∈C}e^{s_K(k_j)}π_{K,r}(k_j)，p_r(j)=e^{s_K(k_j)}π_{K,r}(k_j)/b_r，α_r(q)=π_{Q,r}(q)b_r/Σ_tπ_{Q,t}(q)b_t，则

\[
\hat p(j\mid q,C)=\sum_{r=1}^{m}\alpha_r(q,C)p_r(j\mid C).
\]

这是m个上下文相关概率分布的凸混合，不是只有m种q/k；α连续变化。此表示适用于一般正feature注意力，混合/低秩解释本身不作为创新。若要发理论贡献，需要在真实上下文分布下建立非平凡的KL/TV或value加权风险界，并显示该构造在有限网络预算下如何达到界。原始κ的L2 Schmidt谱尾不能直接当作归一化KL下界。

**当前证据和论文判断**

以[最新受控评估](../normalized_attention_assessment/REPORT.zh.md)为准：纯删除AD在8k的PPL9.69161，对照HH-exp9.75236、HH-softmax9.75233，约0.62%的局部替换收益；m64状态相对m576为1/9，参考单步耗时约低34%–38%；对FAVOR+质量更好但慢约8.7%。这些均是冻结Qwen2.5-1.5B、24/336heads、既有留出数据和3种子。AD的8kKL3.42287比两HH的约2.55/2.57差，12.23%教师概率质量被低估一万倍，不能宣称全面更准或稳健长度泛化。

形式和训练目标创新：偏弱。有限预算的结构/优化创新：有合理候选，关键同深度对照未完成。机制诊断：有具体结果，尚不是一般理论。当前整体证据不足以支持以“全新kernel及理论最优性”为主张的ICLR/NeurIPS/ICML论文；若预算匹配的近邻对照、长上下文问题及更大模型实证得到解决，简单参数化也可能形成有价值的顶会方法贡献。

建议采用的准确定位是“针对归一化线性注意力的紧凑幅度—方向正特征参数化”，以状态—计算—质量曲线和可验证机制为主张。暂不使用“首次幅度方向分解”“首次查询尺度约消”“首次KL学习正kernel”“基于SVD保证接近总体最优”等表述。

本次未新增训练或更改模型。equivalence_check.py在真实1k缓存Q/K和现有AD权重上核验指数重写、因果混合重写，并对一般正feature作数值示例；结果见equivalence_check.json。数值检查辅助验证代数实现，不替代一般性证明，也不证明有限网络之间的等参数表达能力等价。
