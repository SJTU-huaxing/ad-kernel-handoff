**AD项目完整交接总册：当前版本、理论、实验、历史原文与双910B接续**

目标已确认：删除C和Q幅度，保留K幅度。研究快照2026-09-08，整理校验2026-09-09。

本文件合并六份交接材料，其中历史全文包含40份原项目文档。它可以独立阅读；源码、JSON、图和检查点仍在独立文件中，不嵌入本册。分段行号见READING_MAP.json。

历史原文保留当时结论，后续纠正以本册前部当前结论为准。历史相对链接以所标注的原文件目录为基准；请用SOURCE_INDEX打开原目录副本。外部大数据/基模型须按资产清单另行迁移。

---

**合订来源：AD_PROJECT_FULL_HANDOFF.zh.md**

**AD幅度—方向正kernel：完整研究与工程交接**

研究快照：2026-09-08；交接打包核验完成于2026-09-09，目录保留快照日期。原工作区`/root/autodl-tmp/kan_attention_theory`。接收设备：用户提供的双昇腾910B，实际显存、主机架构、驱动/CANN版本和卡间互连尚待接收端读取。本轮仅整理、打包、核验交接材料，没有启动预训练，也没有访问目标设备。

本文给出可以独立阅读的研究主线；全部历史原文另存于[全文备份](HISTORICAL_REPORTS_FULL.zh.md)，包括早期完整表格、各次验证协议与复现入口。原文中的相对路径在`snapshot/kan_attention_theory/`中保持原目录；本包未携带的大型资产在清单中明确标记。不要用摘要覆盖原始负结果。

**1．接续目标与用户已经确定的约束**

用户最初希望用两个KAN或mulKAN分别处理真实LLM的Q/K，用正特征内积拟合未归一化的指数kernel，并建立分布相关的rank-m理论下界。研究经过谱诊断、条件期望构造、MLP正特征和AD消融后，当前聚焦连续AD。

用户后来明确把当前实用目标转向归一化attention和语言模型质量，并要求删除在该目标中无效的查询幅度/C。迁移请求中的“删除K幅度”已由用户确认是笔误。**唯一默认接续版本是：删除C和Q幅度，保留K幅度。**

持续约束：不使用子agent；仔细核验而不制造实验结果；结构比较匹配有效参数，并同时报告m、计算与状态预算；feature拟合采用非MSE训练；单遍新数据约束不能被重复采样隐藏破坏；区分原始kernel风险、归一化attention指标和完整模型PPL；重要实现检查通过后推进，不反复要求确认已授权事项。目标是获得有顶会潜力的可靠研究结果，不以包装代替新颖性、强基线、正确理论或复现证据。

“可以开始预训练研究”是已给出的Go判断；具体预训练token预算、模型配置、数据集、输入归一化和双卡软件栈仍没有冻结。接收端不能把它说成已经完成100M–1B从零训练。

**2．术语、测度和风险：后续所有讨论的共同语言**

q/k指模型Q/K投影和RoPE之后的真实head向量，当前维度d=128。教师目标为

\[
\kappa(q,k)=\exp(q^\top k/\sqrt d).
\]

将q、k各乘d^(-1/4)后可写为exp(q̃ᵀk̃)。原始未缩放Q/K上的exp(qᵀk)对应不同温度，不能混用。输入标准化只作用于学生feature网络，不能替换教师目标的q/k。

P_Q与P_K由指定模型、文档分布和位置采样规则诱导。P_Q×P_K允许独立抽样和跨文档配对；真实同文档因果joint分布通常不同。一个小测试矩阵、整个有限经验边缘乘积、未知总体分布，是三个不同对象。对每张测试矩阵重新SVD/NMF，与一个训练后固定的函数在新Q/K上的泛化，也不是同一问题。

当前学习的“kernel”是可分离函数h(q,k)=Σ_r f_r(q)g_r(k)>0；Q/K两侧可以不同，不要求共享特征或Mercer意义的对称半正定性。任意未mask采样矩阵的rank≤m；对固定key集合逐行归一化仍只是左对角缩放。然而，因果mask可提升整张矩阵的rank，不能把mask后的attention矩阵也当作rank≤m。

以下评价不能混同：

| 名称 | 精确定义或解释 | 用途 |
|---|---|---|
| raw L2/NMSE | E[(κ−h)²]/E[κ²] | 原始kernel平方风险，与Schmidt理论对应；现阶段只评估 |
| raw I | E[d_I(κ,h)]/Eκ，d_I(a,b)=a log(a/b)−a+b | 未归一化质量和相对形状共同误差 |
| balanced I | 对每个Q先除以该行真实总质量，再平均 | 改变Q权重的原始I目标 |
| causal KL | KL(教师因果attention‖学生因果attention)，按Q平均 | 当前feature拟合主损失 |
| TV | ½Σ_j|p_j−p̂_j| | attention概率绝对质量偏差 |
| 系数L2² | Σ_j(p_j−p̂_j)² | attention系数平方误差，非训练损失 |
| 输出NMSE | 对AV输出计算SSE/目标能量 | 受V几何影响；聚合口径必须标明 |
| W_O后NMSE | 同层heads拼接后经过真实输出投影的误差 | 包含head间交互，仍非最终PPL |
| PPL | exp(全部评估下一token的总NLL/总token数) | 完整模型质量；每种子先计算再报告种子均值 |

balanced I、raw I和KL即使无约束零点相关，在有限模型中也可能有不同最优解。PPL的种子算术平均、平均NLL的指数、head平均NMSE和全局能量加权NMSE不可偷偷互换。

**3．当前AD的精确数学形式与参数布局**

定义固定训练输入统计量μ_Q、σ_Q、μ_K、σ_K，以及

\[
\bar q=(q-\mu_Q)/\sigma_Q,\quad
\bar k=(k-\mu_K)/\sigma_K,
\]
\[
z_Q=W_{Q,2}\operatorname{SiLU}(W_{Q,1}\bar q)\in\mathbb R^{63},
\quad
u_K=W_{K,2}\operatorname{SiLU}(W_{K,1}\bar k)\in\mathbb R^{64}.
\]

令z_K为u_K的前63维，s_K为最后一维，补入固定的零方向logit：

\[
\boxed{h_{AD}(q,k)=e^{s_K(k)}\pi_Q(q)^\top\pi_K(k)},
\quad \pi_X=\operatorname{softmax}([z_X,0]).
\]

φ_Q=π_Q，φ_K=e^{s_K}π_K。不存在s_Q、外部C、sqrt(m)因子或额外查询偏置。当前默认是`reduced_plain`，不是`reduced_matched`。后者补入偏置以维持旧登记参数量，是另一实验版本。

每head矩阵大小：W_Q1=(192,128)，W_Q2=(63,192)，W_K1=(192,128)，W_K2=(64,192)。总参数为24576+12096+24576+12288=73536。24heads独立堆叠，共1764864参数。没有隐藏偏置或输出偏置。虽然输入K来自GQA共享组，不同Q head的K-feature网络仍独立，不能在接续时擅自把它们共享来“保持同模型”。

主检查点保存`qnet.w1`、`qnet.w2`、`knet.w1`、`knet.w2`，以及四个mean/std和`numerical_scale`buffer。Q/K均值、标准差已随检查点保存，复现时直接加载。历史计算统计使用训练张量每8个位置取样后double归约、std下限.03；这不同于后续Gaussian诊断使用全部缓存向量，不能误称所有统计都用全量向量。

`raw_log_feature`返回当前数学函数的log特征。`log_feature`在`runtime_raw=False`时两侧分别减去训练head数值尺度c/2；教师logit也减c。这只是共同计算单位，归一化结果不变。`runtime_raw=True`用于实际当前attention执行。**这个buffer c不是已删除的模型C，也不是额外可学习参数。** raw评价必须正确恢复单位，KL不识别共同幅度。

log域计算先分别减各向量最大值再做exp内积、加回相应log尺度。特征维softmax仅作用于m个feature坐标，不是沿序列tokens做softmax，因此仍可线性汇总。输出π连续变化，不存在“只有64种q和64种k”的限制；m限制共享feature/state维数。

**4．为什么删Q幅度/C，却必须保留K幅度**

旧完整AD为C exp(s_Q(q)+s_K(k))π_Qᵀπ_K。固定查询i和实际合法前缀J_i，定义a_i=C exp(s_Q(q_i))。归一化后

\[
\hat p_{ij}=\frac{a_i e^{s_K(k_j)}\pi_Q(q_i)^\top\pi_K(k_j)}
{a_i\sum_{\ell\in J_i}e^{s_K(k_\ell)}\pi_Q(q_i)^\top\pi_K(k_\ell)}.
\]

a_i完全约去，对任意上下文成立，不依赖它是否服从训练P_K。专门的查询幅度输出行及C在纯attention KL/语言模型损失中没有该路径的有效梯度；Q隐藏层仍可通过方向正常训练。删除独立Q幅度不等于删除输入q范数信息，完整q仍进入方向MLP。

K幅度随j变化，通常不能从求和提出。它改变token在状态中的写入权重和最终attention比例。它不是遗忘门：没有乘在旧状态上衰减历史，也不等同于GDN的擦除。

对单个查询的KL，∂L/∂s_j=p̂_j−p_j。令r_jr为第r个feature在该query-key内积中的后验贡献，则AD方向坐标的导数为(p̂_j−p_j)(r_jr−π_K,jr)；直接EXP输出导数为(p̂_j−p_j)r_jr。当前autograd核验这些恒等式到约10^−16。

从零语言模型训练中，y_i=Σ_j p̂_ij v_j，有∂y_i/∂s_j=p̂_ij(v_j−y_i)。所以LM交叉熵可以直接训练K幅度，不需要恢复Q幅度/C或另加原始质量损失。共享隐藏层仍使实际参数更新耦合，这不是两个完全独立的优化子问题。

如果未来真要删除K幅度，会得到π_Qᵀπ_K；这是新的方向-only消融，不继承当前保留K幅度的结果。若引入精确窗口与近似分支共同归一化，查询标度可以改变分支相对质量，此时又是另一模型，不能套纯线性约消结论。

**5．线性状态、数值稳定性、训练与推理的区别**

持久状态为

\[
S_t=\sum_{j\le t}\phi_K(k_j)v_j^\top\in\mathbb R^{m\times d_v},
\qquad z_t=\sum_{j\le t}\phi_K(k_j),
\]
\[
y_t=\frac{\pi_Q(q_t)^\top S_t}{\pi_Q(q_t)^\top z_t}.
\]

每token更新和读出O(md_v)，状态O(m(d_v+1))，不需要保存全部KV。不存在新的注意力外部exp(dot feature)步骤；那样会破坏可分离形式。

实际稳定实现额外保存逐feature的log缩放g_r。令g_r为已见logφ_K,r的最大值，存储S̄_r=e^(−g_r)S_r、z̄_r=e^(−g_r)z_r。新g变大时旧状态乘exp(g_old−g_new)，新key贡献exp(logφ_K−g_new)。查询使用与g互补的exp(logφ_Q+g)再作公共归一化，最终输出不变。这是可逆数值重标度，不是截断logit或改变模型温度。

当前`causal_direction/operators.py`实现chunk64 prefill：块内因果矩阵、块间前缀状态累加。全段g可由待处理段先计算，理论上只是互补缩放，因果mask与前缀求和仍必须严格保留。`prefill`内存在finfo.tiny的分母保护；已测有效位置分母均正、保护未被当作拟合工具。迁移要记录保护是否触发，不能让大量零分母被静默掩盖。

此外，`mlp_direction/core.py::Pair.log_matrix`对缩放后feature内积使用`product.clamp_min(torch.finfo(product.dtype).tiny).log()`。这是另一处实现下限，不应在交接后声称当前源码完全没有数值保护。NPU移植须分别统计内积下限与分母下限的触发；若触发后改变了概率或梯度，须报告误差，不能把受截断的实现无条件等同于实数域公式。早期raw阶段的“FP64且不用epsilon”结论只对应其指定的修正实现。

该文件导入`triton.language.extra.cuda.libdevice`并含NVIDIA Triton内核。即使只调用PyTorch参考分支，模块级CUDA依赖也可能阻碍NPU导入。更重要的是，`prefill/step`有`torch.inference_mode()`，step原地更新状态：**它们是推理实现，不是已验证的端到端预训练反向实现。** 当前feature训练构造64Q×1024K的log矩阵来反向传播，未使用这些scan内核反向。

接收端必须将设备无关参考、训练autograd路径和NPU高效实现分离；对训练scan做输出、梯度和边界一致性检查。训练中需要保存/重算激活，不可把decode固定状态内存当作训练显存。分块detachment会改变梯度，属于截断BPTT，不能默认为等价训练。

早期只替换4个Q heads时，原GQA组仍有其他head使用KV，HF路径也可能在attention回调前先更新cache，所以KV未释放、解码更慢。后来扩为完整两层的24heads，并另行改写原KV路径，才验证了被替换层的固定状态。最新PPL使用use_cache=False的整段评价，速度表则是单层微基准；两者不能当成新完成的全模型decode/cache实验。

**6．理论体系：成立部分及不能跨越的边界**

**6.1 一般Schmidt谱框架。** 对κ∈L²(P_Q×P_K)，T:L²(P_K)→L²(P_Q)，Tf(q)=E_Kκ(q,K)f(K)。允许独立有符号特征时

\[
\kappa=\sum_r\sigma_r u_rv_r,\qquad E_m^\star=\sum_{r>m}\sigma_r^2.
\]

非负函数类最优值E_m^+≥E_m^⋆，有限宽度AD/MLP再有表示、统计和优化约束。没有证明AD、KAN或mulKAN达到这个下界。乘法节点只增加单侧特征交互，不突破m维可分离rank限制；KAN通用逼近定理不提供本问题的等预算优势。

固定Qwen的RMSNorm、有限线性投影和保范数RoPE给出Q/K有界，因此真实固定模型κ二阶矩存在。但保守logκ上界可非常大，不会自动给出有用的集中界或样本量。

**6.2 Gaussian可解析对照。** 独立零均值Gaussian，A=Σ_Q^(1/2)Σ_K^(1/2)/√d，奇异值a_i，须max a_i<1/2，才有

\[
E\kappa^2=\prod_i(1-4a_i^2)^{-1/2},\quad
\rho_i=\frac{2a_i}{1+\sqrt{1-4a_i^2}},\quad
\sigma_\alpha=\prod_i\sqrt{1+\rho_i^2}\rho_i^{\alpha_i}.
\]

闭式来自Mehler展开，解析与求积校验成立。初筛16heads只有4个满足条件；后期24heads使用全部缓存covariance时0/24满足，a_max=.552～1.847。不能把a截到.499或删尾部后声称原主线成立。Gaussian替代模型发散不表示真实模型发散。

真正TT*的kernel为M_K((q+q′)/√d)，取决于完整矩母函数而非仅covariance。同均值/方差的有界离散反例可有不同谱和极大二阶矩。因此“covariance→精确真实总体谱”在当前数据未通过Go门槛。

**6.3 有限矩阵SVD/NMF与总体。** 测试矩阵SVD给该矩阵自由rank-m oracle；数值NMF给非负矩阵最优值的可行上界，非下界，也非可泛化feature。512²矩阵即使来自真正Gaussian，也可严重低估总体风险。经验乘积全量积分仍只是有限经验测度，不因pair达到数十亿而拥有同数量独立样本。

**6.4 条件期望分区构造。** 历史正kernel为κ_C^+(q,k)=Σ_r1_Cr(q)E[κ(Q,k)|Q∈C_r]，来自L2正投影。其误差R_C满足E_m^⋆≤R_C≤E_m^⋆+D_Q，D_Q是前m个加权奇异函数的分区内方差和。两侧分区实现h=e_c(q)^T B e_d(k)，B是cell条件均值，R_CD≤2E_m^⋆+D_Q+D_K。估计误差还可与固定分区表达误差正交分离。只有附加项足够小，才有近最优保证；当前未满足。

这个旧版确实把q/k按固定cell映射，B有m²参数但feature仍m维。它与现在连续AD不同。“主要来自固定分区表达误差”指同cell内函数变化丢失，不是数据均值估计再多训练几次就能解决。

**6.5 Galerkin。** 训练确定字典与Gram正交化，压缩C=E[κ A Bᵀ]，截断C的SVD产生有符号可泛化特征。这是一般投影方法，不是FAVOR高斯积分；Galerkin也不必使用分区。其训练空间内最优不等于总体最优，带符号预测可能导致分母抵消。压缩谱只给Σ_{j>m}s_j²≤E_m^⋆≤H−Σ_{j≤m}s_j²，区间宽度H−||C||_F²反映未捕获能量。当前没有总体近最优证书。

**6.6 I散度质量—方向分解。** 对一行a=κ、b=h，Z=Σa、Ẑ=Σb，p=a/Z、p̂=b/Ẑ：

\[
\frac{\sum_jd_I(a_j,b_j)}Z=KL(p\|\hat p)+\frac{\hat Z}Z-1-\log\frac{\hat Z}Z.
\]

这是经典恒等式。原始质量与方向都对才拟合raw kernel；仅KL不识别查询尺度。对固定正g、G(q)=E_Kg、Z(q)=E_Kκ和有限相关积分：

\[
R_I(ag)=R_I((Z/G)g)+E_Qd_I(Z,aG).
\]

所以a*=Z/G在固定attention方向等价类中最优。它不会改善实际attention或PPL，不是整个正rank-m的最优构造，也非L2最优标度；L2固定g时标度为E[κg]/E[g²]。原研究实现了KL方向后再校准Q质量的两阶段方法，需披露额外训练计算。现在主方法已删除Q幅度，这个定理保留为理论来源，不再作为当前方案的PPL收益来源。

一般正函数可写h=A(q)Σπ_r(q)ψ_r(k)，其中π在simplex、E_Kψ_r=1。这说明有限m瓶颈是共享正密度的条件混合。对原始I，Q按Z重加权；对L2，按Z²重加权：若M2=E_QZ²，则σ_j(T)=√M2 σ_j(S2)，S2的kernel是κ/Z、Q测度为Z²P_Q/M2。不能把原Q测度下的归一化KL与此谱尾直接等同。

**6.7 正性、KL和因果下界。** 非负rank-m全局归一化后是至多m个乘积分布的混合，得到[I(Q;K)−log m]_+下界；当前m64实数值多数/全部为0，无近最优证书。进一步对概率联合分布粗粒化得到矩阵C，可由数据处理、Pinsker、核范数rank-m逼近推出

\[
R_I/E\kappa\ge\tfrac12(\sum_{r>m}\sigma_r(C))^2.
\]

固定key参考测度下的query平均KL也有相应界。三角mask场景另将合法区域分解为互不重叠矩形b：B_m(Ctxt)=½Σ_b w_b(Σ_{r>m}σ_r(C_b))²，平均因果KL≥E_context B_m。这个合法界没有要求整张因果矩阵rank≤m。其松弛忽略了不同矩形共享feature的约束，可能很松。

真实128篇前512位置的m64因果见证，head均值范围约2.89e−9～1.32e−4，中位3.94e−6；加所列保守iid文档置信修正后，因果和乘积两类总体下置信界都为0。**不能报告接近已知总体下界，也不能因松下界很小就证明方法离真实最优很远。**

**6.8 KL、V几何与PPL。** δ=p̂−p，则输出差δV，平方为δVVᵀδᵀ。KL可能由真实概率不小却被预测极低的tokens主导；这些tokens的V又可能冗余，因此KL排序和输出/PPL排序可以不同。Pinsker提供充分上界，不是排序等价或PPL单调定理。当前对三个heads及尾部概率的定位是具体诊断，不能把这一基本现象声称首次发现。

**6.9 AD究竟具有什么理论优势。** 已证明的是log||φ_K^AD||_1=s_K，而EXP为LSE(u_K)，以及输出坐标的梯度结构。固定隐藏表示h后，匹配AD的EXP输出需u=[z_K,0]+(s_K−LSE([z_K,0]))1，含共享非线性校正；同一h上的线性输出层一般不能免费完成。允许改变隐藏网络后，没有证明严格函数类包含关系、统一更小的最优风险或更好的收敛率。

AD是受分布正混合表述启发的有限网络参数化，当前实验支持其归纳偏置/优化组合的条件性收益。它没有从Schmidt截断唯一推导出来，更不是原始理论下界的达界构造。上述许多工具都是经典数学，创新性必须另行论证。

**7．从最初KAN到当前AD的完整阶段账本**

下面按研究发展逻辑排序；不把文件修改时间当作正式实验日期。每个阶段的完整所有数值均在对应原文与JSON中。本章只抽取关键问题、结果和下一步原因。

| 阶段/目录 | 做了什么 | 结果与演进 |
|---|---|---|
| real_llm_pilot | 真实Qwen post-RoPE Q/K，早期row目标及raw指数目标；KAN/MLP、signed/positive、m扫描、长训练、重要性采样、4head替换 | row目标归档为辅助；raw实验未证KAN普遍更优。早期FP32小分母保护影响的结果另存legacy备份，正式离线表改FP64真实分母 |
| single_pass_mulkan | 两层KAN、两层带乘法mulKAN、两层MLP，73856有效参数，双倍预算、4096文档单遍 | I训练下MLP输出NMSE .3961、KAN .4371、mulKAN .4196；没有KAN/mulKAN全面优势。更大数据与更多更新同时变化，不能只归因数据 |
| single_pass_mulkan/diagnostics | 收敛曲线、极端样本、sink/尾部分解 | 最大0.1%样本贡献90.74%–99.69%平方能量；稳健平均误差与尾部风险差异大；单遍未证明收敛 |
| gaussian_go_nogo | 16heads、m16/32/64/128，Gaussian总体谱/有限Gaussian/SVD/NMF/FAVOR+ | Gaussian原主线No-Go；4/16 HS有效，相关性与量级不足。有限NMF优于FAVOR展示矩阵空间，但不是总体函数空间证据 |
| positive_generalization | covariance适配的正指数quadrature、shared/untied节点、单遍训练，24主拟合+6损失消融 | 当前构造No-Go，未兑现NMF空间；属于FAVOR/DERF相关路线，不是非FAVOR新原理 |
| distribution_operator | 总体算子、条件期望正分区、cross-Nyström谱坐标、全经验积分、训练Galerkin | 数学误差分解成立，固定分区表达成为主瓶颈；未达总体下界 |
| kernel_comparison | 与FAVOR+、SDERF/ADERF、Galerkin按同协议比较质量和计算 | 必须区分raw风险/attention、signed分母风险；1024-landmark计算高，不能以GEMM吞吐解释为低延迟优势 |
| deployment_validation | 4/336head实际替换、双方融合优化、cache与decode、扩大经验谱 | 局部质量有收益但没有推理加速；部分GQA替换保留KV。谱收紧仍未证明总体近最优 |
| candidate_validation | 连续积分/非负锥投影、Nyström、谱模态成对正化、key-VQ、MLP/KAN/mulKAN基 | 保留K幅度改善部分raw拟合；当前谱正化全局包络失败，连续候选未建立优于普通MLP的质量/效率优势 |
| mlp_direction | 非MSE损失、softplus/exp、Q和双侧幅度方向、value加权、三轮文档确认 | 连续factorized_both显现收益；固定方向Q校准严格保attention但不能改善PPL；29主拟合及校准，原版73856参数 |
| causal_direction | 扩到完整两层24heads、全合法因果prefix、强KL/softplus控制、窗口/标度门控、三项收紧 | 真实固定cache得证；原始质量目标未稳胜强KL。Gaussian全量0/24有效，总体置信下界0；共同RoPE旋转揭示泛化缺口 |
| orbit_direction | 6个旋转增强拟合，64篇新1k+16篇新8k确认 | 增强改善长文，短文变差，同增强KL也受益；不是raw质量独有优势。未纳入当前默认AD |
| hedgehog_matched | 修正独立Q/K的HH实现；等登记参数ADm64/HHm576，I和KL、补跨文档product I | 原AD在product I比HH低42%–43%；KL下AD查询幅度192权重无有效监督，需更紧消融；HH为适配版非完整论文复现 |
| gated_integration | 340M GLA/360M GDN冻结主体，两层局部CE适配、正AD/EXP/残差MLP | 独立GDN写入幅度相对正方向有效，但强signed残差MLP更好；GLA AD劣于原模型。不能以静态正kernel谱直接控制delta记忆 |
| query_amplitude_ablation | 删除Q幅度/C，pure与补偏置同参版，product I、causal I、causal KL，24新增拟合 | I下删除损害raw拟合；KL pure与原AD基本持平。当前选择73536参数纯删除KL，不是补偏置版 |
| normalized_attention_assessment | 以归一化目标重新审计当前AD/HH/FAVOR，复测FAVOR PPL，TV/V/W_O和尾部归因 | AD的8k KL劣于HH但PPL更好；12.23%教师概率被低估1万倍，需保留这一局限 |
| novelty_review_20260908、normalized_kernel_novelty_20260908 | 对照Hedgehog、NaLaFormer、STILL、MALA、DoF等原文，代数等价核验 | 不支持“首创幅度方向/查询尺度约消/KL正特征”；有限预算K参数化归因成为关键缺口 |
| key_parameterization_attribution | 同Q网络、同两层K网络、同m64/73536参数、同KL/数据，AD vs EXP，两种初始化、3种子 | 12新增拟合、9新选定+3旧AD；8k KL/输出/PPL均改善，1k PPL略差，AD参考步骤慢10.7%；支持预训练Go |

早期raw拟合与当前归一化目标不同，是用户明确调整研究目标后的发展，不能回写为从始至终只训练同一种损失。大数据单遍阶段也不是从零训练LLM。

**8．数据、训练和检查点：当前实验如何复现**

基模型Qwen/Qwen2.5-1.5B，revision `8faed761d45a263340a0528343f099c05c9a4323`，28层、12Q heads/层、2KV heads/层、head d=128，隐藏维1536。全部层/head编号从0开始。研究核心层14/27，HEADS=[(14,0..11),(27,0..11)]，全局KI=[i//6 for i in range(24)]映射4个缓存KV heads。两层内部切片要分别用0:12和12:24。

当前`causal_direction/data/manifest.json`含4504条记录：4096 train、128 validation、128 confirm_wiki、24 confirm_long、128 confirm_prose。训练/验证来自先前文档划分；Wiki确认集从同一WikiText-103源train库另选未进入前序实验的文档，因此是研究内部留出，不是官方WikiText test benchmark。历史阶段另有官方WikiText-2固定20篇子集、SWDE、LEval和文学前缀，不能把不同阶段绝对PPL直接拼接。模型原预训练是否见过这些文章未知。

train为4096×1024 tokens，共4194304 tokens。每篇选64个互异Q位置；所有合法前缀K参与loss。每head262144个选定Q、134360517合法pairs。每个选定pair在每次运行中只进入一次梯度更新；向量在不同pair中复用。校准和统计读取不是SGD epoch，却仍是数据使用。文档/pair相关性不能忽略。

Q缓存形状train=(4096,24,64,128)，K=(4096,4,1024,128)；确认长文K/V长度8192。缓存为真实BF16投影后、RoPE后激活。训练不存V，验证/确认存V。每32文档一分片；按文件名排序连接，不能打乱manifest与张量对应。精确input_ids、Q位置、文本/token hash与模型revision都在manifest中。

标准化buffer与数值尺度只由训练集确定。feature网络初始化：每headW1~N(0,1)/√128，原64行W2~.1N(0,1)/√192；Q复制并保留前63行。要复现旧种子的随机序列，不能直接独立初始化一个63行层后宣称随机张量完全相同，因为RNG消耗顺序会变化。

优化：AdamW，weight_decay=1e−4；每文档一次更新，4096步；lr候选.002/.0005，余弦降至lr/10；按head梯度norm裁剪10；无TF32，线程4。顺序由manual_seed(85000+seed)的randperm产生；种子11/29/47。每配置seed11在相同前32验证文档上选择学习率，其他种子固定；最新所有配置选.002。不要在已有confirm集上继续调参又称新盲测。

最新归因包含standard_AD、standard_EXP、matched_zero_AD、matched_zero_EXP。standard只保证相同初始权重，输出变换造成初始attention不同。matched_zero将双方K第二层全部置0，Q与K第一层同种子；AD Kfeature为1/64，EXP为1，归一化attention均为合法前缀均匀分布，最大初始差9.72e−17。两方Q均正常更新，并不是冻结同一Q权重到训练结束。

默认主检查点为`query_amplitude_ablation/fits/causal_kl_reduced_plain_s{11,29,47}_lr0.002.pt`。最新比较检查点在`key_parameterization_attribution/fits/{standard_exp,matched_zero_ad,matched_zero_exp}_s{11,29,47}_lr0.002.pt`。后者是归因对照，不应因8k测试分数更好而未经验证规则改选matched_zero作为唯一主模型。HH-exp/softmax来自`hedgehog_matched/fits/`，FAVOR来自`causal_direction/fits/favor_s*.pt`。本包另附这21个关键检查点，精确SHA256见清单。

**9．当前最重要的结果与统计解释**

最新归因，三种子均值：

| 初始化 | K参数化 | 1k KL | 1k PPL | 8k KL | 8k输出NMSE | 8k PPL |
|---|---|---:|---:|---:|---:|---:|
| standard | AD | .347341 | 8.858055 | 3.422874 | .641975 | 9.691610 |
| standard | EXP | .348753 | 8.854966 | 4.094250 | .698279 | 9.733331 |
| matched_zero | AD | .347406 | 8.862187 | 3.380424 | .635261 | 9.694544 |
| matched_zero | EXP | .350020 | 8.855541 | 3.974762 | .694563 | 9.726658 |

AD的8k KL分别降低16.40%和14.95%，输出NMSE降低8.06%和8.54%，PPL降低.429%和.330%。两个初始化下三个种子配对方向全部一致。EXP−AD的8k NLL/token差：standard .00429654，条件文档95%区间[.00318318,.00542132]；matched_zero .00330604，区间[.00180787,.00469175]。分别24/24与21/24文档平均方向有利AD。

1k PPL AD分别比EXP高.035%和.075%。standard条件文档区间包含0；matched_zero有利EXP但训练种子方向不一致。因此当前优势应限定长上下文，不宣称全面占优。8k静态KL AD优于EXP的head数为21/24和22/24，输出为18/24和16/24，也不是逐head全胜。

与此前精确匹配适配HH及固定FAVOR的最新同数据比较：

| 方法 | 参数/head | m | 1k PPL | 8k KL | 8k输出NMSE | 8k PPL |
|---|---:|---:|---:|---:|---:|---:|
| 当前AD pure | 73536 | 64 | 8.858055 | 3.422874 | .641975 | 9.691610 |
| HH-exp适配 | 73728 | 576 | 8.870295 | 2.551741 | .894505 | 9.752359 |
| HH-softmax适配 | 73728 | 576 | 8.880211 | 2.570944 | .745977 | 9.752327 |
| 固定FAVOR+ | 0学习参数 | 64 | 9.857622 | 5.902980 | 1.652875 | 10.644294 |

HH为独立Q/K、128→288、拼接正负指数得到576feature的同登记参数适配版；当前pure AD因删192个无监督权重比73728少192，严格73536匹配的归因对象是最新EXP。FAVOR存随机系数但不训练；若接续预训练，要将随机系数显式冻结，同时允许原Q/K投影学习。不能把“胜过固定FAVOR”当作胜过所有分布适配随机特征。

教师PPL1k=8.664559867、8k=9.066278957。最新重测与历史完全一致。所有表的学生都仍低于原softmax模型质量；只替换24/336 heads，不能外推全部替换损失。

**9.1 KL比HH差、PPL却较好的归因。** 当前AD8k严重低估一万倍的教师概率质量为12.226%；HH-exp约2.373%、HH-softmax约.861%。这指教师概率质量，不是token比例。该尾部贡献约1.776的正KL；全部KL还需加其他正项和负项。L14H8、L14H9、L27H10解释AD相对HH-exp平均KL差约94.08%。

AD8k TV=.625747、系数L2²=.107785、W_O后NMSE=.567698；HH-exp分别.664561/.147266/.695810，HH-softmax.678435/.153673/.610394。静态输出和PPL可有更好排序。另一个全局能量加权pre-W_O口径中，HH-softmax .448660优于AD .471712，所以完整报告必须同时保留各口径。

最新同深度EXP的严重低估质量为14.732%，matched_zero EXP为14.364%，对应AD为12.226%/12.026%。这支持当前具体尾部校准改善，但AD自身仍有明显长文误差，不能称尾部已解决。

**9.2 删除Q幅度/C为何raw实验变差。** product I下旧完整AD的A→B/B→A原始I为.27245/.266388；纯删除为.491535/.482388；HH为.46997/.470631。补偏置同参删除版也明显差。原始I需要标度自由度，初始化raw尺度也同时变化，不能把该试验单独归因C或某一输出行。causal KL下旧完整AD和pure删除的PPL基本持平，这才支持当前目标下的取舍。旧raw优势不应转写为当前无Q幅度版本的raw优势。

**9.3 配对统计范围。** 当前文档bootstrap先对3个固定种子的同文档差取平均，再重采样整篇文档10000次。它条件于这些拟合，不覆盖完整训练种子分布、模型家族和未知P_K；head固定，不将24heads视为24次独立模型重复。8k只有24篇且已反复用于前序评估；最新协议没有按它调参，但仍不能称真正新盲测。PPL表是种子PPL均值，统计检验用NLL/token。

**10．计算、cache、时间：哪些测过，哪些只是估算**

乘加按2 FLOPs，当前AD/同深度EXP投影主导量147072/head/token；m64状态读写主导量32768。AD额外K log-softmax和幅度广播未包含在主导乘加计数。两方法状态完全相同：每head S、z和数值g共m(d_v+2)个FP32数；12head单层399360bytes=.380859MiB，两层.761719MiB。若只算数学S+z，少一个m向量；历史10.582MiB全head估算采用S+z，不能与含g实测直接视为同口径。

最新同期RTX3090、FP32、batch1、单层12heads、CUDA Graph热缓存、同参考state算子、两轮正反顺序：

| 方法 | log-feature计算μs | feature+state μs |
|---|---:|---:|
| standard AD | 32.784 | 64.384 |
| standard EXP | 26.624 | 58.175 |
| matched_zero AD | 32.848 | 64.416 |
| matched_zero EXP | 26.656 | 58.192 |

AD步骤慢10.67%–10.70%。此前HH-exp/softmax参考步骤98.016/103.392μs，FAVOR+59.264μs，来自另一轮相同设备协议；HH状态m576是m64的9倍，AD更快主要包含状态成本差异，不能说AD纯feature更快。不同轮次小差异不代表新算法速度变化。

历史全模型估算保留原Qwen投影/FFN：原decode约3.08714+.000172032T GFLOPs/token，旧完整AD约3.14769；8k算术量下降约30%。这只是全head理想计算估算，当前pure版只略少查询幅度计算；没有全headPPL或速度实测，也没有910B速度。旧21–27ms等区间只是RTX3090条件情景，禁止迁移为NPU预测。

全部28层、每层12head若均用当前m64且含g，状态约10.6641MiB；当前feature参数共24708096，FP32权重约94.254MiB（不含norm buffers）。原BF16 GQA KV为T×28×2×2×128×2bytes，8k=224MiB。状态不增长是真实算法性质，但附加权重、激活、workspace、未替换层KV和训练优化器显存必须另计；不能称总显存缩小21倍。

旧landmark分区kernel1024节点的高FLOPs在融合后仍很慢，GEMM有效吞吐更高不等于延迟更低。现在AD没有这些landmark。当前新方案是否适合NPU融合，需要实际profile验证，不能从3090热缓存微基准推断910B优势。

**11．必须保存的支线与负结果**

旋转增强：同一RoPE正交旋转R同时作用q/k，目标κ不变，推前分布的总体谱也不变，固定学习feature却可能改变。训练中共同旋转改善了新长文结果，但短文变差、同增强KL也受益。当前默认AD没有该增强；共同位置平移也不等于真实相对距离变长。

GLA/GDN接入：另一个独立实验冻结340M/360M原生线性模型，只做两层CE适配，不是当前正kernel加分母模型。GLA采用逐通道门，delta记忆还依赖key-key几何和擦除，静态QK kernel相同也可能有不同输出。原生L2会消掉AD幅度；把幅度改乘V可形成独立写入门，但不是原公式。

该支线中GLA AD的2048 PPL18.49160，原模型17.43497、signed残差MLP16.65899；GDN AD写入14.95613，原模型15.63878、方向-only15.15213、signed残差MLP14.34615。AD有正方向条件下幅度增益，但强残差控制更优。端到端参考耗时增加约3%–6%，原模型本来就固定state，cache不减少。GLA/GDN branch对“直接接入独特优越性”No-Go，并未完成该架构从零训练，也不能用它证明当前AD普遍成立或普遍失败。

谱正化、固定partition、covariance quadrature和KAN不能因为曾有局部好数字就默认复活为主方法。后续可研究，但必须有新动机、独立协议和预算匹配对照。

**12．新颖性定位、顶会潜力和文献边界**

当前准确定位是“针对归一化线性attention、有限feature状态下的紧凑K幅度—方向正特征参数化”。经典正特征分解、softmax约消、KL蒸馏、条件期望和SVD本身不构成新贡献。最新归因补上了有限预算参数化的实证证据，但预训练、强基线、新域泛化、有效理论界和NPU高效实现仍待完成。

| 近邻 | 已有内容 | 当前区别/限制 |
|---|---|---|
| [Hedgehog](https://arxiv.org/abs/2402.04347) | 学习正指数特征、尖锐性、attention KL/交叉熵蒸馏 | 当前两层K显式质量与方向，已有同深度EXP归因；不是首次正MLP或KL |
| [Performer/FAVOR+](https://arxiv.org/abs/2009.14794)、[FAVOR#/DERF](https://arxiv.org/abs/2302.00787) | 正随机特征、优化随机特征参数/统计适配 | AD为确定性学习feature，无随机估计无偏性；不能只胜固定FAVOR就称领先全部相关方法 |
| [NaLaFormer](https://arxiv.org/html/2506.21137v3)、[STILL](https://arxiv.org/html/2602.02180v1) | 范数/方向及归一化映射已有直接研究 | 其范数作用位置不同；“幅度方向”名称不新，须比较精确参数化与机制 |
| [MALA](https://arxiv.org/html/2507.00698v3) | 查询范数被归一化消去 | 删除query共同幅度不是新发现 |
| [DoF for Linear Attention](https://arxiv.org/html/2507.03340v1) | 分布相关有效维数、逼近界与特征预算 | 当前并非首次分布理论指导线性attention；尚无新的紧界 |
| [LoLCATs](https://arxiv.org/abs/2410.10254) | 输出拟合、转换、LoRA、混合attention | 当前冻结模型无LoRA；输出与KL非单调不是首次价值感知问题 |
| [ReGLA](https://arxiv.org/abs/2502.01578)、[The Key to Going Linear](https://arxiv.org/abs/2607.07706) | 指数稳定性/门控、冻结骨干机制分析 | 静态K质量不同于递推门，冻结研究设定本身不新 |
| [Gated DeltaNet-2](https://arxiv.org/abs/2605.22791) | 擦除与写入分离 | 不能把旁支独立写入门单独包装为首创 |

上述文献关系沿用本机2026-09-08原文核对；预印本不冒称已录用，未运行的论文不参与SOTA排名。详细版本/原文辨析见两份novelty报告。主线理论需要证明可检验分布/网络约束下的非平凡优势，而不是重新命名经典恒等式。

**13．三项必须持续收紧的地方与证据状态**

第一，理论对象与证书：始终明确raw L2、I、normalized KL、causal/context/序列测度。已有合法关系，但未知总体的非零紧下界和AD近达界仍未得证。Gaussian covariance捷径已失效。

第二，强控制与泛化：同深度EXP归因已完成；原HH共享权重的问题已修正；m/状态和有效参数必须一起报告。旧确认集不再是新盲测，后续扩大规模需要新数据与预训练。当前不能分清优化路径和表示约束各自贡献，不能声称已充分收敛。

第三，真实部署与预算：当前分母和固定状态数学/数值已核验；最新只测局部PPL和参考单层步骤。预训练反向、全层模型、双910B吞吐、长序列稳定性和实际cache路径均是接收端新增工作，不能把旧代码中的inference_mode当训练支持。

现阶段结论：**小规模从零预训练Go；普遍理论优势、总体达界、完整模型加速和顶会充分性均未证明。** 不需要等待全部理论完成才做预训练，但不能将这些未证主张写作实验前提。

**14．文件关系、执行顺序与复现实物**

当前类定义入口`query_amplitude_ablation/ablation.py::Reduced`。父网络`hedgehog_matched/models.py`，数据与KL`causal_direction/common.py`，log矩阵的父实现`mlp_direction/core.py`，推理state`causal_direction/operators.py`。最新归因`core_attribution.py::Features`只覆盖EXP的K输出与匹配初始化。完整模型替换复用`hedgehog_matched/assess.py::Replacement`，通过adapter设置runtime_raw。

导入大量依赖`sys.path.insert`与通用模块名core/common/models；迁移时建议明确包名或importlib加载，防止错误导入别的阶段同名模块。原路径含`/root/autodl-tmp/hf-cache`和绝对Python路径，需要改配置。保留旧脚本只读，不修改旧JSON来覆盖NPU结果。

最新NVIDIA原复现顺序：`proof_checks.py`、`mechanism_checks.py`、`train_run.py`、`eval_run.py all`（verify→precision→kernels→ppl）、`diagnose.py`、`benchmark_run.py`、`aggregate.py`、`plot_results.py`、`write_report.py`、`audit.py`。绘图环境与torch环境历史上不同；不要把缺matplotlib理解为模型错误。

核验结果：初始化函数一致、参数数目、数据顺序/hash、norm buffer、valid-only学习率、FP64矩阵/scan/递推、真实8k FP32误差约1.6e−6～2.0e−6、教师PPL完全一致、诊断与父KL/output指标<1e−9差异，最新audit全部通过。数值核验不是统计泛化证明。

精确数据manifest约68.6MB，整个最终causal数据目录8743844018bytes（8.744GB，约8.143GiB）；没有装入文档包。基模型完整snapshot为3098972223bytes（3.099GB，约2.886GiB），权重及tokenizer/config/generation_config等需同revision迁移。HF snapshot可能是指向blobs的相对符号链接，单独复制链接会断；在目标目录保留整个cache结构或复制解引用后的快照。外部资产与文件SHA256见清单，禁止携带HF token、账号配置或旧设备完整环境目录来替代安装。

本次整理附带21个关键检查点独立包，源码/报告/汇总轻量包和大数据迁移清单。原始历史大矩阵、所有未选中拟合和全部模型cache仍在旧工作区，有需要再按清单迁移；文档已经完整保留各阶段结论与负结果。下一步的910B验证、双卡训练和预训练研究顺序见[迁移文档](ASCEND_910B_MIGRATION.zh.md)。

---

**合订来源：ASCEND_910B_MIGRATION.zh.md**

**双昇腾910B迁移与预训练接续方案**

本文是待在目标设备执行的工程方案，不是NPU实测报告。已知仅为两张910B；显存容量、910B子型号、主机CPU架构、互连、驱动、固件和CANN均未知。应先运行本包[inspect_target.py](inspect_target.py)，记录输出，再选择软件组合。不要根据“910B”三个字承诺可训练的模型规模、batch或耗时。

**1．必须保持不变的研究对象**

当前h(q,k)=exp(s_K(k))softmax([z_Q(q),0])ᵀsoftmax([z_K(k),0])；删除C与Q幅度、保留K幅度。源版m64、d128、hidden192、无偏置、73536参数/head。迁移复现阶段先原样保留，不能偷偷加入幅度tanh/clamp、epsilon kernel、旋转增强、窗口、遗忘门、GQA共享feature或不同归一化。

复现实验和从零预训练是两个任务。前者验证相同数学函数、源检查点及源BF16 QKV在目标后端的前向/梯度/指标；后者允许定义新的完整LM配置并训练全部权重，必须单列协议和结果。将NPU重新提取的QKV与原GPU缓存混合，会同时改变数值路径与输入数据，不能称只更换设备的严格对照。

**2．软件栈选择及当前官方依据**

源机是x86_64，Python3.11.16，torch2.13.0+cu130、Transformers5.16.1、Triton3.7.1；这些是源环境记录，**不是NPU安装命令**。torch_npu需要与PyTorch、CANN以及驱动/固件匹配。官方版本表持续变化，应以目标设备实际CANN和当前支持矩阵为准，不照搬CUDA wheel或旧conda目录。[TorchNPU兼容矩阵](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)

官方TorchNPU提供PyTorch的NPU后端；建议显式import torch_npu完成设备初始化。接收端先验证两个NPU可见及小型矩阵乘，再创建隔离环境，不擅自升级系统驱动。若主机为aarch64，选择对应wheel，不能复制x86_64二进制。[TorchNPU官方入口](https://github.com/Ascend/pytorch)、[安装说明](https://ascend.github.io/docs/sources/pytorch/install.html)

源状态内核含CUDA libdevice，不能直接在910B执行。Ascend提供Triton-Ascend，其安装和支持硬件/版本需要单独匹配；存在后端支持不等于某个项目内核已经兼容。[Triton-Ascend安装指南](https://github.com/Ascend/triton-ascend/blob/main/docs/en/installation_guide.md)

FLA官方已包含Ascend相关适配/发布记录，因此不能笼统说“FLA完全不支持NPU”；应核对所选commit、所需linear-attention操作、归一化语义和backward。[FLA发布记录](https://github.com/fla-org/flash-linear-attention/releases)

这里有两个不同的版本证据：历史`gated_integration/checks/environment.json`记录该支线实验的FLA commit为`78254ec52c2981370260f606bb82d3da28aee9e3`；交接时实际工作树HEAD为`8e84ed4a6727be082c34a3855c60623fd11411e9`，见[SOURCE_ENVIRONMENT.json](SOURCE_ENVIRONMENT.json)。不能用当前HEAD替代历史运行出处。源FLA文件树未装入交接包，复现该支线时应先固定对应历史revision；新的NPU移植另记实际采用的revision。

双卡PyTorch分布式使用与NPU匹配的HCCL后端，rank绑定到对应设备。具体初始化参数按选定TorchNPU版本；不沿用NCCL配置。[Ascend手工迁移指南](https://www.hiascend.com/document/detail/zh/Pytorch/710/ptmoddevg/trainingmigrguide/PT_LMTMOG_0016.html)

以上官方页面于交接时核查。本文不冻结一个未经目标设备验证的版本号；源Transformers5.16.1的attention registry接口也需与目标兼容PyTorch版本共同测试。若降版本，模块路径、forward签名、cache对象和dtype参数均可能变化，应写adapter并校验，而不是忽略警告继续报结果。

**3．资产迁移：文档包能做什么、还缺什么**

文档包包括40份项目报告/理论/协议原文、183份项目Python源码、关键汇总/检查、图和完整文件清单。另一个检查点包包括当前pure AD3种子、最新同深度归因9个、HH6个、FAVOR3个，共21个。所选.pt保存state_dict/metadata，加载时用weights_only=True、map_location='cpu'并校验预期字段和形状；不要将PyTorch序列化文件当成普通文本读取。

最终causal_direction/data全目录精确字节数与每文件SHA256在EXTERNAL_DATA_MANIFEST.json，合计8743844018 bytes；该目录包含约68.6MB的manifest和原BF16激活。应另行迁移到新项目同相对目录。目标容量不足时可先只迁移少量确认分片做烟测，但不能以它代替完整确认数据重报表中指标。

基模型Qwen2.5-1.5B指定revision及tokenizer/config/权重文件见EXTERNAL_MODEL_MANIFEST.json。可迁移解引用后的完整snapshot，或在目标按同revision重新获取；只拷HF snapshot中的符号链接可能丢失blobs。模型权重和最终数据均未放进两个交接压缩包，避免把轻量文档包误解为可离线复现全部实验的完整镜像。

其他阶段所需single_pass_mulkan/data、real_llm_pilot/data、mlp_direction/values、candidate大矩阵、GLA/GDN模型等不属于当前最小复现集；路径和大小在ARTIFACT_INVENTORY.json。若要重新计算某历史阶段，再迁移相应资产。原文表格与该阶段摘要已全部保存，不需要先转移34GB原工程才能理解研究结论。

迁移后可将snapshot中的kan_attention_theory复制成新的工作目录，再把检查点包同名目录合并，将外部data放入对应位置。保留原snapshot作为证据。新运行写入例如ascend_reproduction或pretrain_ad的新目录，禁止让旧“文件存在则跳过”逻辑生成伪复现。

**4．源码需要改的具体位置**

| 源位置/问题 | 迁移动作 | 保持的语义 |
|---|---|---|
| 各load_fit/data/训练脚本中的.cuda()、device='cuda' | 明确device参数，checkpoint先CPU加载再.to(device) | 相同模型与数据，不静默改dtype |
| common/core/models等通用模块名和sys.path.insert | 改为明确包名或显式importlib路径 | 防止导入错误阶段的同名类 |
| causal_direction/operators.py模块级Triton CUDA导入 | 拆出纯torch参考和可选后端模块 | 不因导入失败阻塞CPU oracle |
| prefill/step的inference_mode及原地state更新 | 推理保留；另实现可微训练scan或验证过的custom backward | 不把推理代码当训练算子 |
| FP64核验中的.double().cuda() | CPU FP64 oracle；NPU FP32/BF16单独对比 | 不要求NPU支持与CUDA相同的FP64路径，也不静默降精度冒充FP64 |
| CUDA Graph、CUDA Event、cuda.synchronize | 用目标后端支持的同步/计时API；先普通eager | 不把异步提交时间当真实耗时 |
| TF32标志、CUDA autocast | 明确NPU混合精度和算子策略，保留FP32累加对照 | CUDA开关不会自动控制NPU精度 |
| 原Qwen registry Replacement | 适配目标Transformers版本、RoPE/GQA/cache调用顺序 | post-RoPE输入、正确head到KV组映射 |
| 所有固定6倍repeat、12head/层、128/64常量 | 复现保留；预训练改配置并测试布局 | 新架构不要沿用错误硬编码 |
| 绝对/root/autodl-tmp和本机cache路径 | 用配置/命令行指向新目录 | 不读取不存在或错误版本的文件 |

不建议使用广泛猴子补丁把torch.cuda全部重定向为NPU后立即开展长训练：dtype、同步、图捕获、Triton、cache和autograd语义仍需逐项验证。当前修改目标是数学等价和可审计，不是让脚本表面无报错。

**5．设备与数值验收顺序**

先建立CPU FP64 oracle。读取源检查点和小型真实QKV，逐项比对μ/σ、Q/K logfeature、logkernel、因果attention、输出。不要把源BF16缓存转FP64后描述为“原始投影就是FP64”，它只是对既有BF16值作高精度后续运算。

随后用NPU FP32实现同一函数；最后评估BF16投影/FP32累加方案。每种精度单独记录feature误差、attention KL、输出相对误差、zero/nonfinite分母、幅度log范围。FP64 oracle保留CPU即可；目标栈各算子的double支持应按实际测试，不假定能像源CUDA路径一样运行。

必须测试：

1. 参数数目、形状、加载strict、buffer哈希；完整AD约消到pure版的实数恒等性；K幅度仍影响输出。
2. 小矩阵显式masked kernel、分块prefill、逐token recurrence三者一致。覆盖1token、非块大小整倍数、空初始state、已有state续接、文档重置、padding、不同head/GQA组。
3. 前向因果性：修改未来K/V不会改变当前输出；有效查询不能读到padding或其他文档。数值g使用未来块最大值仍需保证最终结果只差容许舍入。
4. 小规模可微dense参考与训练scan比较dQ/dK/dV、全部feature参数梯度。检查没有inference_mode、detach或不可见原地覆盖破坏反向。有限差分/CPU gradcheck仅对小尺寸高精度参考使用。
5. AMP下FP32状态与log归约稳定性；扩到1k和8k，报告实际误差而不直接复用3090阈值。阈值在看候选质量排名前按参考数值误差冻结；新失败不可通过临时放宽阈值掩盖。
6. 整段前向与prefill后cache逐token延续比较logits/NLL。实际cache对象张量字节应按存储去重统计，确认新token只增长未替换层KV，不增长替换层线性state。
7. 双卡与单卡同global batch的小步训练比较loss/更新方向，正确聚合有效token数和梯度；通信精度及归约顺序造成的差异单列。

源最新8k FP32/FP64输出最大相对差约2e−6、零分母0，是源平台记录，不能强求所有NPU低精度路径达到完全相同bit。质量差异大于数值噪声后才能讨论算法差异。真实teacher在NPU上的PPL也要重测，不能拿GPUteacher数值作NPU学生的唯一参考。

**6．双卡执行策略与公平计时**

先一张卡验证算子、梯度和短训练。两卡可在独立研究拟合时分别跑AD与EXP，固定配置和seed，但这样得出的壁钟不是同期无干扰的严谨速度benchmark；性能计时单独占用设备并轮换顺序。

完整LM预训练优先采用每卡一进程的HCCL DDP。是否需要参数/优化器切分、激活检查点或张量并行由实际显存和模型配置决定，不能因有两卡就直接选择1B。初期不同时改变并行策略和kernel设计来解释收益。

相同global tokens/batch、累计步数、文档流、随机种子、tokenizer和序列长度必须记录。DistributedSampler默认补齐样本可能造成重复，严格单遍时要使用明确无重复分片并处理最后不齐批；丢弃数据也要计数。预训练中的多epoch若以后被允许，应明示新协议，不沿用“每个pair只训练一次”的旧结论。

验证PPL按总NLL/总有效token归约，不能直接平均不同长度文档或rank的PPL。故障续训保存optimizer/scheduler、RNG、数据游标和global token计数，避免恢复后重放数据却仍宣称单遍。

计时分别报告feature、attention、整步训练、prefill、decode。NPU需预热和同步；排除编译、首次内存分配及数据读取干扰，另外记录冷启动代价。测多个batch和1k/8k/更长序列，报告吞吐、延迟分位数、峰值/持久显存。不能用3090的CUDA Graph热缓存μs预测910B；不能把PPL评估总秒数当推理benchmark。

**7．预训练研究的冻结方案应怎样制定**

建议分三步；这是接续方案，尚未执行或确定具体token预算。

| 阶段 | 目标 | 进入下一步的依据 |
|---|---|---|
| 实现烟测 | 小模型/短序列、dense和scan前向梯度、单/双卡训练 | loss/梯度/因果性/cache正确，无数值故障 |
| 100M–150M完整预训练 | AD vs同深度EXP主归因，加HH、FAVOR及softmax参考，多个seed | 独立验证数据的质量、长上下文能力和训练成本形成可复现取舍 |
| 350M再到1B | 规模、数据和长期稳定性复现 | 小规模收益经新域和计算预算对齐后仍有价值，硬件实测支持规模 |

从零训练用LM next-token交叉熵。当前attention KL是有教师的feature拟合目标，不必带入主预训练，更不需要为删除的Q幅度/C添辅助损失。若做教师蒸馏或转换，单独统计教师计算和loss权重，不能与无教师从零训练混为一组预算。

主归因两者保持同Q网络、同K深度/宽度、同m、同参数和初始化方案，仅切换K输出。HH/FAVOR和softmax不能同时满足所有参数、state和FLOPs约束，须分别展示等状态维数、总参数及等训练计算的比较；清楚标出各项，而非补无效参数。新骨干必须计算包含feature网络的总参数。

旧Qwen检查点仅适用于其d128/head布局和post-RoPE输入；从零新模型不要直接加载它们来暗中引入教师初始化。m64可作为起点，再预先规划m32/64/128及宽度消融，避免根据新测试集临时挑选。来源训练标准化buffer不应直接用于完全不同骨干。预训练输入归一化必须统一定义：是否沿用固定统计、何时校准、或使用统一可学习/RMSNorm，都要视为新配置并公平应用；这项尚未冻结。

全部需要比较的层使用真正线性attention实现后，才可以称全模型线性预训练；如加入local window、sink、门控、短卷积或hybrid比例，要全基线统一或设明确消融。当前pure AD没有这些组件。

主要指标：in-length与长度外推PPL、独立领域PPL、检索/关联回忆、训练NLL随tokens与实际计算下降曲线、训练/解码吞吐、cache和峰值显存、数值异常率。原始kernel谱和I只作为诊断，因为从零训练时Q/K分布也共同改变；不能把“更接近原softmax exp(qk)”视为LM CE训练的隐含必需目标。

Go规则应在新确认结果前冻结。不能把旧约.3%–.4%PPL增益设成保证，也不必只看是否有统计显著性；需要看效应是否大于数值差异、是否跨seed/新数据保留、是否值得实际额外计算。若只在8k收益、1k损失，就把它作为长度泛化取舍研究，不包装为全指标优越。

**8．后续理论应围绕什么完善**

保留三个层次：总体正rank-m条件混合、有限宽度AD/EXP函数族、有限数据/优化过程。现在只确认输出坐标结构与实证差异，没有分别识别它们。可以研究在真实或可检验的有界/混合分布及分布漂移下，显式K质量读出如何限制log质量随方向变化产生的误差，并与同预算EXP/softplus等对照。

要提出“理论上更好”，需明示分布/网络/预算假设、比较对象、评价风险和可验证条件。原Schmidt L2尾不自动成为归一化KL下界；旧非零经验见证经总体修正为0仍须保留。预训练分布随参数演化，固定教师分布的定理不能无条件扩展到联合训练。

本阶段最有希望的论文主张是有限状态下的可复现质量—计算优势及其机制；不是“首次幅度方向”“首次query缩放约消”或“已近总体理论极限”。理论、实验与NPU工程可以并行推进，但每项已完成/待验证状态要持续更新。

---

**合订来源：CURRENT_RESULTS.zh.md**

**当前有效结果账本：由审计后JSON自动生成**

没有新增训练或计算新的模型指标。本表不跨不同数据集拼排名；所有条件和限制见完整交接文档。数字来源为snapshot中的两份summary.json及query_amplitude_ablation的benchmark.json。

**A．最新K参数化归因：三种子均值**

| split | group | kl | output_nmse | ppl | head_tv | head_coeff_l2 | layer_projected_nmse |
|---|---|---|---|---|---|---|---|
| confirm_wiki | standard_ad | 0.347341452 | 0.220416135 | 8.858054832 | 0.263031433 | 0.041484233 | 0.258934245 |
| confirm_wiki | standard_exp | 0.348752612 | 0.221531969 | 8.854965645 | 0.262249243 | 0.042448547 | 0.262741101 |
| confirm_wiki | matched_zero_ad | 0.347405666 | 0.221244541 | 8.862187380 | 0.263506777 | 0.041650597 | 0.261007339 |
| confirm_wiki | matched_zero_exp | 0.350020290 | 0.222377175 | 8.855540686 | 0.262164699 | 0.042450578 | 0.265262763 |
| confirm_long | standard_ad | 3.422874068 | 0.641975163 | 9.691610484 | 0.625747287 | 0.107785455 | 0.567697869 |
| confirm_long | standard_exp | 4.094249851 | 0.698279267 | 9.733331121 | 0.631699205 | 0.112674802 | 0.616000478 |
| confirm_long | matched_zero_ad | 3.380424084 | 0.635261297 | 9.694544312 | 0.621531622 | 0.106134736 | 0.565052171 |
| confirm_long | matched_zero_exp | 3.974761869 | 0.694563156 | 9.726657820 | 0.632034799 | 0.113095200 | 0.612478342 |

每种子PPL顺序固定为11、29、47。

| split | group | s11 | s29 | s47 |
|---|---|---|---|---|
| confirm_wiki | standard_ad | 8.849866025 | 8.859123760 | 8.865174713 |
| confirm_wiki | standard_exp | 8.846558680 | 8.856743624 | 8.861594630 |
| confirm_wiki | matched_zero_ad | 8.859625934 | 8.861511934 | 8.865424272 |
| confirm_wiki | matched_zero_exp | 8.862835999 | 8.860263049 | 8.843523011 |
| confirm_long | standard_ad | 9.666692751 | 9.708363646 | 9.699775056 |
| confirm_long | standard_exp | 9.742827637 | 9.740778129 | 9.716387597 |
| confirm_long | matched_zero_ad | 9.695719095 | 9.688844400 | 9.699069442 |
| confirm_long | matched_zero_exp | 9.729756266 | 9.742666712 | 9.707550482 |

**B．归因配对统计：EXP−AD，正值有利AD**

| split | init | metric | Δ | 文档95%CI | 三种子Δ | AD胜文档 | AD胜head/24 |
|---|---|---|---|---|---|---|---|
| confirm_wiki | standard | kl | 0.001411160 | [0.000714889, 0.002112432] | -0.000884093,0.001673650,0.003443924 | 79/128 | 10 |
| confirm_wiki | standard | output_nmse | 0.001115834 | [0.000516936, 0.001715239] | -0.000097509,0.001126611,0.002318399 | 83/128 | 8 |
| confirm_wiki | standard | nll | -0.000348802 | [-0.000841932, 0.000146072] | -0.000373787,-0.000268701,-0.000403918 | 63/128 | — |
| confirm_wiki | standard | head_tv | -0.000782190 | [-0.000969626, -0.000597461] | -0.001413556,-0.000390240,-0.000542774 | 36/128 | — |
| confirm_wiki | standard | head_coeff_l2 | 0.000964314 | [0.000834605, 0.001095846] | 0.000733837,0.001007557,0.001151548 | 115/128 | — |
| confirm_wiki | standard | layer_projected_nmse | 0.003806856 | [0.002825641, 0.004799173] | 0.000298615,0.006670653,0.004451300 | 89/128 | — |
| confirm_wiki | matched_zero | kl | 0.002614623 | [0.001925912, 0.003318900] | 0.002844041,0.000831602,0.004168227 | 97/128 | 11 |
| confirm_wiki | matched_zero | output_nmse | 0.001132634 | [0.000376466, 0.001882505] | 0.003324723,-0.001458171,0.001531351 | 78/128 | 11 |
| confirm_wiki | matched_zero | nll | -0.000750718 | [-0.001227837, -0.000256494] | 0.000362260,-0.000140944,-0.002473470 | 50/128 | — |
| confirm_wiki | matched_zero | head_tv | -0.001342079 | [-0.001529455, -0.001159793] | -0.001360313,-0.002147472,-0.000518451 | 15/128 | — |
| confirm_wiki | matched_zero | head_coeff_l2 | 0.000799981 | [0.000651455, 0.000948285] | 0.000885744,0.000637773,0.000876425 | 111/128 | — |
| confirm_wiki | matched_zero | layer_projected_nmse | 0.004255424 | [0.003000690, 0.005506881] | 0.006650758,0.003334522,0.002780992 | 89/128 | — |
| confirm_long | standard | kl | 0.671375783 | [0.647896176, 0.695285174] | 0.655982430,0.594831662,0.763313257 | 24/24 | 21 |
| confirm_long | standard | output_nmse | 0.056304103 | [0.048399472, 0.065156899] | 0.073492829,0.041970922,0.053448559 | 24/24 | 18 |
| confirm_long | standard | nll | 0.004296538 | [0.003183176, 0.005421323] | 0.007845148,0.003333259,0.001711208 | 24/24 | — |
| confirm_long | standard | head_tv | 0.005951918 | [0.004612371, 0.007284317] | 0.009065013,0.003197626,0.005593113 | 23/24 | — |
| confirm_long | standard | head_coeff_l2 | 0.004889347 | [0.003924376, 0.005803254] | 0.007733344,0.002291933,0.004642764 | 22/24 | — |
| confirm_long | standard | layer_projected_nmse | 0.048302609 | [0.040862158, 0.056202216] | 0.053609932,0.039832766,0.051465130 | 24/24 | — |
| confirm_long | matched_zero | kl | 0.594337785 | [0.573010943, 0.616601326] | 0.592216106,0.674306976,0.516490273 | 24/24 | 22 |
| confirm_long | matched_zero | output_nmse | 0.059301859 | [0.050638929, 0.069295233] | 0.056795925,0.071199478,0.049910174 | 24/24 | 16 |
| confirm_long | matched_zero | nll | 0.003306044 | [0.001807866, 0.004691746] | 0.003504388,0.005539708,0.000874036 | 21/24 | — |
| confirm_long | matched_zero | head_tv | 0.010503177 | [0.009415846, 0.011599640] | 0.009785235,0.011556504,0.010167791 | 24/24 | — |
| confirm_long | matched_zero | head_coeff_l2 | 0.006960464 | [0.006202444, 0.007793424] | 0.004527918,0.010184755,0.006168720 | 24/24 | — |
| confirm_long | matched_zero | layer_projected_nmse | 0.047426171 | [0.038646998, 0.056822890] | 0.047070415,0.050084797,0.045123302 | 24/24 | — |

区间为条件于3个固定拟合的整文档bootstrap；不覆盖未知训练随机性/模型家族，无多重比较校正。

**C．学习率选择与训练曲线记录**

| group | lr | seed11 validation KL | selected |
|---|---|---|---|
| standard_ad | 0.002 | 0.353309632 | True |
| standard_ad | 0.0005 | 0.478882006 | False |
| standard_exp | 0.002 | 0.352812785 | True |
| standard_exp | 0.0005 | 0.496450013 | False |
| matched_zero_ad | 0.002 | 0.353002038 | True |
| matched_zero_ad | 0.0005 | 0.488211340 | False |
| matched_zero_exp | 0.002 | 0.357205939 | True |
| matched_zero_exp | 0.0005 | 0.498428492 | False |

`standard_ad`；steps=[512, 1024, 1536, 2048, 2560, 3072, 3584, 4096]；每512文档平均KL曲线（3种子）=[[0.7263801246881485, 0.4739738143980503, 0.43088475863138836, 0.40007366115848225, 0.38227002633114654, 0.36582383203009766, 0.3578212441255649, 0.3506211632241805], [0.7203923091292381, 0.4737651323278745, 0.42778696057697135, 0.3973876864959796, 0.38127755497892696, 0.36642860372861225, 0.35174179698030156, 0.3507952441771825], [0.7242023957272371, 0.4780319556593895, 0.4288492215176423, 0.4002997409552336, 0.37364640769859153, 0.36126883886754513, 0.35346464812755585, 0.34743626539905864]]；validation KL=[0.3533096317762738, 0.353640024927612, 0.3491020676145977]；训练循环秒数=[32.18901946209371, 30.68344553746283, 29.359612295404077]。

`standard_exp`；steps=[512, 1024, 1536, 2048, 2560, 3072, 3584, 4096]；每512文档平均KL曲线（3种子）=[[0.7483264642457167, 0.4797905298570792, 0.43384090748926, 0.39959160548945266, 0.38184993900358677, 0.3649140087266763, 0.3573954838017623, 0.3496781413753827], [0.7408600375056267, 0.4809233422080676, 0.4312120887140433, 0.3986082822084427, 0.3822189973046382, 0.3672678265720606, 0.3527536727488041, 0.3522798400372267], [0.74187833070755, 0.4829679951071739, 0.43202414860328037, 0.4039328309396903, 0.37639843858778477, 0.3642876725643873, 0.3568343836814165, 0.35041124870379764]]；validation KL=[0.35281278530174626, 0.353585403498811, 0.3534613256864789]；训练循环秒数=[28.972096603363752, 28.751284204423428, 27.95178009569645]。

`matched_zero_ad`；steps=[512, 1024, 1536, 2048, 2560, 3072, 3584, 4096]；每512文档平均KL曲线（3种子）=[[0.7368550884226958, 0.4774505378057559, 0.43108449193338555, 0.39990180792907876, 0.38234706843892735, 0.36570513620972633, 0.3580151417603095, 0.3512288040171067], [0.7320571889479955, 0.4805584376056989, 0.43008329595128697, 0.3978986721485853, 0.38180644313494366, 0.3667921635011832, 0.35236654430627823, 0.3517198208719492], [0.7354759623607, 0.47931768496831256, 0.4276965477814277, 0.398336252818505, 0.37200738800068694, 0.359372990205884, 0.35172929987311363, 0.3461409385005633]]；validation KL=[0.3530020380707408, 0.35286668078780203, 0.3481738325968195]；训练循环秒数=[31.046193286776543, 30.72222498804331, 30.496870944276452]。

`matched_zero_exp`；steps=[512, 1024, 1536, 2048, 2560, 3072, 3584, 4096]；每512文档平均KL曲线（3种子）=[[0.7615259103477001, 0.4862556792795658, 0.437129070982337, 0.40283143892884254, 0.3850159210463365, 0.3677578754723072, 0.36086983792483807, 0.35331404705842334], [0.752001008639733, 0.48451678454875946, 0.43372926364342373, 0.40061739025016624, 0.3834811033060153, 0.3680503536015749, 0.35322110913693905, 0.3524506955097119], [0.7510509168108305, 0.485097033282121, 0.43216509620348614, 0.40309234770635766, 0.3754226602613926, 0.3629657117029031, 0.35517572797834873, 0.349221583455801]]；validation KL=[0.35720593945765566, 0.3539857203283472, 0.3517516021168534]；训练循环秒数=[28.73698396794498, 28.181755408644676, 27.48097581230104]。

训练秒数包含历史AD复用，不能作同期算法加速比较。每组全量epoch=1，最终checkpoint；验证集选择学习率，不按确认数据选择。

**D．最新同深度时延和状态：RTX3090参考微基准**

| group | feature μs | feature+state μs | 两轮step中位μs | state bytes/12heads |
|---|---|---|---|---|
| standard_ad | 32.784000 | 64.383999 | 64.319998,64.383999 | 399360 |
| standard_exp | 26.624000 | 58.174500 | 58.176000,58.173001 | 399360 |
| matched_zero_ad | 32.848001 | 64.415999 | 64.383999,64.447999 | 399360 |
| matched_zero_exp | 26.656000 | 58.192000 | 58.208000,58.176000 | 399360 |

FP32、batch1、单层12heads、CUDA Graph热缓存；不是完整模型时间或910B测量。

**E．当前pure AD对Hedgehog/FAVOR：相同确认数据，历史匹配对照**

| split | group | KL | output NMSE | PPL |
|---|---|---|---|---|
| confirm_wiki | ad_plain | 0.347341452 | 0.220416135 | 8.858054832 |
| confirm_wiki | hh_exp | 0.377019423 | 0.231022310 | 8.870294885 |
| confirm_wiki | hh_softmax | 0.402539411 | 0.264257113 | 8.880210804 |
| confirm_wiki | favor | 4.163506329 | 1.792436921 | 9.857622151 |
| confirm_long | ad_plain | 3.422874068 | 0.641975163 | 9.691610484 |
| confirm_long | hh_exp | 2.551740747 | 0.894505332 | 9.752359089 |
| confirm_long | hh_softmax | 2.570944036 | 0.745977305 | 9.752327014 |
| confirm_long | favor | 5.902979496 | 1.652874828 | 10.644294280 |

AD73536参数/m64；HH-exp与HH-softmax73728参数/m576；FAVOR固定随机feature/m64。HH是本项目参数匹配适配版；不存在所有方案同时同参数、同state、同计算的单一比较。

| split | group | head_tv | head_coeff_l2 | layer_projected_nmse | head_output_global_nmse | layer_projected_global_nmse |
|---|---|---|---|---|---|---|
| confirm_wiki | ad_plain | 0.263031433 | 0.041484233 | 0.258934245 | 0.177327231 | 0.075436308 |
| confirm_wiki | hh_exp | 0.274934131 | 0.042883090 | 0.269638871 | 0.191248381 | 0.088897938 |
| confirm_wiki | hh_softmax | 0.290151636 | 0.051391735 | 0.302496230 | 0.189993671 | 0.089106365 |
| confirm_wiki | favor | 0.653947710 | 0.259675366 | 1.462966743 | 1.672933536 | 1.017235009 |
| confirm_long | ad_plain | 0.625747287 | 0.107785455 | 0.567697869 | 0.471711854 | 0.173465241 |
| confirm_long | hh_exp | 0.664561389 | 0.147266351 | 0.695809564 | 0.514514859 | 0.188792035 |
| confirm_long | hh_softmax | 0.678435350 | 0.153672605 | 0.610394139 | 0.448660476 | 0.185173451 |
| confirm_long | favor | 0.750985835 | 0.245490310 | 1.278359807 | 1.401633787 | 0.707747308 |

**F．严重低估尾部：8k**

| group | predicted < teacher/10000的教师质量 | 该部分正KL |
|---|---|---|
| standard_ad | 0.122264049 | 1.776189280 |
| standard_exp | 0.147321130 | 2.510289860 |
| matched_zero_ad | 0.120260041 | 1.724773326 |
| matched_zero_exp | 0.143638838 | 2.381087361 |

注意这是教师概率质量，不是token占比；正KL项不加负项不能还原完整KL。Hedgehog/FAVOR完整阈值表保存在normalized_attention_assessment的summary.tail。

**G．教师和实现审计**

| split | 重新测得PPL | 历史PPL | 差 |
|---|---|---|---|
| confirm_wiki | 8.664559867077683 | 8.664559867077683 | 0.0 |
| confirm_long | 9.066278957389708 | 9.066278957389708 | 0.0 |

最新审计passed=True，新增拟合=12，新选定=9，复用=3。

所有源文件SHA及结果SHA见对应checks/audit.json；它记录的是旧设备绝对路径，迁移后用相对目录定位，不应修改旧审计文件中的历史路径。

---

**合订来源：NEXT_CODEX_PROMPT.zh.md**

你正在接续一个已有真实实验和理论记录的研究项目。先阅读START_HERE.zh.md、AD_PROJECT_FULL_HANDOFF.zh.md、ASCEND_910B_MIGRATION.zh.md、CURRENT_RESULTS.zh.md；需要原始细节时读取SOURCE_INDEX及snapshot对应文件。不要仅凭本段摘要开始改模型。

用户明确不允许使用子agent。目标设备有两张昇腾910B，但环境尚未检查。当前已确认目标kernel为：

h(q,k)=exp(s_K(k))*softmax([z_Q(q),0])^T*softmax([z_K(k),0])。

删除的是C和Q幅度，保留K幅度。当前pure版本无bias、Q128→192→63、K128→192→64、SiLU、m64、73536参数/head；真实post-RoPE输入。不是固定分区、不是KAN、没有旋转增强或GDN门作为默认组件。

主结果是冻结Qwen2.5-1.5B两层24/336heads的feature单遍KL拟合和局部替换PPL，尚未从零预训练。最新同深度EXP控制在两种初始化、三个种子下支持8k收益，但1k PPL略差、当前参考step约慢10.7%。理论仅确认结构/梯度与适用域明确的风险关系，没有证明普遍优越或总体达界。历史Gaussian与某些kernel构造失败、GLA/GDN的strong residual控制更好，必须保存。

现在先校验交接包，读取目标设备、CANN/torch_npu/驱动及显存，整理明确的NPU移植清单。不要安装源CUDA wheel。源码中的模块级Triton CUDA导入、inference_mode scan、原地state和HF cache更新都需处理；不能把换.cuda()为.npu()当成已支持预训练。

保持snapshot和历史结果只读，在新目录进行设备无关CPU FP64 oracle、NPU FP32前向/梯度及BF16稳定性检查。先确认源检查点函数和GQA/RoPE正确，再做小规模可微训练scan、单卡和HCCL双卡烟测。不得仅复用已有JSON就报告NPU复现。

之后围绕100M–150M完整LM预训练制定具体预算和协议：AD与同深度EXP是主归因，同数据、m、参数与计算记录；加入HH/FAVOR/softmax参考。主损失为LM交叉熵，K幅度可由任务梯度直接训练，不需要Q幅度/C辅助loss。是否混合窗口、换归一化、clip幅度等都是新设计，须显式消融。目标是跨seed、新数据、长上下文与质量—计算曲线的可靠结果，稳定后再考虑350M–1B。

不需要重新问用户已明确的“删除哪一侧幅度”、是否保留原始负结果、是否允许只做文档读取和可逆迁移核查。缺失的预训练总预算或系统级操作条件，应在实际设备证据和具体可审核方案形成后再处理；不要凭猜测宣称已经授权任何付费云服务。所有新结果保存协议、源码/数据/检查点哈希、逐文档指标与数值精度记录。

---

**合订来源：SOURCE_INDEX.zh.md**

项目原文索引。每份原文保持原相对目录、原内容和SHA256。正文阶段概述不替代原始逐head/逐种子表。

| 原项目文件 | 字节 | SHA256 |
|---|---:|---|
| [candidate_validation/REPORT.zh.md](snapshot/kan_attention_theory/candidate_validation/REPORT.zh.md) | 15549 | `3468d3b0cdfa894402dd9e2f385d6f8e761de64e8b78127f7302abe51eab521b` |
| [causal_direction/PROTOCOL.zh.md](snapshot/kan_attention_theory/causal_direction/PROTOCOL.zh.md) | 3678 | `e00b12e88b58738119bff55baacad719aa4b69d91ab103190e75a15e5c55673d` |
| [causal_direction/REPORT.zh.md](snapshot/kan_attention_theory/causal_direction/REPORT.zh.md) | 12857 | `09c8a5b26034f9b2e8b4c0785e127235b6cc87658ad7d777f6de6ac3ab9a7e4c` |
| [causal_direction/REPRODUCE.zh.md](snapshot/kan_attention_theory/causal_direction/REPRODUCE.zh.md) | 2502 | `516c2a36653f3ef29e1cb52dcb475e4d1c79ff6c0d522d2fdd05facdb98d535f` |
| [causal_direction/THEORY.zh.md](snapshot/kan_attention_theory/causal_direction/THEORY.zh.md) | 12004 | `123a984745efab8f3ea6e8f6e7268d668cf9b54bcee1677023d32b36865f5008` |
| [deployment_validation/ALL_HEADS_ESTIMATE.zh.md](snapshot/kan_attention_theory/deployment_validation/ALL_HEADS_ESTIMATE.zh.md) | 6843 | `9d0cebcf49e95cc7bfbb05cc0b409846aabe58e9f3996b2b55dc7d38c1b0b8dd` |
| [deployment_validation/REPORT.zh.md](snapshot/kan_attention_theory/deployment_validation/REPORT.zh.md) | 14649 | `a63b2c50c6e04281061d6c7d8ade716d57bd7111f48934ae90f4c8dc273dd05e` |
| [distribution_operator/ALTERNATIVE_KERNELS.zh.md](snapshot/kan_attention_theory/distribution_operator/ALTERNATIVE_KERNELS.zh.md) | 11519 | `c599a03b7e38dc5e6aed4a96e9edac24f1187ecc2e7988f9bdbc5ad4075b53b6` |
| [distribution_operator/REPORT.zh.md](snapshot/kan_attention_theory/distribution_operator/REPORT.zh.md) | 17676 | `f3509ddfe789dd5c92da8dc95a7c2977c04351cf44d5140364982a1e3c316f17` |
| [distribution_operator/THEORY.zh.md](snapshot/kan_attention_theory/distribution_operator/THEORY.zh.md) | 11863 | `6b803ac3ea67a46be51fbbba766be9b93b149a09cfb213f7725b51b1c8a09188` |
| [gated_integration/PROTOCOL.zh.md](snapshot/kan_attention_theory/gated_integration/PROTOCOL.zh.md) | 5439 | `656f1bdf3f21f16c9650b4d188ebeb27ca20ed4eff1dbe092d380ff991860e40` |
| [gated_integration/REPORT.zh.md](snapshot/kan_attention_theory/gated_integration/REPORT.zh.md) | 15082 | `87aae7c40741cab98c33a87017370e39e82201d33b7d9dee5a32334cc9793aea` |
| [gated_integration/THEORY.zh.md](snapshot/kan_attention_theory/gated_integration/THEORY.zh.md) | 7007 | `49b85a034e53e658ad989cc61b6a95360fb04c046738393d01545c9a2180821d` |
| [gaussian_go_nogo/REPORT.zh.md](snapshot/kan_attention_theory/gaussian_go_nogo/REPORT.zh.md) | 13347 | `fcdc47d48c8a058372c6ae10b07dbacc871d95ba793dfe29f554af51a0f81305` |
| [hedgehog_matched/PRODUCT_PROTOCOL.zh.md](snapshot/kan_attention_theory/hedgehog_matched/PRODUCT_PROTOCOL.zh.md) | 2109 | `0b7a80fca03d301709989a87969fe757a58e37dfc3b87f562782ad36cc4a6c84` |
| [hedgehog_matched/PROTOCOL.zh.md](snapshot/kan_attention_theory/hedgehog_matched/PROTOCOL.zh.md) | 3641 | `d1d482a5d046358acc6a3ced017fd632e1a2fa77be2aa2d6c4fab8e323c75214` |
| [hedgehog_matched/REPORT.zh.md](snapshot/kan_attention_theory/hedgehog_matched/REPORT.zh.md) | 12380 | `e3d60ac93a6237d88db07f78d9ee627c0b509be7003757b2ba05f2943caf0e9e` |
| [hedgehog_matched/THEORY.zh.md](snapshot/kan_attention_theory/hedgehog_matched/THEORY.zh.md) | 2576 | `917022e4c034ac8344f02566f29f763e05b4a437ec61423a3939f671465e2e6a` |
| [kernel_comparison/REPORT.zh.md](snapshot/kan_attention_theory/kernel_comparison/REPORT.zh.md) | 17556 | `f47936f7c969d96393f4eff11eea14f4a0f978ceab6a511316f01376d044eaac` |
| [key_parameterization_attribution/PROTOCOL.zh.md](snapshot/kan_attention_theory/key_parameterization_attribution/PROTOCOL.zh.md) | 3683 | `bc9ed945214239737d7a41dd7d343ad915dd9da64986ac8b8113085f9e61262b` |
| [key_parameterization_attribution/REPORT.zh.md](snapshot/kan_attention_theory/key_parameterization_attribution/REPORT.zh.md) | 11921 | `7109eb5e2e8fc9f1d6184600f5f9c544d222b5b261ee2a7fdf7473fff7a5ba3d` |
| [mlp_direction/ALL_HEADS_ESTIMATE.zh.md](snapshot/kan_attention_theory/mlp_direction/ALL_HEADS_ESTIMATE.zh.md) | 5045 | `601efad87ab71f568956bf9583a6becfe80bc882d8e9968a1c725d6e0f887f89` |
| [mlp_direction/AMPLITUDE_DIRECTION_FOCUS.zh.md](snapshot/kan_attention_theory/mlp_direction/AMPLITUDE_DIRECTION_FOCUS.zh.md) | 10494 | `0c1378eb798b3921999a7ceaf8e27cb137d4817c20cb3844eb9f10c050f4adbd` |
| [mlp_direction/PROTOCOL.zh.md](snapshot/kan_attention_theory/mlp_direction/PROTOCOL.zh.md) | 4901 | `919ab4212b98e37edb91348a0eb9b6704409bd2981cb3edee60e0022e25c0064` |
| [mlp_direction/REPORT.zh.md](snapshot/kan_attention_theory/mlp_direction/REPORT.zh.md) | 14281 | `5ab3b1eddbf43f39eb8924ded56da7ed66c36f94d94a160b0c29087bb601d036` |
| [mlp_direction/REPRODUCE.zh.md](snapshot/kan_attention_theory/mlp_direction/REPRODUCE.zh.md) | 2309 | `7fb552fdeba23d6fcf69bfe1e93367f8503f6f917e1f8cd67c1aaaaafc3e259e` |
| [mlp_direction/THEORY.zh.md](snapshot/kan_attention_theory/mlp_direction/THEORY.zh.md) | 14560 | `27d33914feac29765acaa89428228ab6edfc077a18edcd01d42002f4dae9d15e` |
| [normalized_attention_assessment/PROTOCOL.zh.md](snapshot/kan_attention_theory/normalized_attention_assessment/PROTOCOL.zh.md) | 1502 | `1e6c5fab22161015d63aff9589b119620c35f809206f472df0d7266318522bb9` |
| [normalized_attention_assessment/REPORT.zh.md](snapshot/kan_attention_theory/normalized_attention_assessment/REPORT.zh.md) | 9740 | `8563eb141858370b80c3522143d751eedee5151d785116ae1c391868a54ec446` |
| [normalized_kernel_novelty_20260908/REPORT.zh.md](snapshot/kan_attention_theory/normalized_kernel_novelty_20260908/REPORT.zh.md) | 9407 | `083a5484db1ce1b6e406125602e8d3804565a0ab9465f8a7e93f73c28c5fbbfa` |
| [novelty_review_20260908/REPORT.zh.md](snapshot/kan_attention_theory/novelty_review_20260908/REPORT.zh.md) | 7920 | `c40a8ba1e7309de510a7d397e63360ea91bd158bdb43bf2d582825e8e240e0cb` |
| [orbit_direction/PROTOCOL.zh.md](snapshot/kan_attention_theory/orbit_direction/PROTOCOL.zh.md) | 1691 | `5364660c9d4ebb93b46a37f11f3703c7d829622b7332baedbd6245b69b4fe1b8` |
| [orbit_direction/REPORT.zh.md](snapshot/kan_attention_theory/orbit_direction/REPORT.zh.md) | 6786 | `986425f6022a25541b062c05c563779530fe08efc4d08b38b4ec7ad02e4143d5` |
| [positive_generalization/REPORT.zh.md](snapshot/kan_attention_theory/positive_generalization/REPORT.zh.md) | 16245 | `3f75b8932c96054fd83b148412fd2c7188323afba798b99c3d874275960f9d2e` |
| [query_amplitude_ablation/PROTOCOL.zh.md](snapshot/kan_attention_theory/query_amplitude_ablation/PROTOCOL.zh.md) | 4104 | `6ebcc9c0fbf2c33bd32fb7493bd2275555490ba9a05d11cce12f02bc4ef96e73` |
| [query_amplitude_ablation/REPORT.zh.md](snapshot/kan_attention_theory/query_amplitude_ablation/REPORT.zh.md) | 12881 | `bdfa3109a3c33c74945ad7c204143a8cf571ac60a9d24fde7f3d7d83e3d3917d` |
| [real_llm_pilot/AUXILIARY_REPORT.zh.md](snapshot/kan_attention_theory/real_llm_pilot/AUXILIARY_REPORT.zh.md) | 19799 | `80b8bfb17997303e5565336c299d2be086d9b34fb06c69343c09b24191d46aaa` |
| [real_llm_pilot/RAW_REPORT.zh.md](snapshot/kan_attention_theory/real_llm_pilot/RAW_REPORT.zh.md) | 19216 | `078d72570a0bce4690af33768bae43457495638e310a7debd5e5a18cd3f36215` |
| [single_pass_mulkan/REPORT.zh.md](snapshot/kan_attention_theory/single_pass_mulkan/REPORT.zh.md) | 19268 | `f101958f8ba3a7e5e47ca76df0d305a79b6002f7f2665dd9162c02912c6f2056` |
| [single_pass_mulkan/diagnostics/REPORT.zh.md](snapshot/kan_attention_theory/single_pass_mulkan/diagnostics/REPORT.zh.md) | 11972 | `cd907ea615b11a20af083f24ad8f2d20d8cb23749b91b61310441643b27474ab` |

---

**合订来源：HISTORICAL_REPORTS_FULL.zh.md**

历史报告、协议与理论全文备份。以下按相对路径排序，不将文件顺序解释为实验时间。已被后续纠正的旧结论仍原样保存，当前有效结论以主交接文档为准。

本合订本中的历史相对链接仍以各段标注的原文件目录为基准，直接点击可能无法定位；请从[SOURCE_INDEX.zh.md](SOURCE_INDEX.zh.md)打开保留原目录的独立文件。大型外部资产未全部附带，存在于原文的链接不代表对应资产已打包。独立快照的原文保持逐字节一致，新增解释只位于本合订本开头。


---

原文件：candidate_validation/REPORT.zh.md

本轮结论：已在冻结 Qwen2.5-1.5B 的真实 Q/K 上验证连续积分、连续非负投影、Nyström 谱截断和谱模态成对正化，并与原分区、Galerkin、FAVOR+、ADERF、key-VQ 及等参数 MLP/KAN/mulKAN 正特征比较。保留 key 特征幅度改善了部分 head 的原始 kernel 拟合，但新连续非负投影没有建立优于普通 MLP 正特征的模型困惑度与推理效率综合优势；谱正化的当前全局包络构造失败。“近总体理论下界”仍未成立。

**范围与训练。** 固定模型 revision 8faed761d45a263340a0528343f099c05c9a4323；L14H0、L14H6、L27H0、L27H6，d=128，主比较 m=64。复用此前真实提取及冻结检查点，无新增 LLM 微调，无新增神经训练。MLP/KAN/mulKAN 每对 Q/K 网络均为 73,856 个活跃参数，4096 个训练文档、每 head 2,097,152 对样本，仅一遍 Poisson/广义 KL 训练，种子 11/29/47 全部报告。这些旧网络训练配对是同文档合法 Q/K；本轮 kernel 评估则为边缘乘积分布，两者不同，不将旧训练目标误称为总体乘积分布风险优化。

**新构造的实际形式。** 连续基 b(k) 取 64 个校准 query 锚点的 softmax 响应，或既有 k-network 输出的连续正特征再归一化。后者各架构固定 seed 11，用于比较基函数形状。对每个 moment-fit 文档取一个 key，共 2048 个节点，h(q)=2048⁻¹Σ_i κ(q,k_i)b(k_i)。积分候选使用 c=h/p；非负投影候选对训练 Gram 的二次子问题做 128 次加速投影梯度迭代。最终 κ̂=c(q)ᵀb(k)。没有把目标改成归一化 attention；二次子问题是谱投影的数值构造，不是神经 MSE 训练。有限节点积分不是未知总体条件期望。

Nyström 使用现有 1024×1024 正训练锚点 kernel 的前 64 个模态。谱正化保留 32 个模态，63 个槽位补零到 64；其包络由正锚点加权平均推导，对固定展开的所有输入成立，不用测试点估计。它不是真实总体的正交 Schmidt 函数，因此实际风险由留出数据测量，不能套用理想 E_r*+δ² 等式。

**追加的基函数消融。** 一般非负锥投影不要求 Σb_s(k)=1，所以追加 cone_raw_mlp/kan/mulkan，直接使用未经分量归一化的原非负 key 特征，保留幅度。全部三种架构固定 seed 11、2048 个节点与 128 次投影，全部评估，没有据测试分数挑选其中一种。它是首轮筛选后的探索性消融；复用了同一组留出文档，不应称为全新独立盲测。两个版本的目标都一直是原始 kernel。

**比较口径。** 新旧全部 33 个设置在验证、训练诊断、内部测试的 8192×8192 经验乘积及官方 10240×10240 经验乘积上评估。内部 8192 个样本覆盖全部 256 篇文档；官方覆盖原有全部 20 篇固定测试文档。随后将表中连续候选及神经对照复核到内部全部 131072×131072 对/每 head。FAVOR+/ADERF 五种子和原分区/Galerkin 的全量数据结果来自相同冻结数据上此前的完整积分，没有混用 8192 结果。全部 kernel 平方风险 FP64；广义 KL 与 log 误差额外使用固定的 262,144 对独立索引抽样，不能把抽样 KL 说成全矩阵精确积分。

两种完整模型候选在独立的 128 篇 validation 文档上，按四个 head 的平均 raw generalized-KL/target-mass 预先选择，为 cone_mulkan 和 cone_mlp。全部测试结果均保留；没有按官方或内部测试挑选候选/种子。

**原始 kernel 的全量内部经验风险。** 下表是 E[(κ−κ̂)²]/E[κ²]，所有行均对应相同的 131072×131072 经验乘积。FAVOR+/ADERF 为五种子中位数。

| 方法 | L14H0 | L14H6 | L27H0 | L27H6 |
| --- | --- | --- | --- | --- |
| 原分区正 kernel | 0.2065 | 0.4677 | 0.7842 | 0.6608 |
| 训练 Galerkin（有符号） | 0.1717 | 0.4776 | 0.5123 | 0.3181 |
| 连续非负投影 / mulKAN 基 | 0.3206 | 0.4690 | 0.8134 | 0.7639 |
| 连续非负投影 / MLP 基 | 0.3526 | 0.4675 | 0.8492 | 0.7788 |
| 连续非负投影 / KAN 基 | 0.3412 | 0.4676 | 0.8610 | 0.7579 |
| 非负投影 / 保留幅度 MLP 基 | 0.3374 | 0.4657 | 0.7715 | 0.6226 |
| 非负投影 / 保留幅度 KAN 基 | 0.3397 | 0.4669 | 0.8039 | 0.6796 |
| 非负投影 / 保留幅度 mulKAN 基 | 0.3277 | 0.4668 | 0.7736 | 0.5857 |
| 连续积分 / MLP 基 | 0.4576 | 0.5678 | 0.8918 | 0.8536 |
| 连续积分 / mulKAN 基 | 0.3391 | 0.7688 | 0.8447 | 0.8537 |
| MLP 正特征（seed 11） | 0.3670 | 0.5210 | 0.9045 | 0.8447 |
| KAN 正特征（seed 11） | 0.3748 | 0.5231 | 0.8719 | 0.8233 |
| mulKAN 正特征（seed 11） | 0.3593 | 0.5208 | 0.8640 | 0.7650 |
| key-VQ kernel 对照 | 0.9630 | 0.9970 | 0.9220 | 0.8655 |
| Nyström 谱截断（有符号） | 1.5310 | 0.4871 | 0.7320 | 0.4017 |
| FAVOR+ m64 | 1.0000 | 3.8765 | 1.0045 | 1.0000 |
| centered_favor_plus | 1.6251 | 50.5762 | 767.7480 | 224.8377 |
| sderf | 22.0497 | 308.6736 | 252.5649 | 804.3158 |
| aderf | 9214.5833 | 45.1858 | 53.3313 | 5.2238 |

**官方测试的失败与泛化检查。** 下表采用全部 10240×10240 经验乘积，不是某个小块矩阵上的重新拟合。

| 方法 | L14H0 | L14H6 | L27H0 | L27H6 |
| --- | --- | --- | --- | --- |
| 谱模态成对正化 | 1.52e+16 | 3.85e+13 | 1.84e+15 | 2.88e+13 |
| 连续积分 / 锚点基 | 0.9808 | 0.9988 | 0.9192 | 0.9147 |
| 连续非负投影 / 锚点基 | 0.9196 | 0.9934 | 0.8586 | 0.8367 |
| 连续非负投影 / mulKAN 基 | 0.3992 | 0.9457 | 0.6752 | 0.7192 |
| 连续非负投影 / MLP 基 | 0.4305 | 0.9452 | 0.7956 | 0.7124 |
| 非负投影 / 保留幅度 MLP 基 | 0.4226 | 0.9450 | 0.6702 | 0.5067 |
| 非负投影 / 保留幅度 KAN 基 | 0.4247 | 0.9453 | 0.7096 | 0.5729 |
| 非负投影 / 保留幅度 mulKAN 基 | 0.4128 | 0.9453 | 0.6406 | 0.4442 |
| 原分区正 kernel | 0.2541 | 0.9456 | 0.6879 | 0.5987 |
| MLP 正特征（seed 11） | 0.4491 | 0.9498 | 0.8414 | 0.8380 |

成对正化的第一模态增量系数 δ 分别为 1.626e+08, 2.665e+07, 6.874e+07, 7.846e+06。这证明该可部署包络构造的偏置不可接受；它不否定所有可能的谱正化或正特征方法。连续积分/非负投影的结果则说明：取消硬分区本身不足以得到更好的基函数。

**完整模型困惑度，只替换 4/336 个 Q head。** 其余模型冻结，所有因果位置均重新计算，内部 256 篇与官方 20 篇各取 1024-token 前缀。官方是固定子集，非标准拼接 WikiText2 benchmark。

| 方法 | 内部 PPL | 官方子集 PPL |
| --- | --- | --- |
| 原模型 | 8.613259 | 8.915608 |
| 原分区正 kernel | 8.689070 | 9.034617 |
| 训练 Galerkin（有符号） | 8.709132 | 9.048217 |
| 连续非负投影 / mulKAN 基 | 8.681544 | 9.027188 |
| 连续非负投影 / MLP 基 | 8.695035 | 9.027527 |
| 非负投影 / 保留幅度 MLP 基 | 8.676708 | 9.020275 |
| 非负投影 / 保留幅度 KAN 基 | 8.681595 | 9.031753 |
| 非负投影 / 保留幅度 mulKAN 基 | 8.680124 | 9.031442 |
| key-VQ kernel 对照 | 8.747594 | 9.071117 |
| MLP 正特征（三种子） | 8.669130 [8.667563, 8.670232] | 9.006821 [9.002877, 9.006879] |
| 两层 KAN 正特征（三种子） | 8.676051 [8.675416, 8.676433] | 9.018936 [9.017791, 9.022684] |
| 两层 mulKAN 正特征（三种子） | 8.676432 [8.675957, 8.677579] | 9.018853 [9.014835, 9.022341] |
| FAVOR+ m64 | 8.747917 [8.736948, 8.770964] | 9.101324 [9.078257, 9.110611] |
| FAVOR+ m640 | 8.742300 | 9.077467 |

神经特征行为本轮三个固定种子的中位数 [最小,最大]；FAVOR+ 为五种子。单独列出的连续投影候选使用 seed 11 的基函数。所有新 PPL 测试的非正分母和非有限 attention 行数：0 / 0。总体 PPL 的小差异不能外推到全部 head 替换。

**数值与表达误差。** 27 组真实前缀的显式因果 kernel 与分块/递推实现核验，最大 FP64 相对 L2 为 3.735e-16。FP32 与 FP64 的最大特征差异为 1.217e-03，主要来自病态非负投影；不能把投影器声称为精确 NNLS。

训练 Gram 的条件数中，MLP 基可达约 2×10^8，KAN 基约 2×10^10，mulKAN 的 L14H0 在相对 10^-10 阈值下只有 40 个有效方向。对 32 个未见 query 再做 2048 次投影迭代，训练积分节点上的平方风险最多进一步降低 2.10%。该诊断不改变部署参数，也不证明 128 次迭代达到总体最优。

报告中的 span_risk 是在留出经验测度上诊断函数空间可覆盖的能量，使用相对 10^-10 的 Gram 谱截断。coefficient_excess 是总风险减该诊断风险的数值，包含非负约束、训练积分估计与数值求解影响；由于还做了谱截断，不将它声称为全 64 维空间的精确正交分解，更不是总体正特征误差下界。

**与同一经验算子的谱界比较。** 用新候选风险除以既有 rank-64 最优风险的上界，得到相对无约束最优解的确定经验差距下限。不是正特征最优值或未知总体的证书。

| 方法 | 内部：各 head 至少多少倍 | 官方：各 head 至少多少倍 |
| --- | --- | --- |
| 连续非负投影 / mulKAN 基 | 14.51×, 5.24×, 11.01×, 13.49× | 18.84×, 249.81×, 30.12×, 104.28× |
| 连续非负投影 / MLP 基 | 15.96×, 5.22×, 11.49×, 13.76× | 20.31×, 249.69×, 35.49×, 103.28× |
| 非负投影 / 保留幅度 MLP 基 | 15.27×, 5.20×, 10.44×, 11.00× | 19.94×, 249.62×, 29.89×, 73.47× |
| 非负投影 / 保留幅度 KAN 基 | 15.37×, 5.22×, 10.88×, 12.00× | 20.04×, 249.71×, 31.65×, 83.06× |
| 非负投影 / 保留幅度 mulKAN 基 | 14.83×, 5.22×, 10.47×, 10.35× | 19.48×, 249.71×, 28.57×, 64.40× |

**计算量。** 以下为每 head、每对 Q/K 特征的主导稠密乘加 FLOPs，乘加计 2；不含 exp/softmax/spline 基函数、比较、索引、元素运算和共同线性状态聚合。KAN/mulKAN 与 MLP 的 GEMM 参数量相同，但 spline/乘法节点还有额外开销。

| 方法 | 主导 FLOPs | 相对 FAVOR+ m64 |
| --- | --- | --- |
| FAVOR+ m64 | 32,768 | 1.00× |
| MLP/KAN/mulKAN 正特征 | 147,456 | 4.50× |
| 原分区/Galerkin/Nyström m64 | 786,432 | 24.00× |
| 连续积分 / 神经基 | 860,160 | 26.25× |
| 连续非负投影 / 神经基，128 步 | 1,908,736 | 58.25× |
| key-VQ kernel | 32,768 | 1.00× |

非负投影还有 2048 个真实 key 节点和节点权重等静态常数；m 相同不等于总参数、静态内存或计算量相同。三种神经架构的直接正特征对照严格保持相同网络参数量；三种神经基的积分/投影版本也使用相同基网络参数预算和节点数。FAVOR+ m640 是沿用的补充对照，仍明显低于当前 128 步投影的算术成本，不称为本轮严格同 FLOPs 对照；本轮也没有宣称候选在同 FLOPs 下获胜。

**实际端到端时间。** RTX 3090、BF16 模型、FP32 特征/状态，batch=1，TF32 关闭，无并行 GPU 工作负载。沿用原协议：prefill 3 次预热 + 7 次；decode 1 次预热 + 5 次，每次固定 32 token。下表为 8192-token 提示词；计时长提示词由真实文章拼接，不构成长上下文质量验证。

| 方法 | Prefill ms | Decode ms/token | 原 KV MiB | 额外线性状态 KiB |
| --- | --- | --- | --- | --- |
| 原模型 | 471.67 | 21.862 | 224.0 | 0.0 |
| 原分区正 kernel | 475.77 | 23.136 | 224.0 | 129.0 |
| FAVOR+ m64 | 473.49 | 23.472 | 224.0 | 129.0 |
| 连续非负投影 / mulKAN 基 | 513.93 | 52.990 | 224.0 | 129.0 |
| 连续非负投影 / MLP 基 | 501.22 | 51.036 | 224.0 | 129.0 |
| key-VQ kernel 对照 | 472.36 | 23.910 | 224.0 | 129.0 |
| MLP 正特征（seed 11） | 474.00 | 23.993 | 224.0 | 129.0 |
| KAN 正特征（seed 11） | 498.94 | 26.644 | 224.0 | 129.0 |
| mulKAN 正特征（seed 11） | 498.84 | 26.986 | 224.0 | 129.0 |
| 非负投影 / 保留幅度 MLP 基 | 501.38 | 50.153 | 224.0 | 129.0 |
| 非负投影 / 保留幅度 KAN 基 | 513.61 | 51.554 | 224.0 | 129.0 |
| 非负投影 / 保留幅度 mulKAN 基 | 513.58 | 52.017 | 224.0 | 129.0 |

所有原 KV 仍由未替换的 GQA query heads 使用。局部替换只能增加线性状态，不能移除共享 KV；因此本轮没有全模型 cache 压缩结论。这是 HF eager 原型计时，不能称为各算法最佳算子性能。

**特征耗时。** 同一 PyTorch 原型、双方均用 CUDA Graph，4 heads 的 Q/K 特征对，单位 μs，20 次中位数。单独展示，避免把特征时间和整模型时间混为一谈。

| 方法 | N=1 μs | N=2048 μs |
| --- | --- | --- |
| favor_1009 | 33.79 | 137.22 |
| favor_640 | 34.82 | 624.13 |
| 原分区正 kernel | 58.37 | 1112.02 |
| 训练 Galerkin（有符号） | 33.25 | 1011.20 |
| 连续积分 / MLP 基 | 41.98 | 1098.24 |
| 连续非负投影 / MLP 基 | 1224.70 | 6581.25 |
| 连续非负投影 / mulKAN 基 | 1298.43 | 9721.86 |
| 非负投影 / 保留幅度 MLP 基 | 1392.64 | 6612.43 |
| 非负投影 / 保留幅度 KAN 基 | 1299.46 | 9805.28 |
| 非负投影 / 保留幅度 mulKAN 基 | 1307.65 | 9720.32 |
| MLP 正特征（seed 11） | 47.10 | 261.60 |
| KAN 正特征（seed 11） | 180.19 | 6617.09 |
| mulKAN 正特征（seed 11） | 196.61 | 6440.90 |
| key-VQ kernel 对照 | 27.65 | 113.66 |

这里没有新写或调优融合算子。此前专门优化的 FAVOR+ N=2048 为约 67.58 μs，可作为进一步的效率参照；不同实现的数字不能当作本轮等优化程度对照。

**FP64 特征/状态补充检查。** 同一 BF16 模型，每个候选在每个数据集前 8 篇文档比较，表中差值为 FP64−FP32 NLL；不是重新训练。

| 候选 | 数据集 | 平均 NLL 差 | 最大逐文档绝对差 |
| --- | --- | --- | --- |
| cone_mulkan | internal | -0.0005241 | 0.0017765 |
| cone_mulkan | official | 0.0000677 | 0.0023944 |
| cone_mlp | internal | -0.0000490 | 0.0025990 |
| cone_mlp | official | -0.0003777 | 0.0032206 |
| cone_raw_mlp | internal | 0.0000611 | 0.0024729 |
| cone_raw_mlp | official | -0.0001037 | 0.0017955 |
| cone_raw_kan | internal | -0.0006908 | 0.0027101 |
| cone_raw_kan | official | -0.0010629 | 0.0035551 |
| cone_raw_mulkan | internal | -0.0003604 | 0.0029204 |
| cone_raw_mulkan | official | 0.0011214 | 0.0023299 |

**理论主张与 Go/No-Go。** 一般 Schmidt 误差下界仍是适用条件下的数学事实；本轮没有验证 covariance-only Gaussian 谱预测，也没有得到未知总体的近最优证书。经验跨度增大时的风险变化和官方 L14H6 的大误差，说明少量文档/尾部及训练函数空间的泛化仍是核心困难。当前结果支持停止把这版成对正化作为实用方案；也不支持把当前连续积分/128 步非负投影作为优于普通 MLP 的新方法。函数空间如何低成本覆盖有效谱，仍可作为研究问题，但需要新的构造或实质证据。

FAVOR+ 与 ADERF 为既有正随机特征公式复现；key-VQ 是用校准 key k-means 形成的 kernel 级对照，没有复现端到端训练的完整 Transformer-VQ。[Performer](https://arxiv.org/abs/2009.14794)、[FAVOR#/DERF](https://proceedings.neurips.cc/paper_files/paper/2023/file/02dec8877fb7c6aa9a79f81661baca7c-Paper-Conference.pdf)、[Transformer-VQ](https://arxiv.org/abs/2309.16354)。当前理论推导和候选定义见同项目 distribution_operator/ALTERNATIVE_KERNELS.zh.md。

复现脚本：prepare.py、evaluate.py、verify.py、choose_candidates.py、ppl.py、benchmark_full.py、benchmark_features.py、raw_basis.py、report.py。GPU 阶段顺序执行；原始结果、检查和文档 bootstrap 在 results/ 与 checks/。



---

原文件：causal_direction/PROTOCOL.zh.md

本轮继续解决三项明确缺口：与原始非MSE风险相容的非零谱界；完整因果位置上的匹配强对照；能够真正释放KV cache的部署。

模型仍冻结Qwen2.5-1.5B。为了使缓存可以释放，扩展原L14/L27到这两层完整24个Q heads、4个GQA KV组；不是全模型替换。训练原4096文档各取64个不同Q，覆盖所有1024位置和合法前缀K，每个选中Q/K配对一遍。验证128篇；新的确认集128篇1024-token Wikipedia、24篇8192-token Wikipedia、128条64-token LAMBADA文学提示，全部在方法选择前固定。LAMBADA是前缀PPL，不冒充标准最后词准确率或按原书独立抽样。

主参数预算两侧128→192→64，每head73856参数。预先比较factorized_both的质量项系数λ=1、0.1、0（后者为同架构KL控制），exp-MLP KL、softplus-MLP KL；若需额外候选必须标明探索并使用新的确认集。主方法λ>0仍以原始κ为唯一无约束目标，λ=0明确为归一化蒸馏控制。全程不使用MSE作神经训练损失。方法采用同一批Q/K和文档顺序、同样的优化设置。validation用于λ方案/部署选择，确认集不调参。

基线同时保留FAVOR+以及论文明确给出的特征公式；原论文流程未完整复现时必须明确标注。训练参数预算和推理状态预算分别报告，不能用不同预算的胜利代替匹配对照。

理论尝试通过固定Q/K可测分组对kernel诱导联合分布粗粒化，由数据处理、Pinsker和核范数最佳秩m逼近，建立I损失的谱尾下界。分组仅用于证书/诊断，实际kernel仍是连续特征。分别报告总体有效的不等式、有限经验分布的数值证书、未知总体的统计不确定性；若严格置信下界为零必须保留零，不能用点估计冒充总体证书。

部署采用真正的线性状态，并在被完整替换的两层跳过KV缓存。必须验证显式因果计算、prefill+decode一致性、存储字节和完整模型PPL；吞吐比较包含原模型、同预算KL和FAVOR+，按当前实现报告。任何局部精确窗口或精确异常项若加入都作为单独混合对照，对所有方法使用相同预算，并明确其已经不是纯rank-m kernel。

确认数据可用性修订：缓存LAMBADA仅20条达到128 tokens，因此在任何模型评价前将文学提示长度改为64 tokens，保持128条；它主要检验短上下文迁移，不承担长上下文证据。

确认评价前追加的构造性检验：固定64-token精确近邻窗口，远处使用可分离kernel；所有参评方法使用相同窗口/状态预算。它作为独立混合模型对照，不声称整张mask矩阵rank64。在冻结exp-KL方向上，比较仅head常数校准、同预算查询幅度校准，以及允许学习分支质量的控制。校准Q取同训练文档中未参与第一阶段的64个位置，配对不重复；两阶段共用文档但不假装总训练计算相同。新增查询读出、全局标度对照共用这份校准数据。它检验raw质量校准是否在混合精确/近似分支时具有实际效用，而非给纯线性attention制造不可能的标度收益。

确认指标读取前的文献驱动补充：2026-08-28预印本《Sliding-window beats linear attention》强调必须比较无需训练的窗口＋sink。加入固定448 recent＋4 sink基线，其两层BF16缓存925696 bytes，与rank64＋64窗口的929792 bytes相差0.44%，不调参、不训练。它不是kernel拟合方法，只作为同持久缓存预算的完整模型实用对照。LoLCATs已经使用近邻精确＋远处线性和共享分母，本轮不以混合架构本身主张新颖性。



---

原文件：causal_direction/REPORT.zh.md

本轮把三项要求分别落实为有条件的理论、严格匹配控制和真实状态部署。结果是：**固定缓存节省已得到验证；主方法没有稳定胜过强KL控制；接近未知总体下界的主张仍缺乏证据。** 共同RoPE旋转揭示了一个明确的表示泛化缺口，后续增强实验独立保存在 [orbit_direction](../orbit_direction/PROTOCOL.zh.md)。

**范围与公平性。** 冻结Qwen2.5-1.5B，revision `8faed761d45a263340a0528343f099c05c9a4323`。为了释放GQA缓存，将原来L14/L27的少数head扩展到这两层全部24个Q heads、4个KV组。其余26层保留原始attention；完整模型PPL指整个冻结模型输出，不代表已经全模型线性化。训练4096文档，各64个不同Q、全部合法前缀K，共每head 134,360,517 个配对，4096次单遍更新。比此前只用后半Q与前半K，覆盖了自注意力、近邻和所有因果位置。

主MLP两侧128→192→64，73856参数/head；λ=1、.1、0及exp/softplus-KL均匹配参数、数据、顺序和优化预算。λ=.1在验证集输出误差上预先选为主方案，所有其他结果保留。FAVOR+、Hedgehog公式、learned PRF都使用m64，后两者参数分别4128、8256/head，属于论文公式控制与共同单遍预算实验，未完整复现或充分调优原论文训练流程，不能将胜过这些控制等同于胜过论文SOTA。校准额外使用同4096文档中不重叠的64个Q，仅优化193参数的幅度读出并折叠回原参数预算；数据配对不重复，文档被再次使用，额外训练计算已明示。

**一、理论收紧后的可用结论。**

真实模型在投影前使用RMSNorm，投影权重有限，默认RoPE为正交变换。因此真实Q/K范数有界，原始κ的L²存在，Schmidt谱尾定理有严格适用基础。它不要求真实分布是Gaussian。

相反，以本轮全部缓存训练Q/K协方差构造的Gaussian替代模型，**0/24 heads**满足a_max<1/2；a_max范围0.552～1.847。该最终检查使用每head全部262144个Q、每KV组全部4194304个K，初始按位置抽样的结果已另外归档。这否定了直接在这些head上使用有限Gaussian二阶谱公式的前提，不能解释为真实kernel二阶矩发散。RMSNorm给出的保守logκ上界仍达2.25万～6.30万，数学上的有限性不会自动提供有用的统计界。

对固定乘积参考分布，令C为原始kernel诱导概率的可测粗粒化矩阵。数据处理、Pinsker与最佳核范数rank-m逼近给出原始I风险的合法下界：

$$ R_I(h)/\mathbb E\kappa\ge\tfrac12\big(\sum_{r>m}\sigma_r(C)\big)^2. $$

对query-balanced目标也有相同形式。它用整个选定经验边缘的完整乘积积分，实际MLP没有查表分区；但固定key bank与未知P_K之间的差异仍未被置信区间覆盖。

因果场景另用合法区域的互不重叠矩形分解。令w_b为矩形真实概率质量，C_b为其中条件概率矩阵，则所有m维正可分离kernel满足

$$\mathbb E_C\mathcal L_\lambda(C)\ge\mathbb E_C B_m(C),\qquad B_m(C)=\tfrac12\sum_b w_b\big(\sum_{r>m}\sigma_r(C_b)\big)^2.$$

这修正了把三角mask矩阵直接当rank-m矩阵的错误。完整证明、原始风险与方向／质量分解、有限样本条件见 [THEORY.zh.md](THEORY.zh.md)。

实际128篇新Wiki的前512位置，m64因果下界按head平均后范围2.89e-09～0.000132，中位数3.94e-06。乘积分布下的粗粒化界也很小。**两类界加入所列保守iid文档置信修正后均为0。** 因此不能声称未知总体最小误差已数值确定，也不能声称新kernel接近该最小误差；一个很松的下界不允许反过来证明方法离真实最优很远。原始无约束rank-m、正特征rank-m、固定宽度MLP三种最优值仍需区分。

**二、拟合的仍是原始kernel，长上下文质量尚未守住。**

主方法数学形式保持为连续正特征：

$$h_{orig}(q,k)=e^{s_h}64e^{s_Q(q)+s_K(k)}\pi_Q(q)^\top\pi_K(k),\quad\pi_X=\operatorname{softmax}([z_X,0]).$$

s_h是训练数据确定的固定head数值标度，可吸收到两侧feature中。它不是每行attention归一化。λ>0的目标为方向KL加λ倍原始质量I项，零损失目标仍是κ；λ=0明确是强KL控制。MSE只在测试中作为诊断，未用于神经训练。

三种子、24heads平均；原始NMSE先按head汇总该集合平方误差／平方能量，再平均head与种子：

| 方法 | Wiki输出NMSE | 长文输出NMSE | Wiki原始kernel NMSE | 长文原始kernel NMSE |
|---|---:|---:|---:|---:|
| split_1 | 0.260565 | 0.604232 | 0.7653 | 1.058e+08 |
| split_01 | 0.228882 | 0.626856 | 0.7782 | 1.321e+17 |
| split_kl | 0.221459 | 0.639990 | 4.097e+18 | 3.455e+23 |
| exp_kl | 0.221352 | 0.703638 | 6.857e+23 | 1.277e+43 |
| softplus_kl | 0.223945 | 0.643441 | 1.004 | 1.006 |
| calibrated | 0.221352 | 0.703638 | 27.03 | 1.757e+29 |
| gate | 0.221352 | 0.703638 | 135.7 | 8.916e+29 |
| favor | 1.792437 | 1.652875 | 1.062 | 1.022 |


原始质量失真和归一化输出失真不能混为一谈。特别是长上下文中的巨大原始误差，一部分来自幅度外推，attention分母会消去query共同标度，所以归一化结果看起来没有同样严重。查询校准保持纯线性方向不变；它相对于原始未校准KL有明显幅度作用，但不能以此主张更好的纯线性attention。完整kernel表和逐文档数据见 [summary.json](results/summary.json)，图见 [kernel_generalization.pdf](figures/kernel_generalization.pdf)。

**三、完整模型PPL与真正的强控制。**

确认集为128篇1024-token Wiki、24篇8192-token新长文、128条64-token文学前缀，排除既有训练/验证/确认文本和token哈希。下表为三训练种子PPL均值，原模型与SWA无训练种子：

| 方法 | Wiki1024 PPL | Wiki8192 PPL | 文学64 PPL |
|---|---:|---:|---:|
| 原模型 | 8.664560 | 9.066279 | 33.434616 |
| FAVOR+ | 9.857622 | 10.644294 | 35.348010 |
| Hedgehog公式控制 | 8.990198 | 9.718644 | 33.964945 |
| 可学习PRF控制 | 9.113363 | 9.981773 | 34.197194 |
| 同架构KL | 8.864630 | 9.691744 | 33.738680 |
| exp-MLP KL | 8.849618 | 9.736733 | 33.749442 |
| softplus-MLP KL | 8.862422 | 9.644074 | 33.741978 |
| 原始质量+KL，λ=.1 | 8.875796 | 9.683294 | 33.765944 |
| λ=.1 + 精确窗口64 | 8.798212 | 9.577636 | 33.450647 |
| softplus-KL + 精确窗口64 | 8.951833 | 9.568173 | 33.450647 |
| 原始质量校准 + 窗口64 | 8.792923 | 9.647021 | 33.450647 |
| 匹配门控KL + 窗口64 | 8.788346 | 9.648020 | 33.450647 |
| SWA448 + 4 sinks | 8.727983 | 9.365205 | 33.430035 |


文学64在混合版本中完全位于精确窗口内，不能用它证明学到了远处泛化。与原模型的细小差异来自BF16／FP32 attention实现的积累次序；所有混合feature在这个集合给出相同结果。它也不是标准LAMBADA最后词准确率。

同架构KL、softplus-KL和同读出预算的门控KL不可省略。原始质量校准与门控KL在长文的差异区间跨0，在短文则略逊于门控KL；不支持“原始质量校准独有的输出收益”。近邻混合优于某些纯线性方案，但SWA448+4的持久缓存与混合方案只差4096 bytes（约0.44%选中层预算），且本轮PPL更好。

| 比较A−B，负值有利A | 集合 | ΔNLL/token | 配对文档95%区间 |
|---|---|---:|---|
| pure/split_01 − pure/split_kl | 8192 | -0.000872 | [-0.001912, 0.000249] |
| pure/split_01 − pure/exp_kl | 8192 | -0.005502 | [-0.006764, -0.004152] |
| pure/split_01 − pure/softplus_kl | 8192 | 0.004059 | [0.002589, 0.005730] |
| hybrid/calibrated − hybrid/gate | 8192 | -0.000104 | [-0.000441, 0.000234] |
| hybrid/split_01 − hybrid/softplus_kl | 8192 | 0.000989 | [-0.000818, 0.002830] |
| hybrid/split_01 − window/swa448_sink4 | 8192 | 0.022429 | [0.019043, 0.025712] |
| hybrid/calibrated − window/swa448_sink4 | 8192 | 0.029646 | [0.025916, 0.033120] |


区间对文档做配对bootstrap，条件于这三个固定训练种子；另外保留每种子的ΔNLL。不能把24个head或重复token当独立模型样本，也不能由三个种子外推完整训练随机性分布。混合softplus-KL的完整模型结果是在其已冻结kernel的首批指标后追加的强控制，未改其训练；因果矩形与旋转测试也明确标记为事后机制诊断。

[LoLCATs](https://arxiv.org/html/2410.10254v2)已使用精确窗口＋远处线性、共同分母和相关算子优化，混合结构本身不构成新颖贡献。[2026-08-28的SWA预印本](https://arxiv.org/html/2608.28444v1)进一步强调了无需训练的滑窗＋sink比较。本轮加入了该类对照；当前结论仅限本模型、两层和这些PPL数据，未完整复现该论文广泛任务结果。

**四、部署收紧：缓存是真的省了，速度没有显著赢。**

RTX3090、BF16模型、FP32 feature/state、TF32关闭，batch1。3次prefill预热、6轮prefill+32固定后续token解码，首轮丢弃；不存在并发GPU工作。FAVOR与MLP都采用融合特征投影和同一融合状态更新，8k实际结果为：

| 模式／方法 | prefill ms | decode ms/token | KV＋状态＋窗口 MiB |
|---|---:|---:|---:|
| pure/teacher | 468.554 | 21.320 | 224.00000 |
| pure/favor_s11 | 471.154 | 21.606 | 208.76172 |
| pure/exp_kl_s11 | 473.768 | 21.281 | 208.76172 |
| pure/split_01_s11 | 473.966 | 21.229 | 208.76172 |
| hybrid/favor_s11 | 553.978 | 23.230 | 208.88672 |
| hybrid/calibrated_s11 | 556.530 | 23.402 | 208.88672 |
| hybrid/gate_s11 | 556.596 | 23.061 | 208.88672 |
| window/swa448_sink4 | 492.594 | 21.645 | 208.88281 |


旧实验的缓存问题已修复：被替换两层的DynamicCache真实存储为0，另存24×64×(128+2)个FP32数，即798720 bytes。n8192时全模型224MiB降至208.7617MiB，节省约6.80%；64近邻窗口另加128KiB。状态与窗口在继续decode时保持固定大小，其余26层KV仍随长度增长。

纯线性主方案21.229ms/token相对原模型21.320ms/token仅约0.43%的中位数差别，处于本次重复波动范围，不能宣布稳健加速；prefill稍慢，混合参考实现也更慢。当前是HF eager局部替换，不是已优化的全模型线性服务系统。短上下文低于约390tokens时，本配置FP32状态还可能比被移除的BF16 GQA KV更大。MLP额外约6.76MiB权重也应计入总显存，不能只报KV而冒充总显存同幅下降。8k的prefill新增峰值显存实测从原模型745.1MiB升至纯MLP773.2MiB，校准混合版本为819.0MiB；向量化前缀状态与局部窗口临时张量尚未优化，持久缓存减少并不等于prefill峰值减少。

剥离Python逐算子派发、以真实Q/K做CUDA graph热缓存微测：FAVOR投影＋状态为5.734µs/层，主MLP为9.902µs/层，MLP约1.73倍。主导FLOPs比为2.75倍，单看feature投影为4.5倍。这说明融合和固定开销可以缩小耗时倍率，**不会减少数学FLOPs，也不能推出MLP比FAVOR天生更适合GPU**。FAVOR微测仍更快。完整模型差距小还因为仅替换2/28层，其他计算与权重读取占主要开销。图见 [inference.pdf](figures/inference.pdf)。

FP64显式矩阵／递推、FP32融合feature、8k完整真实Q/K的显式参考、完整模型prefill→decode和真实storage均已检查。8k数值检查最大相对输出误差1.46e-6，没有零分母，说明这批质量问题不能归咎于已测到的递推数值错误。详见 [检查](checks/long_precision_and_window.json)。

**五、下一步的证据来自不变性缺口。**

共同RoPE旋转保持所有原始kernel值和对应分布的Schmidt谱不变，验证中最大logit差为5.68e-14。然而固定主特征在offset0→1024下，方向KL约0.364→1.973；exp-KL约0.348→2.267。FAVOR的平均KL在该诊断下变化很小，但其原始水平更差。这个受控变化说明，固定feature子空间的对齐／泛化是谱下界之外的一项实质问题。它不能证明共同平移是所有真实长上下文失败的唯一原因，RoPE不变性本身也并非新发现。

据此单独启动相同m、参数量、更新数的共同旋转增强与增强KL控制，并使用另外80篇全新文档确认，见 [后续实验协议](../orbit_direction/PROTOCOL.zh.md)。它是否值得继续，必须由新确认结果决定；单纯“MLP正特征＋一般谱框架”，当前仍不足以支持顶会方法论文。主张创新需要同时拿出超越同样增强的KL、匹配缓存的SWA的质量／效率结果，并解决证书过松的问题。

复现入口：[REPRODUCE.zh.md](REPRODUCE.zh.md)。本轮审计通过：36个feature/checkpoint构造、108组kernel评价、132组完整模型PPL、24个全模型计时条件；[final_audit.json](checks/final_audit.json)与[artifact_manifest.json](results/artifact_manifest.json)保留预算、哈希和结果完整性证据。



---

原文件：causal_direction/REPRODUCE.zh.md

工作目录 `/root/autodl-tmp`，Python `/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python`，模型与数据从固定本地Hugging Face缓存读取。所有训练LLM权重冻结、TF32关闭、单卡RTX3090。

依次运行 `data.py select`、`data.py extract`、`train.py`、`calibration_data.py`、`calibrate.py`、`verify.py`、`confirm_driver.py`、`verify_more.py`、`causal_witness.py`、`rotation_diagnostic.py`、`gaussian_full_covariance.py`、`benchmark.py`（文件均在本目录）。其中后三项机制/条件诊断是在初始确认后追加，不能回写为预先的模型选择规则。已有训练和逐配置结果会跳过，重做应使用新的输出目录或先移走相应旧结果。不能在确认结果上继续调参再将同一确认集称为新测试。

报告用 `/root/miniconda3/bin/python analyze.py` 生成统计和图，再用研究Python运行 `audit.py`，最后用前者运行 `write_report.py`。`gaussian_full_covariance.py`应在`rotation_diagnostic.py`之后运行：最终条件审计使用全部262144个缓存Q/head、4194304个K/group；初始按位置抽样的检查另行归档。因果矩形的FP64小矩阵SVD使用CPU LAPACK，避免RTX3090的Jacobi分解开销，不改变矩阵或样本数。

`confirm_driver.py` 顺序执行确认激活提取、kernel评价、总体谱见证数值、完整模型PPL；不并发GPU进程。`benchmark.py` 单独运行，输出实际缓存存储与全模型时间。数据分片包含真实post-RoPE Q/K，原训练64 Q/文档与校准64 Q/文档位置严格不重叠；K可以重复作为不同Q的配对对象，同一配对不重复训练。训练统计、校准初始化/标准化并不算额外SGD epoch，但确实使用对应训练数据统计；报告不能声称每个datum只被程序读取一次。

主损失 λ>0 的零点仍为原始指数kernel。固定head log_scale仅作数值标度，精确混合分支也减同一个值。测试raw NMSE只作诊断，不作神经训练目标。谱分析固定参考key bank的完整乘积，不把mask的rank性质混用，也不把有限经验数值证书冒充未知总体下界。

完整模型PPL覆盖每篇文档所有下一token位置，Wiki128×1024、长文24×8192、文学128×64；最后一种在64精确窗口内，混合/窗口结果不能作为学到远处能力的证据。所有指标保留逐文档值以便配对统计；训练种子11、29、47，头和文档相关性须分别处理，不能将24heads当24次独立模型训练。



---

原文件：causal_direction/THEORY.zh.md

本轮的总体对象依旧是 κ(q,k)=exp(qᵀk/√d) 在 P_Q×P_K 上的逼近。以下谱下界不假设Gaussian，也不把一次NMF拟合的数值当成下界。这里应用的粗粒化、数据处理、Pinsker和最佳低秩逼近都是经典数学工具；新组合是否足够新颖尚未确立。

**一、与原始I损失相容的谱下界。**

假设 M=Eκ∈(0,∞)，预测 h(q,k)=Σ_{r=1}^m f_r(q)g_r(k)≥0 且预测总质量 H=Eh∈(0,∞)。目标联合概率测度P与预测R相对于参考P_Q×P_K的密度分别是κ/M、h/H。标准质量分解给出

$$\frac{\mathbb E d_I(\kappa,h)}{M}=D_{KL}(P\|R)+\frac HM-1-\log\frac HM\ge D_{KL}(P\|R).$$

固定任意Q分组A_1,…,A_a与K分组B_1,…,B_b。令C_ij=P(A_i×B_j)、D_ij=R(A_i×B_j)。这两个矩阵的元素为概率，总和为1。注意C是分布的积分，不是对某个测试矩阵单独拟合参数。

因为参考测度是乘积测度，D_ij=(1/H)Σ_r(∫_{A_i}f_r dP_Q)(∫_{B_j}g_r dP_K)，所以rank(D)≤m。由数据处理和自然对数版本Pinsker：

$$D_{KL}(P\|R)\ge D_{KL}(C\|D)\ge\tfrac12\|C-D\|_{\mathrm{entry},1}^2.$$

每个矩阵基元素e_ie_jᵀ的核范数为1，三角不等式给出 ||E||_*≤Σ_ij|E_ij|。而最佳rank-m核范数逼近误差为Σ_{r>m}σ_r(C)。因此

$$\boxed{\frac{\mathbb E d_I(\kappa,h)}{M}\ge\tfrac12\left(\sum_{r>m}\sigma_r(C)\right)^2.}$$

这是对任意满足条件的m维非负可分离kernel都成立的总体下界。它不需要Eκ²有限。与原L²谱尾Σσ²不同，这里先对概率粗粒化矩阵的奇异值尾求和再平方。有限经验边缘分布也满足同一定理，但经验数值不能直接改称未知总体的数值证书。

**二、与query-balanced目标和当前主方法连接。**

令Z(q)=E_Kκ(q,K)，Z_h(q)=E_Kh(q,K)。定义P(dq,dk)=P_Q(dq)κ(q,k)P_K(dk)/Z(q)，R相应替换为h/Z_h。它们有相同Q边缘，且h/Z_h仍是最多m项的可分离非负和。因此同样的C矩阵构造给出

$$\mathbb E_Q D_{KL}(p(\cdot|q)\|\hat p(\cdot|q))\ge \tfrac12\left(\sum_{r>m}\sigma_r(C)\right)^2.$$

本轮λ>0的原始目标为L_λ=E_Q[KL(p||p̂)+λ(Z_h/Z−1−log(Z_h/Z))]。其唯一无约束零点仍是h=κ，且L_λ≥方向KL，所以直接继承该谱下界。减小λ是在有限容量下调整质量与方向的权重，未改变零损失原始目标，但其最优解与λ=1一般不同。λ=0明确为不识别原始标度的KL对照。

参考P_K必须固定。该证明不能直接应用于包含query相关三角mask的整张矩阵；因果矩阵可有高秩，即使未mask的kernel仅rank-m。主训练使用合法因果位置，谱证书另在固定边缘乘积上计算，两项测试承担不同用途。

**三、旧互信息界为零而此界为正的例子。**

n个离散点均匀参考，C=(1−ε)11ᵀ/n²+εI/n。对应原始指数kernel可取对角exp(t)、非对角1，其中exp(t)=1+nε/(1−ε)，用正交q_i,k_i即可构造。C的奇异值为1/n及n−1个ε/n；m≥1时新界为½[ε(n−m)/n]²。n=256,m=64,ε=.1时新界为0.0028125，而[I(Q;K)−log m]+=0。这说明低互信息并不等同于容易被少量feature拟合；许多弱模式也可能需要维度。该例不代表真实LLM已取得同样数值。

**四、有限样本与不能省略的误差。**

若C_hat是C的估计，核范数尾距离对核范数扰动是1-Lipschitz。因此只要以概率至少1−δ有||C_hat−C||_*≤ε_n，则

$$\frac12\big[\sum_{r>m}\sigma_r(\hat C)-\epsilon_n\big]_+^2$$

是相应风险的下置信界。分组若学习而来，必须只用独立训练集选择，随后在独立样本上估计C；不能在同一噪声矩阵上选择分组再假设它固定。

为给出完全明确且保守的例子，固定有限key参考bank，假设独立文档D_l同分布，并在每个文档内平均有界的query-conditional概率外积得到X_l（非负、entry sum=1）。每个||X_l||_F≤1，任意两取值距离≤√2。令C_hat=n⁻¹ΣX_l，利用Hilbert方差界和McDiarmid可得

$$\|\hat C-C\|_*\le\sqrt{\min(a,b)}\,\frac{1+\sqrt{\log(1/\delta)}}{\sqrt n}$$

以至少1−δ概率成立。它可能很松，若代入后为零必须如实报告。该置信界只覆盖文档query分布相对固定key bank的泛化；还没有覆盖bank与真实P_K之间的差异。对未知重尾κ总体，单靠有限协方差和有限数据不能获得可靠分布无关的指数矩估计；跨bank和跨域稳定性只是经验检验，不能替代额外尾部假设。

**五、实际kernel仍然连续。**

主方法为m exp(s_Q+s_K)π_Qᵀπ_K，π_X=softmax([z_X,0])，每侧两层128→192→64。上述Q/K分组只是用于粗粒化下界，推理feature没有查表分区。当前谱理论提供风险证书与难度诊断，不直接保证该MLP接近其下界，也不证明它胜过KL训练或其他可学习kernel。

与文献的边界：[Hedgehog §4及Appendix A](https://arxiv.org/html/2402.04347v1) 已给出共享线性层、指数/negation特征及KL蒸馏；[DoF for Linear Attention §3.3](https://arxiv.org/html/2507.03340v1) 已训练PRF节点与权重。本轮用这些公式作补充对照，主要严格匹配对照仍是同两层网络、同数据、同更新预算的KL方案。λ调整、粗粒化和谱不等式本身不能自动算作顶会贡献。

**六、原始质量在精确近邻混合中的可观察作用。**

将合法前缀分为近邻 L 和远处 R，L 使用精确 κ，R 使用正可分离 h。令 α=Z_L/(Z_L+Z_R)，预测 α̂=Z_L/(Z_L+Z_h,R)。由于两集合不相交且近邻条件分布精确，直接展开 KL 得

$$D_{KL}(p\|\hat p)=d_{KL}(\operatorname{Bern}(\alpha)\|\operatorname{Bern}(\hat\alpha))+(1-\alpha)D_{KL}(p_R\|\hat p_R).$$

对固定远处方向 g，最优上下文标量 c*=Z_R/Z_g,R 同时消除混合质量误差、最小化远处原始 I-divergence。实际可部署的 c(q) 只依赖 q，无法访问未知精确远处总质量，因此这是标签和构造依据，不能声称已实现 oracle。纯线性模型中 c(q) 被分母消去；混合模型中它改变与精确分支的相对权重。所有分支必须使用同一个固定 head 标度，不能分别任意归一化。

本轮冻结 KL 网络的方向，以训练 key bank 的 μ_r=E g_r(K) 作可逆配对 rescaling，构造

$$h_{\rm cal}(q,k)=e^{\ell(q)}\sum_{r=1}^{64}\pi_r^\mu(q)\frac{g_r(k)}{\mu_r},\quad \pi_r^\mu(q)=\frac{f_r(q)\mu_r}{\sum_s f_s(q)\mu_s}.$$

ℓ 是冻结第一层 SiLU 隐状态的线性读出，用未参与方向训练的 Q/K 对拟合远处原始质量。相对 logits 用63坐标，读出用最后1坐标，可折叠回相同73856参数/head、64维状态；它保持父网络的纯线性注意力方向。对照使用完全相同读出类、初始化、数据和更新次数，改用教师混合 Bernoulli KL。只有跨域/长上下文及完整模型结果才能判断原始质量目标是否具有超出通用门控的价值。

近邻窗口固定64时，远处状态大小固定；全算子包含精确分支，不能把它称作整张因果矩阵的rank64逼近。固定近邻＋远处线性、门控与蒸馏已有相关工作；这里的质量分解和训练目标尚未形成新颖性结论。

**七、因果分布的矩形分解下界。**

为避免把乘积测度定理直接套到因果mask上，固定一个上下文C，定义P_C(i,j)=p_C(j|i)/N，其中i均匀且j≤i；预测R_C同理。将三角区域分成互不重叠的笛卡尔矩形I_b×J_b，另加对角单点。按二分递归取“右半Q×左半K”即可构造覆盖；无需对kernel做固定分区。

对任何原始rank-m正kernel h，预测概率在一个矩形内是 h(q_i,k_j)/(N Z_h,C(q_i))，仅增加依赖i的左对角缩放，所以条件矩阵D_b=R_C(·|b)仍rank≤m。令w_b=P_C(b)，C_b=P_C(·|b)。对矩形标签应用KL链式分解，并舍去非负的矩形质量KL，有

$$\boxed{\mathbb E_i KL(p_C(\cdot|i)\|\hat p_C(\cdot|i))\ge B_m(C):=\frac12\sum_b w_b\big[\sum_{r>m}\sigma_r(C_b)\big]^2.}$$

因此本轮非MSE原始kernel损失满足 E_C L_λ(C)≥E_C B_m(C)，所有λ≥0均成立。上下文内Q/K允许来自相关token，因果归一化也已包含在证明中。该定理以随机上下文总体的期望为对象；数值上先求每个上下文的合法谱见证再平均，并未分别训练NMF或把一个矩阵最优解当作总体极限。

它仍是下界，可能很松：矩形大小≤m的部分贡献为零，不同矩形共享feature的约束也被放松。不能把该界称为准确的rank-m最小误差。B_m(C)∈[0,1/2]，对iid上下文，Hoeffding给出 E B_m≥mean(B_m)−½sqrt(log(1/δ)/(2n))；文档有依赖时iid置信解释仍需收紧。

`causal_witness.py` 在首批确认指标读取后追加，作为固定候选的事后理论诊断，明确不作为新的方法确认。它核验128篇新Wiki前512位置的完整因果分布、精确矩形覆盖、rank-m预测的矩形谱尾为零及下界不超过实际KL。不会把512位置的数值扩张为8192位置或未知总体非零证书。

**八、真实冻结模型的L²存在性应与Gaussian模型区分。**

核验本机Qwen2实现可见，Q/K投影之前是RMSNorm，且默认RoPE为正交旋转。写r(x)=Γx/sqrt(||x||²/D_model+ε)，则||r(x)||≤||Γ||op sqrt(D_model)。更紧地，任意head满足

$$\|q_h\|\le\sqrt{D_{model}}\|W_{Q,h}\Gamma\|_{op}+\|b_{Q,h}\|=:B_{Q,h},$$

K同理；RoPE保持该界。因此真实固定模型在所有定义良好的输入上的κ≤exp(B_Q B_K/sqrt(d))，从而κ∈L²(P_Q×P_K)。这说明原始Schmidt框架在本模型上有严格存在性基础；Gaussian近似不满足a_i<1/2，并不意味着真实kernel的二阶矩发散。问题是Gaussian尾模型失真、谱预测不可靠，而非真实有限权重RMSNorm模型不存在L²谱。

这个结构上界通常极大，直接代入集中不等式无法给出有用样本量或数值下界。它不修复Gaussian谱预测，也不证明现有MLP接近最优。当前核范数和因果矩形下界的价值是避免错误假设、提供合法但仍可能很松的证书。

**九、同一RoPE旋转的可检验不变性。**

对同一正交R_δ，κ(R_δq,R_δk)=κ(q,k)。将两个边缘都推前到同一旋转后的分布，原始积分算子通过两个酉坐标变换相连，奇异值及rank-m最小误差完全相同。因此若固定已学习feature在这种变换下误差明显改变，不能解释为目标kernel本身的谱变难，而是表示/泛化不满足相应不变性。

`rotation_diagnostic.py` 仅作事后诊断，选固定前16篇Wiki、种子11与预设offset。它不是新训练结果，也不将共同位置平移等同于增加相对距离的真实长上下文。RoPE与线性注意力相容性已有研究，任何后续不变性构造须重新核对文献并作固定feature维度与状态预算比较。

在谱保持不变时，还需区分“最优rank-m”和“当前feature子空间”。对固定实值子空间F,G（各至多m维），允许自由双线性系数的最佳L²核是P_F T P_G，误差为

$$\|T\|_{HS}^2-\|P_FTP_G\|_{HS}^2=E_m^\star+\underbrace{\sum_{r\le m}\sigma_r^2-\|P_FTP_G\|_{HS}^2}_{\text{特征子空间未对齐造成的额外误差}}.$$

这是放松正性后的诊断，不是正feature可实现的等式最优值。对共同旋转后的分布，T的谱相同，但固定网络经坐标拉回后的F,G可以改变，所以仅知道谱无法预测这个固定网络的泛化。

对于同样保持目标κ不变的共同变换，记δ=log h(Rq,Rk)−log h(q,k)，原始I风险的变化有更直接的恒等式：

$$R(h_R)-R(h)=\mathbb E[(h-\kappa)\delta+h(e^\delta-1-\delta)].$$

若δ只依赖q，纯线性attention方向不变，但原始质量风险仍可显著增大；混合精确分支时也会改变输出。这个恒等式与前述质量分解共同说明：数值上较好的归一化输出不能替代原始kernel泛化的验证。它是经典投影/散度代数的应用，本轮不将这些恒等式本身视为已确立的新颖贡献。



---

原文件：deployment_validation/ALL_HEADS_ESTIMATE.zh.md

本文件仅做解析计算及历史计时外推，没有运行新模型或 kernel 实验。它不预测全 head 替换后的困惑度。

当前两种构造不同。正 kernel 使用谱坐标上的硬分区，再用条件均值表；Galerkin 使用连续的谱特征，没有分区。两者都有 1024 个训练 landmark，不需要访问历史 token 来计算新 token 的特征。

令 l_Q(q)_j=exp(q^T kbar_j/sqrt(d))，l_K(k)_i=exp(qbar_i^T k/sqrt(d))。以下固定数值缩放均吸收入投影矩阵、阈值和系数。

正 kernel：z_Q=W_Q^T l_Q(q)，z_K=W_K^T l_K(k)，c_Q=tree_Q(z_Q)，c_K=tree_K(z_K)，

    kappa_hat_P(q,k) = B[c_Q(q),c_K(k)],
    B_rs = E[kappa(Q,K) | c_Q(Q)=r,c_K(K)=s] >= 0.

实际 B 用独立训练文档估计。等价非负特征为 phi_Q(q)=B^T e_cQ、phi_K(k)=e_cK；维度 m=64。谱嵌入维度 e=64，W_Q,W_K 各为 1024×64。

Galerkin：投影训练积分算子、白化基函数并做 SVD 后，

    kappa_hat_G(q,k) = sum_{r<=m} sigma_tilde_r u_tilde_r(q)v_tilde_r(k)
                    = l_Q(q)^T A_Q A_K^T l_K(k),

其中 A_Q,A_K 各为 1024×64，吸收 sqrt(sigma_tilde)。这给出连续的 64 维特征，但不保证非负。这里描述的是本项目的 Galerkin-Schmidt 构造。

计算量假设：Qwen2.5-1.5B，28 层，12 Q heads/层，2 KV heads/层，D=1536，I=8960，d=d_v=128，vocab=151936。全部 336 个 Q heads 使用同规模、各自独立的新 kernel，a=1024，e=m=64。原模型 Q/K/V/O 投影、MLP 与 LM head 不变，删除所有原始 attention 的历史 KV 依赖。动态状态按每个 Q head 一份计算。

每 MAC 计 2 FLOPs，只计主导矩阵乘加；不把 exp、比较、除法等折成普通 FLOPs。原模型因果 prefill 按合法的三角区域计算，实际算子边界块可能有额外工作。Prefill 仅计算最后位置 LM logits，与此前缓存推理计时一致。

每 token 的全模型投影/MLP：

    F_base = 28 * [4D^2 + 4D*H_KV*d + 6D*I]
           = 2,620,391,424 FLOPs.

最后位置 LM head：F_lm=2D*vocab=466,747,392 FLOPs。

新 kernel 每 head、每新 token 的 Q/K 特征：

    F_phi = 4a(d+e) = 786,432 FLOPs,

另外需要 2a=2048 个 exp。全部 heads 合计每个 decode token 有 688,128 个 exp。

正 kernel 的稀疏桶更新 O(d_v)，输出读出主导成本约 2m*d_v=16,384 FLOPs/head。因此，T 为当前可见 token 数时：

    F_decode_old(T) = F_base + F_lm + 336*2T(d+d_v)
                    = 3.087138816e9 + 172032*T.
    F_decode_new    = F_base + F_lm + 336*(F_phi+2m*d_v)
                    = 3.356884992e9.

标准线性聚合的 prefill 主导成本约 4m*d_v/token/head：

    F_prefill_old(N) = N*F_base + F_lm + 336*N*(N+1)*(d+d_v).
    F_prefill_new(N) = N*F_base + F_lm + 336*N*(F_phi+4m*d_v).

当前块长 64 的分块因果实现另有约 336*N*2*64*(m+d_v) FLOPs，8k 时使整模型估计增加约 0.068 TFLOPs。结果保存在 results/all_heads_analytic_estimate.json。

| 上下文长度 | 原模型 decode GFLOPs/token | 新模型 decode GFLOPs/token | decode 变化 | 原模型 prefill TFLOPs | 新模型 prefill TFLOPs | prefill 变化 |
|---|---:|---:|---:|---:|---:|---:|
| 1024 | 3.263 | 3.357 | +2.9% | 2.774 | 2.966 | +6.9% |
| 2048 | 3.439 | 3.357 | -2.4% | 5.728 | 5.931 | +3.5% |
| 4096 | 3.792 | 3.357 | -11.5% | 12.177 | 11.861 | -2.6% |
| 8192 | 4.496 | 3.357 | -25.3% | 27.240 | 23.722 | -12.9% |
| 16384 | 5.906 | 3.357 | -43.2% | 66.024 | 47.443 | -28.1% |
| 32768 | 8.724 | 3.357 | -61.5% | 178.227 | 94.885 | -46.8% |

逻辑 FLOPs 的 crossover 约为 decode 1568 tokens、prefill 3200 tokens；它们不是实测速度 crossover。Galerkin 的主导特征成本相同，dense 状态更新使 decode 每 head 多约 2m*d_v FLOPs，整模型差异很小。

时间必须另作条件性外推：当前原模型使用 BF16，新增 landmark 特征使用 FP32，TF32 关闭；GPU 吞吐、启动和访存均不同。[NVIDIA 矩阵乘法性能说明](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)

Prefill 8192 的历史数据：原模型 472.15 ms，新 kernel 的 4-head Q/K 特征 2.7064 ms（融合、CUDA Graph）。若全 heads 的吞吐可按 head 数近似线性外推，新特征总预算为 84*2.7064=227.34 ms。

仅作为敏感性模型，用原模型 N=1024/4096/8192 的三个历史计时拟合 t(N)=b0+b1*N+b2*N^2，得到 b0=5.907 ms，b1=0.050737 ms/token，b2=7.5408e-7 ms/token^2。8k 的非二次部分约 421.55 ms。若把二次项近似视为被删除的 attention 工作，替换后为：

    t_prefill_new(8192) ≈ 421.55 + 227.34 + t_linear_aggregation
                       ≈ 648.89 ms + t_linear_aggregation.

这不是 attention profiling：三个点无法区分算子效率变化与真正的二次工作。若给线性聚合及其他误差预留 0–100 ms，得到约 0.65–0.75 秒，较历史原模型约慢 40%–60%。这个区间是明确假设下的预算范围，不是置信区间或实际全 head 结果。

Decode 的历史原模型约 22.07 ms/token（8k）。4-head 特征的 graph 时间为 0.040416 ms，eager 为 0.366048 ms。

若 12-head 批处理花费为现有 4-head 的 1–3 倍，则全部 28 层的新增特征 GPU 预算约 1.13–3.39 ms/token。若 eager 的主机开销每层保留一次、GPU 工作按 3 倍扩展，则新增特征预算约：

    28 * [(0.366048 - 0.040416) + 3*0.040416] = 12.51 ms/token.

总时间应为 t_old - t_removed_attention_and_cache + t_features + t_linear_and_glue。被删除路径尚无独立 profile，故不能唯一确定净变化。作为显式敏感性场景：

| 场景 | 假设删除的原 attention/cache | 新线性聚合及连接开销假设 | 预计总 decode |
|---|---:|---:|---:|
| 新 attention 路径 graph/fused | 1–3 ms | 0.1–0.5 ms | 约 20–25 ms/token，约 -10%～+15% |
| 延用 eager 组织 | 1–3 ms | 0.5–2 ms | 约 32–36 ms/token，约 +45%～+65% |

表中删除成本与连接开销是敏感性假设，不是测量。不能用四个 heads 的端到端时间差直接乘 84，因为原混合适配器还包含分组/索引开销，全替换会移除它们；并行度和数据复用也会改变。

全部 heads 独立维护 FP32 状态时，动态状态为 336*64*129*4=10.58 MiB，8k 原 BF16 KV 为 224 MiB。但当前 kernel 还新增约 509.25 MiB 的主要静态参数（anchors、投影、B 表，未计少量树元数据）。这些静态参数不随上下文增长，却会影响单 token 访存。因此固定状态带来的缓存优势与实际速度优势需要分别评估。

这些推算不验证：全 heads 的精度/数值稳定性、低精度特征可用性、最优算子或生产引擎吞吐。它们说明当前形式在长上下文能减少逻辑计算和动态缓存，但现有 FP32 landmark 实现不保证更低延迟。



---

原文件：deployment_validation/REPORT.zh.md

本轮结论：按用户选定的范围，在完整冻结 Qwen2.5-1.5B 中替换现有 4/336 个 query head。新正 kernel 的困惑度优于这里复现的五组 FAVOR+，但仍劣于原模型；没有测得端到端推理收益。公平融合和去除主机启动开销后，新 kernel 相对 FAVOR+ 的特征计算差距反而更明显。“接近未知总体 rank-m 理论下界”尚未成立；本轮收紧了完整留出经验算子的谱界，用于进一步检验这一主张。

**实验范围与复现条件。** 模型固定 Qwen2.5-1.5B，revision `8faed761d45a263340a0528343f099c05c9a4323`；L14H0、L14H6、L27H0、L27H6，m=64，head dimension=128。没有训练/微调模型，也没有重新拟合新 kernel。模型使用 BF16，特征及因果状态 FP32，TF32 关闭。GPU 为 RTX 3090。所有计时期间 GPU 上只有这一项测试。原始拟合目标始终为 exp(qᵀk/√128)；数值上的公共尺度/互补特征尺度在 attention 分子与分母中抵消，没有把目标换成归一化分数。

**FLOPs 与时间的关系。** 较大的 GEMM 能提高吞吐，启动和读写开销也会使时间不与 FLOPs 成比例。这与 NVIDIA 的矩阵乘法性能说明一致，但这种现象不是新 kernel 独有的优势。此前 FP32 实现中总 FLOPs 约 12.75 倍、总耗时约 2.72 倍，包含小算子启动及公共聚合开销，不能据此推断算子优化后差距必然缩小。[NVIDIA 官方说明](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)

实际实现了双方的融合算子：新 kernel 将 exp epilogue 融合，用 Triton 树遍历替代稠密路径评分 GEMM，单 token 使用 GEMV+exp；FAVOR+ 融合范数、bias、exp，单 token 融合 GEMV。Galerkin 使用同样的 landmark 优化。双方同时测试 eager 与 CUDA Graph；未使用低精度近似偷偷减少任何一方的工作量。融合主要减少中间读写与启动，无法消除 1024 个 landmark 的数学计算。[Triton 官方融合示例](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)

下表是 **Q/K 特征对的耗时，微秒**，4 heads，FP32，双方已优化，CUDA Graph，30 次重复的中位数；不是完整 attention 或完整模型耗时。

| N | FAVOR+ μs | 新正 kernel μs | Galerkin μs | 新 / FAVOR+ |
| --- | --- | --- | --- | --- |
| 1 | 14.34 | 40.42 | 21.50 | 2.82× |
| 128 | 44.03 | 299.01 | 277.98 | 6.79× |
| 2048 | 67.58 | 701.44 | 685.06 | 10.38× |
| 8192 | 228.35 | 2706.43 | 2588.66 | 11.85× |

用同一特征计时口径比较 N=2048：本轮未融合 eager 的 FAVOR+/新 kernel 分别为 0.2847/1.1623 ms，约 4.08 倍；融合加 CUDA Graph 后为 0.0676/0.7014 ms，约 10.38 倍。不能将后一特征比值与此前包含公共聚合的总耗时比值直接当成同一个指标比较。

48 组实现比较的最大特征相对 L2 差异为 2.5e-06；检查的分区 cell 一致。以 N=2048 为例，新方案的主导特征 GEMM 约 6.442 GFLOPs、FAVOR+ 约 0.268 GFLOPs，按表中时间计算的有效吞吐约为 9.18 / 3.97 TFLOP/s。较大 GEMM 的吞吐优势仍然存在，但这不等于延迟优势，也不是实际 SM occupancy 测量。N=1 的差距更小，不能外推到其他 batch、GPU 或部署引擎。尚未完成联合 GEMM-exp-GEMM 的最优融合或穷尽 autotuning，所以这也不是硬件性能上界；可以确认的是，当前证据不支持“新 kernel 比 FAVOR+ 更适合 GPU，因而能靠写算子抹平计算差距”。

**分区 kernel 的实际稀疏状态。** 令 κ̂(q,k)=B[c(q),d(k)]，B≥0。定义

\[
S_s(t)=\sum_{j\le t:d(k_j)=s}v_j,\qquad n_s(t)=\sum_{j\le t}1_{d(k_j)=s}.
\]

则

\[
y_t=\frac{\sum_s B_{c(q_t),s}S_s(t)}{\sum_s B_{c(q_t),s}n_s(t)}.
\]

每一步仅更新一个 key cell 的 value 桶和计数，更新 O(d_v)，读取输出 O(m d_v)。已在 Triton 中实现只写这一桶，并融合读取输出；FAVOR+ 也使用融合的状态更新与读取。该结构优势针对状态更新，1024-landmark 的特征成本仍是主要问题。Prefill 使用并行块状态的前缀和，加上块内精确因果乘积，固定块长 64 时为线性复杂度。

因果验证使用一份真实 1024-token Q/K/V：显式下三角 kernel、分块计算、逐 token 更新及截断前缀互相比较；Float64 最大相对 L2 误差 1.88e-15。最终融合递推 FP32 对 Float64 的最大相对 L2 误差 1.48e-06。完整 BF16 模型中 prefill 与 cached decode 的最后 token logits 并非逐位相同，原模型本身也存在该差异；五种情形的最后 token top-1 均一致。这只是实现检查，不是生成质量评估。

**完整模型的局部替换困惑度。** 使用全部 256 篇内部留出 WikiText103 文章、20 篇固定 WikiText2 官方 test 文章，各取 1024-token 前缀，分别计 261,888 与 20,460 个 next-token loss。两者都与 kernel 构造文档分离。官方一列是固定子集困惑度，不是标准拼接 WikiText2 benchmark 分数。所有因果位置都参与替换，下游隐藏状态和 Q/K 全部重新计算。五个 FAVOR+ 种子全部报告，没有挑最好的种子。

| 方案 | 内部 256 篇 PPL | 官方 test 子集 20 篇 PPL |
| --- | --- | --- |
| 原模型 | 8.613259 | 8.915608 |
| 精确 attention 拆分控制 | 8.612553 | 8.914776 |
| 新正 kernel | 8.689070 | 9.034617 |
| Galerkin | 8.709132 | 9.048217 |
| FAVOR+（m64） | 8.747917 [8.736948, 8.770964] | 9.101324 [9.078257, 9.110611] |
| FAVOR+（m640 预算对照） | 8.742300 | 9.077467 |

FAVOR+ m64 单元格为五种子的中位数 [最小值, 最大值]。m640 是预先固定的五个 128-node block 的联合，不是筛选的最好种子，也不是五个独立 m640 实验。它的原有矩形 attention 主导 FLOPs 约为新 m64 kernel 的 78%，因此只是接近计算预算的补充质量对照，不是严格相同 FLOPs；本轮没有对 m640 写对应的全部融合算子或计入完整模型速度表。精确拆分控制组保留相同的 Q-head 拆分方式，但每一组都使用精确 SDPA，用于区分拆分实现造成的数值变化。配对文档 bootstrap 使用 10,000 次重采样；每个 FAVOR+ 种子的“新 kernel NLL − FAVOR+ NLL”区间记录在 `results/summary.json`。这衡量样本文档的不确定性，不是未知总体谱的置信区间。

Galerkin 的完整模型结果没有 NaN loss，但内部/官方分别累计出现 570 / 67 个非正分母。没有加 epsilon、截断负数或替换失效输出。其带符号 kernel 仍存在分母抵消问题，有限 PPL 不能证明所有输入下稳定。FP64 特征/状态的额外 4+4 文档检查及逐文档差异已记录；不能把单块约 1e-6 的算术一致性外推为完整 BF16 模型逐位一致。

**真实推理测试。** 下表包含整个模型、KV cache 更新、最后位置 LM head。Prefill 3 次预热+7 次测量，decode 1 次预热+5 次测量，每次 32 个固定后续 token，batch=1。计时提示词由真实留出文章拼接，4096/8192 长度只用于性能测量，本轮没有验证这些长度的困惑度。实现是 Hugging Face eager model，不是生产服务引擎。

| 提示词长度 | 方案 | Prefill ms | Decode ms/token | token/s | KV cache MiB |
| --- | --- | --- | --- | --- | --- |
| 1024 | 原模型 | 58.653 | 21.688 | 46.11 | 28 |
| 1024 | 精确 attention 拆分控制 | 59.685 | 22.228 | 44.99 | 28 |
| 1024 | 新正 kernel | 61.576 | 23.452 | 42.64 | 28 |
| 1024 | FAVOR+（m64） | 61.654 | 23.041 | 43.40 | 28 |
| 1024 | Galerkin | 61.738 | 23.203 | 43.10 | 28 |
| 4096 | 原模型 | 226.378 | 22.239 | 44.97 | 112 |
| 4096 | 精确 attention 拆分控制 | 227.855 | 22.424 | 44.59 | 112 |
| 4096 | 新正 kernel | 229.460 | 23.464 | 42.62 | 112 |
| 4096 | FAVOR+（m64） | 229.904 | 23.145 | 43.21 | 112 |
| 4096 | Galerkin | 231.283 | 23.290 | 42.94 | 112 |
| 8192 | 原模型 | 472.151 | 22.069 | 45.31 | 224 |
| 8192 | 精确 attention 拆分控制 | 471.528 | 22.488 | 44.47 | 224 |
| 8192 | 新正 kernel | 476.983 | 23.477 | 42.60 | 224 |
| 8192 | FAVOR+（m64） | 474.004 | 23.847 | 41.93 | 224 |
| 8192 | Galerkin | 477.298 | 23.556 | 42.45 | 224 |

局部替换无法消除 K/V cache：每层被替换的两个 Q head 分别与另外五个未替换 Q head 共用一个 KV head，因此所有 KV 仍需保留。新方案还增加 132,096 字节的 FP32 线性状态，外加约 6 MiB 的静态 landmark/projection/分区参数。拆分控制也包含选择剩余 Q head、复制对应 K/V 张量的开销。当前 eager 适配器不是最优融合 hybrid attention 算子，速度表不能作为不可加速的数学证明。各 head 成本相近的 Amdahl 估计下，即使所选 head 完全免费，也只消掉 attention 部分的 4/336=1/84，整模型改善空间有限。4/336-head 的结果不能用于预测全 336-head 线性化后的性能或质量；但这次部署范围内，收益主张没有得到支持。[GQA 原论文](https://arxiv.org/abs/2305.13245)

**收紧经验谱界，而非把有限样本称为总体。** 原本固定训练基底给出的区间较宽。本轮对官方全部 10240×10240、内部全部 131072×131072 的经验乘积分布做流式算子乘法，未将 kernel 全矩阵放入显存。随机 range 取 256 维并做一次 subspace iteration，只用于诊断，不改变任何待评估 feature map。[随机子空间方法原论文](https://arxiv.org/abs/0909.4061)

对 n×n 的 kernel 数值矩阵 K 和任意正交列矩阵 U，设 τ_j=s_j(UᵀK)/n，H=||K||²_F/n²。H 是经验乘积分布下的平方能量，E_{m,emp}* 也按经验均值定义，则有确定性的经验界

\[
\frac{\sum_{j>m}\tau_j^2}{H}
\le E_{m,\mathrm{emp}}^\star/H
\le 1-\frac{\sum_{j\le m}\tau_j^2}{H}.
\]

这里的随机性只影响区间紧度，不影响正交子空间投影所给区间的数学有效性。左界来自投影的奇异值不大于原算子奇异值；右界来自 U(UᵀK)_m 的可行 rank-m 近似。区间宽度恰为漏掉的平方能量。全部运算 Float64，重新积分的 H 与既有穷举结果核对，正交误差检查通过。

| 经验分布 | head | rank-64 最优相对误差区间 | 新正 kernel 误差 | R / E* 至少 | 区间宽度 |
| --- | --- | --- | --- | --- | --- |
| internal | L14H0 | [0.010434, 0.022096] | 0.206549 | 9.35× | 0.011661 |
| internal | L14H6 | [0.049695, 0.089491] | 0.467656 | 5.23× | 0.039796 |
| internal | L27H0 | [0.048311, 0.073881] | 0.784200 | 10.61× | 0.025570 |
| internal | L27H6 | [0.039641, 0.056611] | 0.660848 | 11.67× | 0.016970 |
| official | L14H0 | [0.017274, 0.021194] | 0.254134 | 11.99× | 0.003920 |
| official | L14H6 | [0.003281, 0.003786] | 0.945591 | 249.78× | 0.000505 |
| official | L27H0 | [0.018302, 0.022420] | 0.687913 | 30.68× | 0.004118 |
| official | L27H6 | [0.005764, 0.006897] | 0.598735 | 86.81× | 0.001133 |

“R/E* 至少”用 R 除以 E* 的**上界**得到，方向不能反过来；它用于证明与经验无约束 rank-64 最优解存在多大差距。它不是正特征最优误差 E_m⁺ 的界，也不是未知总体的倍数证书。

分区模型还有更直接的误差分解：固定 Q/K 分区下，条件均值 B* 是 L2 投影；估计的 B̂ 满足

\[
R(B̂)=R(B^*)+\sum_{r,s}p_rq_s(B̂_{rs}-B^*_{rs})^2.
\]

内部完整经验分布对应结果如下；后两列之和为第一列，说明目前主要损失来自分区函数空间，而非仅仅条件均值样本数不足。

| head | 冻结新 kernel 风险 | 固定分区投影误差 | B 估计误差 |
| --- | --- | --- | --- |
| L14H0 | 0.206549 | 0.203375 | 0.003175 |
| L14H6 | 0.467656 | 0.467362 | 0.000294 |
| L27H0 | 0.784200 | 0.781123 | 0.003077 |
| L27H6 | 0.660848 | 0.644438 | 0.016410 |

**总体主张需要什么。** Schmidt 尾谱定理在 κ∈L2(P_Q×P_K) 下成立；这项数学结论与经验估计是否可信是两件事。要认证 R_P(κ̂)≤(1+ε)E_m*(P)，至少要给出同一总体下的风险上置信界 U_R 与谱下置信界 L_*，再检验 U_R≤(1+ε)L_*。当前没有这样的证书。即使 131072² 个 kernel pair 都算完，也只有 256 篇留出文档，不能把 pair 数当成独立样本量。

举例：若 Q 与 K 来自相互独立的 n_Q、n_K 篇文档，且逐文档交叉平均损失落在 [0,L]，有界差分能给出误差量级 L√((1/n_Q+1/n_K)log(2/δ)/2)。文档内 token 可相关。应用到核平方能量需要已知的全分布 κ² 上界，应用到谱还需要投影矩、Gram 矩阵及其扰动控制。样本最大值不是合法的全分布上界；重尾下该证书可能十分宽甚至无实际用途。原模型的有界投影可以保证存在有限数学上界，并不保证这个上界足够紧。

总体 Schmidt 定理并不要求 Gaussian；本轮经验谱收紧也没有验证“只用 Q/K covariance 即可预测真实谱”的 Gaussian 外推。此前非 Gaussian 尾部及少数文档主导平方能量的问题仍然存在。

原始 kernel 的 L2 风险与模型质量也不等价。对一个上下文，若 κ、κ̂≥0，Z=Σ_jκ_j，κ̂ 的分母非零，且 ||v_j||≤V，则归一化输出误差满足 ||y−ŷ||≤2V Σ_j|κ_j−κ̂_j|/Z。这个不等式解释了为何仍需逐行相对误差及真实上下文测试；它没有改变 raw-kernel 拟合目标。P_Q×P_K 的边缘乘积分布还不同于真实同文档、带位置及因果限制的 joint distribution。对有符号 Galerkin，正性所提供的这一分母稳定性保障不再自动适用。

因此，本轮支持“该冻结正 kernel 在这些未见文档上较五组 m64 FAVOR+ 和预先固定的 m640 对照保留了更多模型质量”；不支持“已经带来推理收益”或“已经接近真实总体理论下界”。下一步更值得研究的是以更低成本逼近有效谱子空间，并在新文档上控制该子空间的误差；单纯继续打磨 1024-landmark 实现，无法保证消除特征成本。

结果文件：`results/ppl_optimized.json`、`results/summary.json`、`results/operator_benchmark.json`、`results/model_benchmark_optimized.json`、`results/tightened_spectrum.json`、`results/runtime_checks.json`。`ppl.json`、`model_benchmark.json` 保留早期实现结果，主表采用最终版本。

复现命令：使用 `/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python` 执行 `verify_runtime.py`、`evaluate_model.py --output ppl_optimized.json`、`benchmark_operators.py`、`benchmark_model.py`、`tighten_spectrum.py`；使用带 NumPy/Matplotlib 的默认 `python` 执行 `summarize.py`、`write_report.py`。GPU 性能测试必须串行进行。



---

原文件：distribution_operator/ALTERNATIVE_KERNELS.zh.md

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



---

原文件：distribution_operator/REPORT.zh.md

后续更新：更紧的完整经验谱界和完整模型 4/336-head 部署验证见[部署与谱界报告](../deployment_validation/REPORT.zh.md)；与 FAVOR+、SDERF/ADERF 的同协议 kernel 比较见[比较报告](../kernel_comparison/REPORT.zh.md)。下面保留本轮算子推导与分布诊断的原始结论范围。

本轮结论：**用户对有限矩阵NMF和此前FAVOR变体的质疑成立。现在已完成从总体算子出发的非FAVOR条件期望kernel推导，并做了完整留出经验乘积分布及不相交文档验证。但尚未验证“covariance准确预测真实总体谱”“正kernel接近最优rank-m下界”这两个核心经验主张。当前正kernel的主要问题是分区表达误差。**

本报告对应`distribution_operator`目录的新实验。以前的Gaussian、有限矩阵NMF、神经特征实验保留在各自目录；它们不与本轮不同采样口径的结果混为一张方法优劣表。

**目标始终是未归一化kernel。** 定义

$$\kappa(q,k)=\exp(q^\top k/\sqrt{128}),\qquad
\mathcal R(\hat\kappa)=\frac{\mathbb E[(\kappa-\hat\kappa)^2]}{\mathbb E[\kappa^2]}.$$

所有表中误差均为这个相对平方误差；分母是总体或经验kernel二阶矩，不是attention行归一化。计算只用一个训练确定的head常数做数值缩放，随后恢复原始kernel幅度。没有截断logit、调低temperature、删除极端Q/K或用归一化attention作为拟合标签。另存广义KL/总kernel质量作为辅助指标。

**总体对象已经重新定义。** 对真实文档及位置采样机制诱导的P_Q、P_K，定义T:L²(P_K)→L²(P_Q)，(Tf)(q)=E_K[κ(q,K)f(K)]。H=Eκ²有限时，Schmidt展开和最优rank-m谱尾定理成立。正特征最优值则为

$$E_m^+=\inf_{f_r,g_r\ge0}\mathbb E\left[\left(\kappa(Q,K)-\sum_{r=1}^m f_r(Q)g_r(K)\right)^2\right]\ge E_m^\star.$$

有限矩阵NMF的数值解给出该矩阵非负最优误差的可行上界，不能替代这个总体函数优化问题。固定权重Qwen的pre-RMSNorm、有限线性投影及保范数RoPE使Q/K有界，所以一般Hilbert–Schmidt框架适用；拟合Gaussian模型的二阶矩发散并不表示真实kernel发散。

决定总体谱的关系是

$$ (TT^*)(q,q')=M_K((q+q')/\sqrt d),\qquad
M_K(t)=\mathbb E e^{t^\top K}.$$

非零特征值为σ_j²。完整矩母函数及另一侧分布决定谱，covariance通常不够。本轮用有界离散分布构造了精确反例：所有分布均值0、方差0.2，最优rank-1相对误差却分别为0.03750、0.14970、约0.5；二阶矩可从1.0811变为1.1769×10¹³。相同covariance的Gaussian surrogate甚至满足a<1/2。这不是采样误差，而是精确加权算子SVD及解析二阶矩同时验证的不可识别性反例。公式、适用条件和证明见[THEORY.zh.md](THEORY.zh.md)。

**新kernel由条件期望投影构造。** 单侧理想形式为

$$\boxed{\kappa_C^+(q,k)=\sum_{r=1}^m\mathbf1_{C_r}(q)
\underbrace{\mathbb E[\kappa(Q,k)\mid Q\in C_r]}_{M_{Q\mid C_r}(k/\sqrt d)}.}$$

它自动非负、rank≤m。把Schmidt展开代入L²条件期望投影，得到

$$E_m^\star\le R_C\le E_m^\star+
\underbrace{\sum_{j\le m}\sigma_j^2\mathbb E\operatorname{Var}(u_j(Q)\mid C)}_{D_Q^{(m)}}.$$

这给出明确的设计原则：让分区在加权谱坐标(σ₁u₁,…,σ_m u_m)内的量化误差小。只有D_Q相对谱尾足够小，才能推出接近最优。低rank本身不保证m个分区满足这个条件。条件期望的L²投影性质是经典事实；谱学习中的经验算子与总体算子近似也需要独立统计条件。[条件期望讲义](https://adembo.su.domains/stat-310b/lnotes.pdf)、[Rosasco等：积分算子学习](https://jmlr.org/papers/v11/rosasco10a.html)。

本轮实际实现两侧分区，使推理不必显式平均训练Q参考点：

$$\boxed{\hat\kappa_m^+(q,k)=e_{c(q)}^\top\hat B e_{d(k)},\quad
b_{rs}=\mathbb E[\kappa(Q,K)\mid Q\in C_r,K\in D_s].}$$

特征取φ_Q(q)=e_c(q)、φ_K(k)=B e_d(k)，维度仍是m；B有m²个系数。Q侧非负且稀疏，K侧严格为正。默认不使用严格正平滑。两侧实际形式的条件保证为

$$R_{CD}\le2E_m^\star+D_Q^{(m)}+D_K^{(m)},\qquad
R(\hat\kappa_m^+)=R_{CD}+\sum_{rs}p_rq_s(\hat b_{rs}-b_{rs})^2.$$

后一式精确分离了分区表达误差和均值估计偏差。当前没有证明或验证前一式中D_Q、D_K相对谱尾很小，也不能把条件性的(2+ε)倍保证当成已实现的结果。

这不是对FAVOR辅助高斯积分改节点或改权重：积分分布是实际Q|C_r，Q侧是分区指示函数，实际部署是两侧分区查表。此前的指数节点方法则确实属于FAVOR/DERF相关方法族。[FAVOR# / DERF原始论文](https://proceedings.neurips.cc/paper_files/paper/2023/file/02dec8877fb7c6aa9a79f81661baca7c-Paper-Conference.pdf)。这里仍然使用经典条件期望、谱分区和Nyström思想，不能仅凭“非FAVOR”声称新颖性。

在固定cell内，广义KL的最优常数也是条件均值，因为∂E D_I(κ∥b)/∂b=1−Eκ/b。因此，本轮直接积分估计均值，既是该函数类的KL最优参数，也是L²投影参数；没有用MSE做神经训练，也没有重复epoch。统计积分会组合同一向量与多个另一侧向量，不能把这些组合称为独立训练样本。

**数据与训练/诊断边界。** 冻结Qwen/Qwen2.5-1.5B，revision为`8faed761d45a263340a0528343f099c05c9a4323`，复用真实投影、RoPE后的BF16 Q/K，head为L14H0、L14H6、L27H0、L27H6，d=128。全部kernel计算和算子累积使用Float64。

| 用途 | 文档 | 每侧向量 | 参数用途 |
|---|---:|---:|---|
| 校准训练 | 2,048 | 65,536 | 锚点、谱坐标字典、分区 |
| 独立moment-fit训练 | 2,048 | 65,536 | 条件均值B、有符号训练Galerkin核心 |
| 内部留出 | 256 | 131,072 | 固定函数风险和经验算子诊断 |
| 官方测试子集 | 20 | 10,240 | 跨来源固定函数风险和诊断 |
| 不相交文档A→B | 128+128 | 每侧65,536 | 固定函数风险和诊断 |
| 不相交文档B→A | 同上反向 | 每侧65,536 | 固定函数风险和诊断 |

内部留出来自WikiText103源train按文档划分；官方子集是以前固定的20个WikiText2 test文章前缀，不是完整官方测试集。训练和内部/官方测试token哈希无重叠。两个训练用途的文档完全不相交。所有输入长度为1024；Q边缘使用后512位置，K边缘使用前512位置；每个训练文档每侧抽32位置，内部测试用全512位置。定义的乘积分布不同于实际同文档因果配对联合分布。

从校准数据每侧抽1,024个锚点，构造原始kernel并SVD。用前256个加权Nyström延拓函数作算子字典；分区用其中前64个加权谱坐标。贪心轴向切分每次降低谱坐标的组内平方误差，最小叶节点32个校准向量；先产生256细分叶节点，保存m=16/32/64/128的嵌套切分。该算法不是全局最优谱量化。

B在独立moment-fit数据完整65,536²乘积上求cell均值。所有报告rank均无空训练cell，所有系数正值。完整内部测试每个head计算131,072²=17,179,869,184个raw kernel值，4个head共68,719,476,736个，分块累积，无需存储整张矩阵。

这个覆盖范围是**全部已收集留出向量的经验边缘乘积分布**，并非未知真实总体。约687亿配对是高度复用向量的组合，不能据此认为有687亿独立样本；采样不确定性仍主要受文档数量与未见尾部控制。

**算子区间代替小矩阵NMF“下界”。** 在固定训练函数字典的子空间内，按评价测度正交化并计算压缩C，记其奇异值s_j，则

$$\boxed{\frac{\sum_{j>m}s_j^2}{H}\le\frac{E_m^\star}{H}
\le1-\frac{\sum_{j\le m}s_j^2}{H}.}$$

左端来自压缩奇异值不超过总体奇异值；右端来自一个实际可行rank-m投影。区间宽度是1−∥C∥²_F/H，明确记录了字典未捕获的能量。测试压缩只作诊断，其参数没有进入部署kernel。以下区间是对完整经验测度的数值区间，未包含总体采样置信误差。

另做了完全训练确定的有符号Galerkin–Schmidt对照：在moment-fit训练分布正交化字典、计算C_train并rank-m截断，得到固定m维函数，再评估未见文档。这比仅在1,024锚点矩阵上截断并求逆更能检验“训练积分算子→新输入函数”的路径，但仍不保证非负或总体最优。

**完整内部留出结果，m=64。**

| head | 经验最优有符号rank-64区间 | 冻结有符号训练Galerkin | 冻结新正kernel | 分区表达误差 | 均值估计偏差 |
|---|---:|---:|---:|---:|---:|
| L14H0 | [0.00004479, 0.165681] | 0.171722 | 0.206549 | 0.203375 | 0.003175 |
| L14H6 | [0.00128578, 0.448789] | 0.477636 | 0.467656 | 0.467362 | 0.000294 |
| L27H0 | [0.00667760, 0.424443] | 0.512328 | 0.784200 | 0.781123 | 0.003077 |
| L27H6 | [0.00511808, 0.266917] | 0.318085 | 0.660848 | 0.644438 | 0.016410 |

分区表达误差按留出测度的精确cell均值计算，是固定分区内最优风险；它不参与训练。表中“均值估计偏差”是相对留出条件均值的偏差，含有限训练样本误差和分布差异，不能全归因于某种独立同分布估计方差。后三列满足严格Pythagoras分解。

即使允许用留出数据最优设置当前B，误差仍为0.203、0.467、0.781、0.644；因此在当前函数类内增加数据主要消除的小项并非主要瓶颈。这个结果支持改进谱表示、分区或条件函数表达，不能推断所有非负kernel都具有同样的误差。

对于L27H6，E*≤0.266917H而R=0.660848H，所以R/E*至少约2.48；L27H0至少约1.85。即使区间仍宽，也足以排除当前构造已经达到接近1倍的rank-m最优误差。该比较针对有符号最优值；非负类自身的最优值仍未确定。

| head | m=16正kernel | m=32 | m=64 | m=128 |
|---|---:|---:|---:|---:|
| L14H0 | 0.271894 | 0.220544 | 0.206549 | 0.203282 |
| L14H6 | 0.469303 | 0.468157 | 0.467656 | 0.467280 |
| L27H0 | 0.798092 | 0.790541 | 0.784200 | 0.779600 |
| L27H6 | 0.676441 | 0.667498 | 0.660848 | 0.658415 |

完整精确数值以[curves.csv](curves.csv)为准。增加m对部分head帮助很小；训练字典之外的kernel能量同样没有消失。字典维度64→128→256时，未捕获能量分别为：L14H0 0.1734→0.1674→0.1656；L14H6 0.4573→0.4531→0.4475；L27H0 0.4930→0.4540→0.4178；L27H6 0.3219→0.2895→0.2618。不能把这些宽区间当作精确总体下界估计。

![完整留出rank曲线](figures/rank_curves.png)

**官方20文档子集与此前小矩阵的差异。**

| head | m=64经验有符号最优区间 | 冻结有符号训练Galerkin | 冻结新正kernel |
|---|---:|---:|---:|
| L14H0 | [0.00014561, 0.199363] | 0.219683 | 0.254134 |
| L14H6 | [0.00018182, 0.840880] | 0.942688 | 0.945591 |
| L27H0 | [0.00410125, 0.243082] | 0.383822 | 0.687913 |
| L27H6 | [0.00111440, 0.170075] | 0.237139 | 0.598735 |

同一官方Q/K来源池，旧L14H0的4张512点产品矩阵NMF m=64报告误差约1.556×10⁻⁵；现在完整10,240×10,240经验乘积的有符号最优误差下界已达1.456×10⁻⁴，约为旧值的9.36倍。旧结果是抽样子矩阵各自拟合后的汇总，并不代表一个函数覆盖整个池的误差。扩大到全池后，连更宽松有符号类的下界都高于旧NMF误差，这直接显示了原有替代的局限。没有把旧FAVOR或旧神经特征的不同配对测试误差搬来充当本轮同协议对照。

**不相交文档验证。** 完整经验乘积允许Q和K来自同一原始文档。虽然这类pair只占1/256=0.390625%，L14H6中却贡献36.3036%的kernel平方能量。因此另将256留出文档随机分成A、B各128篇，计算Q_A×K_B及Q_B×K_A的全512位置乘积，保证两侧没有同文档配对，训练函数及系数不变。

| head | A→B最优有符号区间 | A→B新正kernel | B→A最优有符号区间 | B→A新正kernel |
|---|---:|---:|---:|---:|
| L14H0 | [0.00004879, 0.123664] | 0.164811 | [0.00004275, 0.194741] | 0.237966 |
| L14H6 | [0.00168295, 0.180255] | 0.209981 | [0.00122311, 0.107643] | 0.125994 |
| L27H0 | [0.01012888, 0.307473] | 0.668047 | [0.00343635, 0.416914] | 0.865141 |
| L27H6 | [0.00545303, 0.176336] | 0.597908 | [0.00378266, 0.282251] | 0.683465 |

两个方向是两个不同经验乘积测度，不能平均区间端点后称为总体置信区间。L14H6误差明显改变，L27H0两方向也差异很大，说明文档组成和尾部覆盖对问题定义及结论影响显著。单纯从同池矩阵中屏蔽同文档pair会形成非乘积联合测度，不能直接对它套原产品测度的Schmidt下界；使用不相交两组边缘保留了乘积分布结构。

![不相交文档验证](figures/disjoint_documents.png)

**样本规模与尾部诊断。** 内部8,192、32,768、131,072的嵌套规模都覆盖全部256文档，变化的是每文档采样位置数。L27H0的log Eκ²为15.8508→13.3748→12.4454；最小样本二阶矩约是全量的30.1倍。样本增加可能稀释一次罕见大值的经验权重，不能预设误差或能量必须单调增加。

以文档为单位bootstrap 2,000次，同时对两侧施加相同重采样文档权重，得到二阶矩相对原样本的2.5%—97.5%描述区间：L14H0 [0.955,1.068]，L14H6 [0.845,2.424]，L27H0 [0.519,2.447]，L27H6 [0.627,1.757]。L27H0删除一篇特定文档即可令二阶矩降到0.629倍。bootstrap无法可靠覆盖尚未观测的尾部，也不是总体谱区间证明。

这些诊断补充了此前“最大0.1%贡献极高平方能量”的现象：指数kernel二阶矩对尾部、文档关联及位置抽样非常敏感。之前0.1%的具体百分比属于之前的paired样本，不能直接沿用为本轮全产品的百分比。本轮没有把减少平方损失敏感性等同于改变理论目标；二阶谱尾仍然对应原始kernel的L²风险。

![分布敏感性](figures/distribution_sensitivity.png)

**哪些理论成立，哪些尚未成立。** 一般Schmidt谱尾、条件期望误差分解、压缩谱区间是有明确条件的数学结论，本轮证明和数值检查均支持实现正确。Gaussian闭式谱在Gaussian且max a_i<1/2时成立，原16-head实验仅4个符合HS条件，12个不适用；有效head的总体相对谱排序也未达到原Go标准。该公式没有被验证为真实LLM总体谱预测器，新的同covariance反例进一步说明不能省略分布假设。

从有限经验算子推广到真正总体，需要对H和压缩C及Gram估计误差给出有效界；仅有HS有限性不能保证当前样本量下的精度。本轮提供了完整经验积分、嵌套规模、官方子集、不相交文档及文档敏感性检查，但未得到紧的总体置信界。因此“已验证理论框架”只能用于数学结构与数值实现，不能用于“总体谱预测准确和正特征近最优”的合并主张。

**Go / No-Go判断。** 原“只用covariance→Gaussian总体谱→接近下界正kernel”的完整主线仍是No-Go；当前分区正kernel作为接近最优方法也未过关。继续研究“真实分布算子→可泛化谱表示→正函数逼近”有数学依据，但主要未解决问题已经明确为字典未捕获能量、谱分区/条件函数表达及总体尾部估计。KAN只能作为其中一种参数化方式；本轮没有证明KAN独占优势，也没有新增KAN、mulKAN与MLP的公平比较。若接回两层KAN，需要保持此前等参数、独立文档和非MSE训练协议，并为分区软化/条件函数逼近增加明确误差项。

这轮结果足以纠正研究对象并建立可复核实验，但不足以支持顶会方法结论。更有价值的贡献需来自新的分布假设下可用的谱/统计界、接近最优的正函数构造条件，或在严格同协议强基线上稳定实现优势。当前实现还依赖1,024锚点谱坐标计算与m²查表参数，没有测端到端LLM推理速度、全模型替换或困惑度；m维线性汇总结构并不自动等于实用加速。

**实现检查与复现。** [checks/aggregate_audit.json](checks/aggregate_audit.json)记录176组谱区间检查、320组固定函数风险下界检查、176组嵌套分区投影检查均通过；最大相对Pythagoras残差1.21×10⁻¹⁵，训练Galerkin投影恒等式误差6.39×10⁻¹⁵。独立小算子检查覆盖流式积分与直接矩阵一致、真正有限SVD尾在上下界中、条件均值KL驻点和非负因子分解。部署特征乘积与直接查表的相对偏差低于4.3×10⁻¹⁶。

已有数据缓存下，可在本目录运行以下命令复现计算和图表。`run.py`会复用已存在的本轮算子缓存；修改协议后应使用新输出目录，不能混用旧缓存。

```bash
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python verify.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python run.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python cross_docs.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python diagnostics.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python covariance_counterexample.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python train_galerkin.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python kernel.py
/root/miniconda3/bin/python summarize.py
```

[kernel.py](kernel.py)提供`ConditionalMeanKernel(m=64)`，输入shape为`[4, n, 128]`，head顺序由`protocol.json`固定，`features(x,'q'/'k')`返回`[4,n,m]`非负特征。`results/`保留训练参数、抽样文档索引、全部算子矩、分区及逐尺度结果，`curves.csv`与`summary.json`给出紧凑汇总。图表同时保存PNG和PDF。



---

原文件：distribution_operator/THEORY.zh.md

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



---

原文件：gated_integration/PROTOCOL.zh.md

2026-09-08，训练与确认评价前固定。本轮测试“幅度—方向结构接入原生GLA/GDN是否有收益”，不把softmax kernel拟合优势自动推广成递推模型优势。

模型主体全部冻结。采用本地GLA检查点fla-hub/gla-340M-15B（6e04029dc090a2c55df712f18814db80aa39894f）和公开GatedDeltaNet检查点linear-moe-hub/Gated-Deltanet-340M（c83bdada453cde56932f37be71338df22ca29b7d）。后者仓库名称不代表精确参数量，以实测计数为准。读取加载缺失/多余键并检查有限前向结果；不允许随机补权重后称为预训练模型。

每模型只改第11、23层的全部4个heads；保留原QKV、短卷积、衰减门、输出门、输出投影和归一化。GLA的key宽度128、GDN的key宽度256；本轮保持原状态维度，排除压缩与门控通道重新映射的混淆。每侧两层无偏置MLP，隐藏宽度192，输出原key维度。主比较固定网络参数预算、训练文档、顺序、更新数和种子。登记参数相同不意味着归一化后所有幅度分支都可识别，失效分支明确报告。

GLA比较：ad_shape（两侧方向softmax，幅度消融），ad_full（方向加K幅度，Q幅度在输出RMSNorm之前保留以核查消去），exp_mlp（相同层数、隐藏宽度、输出维度的正指数控制）。为避免无意义的极端数值，全部指数标量使用B(x)=4tanh(x/4)；方向softmax使用未裁剪logits。AD使用m−1方向logits及一个幅度坐标，与既有主方法一致。

GDN比较：ad_norm（正特征接原生Q/K L2归一化，幅度理论消去）；ad_write（归一化方向不变，K幅度通过V→exp(B(sK))V进入写入；保留原beta，所以擦除系数不受幅度影响）；exp_mlp（相同两层正指数特征接L2归一化）。ad_write是稳定的幅度接入扩展，不再把状态递推称为原始正kernel的归一化求和，也不声称两门分离是首次提出。

原生GLA/GDN的目标函数与softmax不同，故本轮局部适配优化语言模型下一token交叉熵；它是非MSE任务损失，不是假称继续拟合exp(qᵀk/√d)。既有softmax核权重不跨模型直接复用。原始kernel理论的适用边界及数值恒等式独立核查。

数据沿用既有causal_direction的文档划分，先由Qwen tokenizer解码原始缓存片段，再用两原生模型共同的tokenizer编码。训练选前1024个满足512token的训练文档，每篇截512token，仅一遍；验证前32篇，确认Wiki64篇512token、长文16篇2048token。确认文档此前已用于研究，不称首次盲测。跨数据划分检查原文hash和token hash；一个文档只提供一个训练序列。最终人数以数据文件记录为准，若不足则从同split后续文档补齐，不复制样本。

每架构每方法seed11、29；相同seed使用相同初始网络权重和同一文档顺序。AdamW lr0.001,wd0.0001，cosine末端lr0.0001，batch4，256次更新，梯度clip1。不根据确认结果改学习率或挑种子。记录初始、训练过程与验证损失；各最终检查点全部进入确认评价。若实现或数值错误导致失败，保存错误并记录修复，重跑受影响训练；不把失败值丢弃。

评价原生基线和局部适配完整模型的PPL、配对文档NLL差、状态字节、特征附加FLOPs及统一实现微测。所测层以外或另一模型的结论均不得外推。小规模且仅2种子，结果是可行性证据。优先完成算子/梯度验证再训练；本轮没有从头训练GLA/GDN或全模型微调。

模型加载修订（GDN尚未训练或确认评价时）：linear-moe-hub版本包含当前FLA未使用的D参数及融合gate/up投影，原样加载出现missing/mismatched keys，故弃用，保留logs/gdn_prepare.log。改为puigde/gated-deltanet-360M-15B-slimpajama（1d1a7bf99323601635f3d63f67c97614f9c2ec10），其原生21层，选10/20层（中间/最后层），同样全4heads。不得丢弃多余参数或随机初始化缺失参数后称为原生检查点。GLA仍为11/23层，其他训练协议不变。

机制补充对照（主12次训练已完成、GLA确认及GDN原生确认已可见后追加）：GDN原生检查点的验证PPL高于所有已适配正映射，存在“额外任务适配”解释。两模型各追加同参数signed_residual控制f_X(x)=x+MLP_X(x)，同两层隐藏192、相同数据与seed11/29、单遍CE及优化设置。该控制保留原生带符号几何，初始化更接近原模型，是有意加入的强对照，不假装与AD初始函数相同。它只作为此轮探索性机制验证，结果不回填为最初预设。新计划单独保存为family_followup_plan.json。

兼容性修复：仅在本研究进程中把FLALayer.get_max_length映射到已有get_max_cache_shape，并刷新抽象方法定义，适配当前Transformers Cache API。此前缓存计时因接口错误中断；不影响use_cache=False的已保存训练/PPL。需在修复后独立核查完整模型prefill+decode与整段前向的一致性。

计时补充：初测为每方法连续3次热运行，signed控制在另一个进程中测得，其小幅时间差容易受运行顺序影响。保留全部初测，追加同进程、五轮循环平衡方法顺序的测量，每轮512token预填充+16token固定延续。质量评价与权重均不改；最终优先报告此补充测量，细微方法间时间差不作算子效率优越性结论。



---

原文件：gated_integration/REPORT.zh.md

**幅度—方向结构接入原生 GLA / Gated DeltaNet：局部适配实验**

2026-09-08。本轮完成16次训练、两种子、完整模型困惑度评价、真实缓存解码、显式递推与分布理论适用边界核查。结论：GDN的独立写入幅度对照有收益，但尚不能证明AD优于一般同预算适配；GLA上直接正特征替换没有超过原模型。

这次测的是“AD参数化结构在预训练原生门控模型中重新学习”的可行性。不是把既有Qwen上的AD权重跨模型搬入，也不是继续用exp(qᵀk/√d)作为原生GLA/GDN的训练标签。原生模型不以该softmax kernel为既有机制，所以冻结主体、仅以下一token交叉熵训练新增分支。这个目标切换必须与原主线的原始kernel拟合结果区分。

**范围与公平比较**

| 项目 | GLA | Gated DeltaNet |
|---|---:|---:|
| 实际原模型参数 | 341,707,776 | 357,781,928 |
| 原生层数 | 24 | 21 |
| 修改层（从0计） | 11、23 | 10、20 |
| 每层修改heads | 全部4个 | 全部4个 |
| feature / key维度m | 128 | 256 |
| 每head新增登记参数 | 98,304 | 196,608 |
| 两层新增登记参数 | 786,432 | 1,572,864 |
| 主要新增矩阵乘 FLOPs/token | 1,572,864 | 3,145,728 |

所有适配器均为独立Q/K、无偏置、d→192→m的两层SiLU MLP。m保持原生状态宽度，保留原衰减、输出门、V投影、短卷积及归一化。FLOPs按一次乘加=2计，上表只计新增MLP矩阵乘；激活、归一化、逐点幅度乘法另有O(m+d_v)开销，模型公共部分未计入新增量。

数据：1,024篇训练文档各512token，共524,288token，每次拟合只用一遍；32篇验证；64篇512token及16篇2,048token测试。文档来自既有划分，经共同tokenizer重编码；训练/验证/测试原文hash和token hash不重复。它们不是本轮新收集的盲测文档，也没有新域测试或数万token长上下文测试。

每方法seed11、29，batch4、256次更新，AdamW lr=0.001，cosine末端0.0001，weight decay=0.0001，clip=1。使用最终检查点，无确认集选点或选种子。主比较与补充控制登记参数、数据、文档顺序、更新数一致。登记参数相等不等于全部分支都可识别：方向消融、原生L2以及RMSNorm会消去或弱化部分幅度自由度。

残差MLP为f_X(x)=x+MLP_X(x)，它保留原生带符号几何且初始化更接近原模型。此强控制在主结果部分可见后追加，属于探索性机制检查；不是与正AD相同的初始函数。因此当前结果也不能区分全部初始化、优化难度与最终表达能力。

模型：[GLA checkpoint](https://huggingface.co/fla-hub/gla-340M-15B)、[GDN checkpoint](https://huggingface.co/puigde/gated-deltanet-360M-15B-slimpajama)。确切revision、环境版本和数据hash在checks及summary.json。首次候选GDN检查点因权重布局不兼容，在训练前弃用，错误与替换过程保存在协议；实际使用的两模型均零缺失/多余/不匹配权重。

**实际接入形式**

令z_X=W₂,X SiLU(W₁,X standardized(x))∈R^m，π_X=softmax([z_X,1:m−1,0])，a_X=exp(4tanh(z_X,m/4))。

GLA完整AD使用f_X=√m a_Xπ_X：

$$S_t=D_tS_{t-1}+f_K(k_t)v_t^\top,\qquad o_t=f_Q(q_t)^\top S_t.$$

方向消融令a_Q=a_K=1。正指数MLP控制为exp(4tanh(z_X/4))/√m，具有相同两层矩阵尺寸。这里的控制不是Hedgehog或FAVOR+原论文实现。

GDN直接把上述AD输入原生Q/K L2归一化，正幅度完全消去。为让幅度真正起作用，写入方案令u=π_K/||π_K||₂、r=π_Q/||π_Q||₂，并采用：

$$\boxed{S_t=\alpha_t(I-\beta_tu_tu_t^\top)S_{t-1}+\beta_t a_K(k_t)u_tv_t^\top,\qquad o_t=r(q_t)^\top S_t.}$$

实现为方向送入原生Q/K归一化，V→a_KV，保留alpha和beta。它是AD的稳定写入扩展，不能再叫原始正kernel加标准linear-attention分母。两个原生模型仍使用各自输出归一化，不额外引入softmax attention分母。

**完整模型PPL；只替换两层，越低越好**

下表为两种子的算术平均±样本标准差；原模型未训练，只有一个固定结果。两模型的绝对PPL不能用来判定架构排名，因为预训练配置与数据不同。

| 模型 | 方法 | 512token | 2048token |
|---|---|---:|---:|
| GLA | 冻结原模型 | 16.01072 | 17.43497 |
| GLA | AD 方向消融 | 16.54511 ± 0.02715 | 18.57573 ± 0.00615 |
| GLA | AD 完整特征 | 16.53492 ± 0.00676 | 18.49160 ± 0.00315 |
| GLA | 同参数正指数 MLP | 16.76497 ± 0.01300 | 18.70428 ± 0.02645 |
| GLA | 同参数带符号残差 MLP | 14.67527 ± 0.04129 | 16.65899 ± 0.00770 |
| GDN | 冻结原模型 | 14.74034 | 15.63878 |
| GDN | AD + 原生 L2（幅度消去） | 13.69574 ± 0.00632 | 15.15213 ± 0.00752 |
| GDN | AD 独立写入幅度 | 13.33647 ± 0.00582 | 14.95613 ± 0.00768 |
| GDN | 同参数正指数 MLP | 13.77786 ± 0.00423 | 15.23072 ± 0.00465 |
| GDN | 同参数带符号残差 MLP | 12.55069 ± 0.02232 | 14.34615 ± 0.01020 |

GLA：AD相对原模型PPL升高3.27% / 6.06%；虽优于正指数MLP，仍落后于带符号残差MLP。幅度相对方向消融在512token几乎无增益，2048token只有约0.45%改善。

GDN：AD写入相对原模型PPL降低9.52% / 4.37%，相对幅度被归一化消去的版本降低2.62% / 1.29%，相对正指数MLP降低3.20% / 1.80%。但相对同参数残差MLP，PPL仍高6.26% / 4.25%。因此“有任务适配收益”和“AD具有独特优势”是不同主张；本轮只支持前者及正方向条件下的写入幅度增益。

**配对文档统计**

先对两种子的每篇NLL取平均，再做20,000次配对文档bootstrap。区间仅条件于这些已拟合检查点和当前文档集，不覆盖训练种子不确定性、总体最优界或新域泛化。表中为AD相对控制的PPL百分比变化，负数代表AD更好；基于平均NLL的比值，与PPL算术均值的比值可能有极小差别。

| 模型 | AD比较对象 | 512token，变化% [95%区间] | 2048token，变化% [95%区间] |
|---|---|---:|---:|
| GLA | 冻结原模型 | +3.27 [+2.68, +3.90] | +6.06 [+5.41, +6.82] |
| GLA | AD 方向消融 | -0.06 [-0.24, +0.11] | -0.45 [-0.58, -0.33] |
| GLA | 同参数正指数 MLP | -1.37 [-1.56, -1.19] | -1.14 [-1.30, -0.97] |
| GLA | 同参数带符号残差 MLP | +12.67 [+12.01, +13.37] | +11.00 [+10.38, +11.72] |
| GDN | 冻结原模型 | -9.52 [-10.09, -8.97] | -4.37 [-4.98, -3.81] |
| GDN | AD + 原生 L2（幅度消去） | -2.62 [-2.86, -2.39] | -1.29 [-1.58, -1.02] |
| GDN | 同参数正指数 MLP | -3.20 [-3.44, -2.97] | -1.80 [-2.14, -1.48] |
| GDN | 同参数带符号残差 MLP | +6.26 [+5.69, +6.87] | +4.25 [+3.95, +4.55] |

![AD配对PPL变化](results/paired_ppl.png)

**理论检查：哪些性质能迁移**

GLA展开后的序列权重为f_Q(q_t)ᵀ[∏_{u=j+1}^tD_u]f_K(k_j)。标量门D_u=α_uI且两方法共享门路径时，d_I(wκ,wh)=w d_I(κ,h)，可在同一个序列抽样测度上转移风险。逐通道门则依赖特征与门控的坐标对应，只有静态kernel的SVD谱和误差不足以控制它。原生GLA的输出RMSNorm还会消去正Q幅度（eps=0精确，实际近似）。这是对[GLA递推](https://arxiv.org/html/2312.06635v2)的接入分析。

GDN的归一化删除幅度；若绕过L2直接使用a_Ku，擦除转移沿u的特征值为α(1−βa_K²)，可能扩张。上述写入扩展保留单位key与擦除beta，因此状态转移仍非扩张。有限幅度只限制每步注入，alpha可趋近1时不能因此声称状态范数有长度无关的绝对上界。这些推导基于[Gated DeltaNet递推](https://arxiv.org/html/2412.06464v1#S3.SS1)。

更强的反例：两组严格为正的单位特征，即使拥有完全相同的完整query-key矩阵，也能产生不同的GDN输出。令两时刻q=(1,1,1)/√3；A的两个key都为u=(2,1,1)/√6，B的key为u与u'=(1,2,1)/√6。query-key矩阵全部等于c=4/√18，但key-key内积为1或5/6。取alpha=beta=1、v=[1,0]，第二次读出分别为0与c/6≈0.157135。这个反例在单位范数条件下成立，不能由保留原生L2来消除。

因此研究GDN需要序列分布P_seq上的读出风险，并纳入key-key几何、擦除、写入。THEORY.zh.md给出共同门/值路径下的方向与幅度扰动界。它是充分界，不是rank-m最优误差，也没有在本轮验证“接近总体Schmidt下界”。GDN的等效权重还可带符号，原始正kernel的I误差分解不能原样用于完整序列算子。

**机制诊断及其限度**

8篇heldout文档、两层、seed11，在固定原生Q/K输入上提取每篇128个key，统计单位key的平均非对角余弦。随后把相同keys用于读写，配独立随机V，alpha=beta=1作一次写入后回读；这项合成关联记忆测量只隔离几何，使用平方误差作评价，并非MSE训练，也不是原生门下的LM检索任务。

| 模型 | 方法 | key平均余弦 | 合成回读NMSE |
|---|---|---:|---:|
| GLA | 冻结原模型 | 0.6627 | 1.5465 |
| GLA | AD 方向消融 | 0.5689 | 1.8274 |
| GLA | AD 完整特征 | 0.6474 | 1.8329 |
| GLA | 同参数正指数 MLP | 0.7192 | 1.8919 |
| GLA | 同参数带符号残差 MLP | 0.6380 | 1.5481 |
| GDN | 冻结原模型 | 0.2583 | 1.0439 |
| GDN | AD + 原生 L2（幅度消去） | 0.8204 | 1.8799 |
| GDN | AD 独立写入幅度 | 0.8273 | 1.8683 |
| GDN | 同参数正指数 MLP | 0.8439 | 1.8965 |
| GDN | 同参数带符号残差 MLP | 0.3428 | 1.4668 |

GDN的AD keys明显更集中，余弦从原生约0.258变为0.827，合成回读也较差。这支持“key几何改变可能增加干扰”的解释，但不能把差距全部归因于正性；有限训练、近均匀初始化及特征坐标迁移也可能参与。残差MLP的合成回读同样弱于原生，尽管PPL更好，进一步说明这项诊断不等价于语言模型质量。

所有方案最终验证损失仍在下降，因此并未证明收敛或固有表达上限。当前负面结果是固定单遍预算下的局部迁移结果，不是从头预训练的架构No-Go。

**速度与固定状态**

RTX3090，batch1，原模型BF16、适配器FP32，TF32关闭，无推理autocast；训练时使用BF16 autocast。真实FLA chunk预填充及fused recurrent解码，512token预填充+16token固定真实延续；所有方法热身后同一进程内五轮循环平衡执行顺序。以下为各项墙钟时间中位数；没有CUDA Graph或新增特征融合算子。初始3轮计时也完整保留，优先用此平衡顺序测量。小于约几个百分点的方法间差距不能作为GPU适配性优越的证据。

| 模型 | 方法 | 512token预填充ms | 解码ms/token | 实际缓存MiB |
|---|---|---:|---:|---:|
| GLA | 冻结原模型 | 36.15 | 28.50 | 12.000 |
| GLA | AD 方向消融 | 37.87 | 29.80 | 12.000 |
| GLA | AD 完整特征 | 38.34 | 30.22 | 12.000 |
| GLA | 同参数正指数 MLP | 37.70 | 30.27 | 12.000 |
| GLA | 同参数带符号残差 MLP | 37.65 | 29.64 | 12.000 |
| GDN | 冻结原模型 | 55.91 | 31.91 | 21.492 |
| GDN | AD + 原生 L2（幅度消去） | 57.80 | 33.21 | 21.492 |
| GDN | AD 独立写入幅度 | 57.55 | 33.41 | 21.492 |
| GDN | 同参数正指数 MLP | 57.22 | 32.88 | 21.492 |
| GDN | 同参数带符号残差 MLP | 57.23 | 32.82 | 21.492 |

AD的两层接入使当前完整模型耗时增加约3%–6%，没有推理速度收益。新增MLP算术量小，但单token路径增加若干小矩阵乘、非线性、转换和Python/算子调用；本轮不把参考实现时间当作最优融合实现的成本。幅度乘法可以与已有处理融合，无法因此保证总体更快。

两模型原本就使用固定状态，本轮m不变，缓存也不变。GLA为12MiB；GDN为21MiB递推矩阵加约0.492MiB短卷积状态。新增适配器FP32权重另占约3MiB / 6MiB，属于模型常驻参数，不在缓存列中。固定状态优势是真实的，但它相对的是随序列增长的softmax KV cache，而非再次节省原生GLA/GDN的状态。

**实现核查**

- GLA：原生及三个主方法、两层的FLA chunk与FP64显式递推相对L2误差最大0.002113；fused recurrent最大0.001682。
- GLA：原生及四种适配seed11、两篇文档的整段前向与128token prefill后逐token缓存输出相比，logits相对L2最大0.005773，平均分布KL最大0.000282；缓存字节不随后续token增加。
- GDN：原生及三个主方法、两层的FLA chunk与FP64显式递推相对L2误差最大0.007810；fused recurrent最大0.001668。
- GDN：原生及四种适配seed11、两篇文档的整段前向与128token prefill后逐token缓存输出相比，logits相对L2最大0.005133，平均分布KL最大0.000295；缓存字节不随后续token增加。

以上是BF16路径数值误差，不是kernel拟合误差。全部16个检查点训练元数据通过单遍、同参数与同seed顺序检查，基模型训练前后SHA256一致。实际使用的两模型完整tokenizer词表一致，63个文本探针编码相同。门控路径、I齐次性、幅度消去、稳定性及单位正特征反例通过FP64核查。具体阈值与原始值均保存在checks文件中。

**Go / No-Go与创新性**

1. “AD写入幅度在正方向GDN中有用”：本轮有限范围内Go，方向消融的配对区间支持改善。
2. “AD直接接入原生GLA/GDN，比一般同预算适配更优”：本轮No-Go；两模型的强残差控制都更好。GLA完整AD也弱于冻结原模型。
3. “原来的softmax kernel谱/I理论已能保证或预测GDN收益”：No-Go；理论对象需要改为序列记忆算子，交叉kernel相同的反例已排除直接推出的可能性。
4. “独立擦除/写入本身足够新”：No-Go。2026年5月的[Gated DeltaNet-2](https://arxiv.org/abs/2605.22791)已研究分离擦除与写入，并给出更一般的通道门与高效算法。本轮未与GDN-2运行对比，不能据此排名。

不建议把这组结果包装成“AD+GDN新SOTA”或现成顶会主结果。其价值是确认幅度进入哪里有效、识别理论迁移障碍，并给出可复现的正负结果。若保留此支线，下一项有判别力的实验应保留原生key几何，仅比较理论约束的幅度写入与同预算普通写入门，再与残差适配比较；这样才能判断优势是否来自AD理论特有约束。当前原始主线的正softmax kernel逼近结论没有因此被否定，但也没有被这组GDN PPL实验进一步证明。

复现入口：native.py（模型/特征）、train.py（主实验或--followup）、assess.py（PPL与初测）、benchmark_balanced.py（平衡计时）、verify_ops.py、verify_cache.py、theory_checks.py、diagnostics.py、summarize.py及make_report.py。GPU训练/评价运行环境见checks/environment.json；make_report.py使用本机/root/miniconda3/bin/python（含matplotlib 3.10.5），GPU环境未安装绘图库。模型与tokenizer需预先缓存到native.py指定目录。原协议及修订见PROTOCOL.zh.md，完整推导见THEORY.zh.md；结果摘要见results/summary.json，图可导出为PDF/SVG。



---

原文件：gated_integration/THEORY.zh.md

本轮采用S∈R^{m×d_v}的记法。幅度—方向特征记为f_Q=a_Q p_Q，f_K=a_K p_K，p_X在正simplex；省略可吸收的固定常数。下文的p_Q/p_K是特征方向，不是原始Q/K输入分布。

**GLA：形式兼容，但静态kernel谱不自动控制门控误差。**

GLA状态S_t=D_t S_{t-1}+f_K(k_t)v_tᵀ，读出o_t=f_Q(q_t)ᵀS_t。D_t是特征通道衰减，元素在[0,1]。展开得到

$$o_t=\sum_{j\le t}f_Q(q_t)^\top\left(\prod_{u=j+1}^{t}D_u\right)f_K(k_j)v_j.$$

若D_t=α_t I且两个比较方法使用同一衰减路径，则有效kernel为w_{tj}h(q_t,k_j)，w=∏α∈[0,1]。对同一序列抽样测度，有d_I(wκ,wh)=w d_I(κ,h)，平方误差也乘w²。因此静态拟合误差可控制同测度下的标量门控误差。不能据此把既有独立PQ×PK风险无条件换成真实因果序列风险。

原生GLA通常是逐通道D_t。此时仅知道Σ_r fQr fKr≈κ，不知道各通道分别承担什么误差，不足以控制Σ_r w_r fQr fKr。甚至把同一个kernel的特征通道置换，若不同时置换D_t，也会改变模型。新的特征方向与预训练门控坐标需要共同适配。原Schmidt谱尾属于固定二元kernel，不能直接作为序列依赖算子的最优误差。

归一化正GLA若另外维护z_t=D_t z_{t-1}+f_K(k_t)，使用o_t/[f_Q(q_t)ᵀz_t]，则Q幅度严格消去。本轮原生GLA并没有这个分母，其输出使用RMSNorm；eps=0时RMSNorm(a_Q o)=RMSNorm(o)，有限eps时为近似不变。K幅度改变各token写入强度，不能同样消去。它也可以代数地吸收到v_t，所以不产生一个自动优于一般写入门的函数类定理。

**GDN：幅度既可能被归一化删除，也可能改变稳定性。**

原生GDN的主要递推为

$$S_t=\alpha_t(I-\beta_t u_tu_t^\top)S_{t-1}+\beta_tu_tv_t^\top,\quad ||u_t||_2=1.$$

在标准β∈[0,1]下，沿u的转移特征值为α(1−β)，其正交补为α，因此算子范数不超过α≤1。若把未经归一化的f_K=a_K u直接代入，则沿u的特征值变成α(1−βa_K²)。保证非扩张的充分条件为0≤βa_K²≤2；考虑α后精确条件是|α(1−βa_K²)|≤1，另需α≤1。未经约束的指数幅度不能保证这一点。

若沿用原生L2归一化，则

$$\frac{a_Xp_X}{||a_Xp_X||_2}=\frac{p_X}{||p_X||_2}.$$

正幅度被完全删除（忽略实现eps和舍入）。因此ad_norm实际只能检验正方向feature的作用，不能把其收益归功于独立幅度分支。

**更强的障碍：静态kernel相同，不代表GDN相同。** 对任意常数c>0，令f'_Q=f_Q/c,f'_K=cf_K，则f'_Qᵀf'_K=f_Qᵀf_K。但delta擦除项包含f_Kf_Kᵀ，变成原来的c²倍。m=1、α=1、β=.5、两次写入v=[1,0]、原f_Q=f_K=1时，读出为[.5,.25]；c=3后仍然同一个恒等于1的静态kernel，读出却为[.5,−1.75]。

所以仅凭“AD比HH更好地逼近exp(qᵀk/√d)”无法预测GDN收益：delta需要同时分析key-key几何、擦除与写入，而非只有query-key二元kernel。

上述缩放反例是绕过L2时的例子；若保留L2，那个常数缩放本身会消去。更强、适用于归一化情形的反例是：两时刻的查询都为r=(1,1,1)/√3。构造A的两个key都为u=(2,1,1)/√6；构造B的两个key分别为u及u'=(1,2,1)/√6。全部向量严格为正且为单位向量，两构造的完整2×2 query-key矩阵完全相同，每个元素都是c=4/√18。令α=β=1、v1=1、v2=0，则构造A第二次读出为0，构造B为c(1−uᵀu')=c/6≈0.157135。两者都稳定，但输出不同。因此，即使固定单位范数、排除幅度不稳定，交叉kernel仍不能单独刻画delta记忆，必须纳入key-key几何。该反例已用FP64显式递推验证。

**稳定的幅度接入扩展。** 令u=p_K/||p_K||₂，读方向r=p_Q/||p_Q||₂，保持擦除β，而单独调制写入：

$$S_t=\alpha_t(I-\beta_tu_tu_t^\top)S_{t-1}
+\beta_t a_K(k_t)u_tv_t^\top,\quad o_t=r(q_t)^\top S_t.$$

它可直接由原生GDN算子实现：Q/K使用方向，V改为a_K V。本轮a_K=exp(4tanh(s_K/4))，幅度在[e^{-4},e^4]内；状态转移仍非扩张，并有

$$||S_t||_F\le\alpha_t||S_{t-1}||_F+\beta_t a_K||v_t||_2.$$

如果α统一小于1、v有界，可进一步求几何级数界；α可趋近1时不能宣称状态有与长度无关的绝对上界。这个扩展对应分离擦除和写入强度，不等于原始AD正kernel在标准归一化linear attention中的计算，也不能主张两门分离新颖性，已有[Gated DeltaNet-2](https://arxiv.org/abs/2605.22791)等近邻。

即使r,u都正，delta最终的等效attention权重也可能为负，因为I−βuuᵀ存在负的非对角项。两维例子，k1=(1,0)、k2=(1,1)/√2、q2=(0,1)、β=α=1，则第二次读出对v1的系数为−1/2。故不能对GDN完整等效权重继续直接使用正kernel的原始I误差分解；需要分析带符号的序列算子风险。

**收益假设与局限。** GLA的K幅度和GDN的独立写入幅度有机会改善写入强度，但原门控/V投影已经有相关调节能力，额外分支是否值得其计算只能实测。L2归一化使直接AD的幅度贡献为零；softmax正方向还可能让keys更相似、增大记忆干扰。正性对稳定的归一化softmax近似很有用，对允许带符号读写的GLA/GDN未必是优势。

本轮在预训练原生模型上固定主体、两层局部CE适配，是有限训练预算下的迁移可行性测试，不是从头训练的架构排名，也不是原始softmax kernel拟合实验。使用原生GLA/GDN的公开FLA chunk/fused recurrent算子，另外做显式递推校验。

**通向分布理论的正确扩展，而非沿用静态谱尾。** 定义序列分布P_seq上的读出风险R_seq=E[(1/T)Σ_t||ô_t−o_t||²]。对于相同α、β、V路径、单位key方向u/û和非负写入幅度a/â，两种状态转移都非扩张。由||ûûᵀ−uuᵀ||₂≤2||û−u||₂可得逐路径界（Δ_t=||Ŝ_t−S_t||_F）：

$$\Delta_t\le\alpha_t\Delta_{t-1}
+2\alpha_t\beta_t\|\hat u_t-u_t\|_2\|S_{t-1}\|_F
+\beta_t\|\hat a_t\hat u_t-a_tu_t\|_2\|v_t\|_2.$$

当读方向也为单位向量时，||ô_t−o_t||₂≤Δ_t+||r̂_t−r_t||₂||S_t||_F。递推展开后在可积条件下对P_seq取期望，才得到与原生GDN机制匹配的分布风险界。这里比较方向必须在共同状态坐标中；允许共同正交变换时可先对齐。这是一个充分扰动界，可能很松，不是rank-m最小误差或接近总体最优的结论；也不能直接把读出误差换成LLM PPL定理。本轮没有通过数值实验估计这一总体界。完整模型改变上游输出后，α/β/V也可能变化，需要在界中再加入对应扰动项。

主要原始来源：[GLA](https://arxiv.org/html/2312.06635v2)、[Gated DeltaNet](https://arxiv.org/html/2412.06464v1#S3.SS1)、[FLA实现](https://github.com/fla-org/flash-linear-attention)。上述缩放与路径推导是对这些递推在当前AD特征下的直接分析，不主张独立新定理。



---

原文件：gaussian_go_nogo/REPORT.zh.md

判定：**原封不动的“真实 covariance→Gaussian 总体谱→真实 head 理论极限”主线暂不 Go；“真实经验谱→正特征近似”的方法探索有条件 Go。** 本次没有训练或微调 LLM。

这不是用户列出的“Gaussian 不准且 NMF≈FAVOR，所以全部停止”的组合。本次结果是：Gaussian 总体谱对多数 head 不适用，但有限矩阵的 NMF 明显优于 FAVOR+，因此两个研究环节需要分开判断。

**先校正理论陈述**

在 κ∈L²(PQ×PK)、允许独立且有符号的两侧平方可积 feature maps 时，Schmidt 分解给出

$$\inf_{\operatorname{rank}\hat\kappa\le m}\|\kappa-\hat\kappa\|_{L^2}^2=\sum_{r>m}\sigma_r^2.$$

它是正特征方案的下界，但不能称作所有正特征方案都可达到的最小误差。共享正指数特征+非负quadrature权重又比自由的两个非负因子更受限制。

对独立零均值 Gaussian，定义 A=ΣQ^(1/2)ΣK^(1/2)/√d，奇异值为 ai，则必须有 max ai<1/2。此时

$$\mathbb E\kappa^2=\prod_i(1-4a_i^2)^{-1/2},\quad
\rho_i=\frac{2a_i}{1+\sqrt{1-4a_i^2}},\quad
\sigma_\alpha=\prod_i\sqrt{1+\rho_i^2}\,\rho_i^{\alpha_i}.$$

该谱公式正确，可以由 [Mehler 恒等式](https://dlmf.nist.gov/18.18.E28)推出；本次使用 Gauss–Hermite 求积独立验证了一维谱、非零线性项及多维模式排序。当 max ai≥1/2 时，Gaussian surrogate 的二阶矩发散，不能再报告有限总体 L2谱尾，更不能把 ai 截到0.499。真实 LLM 的有限权重、RMSNorm 和 RoPE 给出不同的有界支持情形；Gaussian 失效不表示真实 kernel 发散。

真实 Q/K 均值不为零。令 q=μQ+BQx、k=μK+BKy，b=BQᵀμK/√d、c=BKᵀμQ/√d、t=μQᵀμK/√d，l=[b;c]，M=[[I,−2A],[−2Aᵀ,I]]。在 HS 条件满足时，将 Gaussian 密度平方根吸收进核并平移 Lebesgue 坐标，得到

$$\sigma_\alpha^{\rm nonzero\ mean}
=\exp(t+l^\top M^{-1}l)\,\sigma_\alpha^{\rm zero\ mean}.$$

因此均值只改变总体谱的公共幅度，**相对谱尾**仍由 covariance 决定；但绝对误差需包含均值。有限样本矩阵却会强烈受到均值及其诱发的权重集中影响，不能把“总体相对谱与均值无关”搬到有限样本矩阵上。

**实验协议**

冻结 Qwen2.5-1.5B，复用真实投影、RoPE 后的 BF16 Q/K；第0、7、14、27层各选query head 0、3、6、9，共16个。选取覆盖不同层且不按Gaussian有效性筛选。正确处理 GQA；d=128；m=16/32/64/128。

校准用64篇训练文章，每个边缘分布32,768个向量，并检查16/32/64篇时的covariance估计。评估用20篇来源于官方WikiText-2 test split的固定1024-token文章前缀，是所选子集，不是完整benchmark。主实验每次从测试边缘池中独立抽取512个Q和512个K，构造P_Q×P_K的经验乘积矩阵，重复4次。矩阵包含跨文档配对，是分布核诊断，不是实际attention运行。另取4篇同上下文的合法512×512矩形块，作为部署相关对照；其分布不同，未混入主实验。

SVD使用Float64，每个矩阵的离散概率权重为1/512，绝对MSE是平方误差和/512²。NMF求解同一原始kernel的Frobenius近似，初值包含NNDSVD、随机正因子以及较低rank解扩展；每种初值1600次乘法更新，取每个head的最低可行误差。它是测试矩阵上的transductive因子化，不是训练LLM，也不是已可泛化的feature network。实际NMF误差是正因子最优误差的可行上界，部分试验仍有优化空间，不能称为精确非负rank下界。

FAVOR+使用 [Performer 论文](https://arxiv.org/abs/2009.14794)的正高斯正交随机特征，Haar正交方向配独立χ_d半径；每个head10个种子，所有m使用同一投影矩阵的前m行。ψω(q)=exp(d^(−1/4)ωᵀq−||q||²/(2√d))。用logsumexp计算原始feature内积，没有epsilon或未恢复的逐行缩放。另报告正确恢复所有线性项和常数项的训练均值中心化版本，作为辅助基线。数学表达式与[官方实现](https://github.com/google-research/google-research/blob/master/performer/fast_attention/tensorflow/fast_attention.py)对应；官方用于归一化attention的数值平移在此必须恢复，才能评价原始K。

全部曲线评估未归一化 exp(qᵀk/√d)。为了跨head比较，正文使用相对平方误差；每个head先按真实kernel平方能量合并4次矩阵抽样，FAVOR再对10个种子平均。绝对MSE的对数同时保存，以免指数幅度溢出。图中≤1e−12的值放在显示下限，极小数不作为精确数值证书。

**四条曲线的结果**

Gaussian公式只有4/16个head满足适用条件；16/32/64篇校准的逐head检查保存在gaussian.json。对有效head以外的12个不填造数值。

下表是主实验16个head的中位数；FAVOR列先在每个head中平均10个种子。提升率定义为(E_FAVOR−E_NMF)/E_FAVOR。

| m | SVD误差 | NMF可行误差 | FAVOR+误差 | NMF相对提升中位数 | 提升≥25%的head |
|---|---|---|---|---|---|
| 16 | 0.0001401 | 0.0001671 | 1 | 99.983% | 16/16 |
| 32 | 4.671e-05 | 5.721e-05 | 1 | 99.994% | 16/16 |
| 64 | 1.328e-05 | 1.553e-05 | 1 | 99.998% | 16/16 |
| 128 | 2.434e-06 | 5.453e-06 | 1 | 99.999% | 16/16 |

这是明显的有限矩阵可改进空间；特别是存在低误差非负因子，说明在这个宽松的矩阵类中，positivity本身不必造成接近FAVOR的误差。但NMF并不约束因子来自同一指数feature族或共享quadrature节点，也不保证新Q/K上的误差。

![主实验四曲线](figures/product_four_curves.png)

m=64的逐head结果：

| head | max ai | Gaussian总体相对误差 | 真实SVD | 真实NMF | FAVOR+ |
|---|---|---|---|---|---|
| [0, 0] | 0.5757 | 不适用 | ≤1e−12 | 1.17e-08 | 1 |
| [0, 3] | 13.4612 | 不适用 | ≤1e−12 | ≤1e−12 | 1 |
| [0, 6] | 4.0236 | 不适用 | ≤1e−12 | ≤1e−12 | 1 |
| [0, 9] | 0.4210 | 0.8009 | ≤1e−12 | 8.198e-08 | 1 |
| [7, 0] | 0.3724 | 0.2564 | 5.179e-05 | 7.966e-05 | 0.9962 |
| [7, 3] | 0.6079 | 不适用 | 1.283e-05 | 1.549e-05 | 1.13 |
| [7, 6] | 0.3757 | 0.1204 | 2.358e-08 | 4.382e-07 | 1 |
| [7, 9] | 0.6560 | 不适用 | 1.405e-05 | 2.562e-05 | 1 |
| [14, 0] | 0.8281 | 不适用 | 1.374e-05 | 1.556e-05 | 1.001 |
| [14, 3] | 1.2499 | 不适用 | 1.892e-08 | 4.248e-08 | 1 |
| [14, 6] | 0.8466 | 不适用 | 0.0004662 | 0.0005385 | 1.371 |
| [14, 9] | 0.5602 | 不适用 | 1.132e-05 | 1.535e-05 | 0.999 |
| [27, 0] | 0.6160 | 不适用 | 0.002231 | 0.003806 | 1.448 |
| [27, 3] | 0.6090 | 不适用 | 0.002782 | 0.004533 | 1.005 |
| [27, 6] | 0.6229 | 不适用 | 0.00113 | 0.00178 | 1 |
| [27, 9] | 0.4056 | 0.3752 | 0.007348 | 0.01276 | 1.586 |

第0层的经验矩阵几乎退化到很低rank，不能让这些容易的head掩盖其余head的情况。同上下文对照单独见下图及curves.csv；原始kernel的极端权重集中使跨文档乘积矩阵与同上下文矩阵的误差、以及不同重复之间都可能明显不同。

![同上下文对照](figures/same_context_four_curves.png)

**用户提出的Spearman门槛是否通过**

比较总体谱时只允许使用4个有效head。绝对风险同时给出用户零均值公式及包含实际均值幅度的Gaussian修正；后者已经不再是只用covariance预测绝对误差。相对风险使用同一个Gaussian谱形状。另有一个完全不同的量：从拟合Gaussian抽取同样大小的有限矩阵，再计算其SVD，称为有限Gaussian预测。

| m | 有效head数 | 总体相对谱ρ | 零均值总体绝对ρ | 含均值总体绝对ρ | 有限Gaussian全16ρ | 有限Gaussian去第0层ρ |
|---|---|---|---|---|---|---|
| 16 | 4 | -0.200 | 0.400 | 1.000 | 0.704 | 0.308 |
| 32 | 4 | -0.200 | 0.400 | 1.000 | 0.722 | 0.350 |
| 64 | 4 | -0.200 | 0.400 | 1.000 | 0.740 | 0.392 |
| 128 | 4 | -0.200 | 0.400 | 1.000 | 0.785 | 0.497 |

原始总体Gaussian相对谱的相关性没有达到0.6，有效head数量也不足以支持跨8–16个head的预测主张。加入实际均值幅度后，4个有效head的绝对误差排序达到1.0，但这不再是原始零均值covariance-only公式，也没有预测准确误差量级。例如m=64时，(7,0)的绝对预测约高77倍，(7,6)约高17亿倍，(27,9)约高479倍。第0层的SVD尾值已经低于数值分辨率，其绝对尾误差与相关性还要额外谨慎。因此绝对幅度造成的高排序不能替代定量校准检验。4个head的置换检验和相关性细节保存在summary.json。

有限Gaussian全16个head的相关性达到约0.70—0.79，但这不是用户给出的闭式总体谱预测。去掉第0层四个极低误差head后，相关性仅约0.31—0.50；m=64时其余12个head的典型量级偏差约8.35倍，只有3个落在真实SVD误差的2倍范围内。不能凭全16个相关性超过0.6，就宣布“Gaussian理论下界预测成功”。

![两种不同的Gaussian预测](figures/gaussian_prediction_m64.png)

**为什么总体公式与有限SVD不能直接等同**

本次补做了真正Gaussian样本的控制实验。下面均为512×512矩阵、m=64，有限矩阵列是3次随机抽样的相对误差算术平均，故与主表的能量合并口径有所不同：

| head | Gaussian总体谱尾 | 零均值Gaussian有限矩阵 | 含均值Gaussian有限矩阵 |
|---|---|---|---|
| [0, 9] | 0.8009 | 0.1857 | ≤1e−12 |
| [7, 0] | 0.2564 | 0.1191 | 0.009318 |
| [7, 6] | 0.1204 | 0.05574 | 9.997e-06 |
| [27, 9] | 0.3752 | 0.1331 | 0.006751 |

即使Gaussian假设完全成立，小矩阵的oracle也明显低估总体最优风险。有限矩阵只看到有限支持，并且可以针对该矩阵重新选择最优因子；总体谱要求一个函数对整个分布成立。权重集中及稀有事件进一步放大差异：例如(0,9)的含均值Gaussian总体log E[K²]约1381，而有限样本中观测到的尺度远低于它。这不是Mehler公式错误，而是总体与有限样本没有对齐。

控制还比较了n=256/512/1024以及真实矩阵n=1024，全部数值在results/controls.json。有限谱随n变化，不支持将512点oracle视为已经收敛的总体谱估计。对于HS无效的Gaussian，任何有限抽样矩阵依然存在，但它的有限误差绝不能冒充总体L2界。

**Go／No-Go的具体决定**

原主线暂不Go的原因有三项：大多数真实head的covariance对应一个HS失效的无界Gaussian surrogate；仅余4个有效head的总体相对谱没有预测真实矩阵的排序和量级；总体谱与有限SVD的直接对照还混入严重有限样本偏差。当前不能基于这些ρ_i给大多数head设计有理论保证的quadrature。

正特征方法探索可以有条件Go，因为已经找到远优于FAVOR+的非负矩阵因子。不过下一阶段真正需要跨过的门槛是：用训练数据确定、并在独立文档上泛化的正feature map或共享正quadrature，也能缩小这个gap。NMF目前只证明宽松矩阵类里有空间；相同ψω和非负wr的quadrature是更小的函数类，不能直接继承NMF误差。

推荐将主线改成“有界/混合真实分布→经样本规模检验的经验或总体谱→正特征可行性→跨文档泛化”，将Gaussian闭式谱保留为适用域明确的对照。对于仍在探索的Gaussian有限矩阵预测，应独立报告它是有限尺度代理模型，而不是理论下界。不要通过降低temperature、截断ai或删掉极端Q/K来隐式改变原始kernel后宣称原主线成立。

NMF≈FAVOR也不应单独作为数学意义的No-Go证据，因为数值NMF可能没优化好；本次实际没有出现这种组合。反过来，优于未经LLM适配的FAVOR+也不足以证明优于学习型正特征方法，更不足以支持顶会结论。

**验证、文件与复现**

主实验128个矩阵，4个rank；FAVOR+及均值修正版各10个种子。SVD下界、NMF非负更新的目标单调性、FAVOR原始kernel的log域计算与直接计算、均值修正恒等式，以及已知精确非负因子的固定点均检查通过。NMF冷启动不保证达到全局最优；method_checks.json特意将这一点与数值正确性分开。

`curves.csv`保存每个head/rank的四曲线及辅助baseline；`summary.json`保存门槛判定、相关性、绝对误差对数及审计。`results/`保存每个抽样矩阵的索引、SVD谱、所有NMF优化轨迹、所有FAVOR种子结果、Gaussian参数及控制实验。模型revision为`8faed761d45a263340a0528343f099c05c9a4323`，原始缓存manifest记录文档来源和哈希。

```bash
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python kan_attention_theory/gaussian_go_nogo/run.py --nmf-steps 1600 --favor-seeds 10
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python kan_attention_theory/gaussian_go_nogo/controls.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python kan_attention_theory/gaussian_go_nogo/finite_gaussian.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python kan_attention_theory/gaussian_go_nogo/check_methods.py
/root/miniconda3/bin/python kan_attention_theory/gaussian_go_nogo/summarize.py
```

运行位置为/root/autodl-tmp。主脚本可跳过已有矩阵结果；从头运行可先指定新的--out目录，并让控制及汇总脚本读取对应目录。根目录早期smoke输出仅用于流程验证，正式结果统一在results/。



---

原文件：hedgehog_matched/PRODUCT_PROTOCOL.zh.md

这是在首轮同参因果训练的跨文档测试暴露异常后追加的、明确标记的机制实验。不能回填为首轮预注册，也不把既有确认集称为新盲测。目的为直接比较真实经验边缘乘积分布上的原始kernel拟合。

架构保持首轮精确同参ad_raw/hh_raw，每head73729参数，m64/m576。LLM冻结。仍用4096原训练文档、各64个唯一Q。

对文档i的Q，训练keys来自16个其他训练文档，每篇抽64个位置，组成1024个K。16个非零文档位移和64个位置在看该阶段结果之前固定；每个Q文档只更新一次，选定Q/K pair不重复。多个pair复用K向量，不冒充独立样本。评价乘积分布则沿用不同的留出文档A/B，禁止在测试上拟合。

具体索引：文档位移[127,251,509,761,1019,1279,1531,1789,2053,2309,2557,2819,3067,3323,3581,3833]，按4096取模；K位置0,16,...,1008。keys选择与Q内容、kernel值无关。覆盖全部4096个key来源文档；有限bank采样并不等于精确未知总体积分。

训练目标是真正未平衡的原始I：
E_pairs[d_I(κ,h)]/M_train，其中M_train为仅在这批固定训练pair上估计的原始kernel均值。每head一个固定缩放，恢复raw幅度后评价；不是按Q或上下文归一化目标。先计算训练标签总量确定缩放，不更新网络、不算第二训练epoch；之后所有拟合pair只反向传播一次。使用logsumexp估计缩放，训练均值缩放还能限制最大训练目标的数值范围。

两种方法获得相同lr搜索：seed11的0.002/0.0005两次单遍训练，按独立validation前32个Q文档与后32个K文档的固定1024×1024经验乘积上的relative raw I选择；然后所选lr训练seed29、47。全部验证参数冻结，无测试配置搜索。AdamW/余弦调度/梯度裁剪沿用首轮。

评价包括：留出A/B两方向完整所选经验乘积；同文档1k/8k实际因果kernel/输出；同两层24head的冻结模型PPL。原始I/L2/attention KL/PPL分开。此阶段损失和训练配对均改变，不能把与首轮之差只归因于一个因素。



---

原文件：hedgehog_matched/PROTOCOL.zh.md

本轮在查看新对照的留出结果前固定：检验幅度—方向与Hedgehog在相同活跃参数量下的质量差异。LLM冻结，不做预训练、LoRA或模型权重微调。只学习真实post-RoPE Q/K的feature maps。

先纠正旧对照：causal_direction/common.py的hedgehog共享Q/K矩阵；原论文附录A.3使用独立Q、K模块。本轮使用独立模块。旧比较保留但不能作为充分复现Hedgehog的证据。

严格同参主配对：
- AD：Q/K各128→192→64，SiLU隐藏层，幅度—方向输出，所有线性层不带可学习偏置。
- HH：Q/K各128→288，独立权重，拼接exp(z)、exp(−z)，最终m576，投影不带可学习偏置。
- 两者线性权重均为73728/head。原始kernel目标配对各增加一个实际学习的head整体log标度，均73729/head；KL配对不增加这个未被KL识别的参数，均73728/head。没有填充无效参数。
- AD去掉了原版128个输出偏置；HH是加宽且去偏置的公式适配，不宣称是论文默认架构。相同输入标准化可折叠进线性映射。旧AD全偏置版本只作参考。
- ADm64、HHm576，后者状态是前者9倍；双侧投影主导FLOPs均147456/head/token，线性状态更新读取分别约32768和294912。相同参数不等于相同状态/总计算。

模型/损失：ad_raw、hh_raw均balanced I（λ=1，原始未归一化κ为目标）；ad_kl、hh_kl均标准方向KL，明确是控制。另加hh_softmax_kl：论文附录讨论的稳定激活，分别对z、−z做特征坐标softmax后拼接，参数仍73728；它用于检验HH的强方向对照，不作为原始标度已拟合的模型。

HH扩宽初始化：首128行单位阵，剩余由种子固定的正交块构成，避免矩形eye初始化导致所有新增零行永久对称。Q/K初始化相同，之后独立更新。AD沿用原两层网络随机初始化规则。不同架构不能声称逐参数同初始化。

数据完全复用既有causal_direction训练/validation，4096篇、每篇64个唯一Q、1024K合法前缀，单遍4096次更新。不同方法同种子使用相同文档顺序与配对。训练目标非MSE。

同等调参：每种方法seed11各试lr0.002和0.0005，只用既有validation前32篇上的训练对应目标选择；随后以所选lr训练seed29、47。AdamW/weight_decay1e−4、余弦末lr=初lr/10、原梯度裁剪不变。完整最终检查点评估，不按测试选epoch、初始化或lr。每方法四次训练（两个lr试验+另两个种子），预算相同。

评价：既有confirm_wiki128篇1k、confirm_long24篇8k，均与训练文档分开。这些集合以前已用于研究，不称新盲测。记录逐文档/逐head原始I、balanced I、平方误差、幅度误差、方向KL、输出NMSE；KL控制未经标度校准的raw误差不用于宣称原始kernel胜利。另用不相交文档Q/K边缘的固定经验乘积做函数评价，所有方法参数冻结，禁止在测试矩阵拟合。

实际模型PPL仅同两层24/336Qheads替换，全部下一token位置，纯线性attention，不混入窗口或增强。三种子和配对文档区间分别报告。速度使用一致的参考特征与状态实现，不能拿旧m64优化算子与未优化m576代码冒充算法速度差；相同CUDA graph条件微测并明确硬件、dtype、状态大小。

分析既允许AD胜也允许HH胜；1k/8k、raw风险/方向/PPL分别判断。原论文完整任务微调不在本轮scope，不能把本实验写成击败整个Hedgehog系统。

来源：Hedgehog https://arxiv.org/html/2402.04347v1 ，尤其A.1—A.3；Performer https://arxiv.org/html/2009.14794v4 。



---

原文件：hedgehog_matched/REPORT.zh.md

本轮完成精确同参训练与真实LLM评价。**在直接乘积分布训练后，幅度—方向在两个留出经验乘积分布上的原始I风险均低于同参Hedgehog-exp。** 这仍是有限模型、有限文档与单遍训练预算下的结论；不同训练目标、长文外推、原始kernel风险和PPL必须分别判断，不能概括为在所有意义上优于Hedgehog。

**先纠正旧基线。** 此前本地Hedgehog共享Q/K权重，而论文转换代码使用独立Q、K映射。本轮已改为独立映射。旧四千参数对照不应作为充分复现Hedgehog的证据。

**参数严格相同，但状态不同。** 原始损失配对均为每head73729个活跃参数，KL配对均73728。AD两侧128→192→64，隐藏SiLU；HH两侧128→288并拼接exp(z)、exp(−z)，得到m576。统一去可学习投影偏置；原始损失配对各增加一个实际训练的head整体log标度，避免HH的固定原始幅度底限造成额外不公平。没有添加无效参数。相同训练输入标准化可以折叠入第一层仿射映射。

AD去掉原版128个输出偏置；HH为加宽、去偏置的公式适配。故这是精确同参的架构比较，不是论文默认配置复现。HH初始首128行单位阵，其余为种子固定正交块，避免加宽的零行在训练中保持完全对称；Q/K初始相同，但参数独立更新。两种架构不可能逐参数使用同一初始化。

计数补充：KL控制的73728是登记的可训练参数量，不是可识别自由度。AD的query幅度输出行有192个权重，方向KL对这一分支不识别；因此不能把KL控制也称为“损失有效自由度完全相同”。原始I主比较中该分支会影响目标，73729的主参数匹配不受此问题影响。共享MLP内部仍存在通常的参数化冗余，不主张统计可识别性定理。

双侧线性权重计算均147456主导FLOPs/head/token。ADm64，HHm576，状态是9倍；加状态更新读取后约180224与442368 FLOPs，即HH约2.45倍。相同参数不意味着相同计算或状态。

**实际执行范围。** 冻结Qwen2.5-1.5B，只替换层14/27全部24/336Qheads，未微调任何LLM权重。训练4096文档，每个主拟合单遍。每个方法相同lr搜索0.002/0.0005、seed11，验证选择后扩展seed29/47；都选0.002。首阶段20次训练，补充阶段8次，总记录网络训练时间约12.78分钟，不含提取、标签预积分和评价。两个阶段的协议在各自训练前固定，补充阶段是看到首阶段乘积失败后的明确追加实验。

**为什么补直接乘积分布训练。** 首阶段沿用同文档因果配对，AD/HH用同balanced I（λ=1）比较raw目标，另有同参KL及HH-softmax方向控制。首阶段在1k因果评价有优势，但跨文档经验乘积的AD风险异常大。独立CPU FP64复算最差head L27H10确认不是实现错误：某pair目标logκ约−2.36、预测约34.72。该例用于事后故障诊断，不是选择测试样本训练。原始结果完整保留。

只在同文档因果测度上学好，不能推出PQ×PK风险低。补充阶段对每个训练Q文档选16个其他文档、每篇64个K，形成1024-key bank；4096文档全部用作Q/K来源，无同文档pair，每个Q及选定pair只反向传播一次。训练pair为268435456/head，向量复用不代表同数量独立样本。

补充阶段使用**未经query平衡的原始I-divergence**，只除以固定head训练kernel均值作数值缩放。该均值仅从固定训练pairs预积分得到，之后恢复原始幅度评价；它不是当前query或attention行归一化。输入目标始终exp(qᵀk/√128)。损失与配对均相对首阶段改变，因此两阶段之差不能只归因于一个因素。细节见[PRODUCT_PROTOCOL.zh.md](PRODUCT_PROTOCOL.zh.md)。

**与你主线最相关：直接训练后的完整所选经验乘积。**

128个留出1k文档均分A/B各64篇；Q每文档取16个缓存位置、K每文档64个位置，1024×4096全部pair/head，另做反向B→A。模型测试前冻结，不在测试矩阵拟合，不求测试NMF。这里覆盖全部所选向量的乘积，不是所有缓存位置或未知真实总体。所有head/seed均值与head中位数一起报告。

| 方向 | 方法 | 原始I相对风险 | 原始L2相对误差 | I的head中位数 | L2的head中位数 |
| --- | --- | --- | --- | --- | --- |
| ab | 幅度—方向／原始损失 | 0.27245 | 0.365211 | 0.245344 | 0.385268 |
| ab | Hedgehog-exp／原始损失 | 0.46997 | 0.370372 | 0.484103 | 0.465624 |
| ba | 幅度—方向／原始损失 | 0.266388 | 0.299519 | 0.246776 | 0.325588 |
| ba | Hedgehog-exp／原始损失 | 0.470631 | 0.363239 | 0.494933 | 0.414812 |


原始I风险相对下降分别为42.03%与43.40%。按三个种子的每head均值，两方向均24/24heads更低；这不是把24个head当作24个独立模型做统计推断。原始L2在A→B的差异区间跨零，因此不能把I风险的稳定优势推广为两个方向都已确认L2优势。

差值为AD−HH，负值有利AD：

| 方向 | 指标 | 平均差 | AD更好head数 | 配对Q文档95%区间 |
| --- | --- | --- | --- | --- |
| ab | relative_raw_i | -0.197521 | 24/24 | [-0.214865, -0.184915] |
| ab | raw_nmse | -0.00516102 | 18/24 | [-0.0496231, 0.070815] |
| ba | relative_raw_i | -0.204243 | 24/24 | [-0.220323, -0.188936] |
| ba | raw_nmse | -0.0637199 | 21/24 | [-0.0753542, -0.0442103] |


这些区间条件于固定测试K bank与三个拟合模型，只重采样Q文档；不覆盖未知PK、完整训练随机性或未观测尾部。它们不是总体误差下界或近最优证书。

**直接乘积分布训练后的因果迁移。**

| 集合 | 方法 | 原始I相对风险 | 原始L2相对误差 | 输出NMSE | 全模型PPL |
| --- | --- | --- | --- | --- | --- |
| confirm_wiki | 幅度—方向／原始损失 | 2.02422 | 0.822823 | 0.446128 | 9.02579 |
| confirm_wiki | Hedgehog-exp／原始损失 | 2.37177 | 0.859789 | 0.655511 | 9.0632 |
| confirm_long | 幅度—方向／原始损失 | 5.24689 | 0.957457 | 0.785691 | 9.80391 |
| confirm_long | Hedgehog-exp／原始损失 | 19.1412 | 146.027 | 0.936142 | 9.86306 |


这张表是同文档实际因果配对及完整模型输出，不能与上一张乘积表混为同一个风险。高原始误差不能用PPL或归一化输出改善掩盖；反之，低原始乘积风险也不自动保证每个实际上下文的方向、输出或PPL。

**首阶段同损失、同参数的因果训练对照。**

| 方法 | 1k输出NMSE | 8k输出NMSE | 1k PPL | 8k PPL |
| --- | --- | --- | --- | --- |
| 幅度—方向／原始损失 | 0.261325 | 0.609218 | 8.90693 | 9.70089 |
| Hedgehog-exp／原始损失 | 0.752239 | 0.865445 | 9.1607 | 9.85453 |
| 幅度—方向／KL | 0.220439 | 0.641392 | 8.85819 | 9.69226 |
| Hedgehog-exp／KL | 0.231022 | 0.894505 | 8.87029 | 9.75236 |
| Hedgehog-softmax／KL | 0.264257 | 0.745977 | 8.88021 | 9.75233 |


原模型PPL为8.664560/9.066279。首阶段AD与HH在相同KL目标下的比较检验的是两种特征参数化；raw训练对KL控制的比较还包含目标差异。KL控制未经query幅度校准的巨大raw误差不作为原始kernel胜利证据。HH-softmax具体为concat(softmax(z),softmax(−z))，是论文所讨论稳定化思路的一个明确实现，不能冒充全部官方实现/训练流程。

| 训练阶段 | 集合 | 比较 | ΔNLL/token | 配对文档95%区间 |
| --- | --- | --- | --- | --- |
| 因果训练 | confirm_wiki | ad_raw − hh_raw | -0.0280928 | [-0.0304508, -0.0258833] |
| 因果训练 | confirm_wiki | ad_kl − hh_kl | -0.00136551 | [-0.0021798, -0.000545376] |
| 因果训练 | confirm_wiki | ad_kl − hh_softmax_kl | -0.00248292 | [-0.00340139, -0.00156855] |
| 因果训练 | confirm_wiki | ad_raw − hh_kl | 0.00412189 | [0.00321102, 0.0050406] |
| 因果训练 | confirm_wiki | ad_raw − hh_softmax_kl | 0.00300448 | [0.00205699, 0.00396533] |
| 因果训练 | confirm_long | ad_raw − hh_raw | -0.0157134 | [-0.0188192, -0.0127479] |
| 因果训练 | confirm_long | ad_kl − hh_kl | -0.0061835 | [-0.0087731, -0.00374532] |
| 因果训练 | confirm_long | ad_kl − hh_softmax_kl | -0.00618004 | [-0.00772019, -0.00464838] |
| 因果训练 | confirm_long | ad_raw − hh_kl | -0.00529156 | [-0.00815108, -0.00285276] |
| 因果训练 | confirm_long | ad_raw − hh_softmax_kl | -0.0052881 | [-0.0068959, -0.00354132] |
| 乘积分布训练 | confirm_wiki | ad_raw − hh_raw | -0.00413638 | [-0.00595658, -0.00241687] |
| 乘积分布训练 | confirm_long | ad_raw − hh_raw | -0.00601573 | [-0.00841601, -0.00342925] |


区间条件于三个冻结种子，文档作为重采样单位；探索性多比较未做统一多重校正。现有confirm_wiki128篇、confirm_long24篇此前已用于研究，不能称首次盲测。没有第二模型或全层线性化确认。

**时间与状态。** RTX3090、FP32、batch1、单层12heads，真实Q/K/V，CUDA graph热缓存，同一PyTorch参考特征及状态实现：

| 方法 | m | 仅特征 μs | 特征+状态 μs | 单层状态 bytes |
| --- | --- | --- | --- | --- |
| ad_raw_s11_lr0.002 | 64 | 42.688 | 74.204 | 399360 |
| hh_raw_s11_lr0.002 | 576 | 28.288 | 98.24 | 3594240 |
| ad_kl_s11_lr0.002 | 64 | 42.656 | 74.208 | 399360 |
| hh_kl_s11_lr0.002 | 576 | 28.256 | 98.208 | 3594240 |
| hh_softmax_kl_s11_lr0.002 | 576 | 33.342 | 103.36 | 3594240 |
| favor_s11 | 64 | 27.52 | 59.04 | 399360 |


这些是统一参考实现的微测，不是融合到最优的算子，也不是全模型decode时间。不能与旧Triton融合m64的约5–10μs结果直接拼接比较。首阶段计时后的乘积训练改变了权重、没有改变算子结构；没有为补充权重另作计时，成本只作同结构参考。

**数值验证。** 两阶段均通过FP64显式矩阵/分块scan/递推一致性，以及实际8k keys与真实64个Q的FP32递推对FP64显式参考检查。首阶段最大相对输出误差约3.5e−6，未出现零分母；补充阶段精确值在checks/long_precision_product.json。PPL使用完整冻结模型、逐文档所有下一token位置，未混用窗口、旋转增强或LoRA。

**理论解释及边界。** 仿射+正指数HH是(q,k)的联合凸函数，而原始exp(qᵀk/√d)一般不联合凸。THEORY.zh.md给出一个真实乘积分布上的三点支持反例：即使增加HH宽度，仍有正的逼近误差，而自由非负rank3可精确表示。两层非线性AD不受同样的联合凸性约束。但该反例不是当前LLM总体的误差下界，也不适用于带特征softmax或非线性query校准的全部HH变体。这涉及非线性深度与输出结构，不是幅度—方向独占优势。

因此，这轮能评价精确预算下的实测质量、训练测度与泛化缺口；不能证明AD在所有分布、充分优化预算或全模型系统上必然优于Hedgehog，也没有接近总体谱下界的证据。

**Hedgehog与FAVOR+究竟怎样训练。** Hedgehog原论文两种都做：从头训练时，feature MLP和原模型权重一起用任务损失学习；转换已有模型时，先冻结Transformer，以相同Q/K的softmax attention为教师，训练feature map匹配归一化权重，再进行任务微调或LoRA。见[原论文A.3](https://arxiv.org/html/2402.04347v1#A3)。

FAVOR+本身是正交正随机特征近似，基础随机投影通常不是通过教师拟合SGD学出来的。Performer可以从头训练，此时主要学习原模型投影/FFN等权重，随机特征可以重采样；也可替换已有Transformer再评价或继续训练。理论可替换不保证任意有限m、任意已训练LLM立即无损。见[Performer原论文](https://arxiv.org/html/2009.14794v4)。

本轮属于冻结模型后的特征拟合与局部替换，只对应转换流程中的特征阶段。不能把Hedgehog经过后续任务微调的论文结果，与本轮无模型微调的分数直接比较。

**复现与证据。** 首阶段依次train.py、driver.py、summarize.py；补充阶段train_product.py后，设置MATCHED_PLAN=product_plan.json运行driver.py与summarize.py。最后运行write_report.py、audit.py。results/summary.json与summary_product.json保留三种子、逐head聚合与配对区间；fits/保留全部28次拟合及曲线，checks/final_audit.json和results/artifact_manifest.json记录预算与来源检查。



---

原文件：hedgehog_matched/THEORY.zh.md

本轮比较的一个表达限制：**单层仿射映射后的正指数Hedgehog，是(q,k)联合凸的函数；原始exp(qᵀk/√d)一般不是。** 这有助于解释为何单纯增加单层指数特征宽度不必然解决原始kernel拟合，但并未证明这个障碍在当前真实Q/K支持上决定了实验误差。

独立Q/K映射的指数Hedgehog为

$$g(q,k)=c\sum_r \left[
e^{a_r^\top q+b_r^\top k+\beta_r}
+e^{-a_r^\top q-b_r^\top k-\beta_r}
\right],\quad c>0.$$

每个指数都是联合变量(q,k)的仿射函数的指数，所以和为联合凸函数。任意宽度、仿射偏置、固定输入中心化/缩放及正的head全局标度都不改变这条性质。

取x≠0，u=||x||²/√d，考虑三对输入(x,−x)、(−x,x)、(0,0)。两端目标为e^(−u)，中点为1；凸性要求g(0,0)≤[g(x,−x)+g(−x,x)]/2。若三处最大绝对逼近误差≤ε，则

$$1-\epsilon\le e^{-u}+\epsilon,\qquad
\epsilon\ge(1-e^{-u})/2.$$

因此增加这类指数Hedgehog的m不能使其在包含这些点的区域上一致逼近目标到任意精度。

也可以把这个反例放入真正的乘积分布。令P_Q=P_K为{-x,0,x}上的均匀分布，设Δ=1−e^(−u)。记上述三点的预测误差为e_+,e_-,e_0，凸性给出e_0−(e_++e_-)/2≤−Δ。Cauchy–Schwarz于是给出

$$E_{P_Q\times P_K}(\kappa-g)^2
\ge\frac19(e_0^2+e_+^2+e_-^2)
\ge\frac{2\Delta^2}{27}>0.$$

相反，这个有限支持上的任意kernel矩阵均有非负rank≤3（使用Q侧三个指示函数和对应正kernel行），所以自由正rank3的最优值为0。该例说明“特征维度充足”与“具体feature map族足够表达”是不同问题。

边界：
- 这不是当前真实LLM分布的误差下界。实际支持未必包含这些三点或其足够概率邻域。
- 这条论证适用于仿射+exp及其正负拼接。论文也讨论特征softmax；其分母随输入变化，上述联合凸性不再自动成立。因此本轮加入hh_softmax_kl。
- 非线性两层网络不受相同联合凸性限制，但不能据此证明同参数AD在任意分布上都更好。
- 这个区别涉及非线性深度与输出结构，不是幅度—方向参数化独占的优势。
- 对固定g乘非线性查询标度也可破坏联合凸性；不能把此限制原封不动用于已做查询函数校准的Hedgehog。
- 这是直接的凸性推论与构造反例，未完成新颖性检索，不主张新定理贡献。

本轮raw比较仍使用非MSE的balanced I训练，上述L2反例只解释函数类，不作为训练损失或真实总体性能证明。



---

原文件：kernel_comparison/REPORT.zh.md

后续更新：已完成双方融合算子、完整模型 4/336-head 替换困惑度、缓存推理和更紧的完整经验谱界，见[部署与谱界验证报告](../deployment_validation/REPORT.zh.md)。新正 kernel 保留了质量优势，但本轮没有推理收益，且在完整经验算子上明显不接近无约束 rank-64 最优误差。

本轮已完成用户要求的同协议kernel对比、可调用Galerkin特征构造与计算量评估。**在这四个真实Qwen head及指定留出协议上，m=64新分区正kernel的raw-kernel误差优于本轮FAVOR+、均值校正FAVOR+、SDERF、ADERF的全部五个测试种子；但它明显更贵。Galerkin的乘积分布kernel误差更低，却因负kernel项及分母抵消，在真实attention块上出现输出不稳定。**

此前“新kernel与FAVOR尚无同协议比较”的状态已由本轮补齐。这个结果不等于新方法接近总体rank-m下界，也不等于全模型困惑度或速度已获改善。

**Galerkin与已有linear attention研究的关系。** Galerkin attention并非没有相关文献。Shuhao Cao的NeurIPS 2021论文《Choose a Transformer: Fourier or Galerkin》提出了与Petrov–Galerkin投影联系的线性attention，典型形式为Q( K̃ᵀṼ )/n，实验针对PDE算子学习。它并不是对冻结LLM的exp(qᵀk/√d)进行训练分布谱截断。[原论文](https://proceedings.neurips.cc/paper/2021/file/d0921d442ee91b896ad95059d13df618-Paper.pdf)

这里的“训练Galerkin”是本项目由真实Q/K训练积分算子构造的kernel，应作为我们自己的候选构造或对照，不能冒充上述论文的复现，也不能替代FAVOR类基线。

**本轮比较对象与协议。** 主基线为FAVOR+的正交高斯正特征，以及DERF论文的SDERF和ADERF解析特征，另加入利用训练均值精确分解原始点积的均值校正FAVOR+。FAVOR+使用高斯边缘的正交节点：Haar正交方向乘独立χ₁₂₈半径，不采用固定半径的regularized softmax替代原始kernel。SDERF和ADERF是论文特征公式的实现，不是完整FAVOR#语言模型系统的复现。[Performer原论文](https://arxiv.org/abs/2009.14794)、[DERF原论文](https://proceedings.neurips.cc/paper_files/paper/2023/file/02dec8877fb7c6aa9a79f81661baca7c-Paper-Conference.pdf)

固定Qwen/Qwen2.5-1.5B，revision为8faed761d45a263340a0528343f099c05c9a4323，使用实际投影和RoPE后的Q/K；d=128，head为L14H0、L14H6、L27H0、L27H6。随机特征的均值、二阶矩与解析参数来自与新kernel相同的4096篇训练文档，每篇每侧32个位置，共131072个训练向量/边缘。新kernel原先将这些文档分为两组：一组构造谱分区，另一组估计cell均值。测试时所有方法参数冻结。

主实验m=16/32/64/128，RF各五个种子1009、1046、1083、1120、1157；不同m使用同一个128节点集合的嵌套前缀并修正1/√m。所有方法使用完全相同的留出向量、原始kernel与经验概率权重。内部留出为256篇文档，每侧131072个向量的完整乘积，四个head共68719476736个pair；官方子集为固定20篇WikiText2 test文章前缀，每侧10240个向量的完整乘积。内部数据来自WikiText103源train按文档划分，官方子集不是完整官方benchmark。统计独立单位不能按pair数量计算。

目标和主要指标为

$$\kappa(q,k)=e^{q^\top k/\sqrt{128}},\qquad
\mathrm{NMSE}=\frac{\sum_{i,j}(\hat\kappa(q_i,k_j)-\kappa(q_i,k_j))^2}{\sum_{i,j}\kappa(q_i,k_j)^2}.$$

没有对标签做行归一化、裁剪或温度调整。分块Float64计算完整产品风险。为避免存储整张矩阵，利用

$$\|FG^\top\|_F^2=\operatorname{tr}[(F^\top F)(G^\top G)],\qquad
\langle K,FG^\top\rangle=\sum_{i,r}F_{ir}(KG)_{ir}.$$

不同方法复用同一批真kernel值。数值用的head/特征常数都精确恢复；不是测试时拟合幅度。新分区和Galerkin风险复用上轮完整产品算子矩，并独立核验优化后的可调用特征与原构造一致。

**同维度raw-kernel结果。** 以下RF列为五个种子的中位数，新分区及Galerkin为固定的一次训练构造。预测恒零的NMSE为1。

内部完整经验乘积，m=64：

| head | FAVOR+ | 均值校正FAVOR+ | SDERF | ADERF | 新分区kernel | 训练Galerkin（有符号） |
|---|---|---|---|---|---|---|
| L14H0 | 1.0000 | 1.6251 | 22 | 9.21e+03 | 0.2065 | 0.1717 |
| L14H6 | 3.8765 | 50.6 | 309 | 45.2 | 0.4677 | 0.4776 |
| L27H0 | 1.0045 | 768 | 253 | 53.3 | 0.7842 | 0.5123 |
| L27H6 | 1.0000 | 225 | 804 | 5.2238 | 0.6608 | 0.3181 |

官方20篇文档完整经验乘积，m=64：

| head | FAVOR+ | 均值校正FAVOR+ | SDERF | ADERF | 新分区kernel | 训练Galerkin（有符号） |
|---|---|---|---|---|---|---|
| L14H0 | 1.0000 | 1.1222 | 2.0182 | 1.27e+03 | 0.2541 | 0.2197 |
| L14H6 | 1.0014 | 1.7381 | 2.2886 | 1.6389 | 0.9456 | 0.9427 |
| L27H0 | 1.0005 | 458 | 239 | 1.8579 | 0.6879 | 0.3838 |
| L27H6 | 1.0000 | 25.8 | 330 | 1.1672 | 0.5987 | 0.2371 |

两个数据集上，新分区kernel都低于每一种RF基线的每个测试种子，m=64共160个逐种子比较均成立。这个陈述只覆盖这次已测试的种子、head和数据，不能推成随机特征方法的理论不可能性。Galerkin在内部3/4个head、官方4/4个head的raw乘积误差低于新分区kernel。

RF误差分布高度偏斜，因此不能只展示中位数。下面同时给出内部m=64的均值及范围：

| head | 方法 | 五种子均值 | 五种子最小—最大 |
|---|---|---|---|
| L14H0 | FAVOR+ | 1.0000 | [0.9999, 1.0003] |
| L14H0 | 均值校正FAVOR+ | 142 | [1.0473, 685] |
| L14H0 | SDERF | 2.06e+05 | [1.1307, 1.03e+06] |
| L14H0 | ADERF | 2.86e+05 | [8.4154, 1.39e+06] |
| L14H6 | FAVOR+ | 296 | [1.0929, 1.37e+03] |
| L14H6 | 均值校正FAVOR+ | 2.98e+03 | [13.7, 1.16e+04] |
| L14H6 | SDERF | 3.57e+04 | [10.1, 1.76e+05] |
| L14H6 | ADERF | 1.1e+04 | [17.4, 5.47e+04] |
| L27H0 | FAVOR+ | 1.0144 | [1.0000, 1.0449] |
| L27H0 | 均值校正FAVOR+ | 2.24e+03 | [53.8, 9.15e+03] |
| L27H0 | SDERF | 1.27e+03 | [37.5, 4.16e+03] |
| L27H0 | ADERF | 69.8 | [2.0985, 145] |
| L27H6 | FAVOR+ | 1.0137 | [1.0000, 1.0646] |
| L27H6 | 均值校正FAVOR+ | 2.5e+03 | [20.9, 6.37e+03] |
| L27H6 | SDERF | 2.71e+05 | [14.9, 1.35e+06] |
| L27H6 | ADERF | 32.4 | [1.0388, 101] |

FAVOR+不少head的典型误差接近零预测；另一些RF配置被过高预测主导，少数种子的误差非常大。SDERF/ADERF的解析方差目标并不保证这些冻结LLM Q/K上的有限节点L²风险优于FAVOR+。这组实验不能代表在其他输入缩放、任务训练、更多节点或其他调参协议下的论文系统结果。

![同协议完整产品曲线](figures/matched_kernel_curves.png)

**更高计算预算的随机特征对照。** 为避免只比较同m却忽略计算量，将原先五个128-node正交块预先全部合并，得到m=640的RF。不是选最优种子，也不是五个独立640维重复实验：它是一个固定的五块并集。完整kernel交叉项可由五个128维kernel交叉项取均值精确获得，预测平方能量另算全部跨块Gram项；没有遗漏跨节点块交互。

FAVOR+ m=640的结果为：

| 数据 | head | FAVOR+ m=640 | 新分区m=64 | Galerkin m=64 |
|---|---|---|---|---|
| internal | L14H0 | 1.0000 | 0.2065 | 0.1717 |
| internal | L14H6 | 16 | 0.4677 | 0.4776 |
| internal | L27H0 | 1.0012 | 0.7842 | 0.5123 |
| internal | L27H6 | 1.0030 | 0.6608 | 0.3181 |
| official | L14H0 | 1.0000 | 0.2541 | 0.2197 |
| official | L14H6 | 6.7828 | 0.9456 | 0.9427 |
| official | L27H0 | 1.0005 | 0.6879 | 0.3838 |
| official | L27H6 | 1.0007 | 0.5987 | 0.2371 |

另外三类m=640结果均保存在CSV。五块并集的平方误差均不超过对应五个128维kernel误差的均值，32组凸性检查通过。由于嵌套节点从64增加到128可能引入新的假峰值，m=640并不保证比m=64的某个种子或中位数更好。

m=640 FAVOR+的主导总FLOPs约是m=64 FAVOR+的10倍；新分区m=64约12.75倍，Galerkin m=64为12.5倍。因此这个参考更接近算子构造的预算，但仍不是严格等FLOPs或等时延。新分区与Galerkin在两个产品数据集的四个head上也都低于这个固定m=640 FAVOR+实现；不能据一个并集配置估计m=640的随机种子置信区间。

**Galerkin已实现为可调用kernel。** 设训练锚点数量a=1024，定义

$$z_Q(q)=[\kappa(q,\bar k_1),\ldots,\kappa(q,\bar k_a)],\quad
z_K(k)=[\kappa(\bar q_1,k),\ldots,\kappa(\bar q_a,k)].$$

先用训练字典的Gram矩阵正交化得到A_Q(q)、A_K(k)，再计算训练分布压缩

$$C_{\rm train}=\mathbb E_{\rm train}[\kappa(Q,K)A_Q(Q)^\top A_K(K)]
=U S V^\top.$$

这里A_Q、A_K写作行向量。取前m个奇异方向：

$$\phi_Q^G(q)=A_Q(q)U_mS_m^{1/2},\qquad
\phi_K^G(k)=A_K(k)V_mS_m^{1/2}.$$

将所有固定线性变换离线合并，可以直接实现为

$$\boxed{\phi_Q^G(q)=z_Q(q)C_Q,\quad
\phi_K^G(k)=z_K(k)C_K,\quad
\hat\kappa_G(q,k)=\phi_Q^G(q)\phi_K^G(k)^\top.}$$

C_Q、C_K均为a×m固定矩阵，运行时不用再显式生成256维中间谱坐标。代码中的head数值缩放是上述原始kernel表达的等价实现。优化后特征与原始多步变换的相对L²偏差低于2.8×10⁻¹⁵。

新分区kernel仍为

$$\hat\kappa_P(q,k)=e_{c(q)}^\top B e_{d(k)},\quad
\phi_Q^P(q)=e_{c(q)},\quad\phi_K^P(k)=B e_{d(k)}.$$

它的分区坐标来自锚点kernel的64维投影。此次将原先逐节点Python遍历改为路径判定矩阵计算，避免把低效遍历开销当作方法本身的成本；FP64划分与原实现逐项一致。该路由实现额外使用O(Nm²)的小矩阵运算，理论上的直接树遍历则可用O(N·depth)比较。

**更低kernel误差没有保证Galerkin的attention输出更好。** 在全部256篇内部文档和20篇官方子集上，额外计算实际同文档矩形块：后512个Q对前512个K，V也取真实缓存。该块内所有key在query之前；它不包含其他可见key，因此不是完整因果attention或完整LLM替换。kernel误差仍对raw exp函数计算，输出评估才使用标准线性attention分母。没有对负kernel项裁剪，也没有修补分母。

内部m=64输出相对平方误差及Galerkin符号诊断：

| head | FAVOR+输出误差中位数 | 新分区输出误差 | Galerkin输出误差 | Galerkin负kernel条目 | Galerkin非正分母行 |
|---|---|---|---|---|---|
| L14H0 | 1.4935 | 0.8558 | 2.4340 | 19.36% | 0.0328% |
| L14H6 | 1.0604 | 0.4926 | 0.3909 | 26.47% | 0.0107% |
| L27H0 | 0.8835 | 0.2829 | 7.4612 | 20.24% | 0.0359% |
| L27H6 | 3.1831 | 0.4227 | 360 | 19.06% | 0.0877% |

Galerkin约19%—27%的kernel条目为负；比例按条目数计，不代表相同的负质量比例。少量行的分母为负或发生严重抵消，会放大输出误差。L27H6的输出NMSE达到约360，明显差于新分区kernel的0.423。新分区kernel在本轮所有同文档块上都没有负kernel项或非正分母。

官方子集对应的Galerkin输出误差为1.935、0.502、0.291、1.526；新分区为0.835、0.426、0.301、0.445。Galerkin只有其中一个head略优于新分区，因此不能因为产品分布kernel误差更低，就把它当作稳定的正kernel替代。它可保留为有符号候选或谱逼近对照；逐元素裁剪kernel一般不能保持当前m维可分离表达，直接softplus两侧特征也会改变已证明/已测试的kernel。

新分区在同文档块的raw-kernel误差也明显高于产品分布：内部四个head为0.7575、0.9941、0.9512、0.9414。即便它比本轮RF基线更好，也不能说原始kernel已经拟合得很准确。P_Q×P_K理论与真实同文档联合测度之间的差异需要继续保留。

![真实同文档输出诊断](figures/attention_outputs.png)

**计算量需要包含特征生成。** 设每侧N个token，d_v为value维度，e=64为分区谱坐标维度，a=1024为锚点数。所有方法在特征已给定后，矩形线性attention的主导计算均为4Nm d_v FLOPs，推理流式状态为m(d_v+1)个标量/head。差异主要来自生成特征：

| 方法 | 两侧特征生成的主导FLOPs | 主要固定系数/head |
|---|---|---|
| FAVOR+ / 均值校正 / 折叠后的SDERF | 4Ndm | dm，另加小偏置 |
| ADERF | 4Nd(d+m) | 2dm+2d²，另加小偏置 |
| 新分区 | 4Na(d+e)+4Nm(m−1) | 2a(d+e)+m²，另加路由表 |
| 折叠后的Galerkin | 4Na(d+m) | 2a(d+m) |

这些是矩阵乘加主导项，按一次乘加2 FLOPs计算，不包含exp、比较、小量elementwise或归一化。SDERF的正交旋转可折叠进投影，且范数不变，因此其推理主导成本与FAVOR+相同。ADERF额外保留两侧一般二次型的范数计算。新分区和Galerkin每对Q/K需要2a次exp，FAVOR+为2m次；m=64时前者的exp数量也是后者的16倍。

**实测耗时。** RTX 3090，四个head，N_Q=N_K=2048，d=d_v=128，batch=1；无反向传播、关闭TF32，4次预热及20次CUDA-event计时取中位数。所有计时期间没有其他GPU实验。下表包括特征生成和矩形线性汇总，GFLOPs按上述主导公式计算。

| 方法 | m | 主导GFLOPs | FP32时间/ms | FP64时间/ms |
|---|---|---|---|---|
| FAVOR+ | 64 | 0.537 | 0.512 | 1.895 |
| 均值校正FAVOR+ | 64 | 0.537 | 0.515 | 1.894 |
| SDERF | 64 | 0.537 | 0.506 | 1.895 |
| ADERF | 64 | 1.074 | 0.572 | 3.029 |
| 新分区kernel | 64 | 6.843 | 1.389 | 15.905 |
| 训练Galerkin（有符号） | 64 | 6.711 | 1.244 | 15.771 |
| FAVOR+ | 640 | 5.369 | 0.938 | 11.923 |

新分区m=64在FP32下约为FAVOR+ m=64的2.72倍时间，Galerkin为2.43倍；FP64下两者约8.39和8.32倍。FLOPs比例与时间比例不同，是因为矩阵尺寸、GPU利用率、内存流量和算子启动开销不同。不能只按m宣称计算量相等，也不能将这些矩形块微基准称为完整LLM吞吐或完整因果prefill速度。

FP32下每head实际模型tensor存储约为：FAVOR+ m=64 32.5 KiB，新分区1585.5 KiB，Galerkin1536.0 KiB。N=2048时测得的增量峰值分别约12.2、66.0、66.0 MiB（四个head合计，含临时特征及锚点矩阵，未含已有模型/输入）。FAVOR+ m=640模型存储为325 KiB/head，增量峰值约81.3 MiB；其流式attention状态比m=64大10倍。固定参数与训练参数不能混为一谈，本轮是同特征维度及额外预算参考，不是等参数KAN/MLP架构比较。

FP64用于主准确度评估，FP32另在固定四篇留出文档的2048×2048产品上检查数值误差：新分区没有改变任何分区归属；新分区/Galerkin的NMSE与FP64最大绝对差均小于5.6×10⁻⁸。该检查支持本次FP32微基准的数值可用性，但不代替全量FP32或BF16模型评估。

![实测特征与汇总耗时](figures/latency.png)

**离线构造也有成本。** RF统计校准主要需要训练二阶矩和d×d矩阵分解。当前两个算子构造还需要1024×1024锚点kernel/SVD；本项目的精确moment-fit实现穷举65536²个训练pair/head，再累积条件均值或Galerkin核心。这个离线过程含二次训练配对成本，不能因为推理对N线性，就把整个构造流程也称为线性成本。实际部署时需要按不同模型/head及分布变化考虑重新构造的摊销。

**本轮判断。** 在已测试协议中，新分区正kernel确实获得了相对FAVOR类基线的拟合优势，也通过了固定参数的未见文档测试；这解决了此前缺少同协议比较的问题。代价是锚点特征计算、固定参数量和临时内存更大，仍未接近已知rank-m界。同文档raw kernel误差与产品测度结果差别很大，完整LLM困惑度与推理收益仍未评估。Galerkin可以作为具体kernel构造保留，但当前有符号版本不宜直接作为标准正线性attention的替换。

因此现阶段更准确的研究主张是“在冻结真实Q/K上，以更高特征计算成本换取比本轮RF基线更低的原始kernel风险”，而不是“已经实现更快、更准确且接近理论最优的线性attention”。还没有建立KAN或mulKAN的独占优势。

**检查和复现。** 736条kernel曲线、208条同文档诊断、168组微基准配置均已保存。704组rank下界检查、32组RF并集凸性检查、80组直接矩阵与矩公式风险检查通过；完整产品真kernel能量与上轮缓存最大相对差2.3×10⁻¹⁶，并集与五个kernel直接平均的预测偏差小于3.0×10⁻¹⁵。代码和数据索引保留在本目录。

```bash
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python compare.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python verify.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python pooled.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python attention_eval.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python benchmark.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python precision.py
/root/miniconda3/bin/python summarize.py
/root/miniconda3/bin/python write_report.py
```

准确度与微基准应顺序运行，避免其他GPU进程影响计时。已有结果文件会被部分脚本复用；改变协议应使用新输出目录。

[kernels.py](kernels.py)提供`PartitionFeatures(m)`、`GalerkinFeatures(m)`及`linear_attention`；输入shape为[4,n,128]，顺序L14H0/L14H6/L27H0/L27H6，`features(x,'q'/'k')`输出[4,n,m]。Galerkin输出带符号。主曲线见[kernel_curves.csv](kernel_curves.csv)，种子统计见[kernel_summary.csv](kernel_summary.csv)，同文档诊断见[attention_curves.csv](attention_curves.csv)，计算量及耗时见[benchmark.csv](benchmark.csv)。



---

原文件：key_parameterization_attribution/PROTOCOL.zh.md

2026-09-08，本轮训练前固定方案。目标是归因AD中K幅度—方向参数化的作用，主比较为AD vs直接exp，两者同m64、同Q结构128→192→63+末尾0、同K结构128→192→64、SiLU、无偏置、73536有效参数/head。Q输入和K输入沿用相同训练标准化。Q网络结构/初始化相同，均允许正常训练；不把“相同Q网络”理解为训练过程中强制权重相等。没有查询幅度/C，没有raw I或其他辅助损失。

AD：phi_Q=softmax([z_Q,0])，phi_K=exp(s_K)softmax([z_K,0])。K网络的前63输出为方向logits，末尾输出为s_K。
EXP：相同phi_Q，phi_K=exp(u_K)，u_K为K网络全部64输出。仅此输出变换不同，没有缩减/增添hidden width或state维度。

两种初始化：
1. standard：相同seed、相同原AD初始化代码产生完全相同Q/K权重；不同K输出变换会产生不同初始attention。AD复用上一轮已固定的纯删除KL检查点；EXP新训练。
2. matched_zero：Q和K第一层仍按同seed初始化，把双方K第二层所有权重置0。AD的K特征为1/64，EXP为1；这只是所有keys共享的常量比例，归一化attention都为合法因果前缀上的均匀分布，对任意Q/K完全相同。此对照排除初始attention函数差异，但无法证明跨所有初始化/优化器的优势。两边Q可正常训练，第一步某些梯度为0是此初始化的性质，不视为失效参数。

冻结Qwen2.5-1.5B，层14/27全部24/336Qheads，训练只更新新feature网络。4096篇1k文档，64个选定Q/文档、全部合法prefixK，共134360517训练pairs/head。每个选定pair单遍反向传播，向量可复用。与父实验严格相同文档顺序seed85000+seed、AdamW wd1e-4、每head梯度clip10、余弦lr至lr/10、纯方向KL。

新增配置standard_EXP、matched_zero_AD、matched_zero_EXP。每个配置seed11搜索lr0.002/0.0005，只按同一validation前32文档的KL选择；选择后训练seed29/47。共12新拟合、9新选中检查点，加复用standard_AD的3种子，共4组12个被比较模型。不会根据heldout结果改epoch、初始化、lr、数据或裁剪。所有方法的2个LR候选validation结果都展示。

评估沿用confirm_wiki128篇1k、confirm_long24篇8k；系数KL为主拟合指标，同时TV、系数L2、输出NMSE、W_O后NMSE、概率低估质量与完整冻结模型两层局部替换PPL。L2/MSE只评价，不用于训练。原先的8k数据已使用过，不能称新盲测。所有下一token的PPL与缓存64Q的静态误差不同，分别报告。

先核验参数数目、标准初始化逐参数相同、zero初始化attention恒等、计算/梯度连接；选定模型做FP64矩阵/scan/递推及真实8kFP32精度检查。所有新模型PPL采用与父AD相同Replacement和算子，teacher重测。禁止用test matrix重拟合来解释泛化。

推理微基准重新在同一RTX3090、FP32、batch1、单层12heads、CUDA graph热缓存、同状态算子下测AD和EXP；包含标准及函数匹配初始化训练权重。两者状态均m64，主导投影FLOPs均147072/head/token，状态计算约32768。不是全模型decodebenchmark。训练时长只列运行记录，不将跨历史运行的壁钟差异解释为算法训练加速。

归因以EXP−AD配对比较为准，分别报告两个初始化下各种子、head/文档差异和条件文档bootstrap区间。只有在同深度、同m、同参数的对照下仍有优势，才支持AD参数化的增益。若结果依赖初始化或评价口径，按条件结论报告；不能据一次单遍实验声称总体最优、渐近容量优势、或把所有与Hedgehog的差距单独归因网络深度。



---

原文件：key_parameterization_attribution/REPORT.zh.md

2026-09-08，K幅度—方向参数化归因实验。

**本轮支持的结论：在当前冻结Qwen、有限网络/状态及单遍KL训练预算下，AD的长上下文收益包含K幅度—方向输出参数化本身的贡献。** 控制网络深度、隐藏宽度、特征维数、有效参数、数据和初始归一化attention后，AD仍同时改善8k attention KL、输出误差与局部替换后的模型困惑度。不能把这一条件性结果提升为抽象正kernel函数类优势、所有初始化/优化器下的优势，或总体最优保证。

**究竟只改变了什么。** 双方Q网络均为128→192→63，补一个固定0后做softmax；K网络均为128→192→64。两层之间用SiLU、无偏置，均为73536有效参数/head、m=64。输入标准化相同。令输出为z_Q及u_K，比较：

\[\pi_Q(q)=\operatorname{softmax}([z_Q(q),0]),\qquad g_{\rm AD}(q,k)=e^{u_{K,64}(k)}\pi_Q(q)^\top\operatorname{softmax}([u_{K,1:63}(k),0]),\]

\[g_{\rm EXP}(q,k)=\pi_Q(q)^\top\exp(u_K(k)).\]

EXP是同深度直接指数MLP对照，不将其冒充Hedgehog原论文的完整实现。双方都没有独立查询幅度/C，最终通过线性attention分母归一化。Q网络结构及初始权重相同，双方都正常训练Q；训练后的Q权重可以不同。本实验归因的是改动K输出参数化所产生的整体训练结果，不是将训练后Q固定相同的推理消融。

随机权重对照沿用原AD初始化，AD复用既有3个检查点。匹配初始attention对照把双方K第二层置零，其K特征分别为1/64与1，因此归一化attention在任何合法因果前缀上都完全相同。两边初始参数张量也逐项相同。数值最大差9.72e−17；零初始化第一步部分梯度为零，随后所有参数均有有效梯度。这个对照排除了初始attention函数不同的解释，没有排除所有优化路径差异。

冻结Qwen2.5-1.5B，层14/27共24/336个Q heads；仅训练feature网络。训练4096篇1024-token文档，每篇64个选定Q及全部合法prefix K，共262144查询、134360517选定pairs/head，单遍反向传播。向量重复参与不同pair，不将这些pair视为独立样本。AdamW、余弦调度、梯度裁剪、文档顺序完全相同。纯因果教师attention KL损失，无raw I、MSE或输出辅助损失。

**训练与模型选择。** 每组仅用seed11在相同32篇验证文档上选择0.002/0.0005；随后固定学习率训练seed29/47。12次新拟合、9个新选定检查点，加3个复用AD检查点，共12个比较模型。方案在训练前写入[PROTOCOL.zh.md](PROTOCOL.zh.md)，没有按留出结果调参。

| 初始化 | K参数化 | 验证KL，lr=.002 | 验证KL，lr=.0005 | 选中lr |
|---|---|---:|---:|---:|
| 随机权重 | AD | 0.353310 | 0.478882 | 0.002 |
| 随机权重 | EXP | 0.352813 | 0.496450 | 0.002 |
| 匹配attention | AD | 0.353002 | 0.488211 | 0.002 |
| 匹配attention | EXP | 0.357206 | 0.498428 | 0.002 |

训练曲线见[training_curves.png](figures/training_curves.png)，实线为3种子均值、阴影为种子范围。横轴是单遍已处理文档数，纵轴为最近512篇的平均KL；它不是重复测量固定训练子集的学习曲线。各组1k训练/验证误差接近，不能据此宣称已经达到最佳拟合或不存在欠拟合；本轮没有扩大epoch或调学习率来追求收敛极限。

**留出质量。** 下表为3种子均值，越低越好。1k为128篇，8k为24篇；这两组文档此前已经使用，不能称本轮全新盲测。KL和输出NMSE使用冻结教师缓存的64Q/文档；PPL则在完整模型上计算全部下一token，只有上述两层采用新线性attention。输出NMSE按文档/head先归一化再平均。

| 初始化 | 方法 | 1k KL | 1k PPL | 8k KL | 8k 输出NMSE | 8k PPL |
|---|---|---:|---:|---:|---:|---:|
| 随机权重 | AD | 0.347341 | 8.858055 | 3.422874 | 0.641975 | 9.691610 |
| 随机权重 | EXP | 0.348753 | 8.854966 | 4.094250 | 0.698279 | 9.733331 |
| 匹配attention | AD | 0.347406 | 8.862187 | 3.380424 | 0.635261 | 9.694544 |
| 匹配attention | EXP | 0.350020 | 8.855541 | 3.974762 | 0.694563 | 9.726658 |

随机权重条件下，AD相对EXP的8k KL降低16.40%，输出NMSE降低8.06%，PPL降低0.429%。这是相同m、相同深度/宽度与参数数目下的差异。

匹配attention条件下，AD相对EXP的8k KL降低14.95%，输出NMSE降低8.54%，PPL降低0.330%。这是相同m、相同深度/宽度与参数数目下的差异。

1k并非AD全面更好：随机权重与匹配attention条件下，AD的PPL分别比EXP高0.035%和0.075%。随机权重的NLL差异区间包含0；匹配attention的条件文档区间有利于EXP，但种子方向不一致。应将本轮主要收益定位为8k泛化，而非普遍质量提升。

原始教师PPL重新测得：1k=8.664559867，与历史差0；8k=9.066278957，与历史差0。因此表中方法都是有质量损失的局部替换，不是超过原softmax模型。

**配对不确定性。** 以下差值均为EXP−AD，正值有利于AD。先对3个固定种子的同一文档取平均差，再按文档配对bootstrap10000次。区间只反映这批文档上的条件不确定性，不能当作跨模型、跨训练种子的总体置信区间；没有多重比较校正。PPL以可加的NLL/token作配对检验。

| 初始化 | 8k指标 | EXP−AD | 条件文档95%区间 | 三种子差值 | AD胜文档 |
|---|---|---:|---|---|---:|
| standard | kl | 0.671376 | [0.647896, 0.695285] | 0.655982, 0.594832, 0.763313 | 24/24 |
| standard | output_nmse | 0.056304 | [0.048399, 0.065157] | 0.073493, 0.041971, 0.053449 | 24/24 |
| standard | nll | 0.004297 | [0.003183, 0.005421] | 0.007845, 0.003333, 0.001711 | 24/24 |
| matched_zero | kl | 0.594338 | [0.573011, 0.616601] | 0.592216, 0.674307, 0.516490 | 24/24 |
| matched_zero | output_nmse | 0.059302 | [0.050639, 0.069295] | 0.056796, 0.071199, 0.049910 | 24/24 |
| matched_zero | nll | 0.003306 | [0.001808, 0.004692] | 0.003504, 0.005540, 0.000874 | 21/24 |

效应图见[attribution_effects.png](figures/attribution_effects.png)。所有1k配对区间、每head差异及位置分段保存在[summary.json](results/summary.json)。

**误差形态。** 下面的严重低估质量指：被模型赋予小于教师概率1/10000的keys，其教师概率质量之和，再对Q/head/doc/seed平均；不是token比例，也不是平方能量。W_O后NMSE把同层12个heads拼接并经过实际输出投影，包含head间误差交互。

| 初始化 | 方法 | 8k TV | 系数L2² | W_O后NMSE | 严重低估的教师概率质量 | KL/输出更优head数 |
|---|---|---:|---:|---:|---:|---|
| standard | AD | 0.625747 | 0.107785 | 0.567698 | 12.226% | 21/24；18/24 |
| standard | EXP | 0.631699 | 0.112675 | 0.616000 | 14.732% | — |
| matched_zero | AD | 0.621532 | 0.106135 | 0.565052 | 12.026% | 22/24；16/24 |
| matched_zero | EXP | 0.632035 | 0.113095 | 0.612478 | 14.364% | — |

误差指标有不同权重，不应只挑最有利的一种。能量全局加权的输出/W_O NMSE也已完整保存；静态诊断只反映教师QKV，与替换模型实际下游激活不同。8k变化同时包含长度、内容、RoPE位置和上下文分布变化，本轮不能将它单独归因于其中一个因素。

**为何这个参数化可能起作用。** 这是可以严格证明的坐标差异，而不是已经证明的优化优势：

\[\log\|\phi_K^{\rm AD}(k)\|_1=s_K(k),\qquad \log\|\phi_K^{\rm EXP}(k)\|_1=\operatorname{LSE}(u_K(k)).\]

AD中，固定幅度坐标而改变方向logits，不会改变正特征的L1质量；EXP中，质量由所有输出logits共同决定。对单个合法key j，令教师/预测attention为t_j/p_j，r_j为该key内各feature对kernel的归一化贡献，则纯KL的局部导数为：

\[\partial_{s_j}L=p_j-t_j,\qquad\partial_{z_{jr}}L=(p_j-t_j)(r_{jr}-\pi_{K,jr}),\qquad\partial_{u_{jr}}L=(p_j-t_j)r_{jr}.\]

AD的方向式用于可训练的前63个logits；最后方向logit固定为0。输出坐标分别控制质量与方向，但共享隐藏层使实际参数更新仍会相互影响。梯度恒等式已由autograd核验到1e−16量级。结合本轮受控结果，可以提出“显式分配K质量与方向有利于当前预算下的分布外行为”的机制假设；尚未区分有限网络表达约束与优化条件各自贡献，也未证明更好的收敛率或某个具体长上下文病因。

任何严格正向量都可以写成L1幅度乘单纯形方向；AD也能重写为带共享log-sum-exp校正的指数特征。因此不能声称开辟了此前不存在的抽象正kernel函数类。另一方面，允许任意函数的代数重写，不等于一个固定宽度的线性输出层可以免费实现这个校正，本轮正是在检验有限网络预算下的这项差异。

更具体地，固定隐藏表示h(k)后，EXP的log质量为$\operatorname{LSE}(W_Kh)$，AD为$w_s^\top h$。若强行匹配AD方向和幅度，指数输出需要$u=[z_K,0]+[s_K-\operatorname{LSE}([z_K,0])]\mathbf1$，其中共享的非线性校正通常不能由同一个h上的线性输出层直接实现。这解释了有限网络参数化为何值得比较；允许隐藏表示改变后，本轮没有证明两类网络的严格包含关系。

**计算代价。** 两者投影主导乘加均为147072 FLOPs/head/token，m64状态读写的主导项约32768；此口径不计SiLU、exp、log-softmax、标准化和数值重标度。AD额外有K端log-softmax与幅度广播，故总计算并非完全一样。FP32单层12heads含S、z及稳定缩放g的状态均399360bytes（0.380859MiB），没有彼此的cache优势。

| 初始化 | 方法 | feature计算 μs | feature+状态步骤 μs | 两轮步骤中位数 μs |
|---|---|---:|---:|---|
| standard_ad | AD | 32.784 | 64.384 | 64.320, 64.384 |
| standard_exp | EXP | 26.624 | 58.175 | 58.176, 58.173 |
| matched_zero_ad | AD | 32.848 | 64.416 | 64.384, 64.448 |
| matched_zero_exp | EXP | 26.656 | 58.192 | 58.208, 58.176 |

AD的feature+状态步骤在两组中分别慢10.67%与10.70%。因此是当前实现中以约10.7%的这部分时间换取长上下文质量收益，不能宣称AD在相同网络/state预算下更快。

同一RTX3090、FP32、batch1、单层12heads、真实单token QKV、CUDA graph热缓存、同一通用状态算子，两轮正反顺序采样。时延差是当前实现结果，不能外推最优融合算子，更不是完整模型decode加速。训练壁钟保存在summary中，但历史AD与本轮不是同期运行，不把其比值当训练速度结论。

**对主线的影响。** 这个对照比“两层AD对一层加宽Hedgehog”更能支持K参数化的贡献，但仍不能把此前与Hedgehog的全部差距归给该分解，也不识别Q深度、K深度与m各自的独立效应。与已有工作的关系沿用[前轮原文核对](../normalized_kernel_novelty_20260908/REPORT.zh.md)；本轮没有新增“首创幅度方向”或“首创KL学习正feature”的主张。

现阶段可以围绕“有限特征状态下、显式K质量—方向参数化的泛化收益”继续推进。顶会层面的缺口仍包括不同模型/层及真正新域复现、收敛预算和优化器敏感性、更多同规模正feature近邻，以及归一化风险的非平凡理论。原始指数kernel的Schmidt L2谱尾不能直接作为归一化KL下界，本轮没有验证接近总体理论最优。当前是方向得到初步支持，不是已具备顶会级充分证据。

数值核验与复现实物：[结构与初始函数检查](checks/structure.json)、[梯度恒等式](checks/mechanism.json)、[FP64矩阵/scan/递推](checks/operators.json)、[真实8k FP32精度](checks/long_precision.json)、[汇总与逐种子结果](results/summary.json)、[审计与源码/结果SHA256](checks/audit.json)。新增训练日志位于logs/train.log；各检查点在fits/。训练不使用后续诊断或留出集。



---

原文件：mlp_direction/ALL_HEADS_ESTIMATE.zh.md

以下只基于真实模型配置、已完成的局部/算子计时和CPU算术估算，没有运行全head替换、模型前向或新GPU实验。对象是本轮连续Q/K幅度—方向kernel，m64、两侧128→192→64，每Q head独立映射；原Qwen2.5-1.5B的28层、每层12Q heads/2KV heads、d128保持。必须真正移除HF原KV cache路径，才能得到这里的固定状态结果；仅注册新的attention回调仍可能保留原cache。

**算术量。** 每次乘加算2 FLOPs，忽略激活、softmax、RoPE、分母标量操作等。原模型Q/K/V/O、FFN和最终词表投影共3.08714 GFLOPs/token。新特征双侧每head147456 FLOPs，状态更新/读取每head约32768 FLOPs，336heads共0.060555 GFLOPs/token。

原模型decode约 C_old(T)=3.08714+0.000172032 T GFLOPs/token；全线性约 C_new=3.14769 GFLOPs/token。这是保留所有原投影和FFN的比较，不是只算attention部分。

|上下文|原decode GFLOPs/token|新decode GFLOPs/token|算术量变化|原KV MiB|新状态 MiB|
|---|---|---|---|---|---|
|1024|3.2633|3.1477|−3.5%|28|10.582|
|4096|3.7918|3.1477|−17.0%|112|10.582|
|8192|4.4964|3.1477|−30.0%|224|10.582|
|32768|8.7243|3.1477|−63.9%|896|10.582|

理想线性prefix聚合下，8192-token prefill总算术量从27.240 TFLOPs降到21.963 TFLOPs，下降19.4%；当前block64聚合的矩阵额外开销后约22.030 TFLOPs。这里prefill只对末token做词表投影，和计时协议一致。并行scan的数据搬运和激活代价不在这些FLOPs内。

**状态和权重。** 每head FP32状态m(d_v+1)，全模型共10.582MiB。新特征参数共24815616，FP32权重94.664MiB。假定没有其他缓存差异，8K时这两项替代原224MiB KV，净节省约118.754MiB，而不是总显存缩小21倍。1K时额外特征权重反而使这部分总显存增加约77.246MiB。若将来共享同组K映射或换精度，预算会改变，但那不属于本估算。

简化的每token读写模型：BF16原dense权重约3.087GB读一次，原KV各元素读一次；新权重读一次，新线性状态读/写各一次。8K总字节比约0.966，即只少3.4%，远小于30%的FLOP降幅。它只是乐观流量模型，不是硬件计时预测；实际GQA复用、cache、KV拷贝及算子启动会改变流量和延迟。

**decode时间：只能给条件情景，当前数据不能唯一识别。**

T_new = T_old − T_removed_attention/cache + T_features + T_linear_state/glue。

旧原模型RTX3090、batch1、8K下约22ms/token。被移除的attention/cache路径没有独立profile，因此不能从四head替换的+2～4ms按84倍推断全模型，更不能直接套用30% FLOP减少。

一个有明确来源但仍偏乐观的工程情景：旧普通MLP（同样的两层尺寸）的四head Q/K特征在CUDA Graph中为0.047104ms。假设每层12heads合并后的时间是四head测量的1～3倍，则28层新增特征约1.32～3.96ms/token。再假设移除attention/cache省0～3ms、线性状态及衔接花0.2～1ms，就得到20.52～26.96ms/token。可粗略记为21～27ms，对比22ms约−5%～+23%。这不是置信区间，也不是全head实测；旧softplus MLP与新幅度—方向的非线性不同，1～3倍扩展和另外两项都是待验证假设。宽松预算可留20～30ms。

该情景只对新增特征路径做按层合并/CUDA Graph，保留其余模型原执行方式。如果同时优化整个模型，原模型也必须采用相同优化再比较。

直接采用eager、逐head调用当前代码不满足上述假设，可能明显更慢。作为开销示例，旧四head合并MLP特征的eager计时0.559584ms，在每层12heads为其1～3倍的情景下，仅新增特征就约15.7～47.0ms/token，尚未加其余模型。它说明实现方式足以改变结论，不能将算法线性复杂度直接等同于低延迟。

**prefill应单独估计。** 旧四head、8192tokens、同尺寸MLP特征CUDA Graph为1.1694ms；按84倍head数线性扩展，新增特征约98.2ms。原模型8192-token prefill约471ms；已有三点曲线中的二次项约50ms（只是曲线代理，没有单独profile）。按此代理移除50ms、增加98ms，得到约519ms，再加新的线性scan。因而沿用当前FP32特征实现，prefill可能仍慢，即使算术量下降19%。新的融合、混合精度或更高效scan可以改变此结果，目前没有可靠的全head毫秒值。

所以对“增加多少”的当前回答应是：合理合并/图捕获实现下，8K decode可暂按约持平至慢20%左右做情景预算，并保留略快可能；当前逐head原型不能套用这个预算。32K以上更有机会从移除长历史读取中获益，但没有实测支持具体加速倍数。所有数字均不构成全head替换的模型质量保证。

计算脚本：estimate_all_heads.py；逐项数值与假设：results/all_heads_mlp_estimate.json。计时来源：results/benchmark_gauge.json、../candidate_validation/results/feature_benchmark.json。旧的分区kernel估算参数为a1024,e64，不适用于本轮MLP kernel。



---

原文件：mlp_direction/AMPLITUDE_DIRECTION_FOCUS.zh.md

本文件收束到既有连续幅度—方向正 kernel。主研究对象为 factorized_both；query-only 校准是理论诊断与受限构造。旋转增强、分区和窗口不作为本轮主方法的定义。以下新内容是推导与已有结果复核，没有启动或宣称完成扩大规模训练。

**保留的数学形式与配方**

$$h_\theta(q,k)=m e^{s_Q(q)+s_K(k)}\pi_Q(q)^\top\pi_K(k),\quad
\pi_X=\operatorname{softmax}([z_X,0]).$$

固定训练 head 缩放吸收到幅度中。每侧两层128→192→64；m−1个方向输出加一个幅度输出，共73856参数/head。原成功配方使用 balanced I，λ=1：

$$\mathcal L_{\rm bal}
=E_Q\frac{E_K d_I(\kappa,h_\theta)}{Z(Q)},\qquad Z(q)=E_K\kappa(q,K).$$

这是未归一化 kernel 的逐项 I-divergence，正权重1/Z平衡查询；它不同于只拟合归一化 attention，也不同于未经加权的 raw I。原训练以同文档合法矩形估计对应条件损失，完整边缘乘积分布另作留出验证。扩大规模时必须分别固定这两个采样口径，不能把它们写成同一个总体风险。

Q幅度在纯线性attention的分子分母中消去，K幅度不能消去。联合学习Q/K方向和K幅度会改变attention；固定g只校准Q幅度则不会。共享隐藏层的梯度仍耦合，不能说实际神经训练已完全解耦。

**固定方向校准定理究竟解决什么**

固定正kernel g，G(q)=E_Kg(q,K)，Z(q)=E_Kκ(q,K)，假设正数有限且相关风险有限。记p_q=κ/Z，r_q=g/G；它们是相对于P_K的密度。对正查询函数a：

$$E_Kd_I(\kappa,ag)
=Z\,KL(p_q\|r_q)+d_I(Z,aG).$$

逐q目标中与a有关的部分为−Z log a+aG，导数−Z/a+G在a*=Z/G处唯一为零。因此

$$R_I(ag)=R_I((Z/G)g)+E_Qd_I(Z,aG).$$

取a=e^ell/G，额外风险恰为E_Qd_I(Z,e^ell)。取epsilon=ell−log Z，额外风险也等于E_Q Z(e^epsilon−1−epsilon)。若|epsilon|≤b，则它位于

$$\tfrac12e^{-b}E_Q[Z\epsilon^2]\quad\text{与}\quad
\tfrac12e^bE_Q[Z\epsilon^2]$$

之间。这是误差的解析描述，不要求改用MSE训练。固定有限宽度读出未必能表示log Z；参考key均值估计也有误差，不能把理想a*当成已学到。

这项定理对任意P_Q、P_K成立，不假设Gaussian。它优化的是整个分布上的查询函数，不是对每张测试矩阵重拟合；但固定了g的全部交互方向，因此不能解决最优rank-m方向问题。在任何实际key集合，a(q)都从归一化分子分母中消去，与该集合是否来自参考P_K无关。

例如两key原始目标为[2,8]，g=[1,1]，理想校准得到[5,5]。总量正确，但注意力仍[.5,.5]，没有恢复[.2,.8]。

**幅度—方向形式与总体正rank-m问题的精确联系**

任意可积正可分离h=Σf_r(q)g_r(k)，令μ_r=E_Kg_r(K)>0，删除μ_r=0的无效分量，并令

$$A(q)=\sum_rf_r(q)\mu_r,\quad
\pi_r(q)=f_r(q)\mu_r/A(q),\quad
\psi_r(k)=g_r(k)/\mu_r.$$

则

$$h(q,k)=A(q)\sum_{r=1}^m\pi_r(q)\psi_r(k),\quad
\sum_r\pi_r(q)=1,\quad E_K\psi_r=1.$$

反向构造f_r=Aπ_r也成立。这是可积函数层面的一种通用坐标表示，不保证有限宽度两层MLP能无误差完成所有坐标变换。

设M1=E_QZ，dP_Q^(1)=Z dP_Q/M1，r_q=Σπ_rψ_r，则

$$R_I(h)=E_Q[Z\,KL(p_q\|r_q)]+E_Qd_I(Z,A).$$

在幅度函数不受额外限制的总体正rank-m类中，可把幅度精确优化掉：

$$\mathcal E_{m,I}^+
=M1\inf_{\pi,\psi}
E_{Q\sim P_Q^{(1)}}KL\left(p_q\middle\|\sum_{r=1}^m\pi_r(q)\psi_r\right).$$

这里π为simplex值函数，ψ为相对于P_K归一化的非负密度，且满足所需可积性。因此核心有限m瓶颈是“m个共享正密度能表达多少种条件方向”；并非只有m种Q或K。标度定理给出了幅度子问题的解，但没有给出这些最优共享密度和混合系数。真实数据上的两层网络是在参数预算内近似这些函数。

以上是结构性重写和受限构造，不单独主张数学新颖性或总体近最优性。实用理论缺口仍是正混合逼近、有限网络逼近、分布估计与优化误差。

**与原始Schmidt谱框架如何准确衔接**

原始T的kernel为κ，若H=Eκ²<∞，则其最佳自由有符号rank-m L2误差为Σ_{j>m}σ_j(T)²。固定方向的I最优标度Z/G通常不是L2最优标度；后者是

$$a_{L2}^*(q)=E_K[\kappa(q,K)g(q,K)]/E_K[g(q,K)^2].$$

不能把两种损失的最优值直接画等号。但可以对原谱给出一个精确的幅度—方向重写：

$$M2=E_QZ^2,\qquad dP_Q^{(2)}=Z^2dP_Q/M2.$$

令S2:L2(P_K)→L2(P_Q^(2))的kernel为p(q,k)=κ(q,k)/Z(q)，令Uu=Zu/√M2。U是酉映射且T=√M2 U S2，因此

$$\boxed{\sigma_j(T)=\sqrt{M2}\sigma_j(S2),\qquad
E_m^\star=M2\sum_{j>m}\sigma_j(S2)^2.}$$

这并未更换原始目标：方向kernel使用的Q测度必须同时改为Z²加权。对幅度已匹配的h=Zr，也有

$$\|\kappa-Zr\|_{L2(P_Q\times P_K)}^2
=M2\|p-r\|_{L2(P_Q^{(2)}\times P_K)}^2.$$

所以原始L2谱研究Z²加权的方向复杂度，raw I研究Z加权的方向KL，balanced I研究未加权的方向KL。三者恰当对应，但不可混用其最优值。任意正rank-m h仍有L2误差≥原始Schmidt谱尾；正密度和有限MLP约束可能使最优误差更大。

此谱恒等式通过坐标变换得到，不需要或证明Gaussian分布。它也不意味着Z、S2或谱已在未知总体上估计准确。

**已有结果的重新定位**

旧四head、m64、完整新经验乘积、固定seed11（FAVORseed1009）：

| 数据／方法 | raw L2相对误差 | raw I相对误差 |
|---|---:|---:|
| Wiki／FAVOR+ | 1.088077 | 15.259118 |
| Wiki／softplus raw I | .712185 | .810376 |
| Wiki／幅度—方向 | .572821 | .395682 |
| Wiki／固定方向分布校准 | .597019 | .360475 |
| SWDE／FAVOR+ | .999495 | 14.273933 |
| SWDE／softplus raw I | .631168 | .859972 |
| SWDE／幅度—方向 | .481184 | .590270 |
| SWDE／固定方向分布校准 | .487164 | .618769 |

因此原始kernel的初步优势真实存在。Wiki两侧65536向量、SWDE两侧10240向量，均为全部经验乘积而非重拟合测试矩阵；相关配对数量不是独立样本数，单seed完整乘积也不是总体置信区间。三seed独立文档矩形另有参数和损失匹配对照，见REPORT。

复核该方法的完整乘积balanced风险：

| 数据 | 总损失 | 方向项 | 幅度项 | 理想Q校准最多消除的比例 |
|---|---:|---:|---:|---:|
| Wiki | .261053 | .230812 | .030241 | 11.58% |
| SWDE | .370600 | .308249 | .062351 | 16.82% |

比例来自四head均值之比，对应此经验测度上的无限制理想幅度校准，不是已训练新校准器的收益，不适用于raw L2或未平衡raw I。该结果说明主要应继续联合学习方向和K幅度，而非只做Q幅度回归。

后续24head实验改变为全因果配对，主目标又选择λ=.1，且增加8k外推，不能直接归因于只扩大head数。它同时保留了λ=1：该版本Wiki raw NMSE约.7653，8k约1.058e8；说明长文原始幅度泛化确实未解决，而非仅λ=.1出问题。不能拿早期短文乘积胜利声称已解决全因果/长文，也不能用后期失败否定早期配方在所测分布上的优势。

**成本与强对照**

同m64,d128、MLP隐藏192：MLP双侧投影147456主导FLOPs/head，FAVOR32768；相同状态更新读取约32768，因此总量比2.75。两者均为线性attention，数学FLOPs没有因融合而改变。

现有RTX3090、batch1、单层12heads真实Q/K、CUDA graph热缓存：FAVOR特征+状态5.7344μs，幅度—方向9.9021μs（1.73倍），exp-MLP9.7072μs。不是完整模型耗时。现有两层24head融合部署8k全模型：teacher21.320ms/token，FAVOR21.606，主MLP21.229；MLP与FAVOR/teacher差别均不能据这一短测宣布稳健速度优势。prefill分别468.554、471.154、473.966ms。

同选中head下FAVOR/MLP持久状态一致，完整两层替换后8k总KV+状态208.7617MiB，teacher224MiB；MLP还增加约6.76MiB权重，prefill峰值更高。早期四head版本不能释放原GQA缓存，旧慢速数据与该版实现应分开。

FAVOR是无数据拟合的公式级基线；学习型方法的离线训练成本额外存在。现有结果不等于完整复现并击败Performer或Hedgehog论文系统。下一轮需要相同m、相近实测时间两种预算；后一项应增大FAVOR的m，而不是只与m64比较。

**下一阶段规模协议草案：尚未执行**

1. 冻结旧factorized_both/λ1配方、采样规则、初始化、数据标准化和优化设置，先在原4head复现。只改变数据量4096→16384→65536篇；每篇与每个选定pair各训练一次。新增独立文档而非重复旧pair。小数据版本同时保存训练过程与固定验证曲线，区分数据、优化与表达瓶颈。
2. 固定m64后扩至48个heads，预先选择层6/14/21/27全部Qheads，保留原4head作为追踪子集。先测冻结网络的kernel泛化；完整GQA组部署的PPL另行逐步评估。不同层替换诱导的上游分布漂移单独报告。
3. m32/64/128容量曲线；与同宽度、同损失的softplus/exp MLP及同参数KL控制比较；FAVOR+加入同m和实测耗时匹配的更大m。先固定一级对照再扩大，不进行全因子组合搜索。
4. 先确认原1k分布扩量是否继续收益，再加入真实4k/8k训练数据并定义新的目标分布；保留原1k数据测试。共同旋转仅作机制诊断，不能代替真实长文覆盖或悄悄改变主配方。
5. 新文档/新来源上的raw I、balanced I、L2、log幅度误差和方向KL共同报告；全模型PPL和prefill/decode/cache分开。所有主配置在打开确认结果前固定，完整乘积先扩到多seed与文档级不确定性。
6. 第二模型/另一架构作为外部确认；先得到规模趋势，再扩大模型权重和替换范围。不得预设扩大规模一定修复长文raw失败或自动带来顶会贡献。

核心Go依据是：同预算下的raw kernel优势随独立文档和head规模保持，并能解释幅度/方向误差各自如何随m、数据量变化；若只Q标度改善而方向、PPL、效率均不改善，应收紧论文贡献为原始kernel近似，不能写成更强attention推理方法。

本次复核脚本focus_audit.py生成checks/amplitude_direction_focus.json，保存结果来源哈希、完整乘积损失拆分与真实缓存Q/K上的FP64代数检查。小样本恒等式检查不是新的泛化实验，也不是扩大规模实验。旧REPORT与所有失败结果保留。



---

原文件：mlp_direction/PROTOCOL.zh.md

本轮预先规定的研究问题：真实 Q/K 的谱信息能否指导正 MLP 的特征容量；原始 kernel 的查询质量加权是否改善未见上下文的注意力输出。

模型冻结为 Qwen2.5-1.5B；仅既有四个 head，禁止将局部替换写成完整线性化。所有主方案拟合 exp(q^T k/sqrt(128))，使用非 MSE 损失。raw、balanced、half、logcosh 四个目标预先列出；balanced 为每个 query 的原始广义 KL 除以真实 kernel 行总质量，half 为中间权重。都不是将预测 kernel 行归一化后拟合。raw 与 balanced 在不受限函数类内都以原始 kernel 为唯一最优值。

训练 4096 文档，每个文档一遍，随机选 64 个互不相同的后半段 Q，与前半段全部 512 个 K 形成 32768 个合法配对。矩形中的向量参与多个不同配对；每个文档、每个选中 Q/K 配对只训练一次，不把 1.34 亿配对当作独立文档数。不同目标采用相同数据、初始化、学习率、更新次数和输出维度。与之前 512 一对一配对/文档的旧 MLP 比较时，必须注明数据与计算增多，并以本轮 raw 为匹配对照。

两层 MLP 为 128→192→m，两侧独立。每个 head 参数数为 49152+386m；固定四个 head 的总 m=256 时总活跃参数为 295424。训练不同 m 的候选需要额外搜索成本，须计入研究阶段成本。原始矩形规模与训练时相同的 validation 输出误差选择目标；不使用内部或官方测试分数选择。初始 m64、seed11 的四目标均保留；选定目标扩展 seed29/47 及 m16/32/96/128，按实际结果记录，不在测试上调整损失系数。

谱预测使用训练文档构造的原始 K、按行总质量平方根加权 K、按行总质量加权 K 三种条件上下文算子；它们是预先规定的不同代理量，不能将上下文依赖的行缩放当作可部署的 q-only feature map。比较均匀分配、谱分配、独立验证误差分配。以不超过总 m=256 为预算，并报告是否实际用满与实现时 padding 的额外成本。

验证来自既有 128 篇内部 validation，内部测试 256 篇，官方 20 篇固定 WikiText2 子集。既有测试已经用于之前探索，所以这些不能称为首次盲测；本轮将另选未曾评估且文档哈希不重叠的新文档做确认。所有新方法必须先固定再打开确认结果。现有数据的边缘乘积分布、同文档合法矩形和完整因果 attention 分开报告，不互相冒充总体。

理论推导须说明：经典 Schmidt 尾和 KL 质量分解自身不是新的定理；带权风险一般不能原封不动继承无权谱尾；只有实际验证且有限样本条件可信的结论才可能成为论文贡献。若谱分配或损失改进失败，保留负结果，不调整到测试上的赢家。

后续探索与确认记录（不回填为初始预注册）：

- 首轮追加 value 加权与匹配 softplus-KL 对照。第一轮新数据为 fresh_wiki128 篇 1024 tokens、FDA96 个源文件各 1024 tokens。FDA 可用独立文件不足 128，在查看结果前减到96。
- 基于 validation 中 KL 对照的优势追加 factorized、factorized_both、exp_control，各三个种子。旧测试与首轮新数据对它们只算探索性评估。第二轮固定 fresh_wiki2 的128篇1024-token文档，以及 SWDE96个不同源文件的512-token文档，在新方案训练/配置固定后评价。
- 第二轮之后追加 query-only 分布标度校准。第三轮固定 fresh_wiki3 的128篇1024-token文档与 swde3 的40个新源文件512-token文档。后者独立文件只剩45个，在评价之前把64目标减到40。所有确认文本/token hash 去重，SWDE 两轮还按源文件去重。
- 最终标度校准保持推理参数预算，但明确增加第二阶段训练计算。使用新查询位置，每个样本配对一遍，不宣称整个文档只访问一遍。仅使用固定训练集估计 key 特征均值与 kernel log-MGF；冻结参数后评价第三轮。全局标度的两种闭式对照也只用同一校准集。
- 第三轮新增完整经验乘积评价：Wikipedia 两侧各65536个新向量，SWDE两侧各10240个新向量，全量分块FP64计算；主网络固定seed11，FAVOR+固定seed1009。它们不是独立样本数，也不是未知总体的精确风险。全部三种子另有逐文档条件矩形与模型PPL结果。
- 随机特征对照沿用此前固定的FAVOR+、centered FAVOR+、SDERF、ADERF，每种5个seed，m64；公式级对照不等于完整原论文的LLM训练系统复现。PPL主表的FAVOR+是固定seed1009，必须注明单种子局限。
- 单独核验显式矩阵、因果scan、递归解码一致性；分布标度校准还核验与其父kernel的attention不变性。报告全部成功/失败分支，并分别讨论原始L²、I-divergence与PPL，禁止用一种指标替代另一种结论。



---

原文件：mlp_direction/REPORT.zh.md

本轮完成了实际训练、三轮独立文档确认、完整模型的四个 head 局部替换，以及全量经验乘积分布验证。得到了两个可复现的初步结果：连续 Q/K 幅度—方向正特征在原始 kernel 拟合上优于匹配的普通 MLP；分布标度校准可以修复原始 kernel 的质量误差并保持已有 attention 输出。现有证据支持继续研究，尚不支持“新 kernel 全面优于现有方法”“接近总体 rank-m 下界”或“已有顶会水平成果”。

**最值得保留的两个构造。**

全程拟合原始目标的主候选为 factorized_both：

$$\hat\kappa(q,k)=m e^{s_Q(q)+s_K(k)}\,\pi_Q(q)^\top\pi_K(k),\qquad
\pi_X=\operatorname{softmax}([z_X,0]).$$

每侧两层 MLP 输出 m−1 个连续方向坐标与1个幅度坐标；损失是原始 I-divergence 除以真实行总质量。它没有固定分区，不是把 q/k 离散成64类。归一化 attention 的分母仍由线性状态公式计算；Q-only 幅度只是在推理时可以精确约去。MLP 中对特征坐标做 softmax 不等于把训练目标换成 attention 分数。

第二个候选使用 exp-MLP KL 控制作为方向初始化，在训练乘积分布上校准：

$$\hat\kappa(q,k)=\frac{e^{\ell(q)}}{G(q)}g(q,k),\quad
G(q)=\mathbb E_K g(q,K),\quad \ell(q)\approx\log\mathbb E_K e^{q^\top K/\sqrt d}.$$

μ=E f_K 可折叠入网络参数，固定隐藏特征上的 log-amplitude 读出采用非 MSE 的凸目标、一遍优化。没有增加推理参数或状态。对任意实际上下文，其 attention 与 g 完全相同；因此校准只能修复 raw kernel，不能凭空改善方向或PPL。此候选明确使用“KL方向预训练 + 原始kernel校准”的两阶段路线；用户要求全程原始目标的比较应使用第一个候选。

固定 g、令 Z=E_Kκ，则在积分有限条件下有精确分解：

$$R(ag)=R((Z/G)g)+\mathbb E_Q d_I(Z,aG),\qquad R(h)=\mathbb E_{P_Q\times P_K}d_I(\kappa,h).$$

这是固定方向等价类内的最优标度与精确 excess-risk 表达式；它不是整个正 rank-m 函数类的最优性。完整假设、推导、下界边界和现有文献归属见 [THEORY.zh.md](THEORY.zh.md)。

**实验预算与确认边界。**

冻结 Qwen2.5-1.5B，只替换 L14H0、L14H6、L27H0、L27H6，4/336个Q head。两层128→192→64的Q/K网络每head73856参数，共295424。主方案共29次单遍训练（含不同m与损失消融），3个校准读出和6个闭式标度对照；计算记录中训练/校准总耗时约15.9分钟，不含激活提取和评估。所有主训练目标均非MSE；KL控制明确单列。

每个主拟合使用4096训练文档、每文档64个不同Q×512K，一遍共134217728配对/head。这是相关配对数，不是独立样本量。校准额外用同一批文档中互不重叠的新64个Q，与16384-key训练bank形成4294967296次kernel计算/head；只优化原网络已有的193个读出参数/head。推理预算相同，训练算力和数据预算并不相同。校准没有重复方向训练的Q/K配对，但同一文档被两阶段访问。

初始验证128篇，既有内部测试256篇与官方WikiText2固定20篇。第一轮新增Wiki128/FDA96；第二轮新增Wiki128/SWDE96；第三轮新增Wiki128/SWDE40。Wiki/FDA每篇1024tokens，SWDE每篇512tokens。所有集合token哈希去重，SWDE额外按源文件去重；这不保证语义近重复完全不存在。第三轮在校准配置固定后测量；早期候选与追加候选的探索/确认界限见 [PROTOCOL.zh.md](PROTOCOL.zh.md)。

**同文档条件矩形：第三轮独立确认。**

下表学习方法为3种子、4heads平均，FAVOR+为5种子。raw平方误差先在每head跨文档汇总分子分母，再平均heads；输出NMSE先逐文档计算再平均。每篇1024tokens使用64Q×512K，512tokens使用32Q×256K，均为合法因果矩形。不能将这些条件矩形冒充乘积分布或完整模型输出。

|方法|Wiki 原始平方误差|Wiki 输出 NMSE|SWDE 原始平方误差|SWDE 输出 NMSE|
|---|---|---|---|---|
|FAVOR+|1.018545|1.908214|2.895324|1.714668|
|Softplus + raw I|0.864592|0.337761|0.801386|0.336683|
|Softplus + balanced I|0.921141|0.280458|0.754563|0.302075|
|Q/K amplitude-direction|0.775266|0.229890|0.715884|0.286543|
|Softplus MLP + KL control|0.973549|0.224105|0.916344|0.281091|
|Distribution calibration|0.793365|0.216739|0.744366|0.292115|

factorized_both 与 balanced 的比较同时匹配损失、数据、参数和更新数，因此较能隔离输出参数化的作用。与 raw 的比较包含损失权重和参数化两项改变。分布校准在Wiki输出上保留exp-KL的优势，但在SWDE不胜过所有KL控制；没有一种新方案在全部指标、全部域上支配其他方案。

沿用此前固定的随机特征bank，额外确认了以下对照。它们在当前冻结模型的原始Q/K分布上误差很大，尤其受有限随机节点和大kernel值影响；这不是关于原论文完整训练系统的否定。各seed数值范围与输出误差完整保存在kernel_summary.csv。FAVOR+的正交正随机特征出处见 [Performer, ICLR2021](https://arxiv.org/abs/2009.14794)。

|公式级随机特征对照（5 seeds）|Wiki raw相对平方误差|SWDE raw相对平方误差|
|---|---|---|
|FAVOR+|1.018545|2.895324|
|Centered FAVOR+|17.675949|38.128035|
|SDERF|98.126948|203.986202|
|ADERF|52.213359|29.942764|

**完整经验乘积分布：不在测试矩阵上拟合。**

Wikipedia两侧各65536个新向量，所有4294967296个配对/head；SWDE两侧各10240个向量，104857600配对/head，全部FP64分块求和。各方法固定网络seed11，FAVOR+固定seed1009，未按测试选seed。下面为四个head相对风险的算术平均；raw I相对误差为 E d_I / Eκ，raw L²相对误差为 E(κ−κ̂)² / Eκ²。

|方法|Wiki raw L²相对误差|Wiki raw I相对误差|SWDE raw L²相对误差|SWDE raw I相对误差|
|---|---|---|---|---|
|FAVOR+|1.088077|15.259118|0.999495|14.273933|
|Softplus + raw I|0.712185|0.810376|0.631168|0.859972|
|Q/K amplitude-direction|0.572821|0.395682|0.481184|0.590270|
|Global balanced calibration|19.805966|2.355906|1.614318|2.732352|
|Global raw calibration|8.657906|2.340883|2.145611|2.843247|
|Distribution calibration|0.597019|0.360475|0.487164|0.618769|

[全量乘积分布图](figures/full_product.png)。这验证了已训练函数在新经验边缘分布上的表现，超出了单个上下文矩阵分解；它仍不是未知总体风险的置信界。不能把数十亿相关配对当成数十亿独立样本。校准与简单标度对照使用同一训练bank，区别是查询相关读出与head全局常数。未校准KL的raw标度不被识别，其很大的raw误差不作为主要胜利证据。

**完整冻结模型困惑度：只有4个head替换。**

|方法|Wiki 第三轮 PPL|SWDE 第三轮 PPL|
|---|---|---|
|Original softmax|8.588208|12.934821|
|FAVOR+|8.719515|13.182934|
|Softplus + raw I|8.637235|13.010605|
|Softplus + balanced I|8.636213|12.996212|
|Q/K amplitude-direction|8.628680|12.996942|
|Softplus MLP + KL control|8.626264|12.995367|
|Exp MLP + KL control|8.625095|13.000088|
|Distribution calibration|8.625095|13.000088|

学习方案表中为3种子PPL均值，FAVOR+为固定seed1009。旧测试曾测量5个FAVOR种子，不能冒充新数据上的多种子PPL。指标通过各文档token加权NLL后取exp；不同域的绝对PPL不能横向解释。样本与种子范围、全部结果保存在 [ppl_summary.csv](results/ppl_summary.csv)。[PPL图](figures/ppl.png) 显示3种子范围。

|数据|比较（前者−后者）|平均 ΔNLL|配对文档95%区间|
|---|---|---|---|
|fresh_wiki3|factorized_both − raw|-0.000991|[-0.001392, -0.000612]|
|fresh_wiki3|factorized_both − balanced|-0.000873|[-0.001110, -0.000642]|
|fresh_wiki3|factorized_both − kl_control|+0.000280|[+0.000081, +0.000480]|
|fresh_wiki3|gauge_calibrated − exp_control|+0.000000|[+0.000000, +0.000000]|
|swde3|factorized_both − raw|-0.001051|[-0.001772, -0.000371]|
|swde3|factorized_both − balanced|+0.000056|[-0.000473, +0.000606]|
|swde3|factorized_both − kl_control|+0.000121|[-0.000357, +0.000581]|
|swde3|gauge_calibrated − exp_control|+0.000000|[+0.000000, +0.000000]|

区间是对文档配对重采样10000次，先对3个固定训练种子求均值；未覆盖未来训练种子的变异，且多项探索性比较未做多重校正。PPL改进总体很小，不能把head局部输出误差下降的百分比写成模型PPL同等比例下降。

原始kernel双精度评价中，校准前后逐文档输出NMSE最大变化为1.44e-15。初次将μ重缩放折叠到两侧FP32特征时，BF16模型中的舍入传播造成小幅PPL变化，已保留 [初次浮点部署结果](results/ppl_gauge_folded_fp32.json)。最终在线实现直接执行校准kernel的代数等价父特征、只保留一套同预算权重；重新测量后PPL最大差异为0。不把舍入差异作为算法质量收益。

**真实递归实现与成本。**

每head只维护 S=Σψ(k)vᵀ 和 z=Σψ(k)，状态m(d_v+1)。m64,d_v128,FP32时四个head总状态129KiB，与FAVOR+相同。前缀scan、逐token更新、显式因果矩阵在FP64核验一致。不同head的m分别分配真实状态，没有padding到最大m。

|方法，8192 prompt|prefill ms|decode ms/token|KV MiB|额外状态 KiB|
|---|---|---|---|---|
|teacher|470.735|21.976|224|0|
|favor_plus|474.208|24.096|224|129|
|factorized_both_m64_s11|475.567|26.108|224|129|
|exp_control_m64_s11|475.991|24.364|224|129|
|gauge_calibrated_m64_s11|475.463|24.301|224|129|

RTX3090，batch1，32个decode tokens，prefill7次、decode5次取中位数；是当前Python/Triton实现的短微基准，没有跨进程硬件方差保证。完整时长曲线见 [benchmark_gauge.json](results/benchmark_gauge.json)。当前四头混合部署仍比原模型慢，所有GQA KV组仍被剩余精确Qheads使用，因此原KV cache保留，外加129KiB状态；不能用这组数据声称全模型cache减少或推理加速。

仅按矩阵乘法计，每token每head双侧MLP特征为73728 MAC，FAVOR+为16384 MAC，约4.5倍；两者还各需约3.3万FLOP的相同状态更新/读取，忽略激活与归一化后总量比约2.75。矩阵乘法可批处理不等于已经证明GPU适配性优于FAVOR+，真实batch1额外kernel launch和GEMV仍有代价。分布标度校准的在线等价执行与父exp-MLP算子相同，校准不会额外扩大在线特征计算。

**本轮没有成立的主张与应停止的分支。**

1. 原始Gaussian covariance-only预测仍未在真实heads验证成立。此前16heads仅4个满足Gaussian L²可积条件；Schmidt尾定理在其假设内正确，不等于Gaussian模型能拟合真实Q/K。
2. 尝试用原始/重加权条件谱分配固定256维预算，原始与sqrt-mass谱均给出均匀[64,64,64,64]；row-mass谱[96,64,32,64]与验证输出配置[32,64,64,96]不同，未获得强PPL增益。容量分配本身也已有论文覆盖。
3. 正rank-m的混合信息论下界[I(Q;K)−log m]+在真实4096²训练经验乘积上、m≥16全部为0。对角toy例子验证其数学紧性，不能补救真实数据上的无效性。没有可用的m64近总体最优证书。
4. value加权原始损失有正确逐查询输出误差上界，但训练效果差于KL控制；不能把上界成立当成该损失一定更好。V重提取首训练分片相对旧Q/K有约1%的BF16差异，其他分片一致；这一失败分支也保留记录，未据此推断普遍无效。
5. 校准后的raw相对平方误差仍约数十个百分点；当前结果远不足以声称逼近原始谱极限。MI、谱、Poisson风险、PPL分别对应不同对象，不能混用。

**顶会潜力与下一阶段判据。**

我对当前完整证据的主观评价是4–5/10；作为继续投入的研究方向约6/10，不是录用概率。已有独立泛化、匹配消融、可部署正特征与精确标度误差分解，强于只展示一张训练矩阵拟合图；但主张的新颖性和实际价值还不足以支持强接收。

已有 [Hedgehog, ICLR2024](https://arxiv.org/abs/2402.04347) 的学习型特征/KL蒸馏、[LoLCATs, ICLR2025](https://arxiv.org/abs/2410.10254) 的注意力迁移、[DoF for Linear Attention, NeurIPS2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c98ef086dc70d528e1c1aa1e66893365-Abstract-Conference.html) 的分布复杂度与维度分配。行标度不变性和KL质量分解是经典事实；[Geometric Attention (2026)](https://arxiv.org/html/2601.11618v1) 也明确讨论行缩放。故“MLP+谱”“幅度方向分离”或“校准不改变attention”单独均不足以确立顶会创新性。这里只做公式/目标级匹配对照，尚未完整复现上述训练系统。

建议保留的主线是：**分布相关的正kernel近似，区分原始标度误差与真正影响attention的交互误差，并用固定预算的连续正特征同时控制两者。** 当前factorized_both是全程原始目标的主要实验支撑；分布校准提供误差可分离的构造性证据与控制实验。它们都是连续学习特征，不是FAVOR随机积分节点的简单替换，但“不是同一个公式”并不自动构成充分创新。

下一阶段应优先取得一个与正特征、非MSE风险一致且在真实heads不为零的总体下界/可预测误差，再检验其对未参与分析的heads是否有效；同时扩展到不同模型、足够多heads及长上下文/检索任务，验证在相同状态和训练成本下可重复超过KL蒸馏，并在覆盖完整GQA组的部署中证明实际cache/延迟收益。当前用户只授权4头局部替换，这些扩大部署未执行。若强对照始终不输、理论界仍不提供可用预测，应停止“逼近理论极限的新kernel”这一论文主张；可保留为kernel评价与校准研究。

复现入口为 [train.py](train.py)、[calibrate.py](calibrate.py)、[product.py](product.py)、[runtime_new.py](runtime_new.py)，推导与协议分别见上述文件。参数、数据去重、风险恒等式、状态递归、PPL加权和有限数值检查见 [final_audit.json](checks/final_audit.json)。全部负结果、原始逐文档测量和中间浮点版本均保留。



---

原文件：mlp_direction/REPRODUCE.zh.md

本轮复现依赖工作区中已经保存的冻结模型、Q/K数据及各轮manifest。完整重新提取/训练会覆盖或跳过已有结果，请先在独立目录保留当前产物。版本与源码校验值见 checks/final_audit.json 和 results/artifact_manifest.json。

训练、评估环境：/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python。绘图环境：/root/miniconda3/bin/python（前一环境没有matplotlib，因此报告使用后一环境；报告只读取JSON，不参与GPU实验）。

已完成的数据和配置记录：PROTOCOL.zh.md；数学假设、推导与边界：THEORY.zh.md；最终解读：REPORT.zh.md。

核心代码职责：

- core.py：两层正特征、原始I损失与条件矩形评价。
- train.py：单遍训练；fits/ 保存全部38组检查点/标度对照及元数据。
- fresh_data.py：三个独立确认轮次，results/fresh_manifest*.json 固定文档/token/hash。
- calibrate.py：16384个训练keys上的log-MGF与既有读出的单遍校准；生成对应全局标度对照。
- product.py：全部新经验Q/K边缘乘积，FP64分块统计原始kernel风险，固定网络seed11/FAVOR seed1009。
- baselines.py：既有FAVOR+/centered FAVOR+/SDERF/ADERF的5随机种子。
- runtime_new.py：各head独立的线性状态；校准kernel在线执行代数等价的父特征。
- gauge_assess.py：第三轮条件kernel与完整冻结模型四头替换PPL。
- benchmark_gauge.py：1024/4096/8192 prompt下的完整模型四头替换计时。
- verify.py、finalaudit.py：代数/递归/数据/参数/聚合检查。

已有GPU实验结果可直接重建表、文档bootstrap和图：

```bash
/root/miniconda3/bin/python /root/autodl-tmp/kan_attention_theory/mlp_direction/report.py
```

无需重新训练的检查：

```bash
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python /root/autodl-tmp/kan_attention_theory/mlp_direction/verify.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python /root/autodl-tmp/kan_attention_theory/mlp_direction/finalaudit.py
```

阶段日志为 results/stages.json、stages2.json、stages3.json、stages_final.json。初次原始μ折叠FP32部署结果单独保存在 ppl_gauge_folded_fp32.json；最终PPL在ppl_gauge.json。所有结果均为指定四个head，不包含全head替换或模型微调。



---

原文件：mlp_direction/THEORY.zh.md

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



---

原文件：normalized_attention_assessment/PROTOCOL.zh.md

本轮不训练、不选择检查点，不改变既有heldout数据。主对象为KL训练的纯删除AD，73536参数/head，m64；对照为既有同预算Hedgehog-exp/softmax，73728参数/head，m576，以及固定随机FAVOR+ m64。AD较HH少192参数/head（约0.26%）；FAVOR没有数据拟合，不能称同训练参数比较。

以1k128文档、8k24文档和种子11/29/47评价。缓存kernel指标基于原教师QKV的64个选定Q/文档；全模型PPL覆盖全部下一token，并让两层24/336个head同时替换后真实传播。头均值、token加权风险、输出误差和PPL分别报告，不能互换。

FAVOR使用与本轮AD/HH完全相同的Replacement、状态算子、精度、模型revision和文档重新运行verify、8k precision、kernel和PPL。原AD/HH文件只读复用；保存父数据来源和hash。

诊断全部4方法×3种子在两种split上的attention KL、TV、系数L2、输出平方误差、按head和位置分解，以及通过冻结W_O后的层输出误差。位置分桶只是既有8k文档中的观察，不能分离长度、内容、RoPE和训练分布等因素。诊断不用于调参。配对文档bootstrap条件于3个固定模型，另报告全部种子差值及胜出文档比例；不把tokens当独立样本，不做未知总体或顶会结论。

运行同一硬件参考计时复用上一轮新测结果。不是端到端解码benchmark。没有全层转换或LLM微调。所有结论限定本次模型、head、语料和训练预算。



---

原文件：normalized_attention_assessment/REPORT.zh.md

此次复核支持AD纯删除版在已测冻结模型局部替换中的质量/状态成本优势，但不支持所有意义上的attention系数近似都优于Hedgehog。8k时AD的KL较差，TV和系数平方误差较好，输出投影后误差和PPL也较好。FAVOR+ m64质量明显较差，但参考单步耗时更低、且不需要拟合训练。

对象：Qwen2.5-1.5B，层14/27的全部24/336 Qheads，4096篇1k训练文档单遍、3种子11/29/47。只使用选定的KL训练纯删除AD，参数73536/head、m64；HH-exp/softmax为本研究既有独立Q/K、同预算扩宽适配版，73728/head、m576，AD少0.26%参数。FAVOR+是固定正交高斯正特征m64，0个训练参数，不能称同训练预算或原论文完整系统比较。此次无训练、无超参选择、无全模型微调。

新kernel为g(q,k)=exp(s_K(k))π_Q(q)ᵀπ_K(k)，查询方向π_Q及K方向π_K位于64维概率单纯形。没有查询幅度或整体标度。通过线性注意力分母归一化。

评估集合为既有confirm_wiki128篇1k和confirm_long24篇8k；不是新的盲测。缓存系数/输出指标基于教师QKV的64个查询/文档，全模型PPL覆盖每篇所有下一token，并在两层同时替换后传播。后者层27的输入可能已受层14替换影响，不能将静态缓存指标直接当作完整模型误差分解。

**1k完整对照，越低越好**

| 方法 | 系数KL | 系数TV | 系数L2² | 输出NMSE | W_O后NMSE | PPL |
| --- | --- | --- | --- | --- | --- | --- |
| AD纯删除，KL训练 | 0.347341 | 0.263031 | 0.041484 | 0.220416 | 0.258934 | 8.858055 |
| Hedgehog-exp | 0.377019 | 0.274934 | 0.042883 | 0.231022 | 0.269639 | 8.870295 |
| Hedgehog-softmax | 0.402539 | 0.290152 | 0.051392 | 0.264257 | 0.302496 | 8.880211 |
| FAVOR+ m64 | 4.163506 | 0.653948 | 0.259675 | 1.792437 | 1.462967 | 9.857622 |

**8k完整对照，越低越好**

| 方法 | 系数KL | 系数TV | 系数L2² | 输出NMSE | W_O后NMSE | PPL |
| --- | --- | --- | --- | --- | --- | --- |
| AD纯删除，KL训练 | 3.422874 | 0.625747 | 0.107785 | 0.641975 | 0.567698 | 9.691610 |
| Hedgehog-exp | 2.551741 | 0.664561 | 0.147266 | 0.894505 | 0.695810 | 9.752359 |
| Hedgehog-softmax | 2.570944 | 0.678435 | 0.153673 | 0.745977 | 0.610394 | 9.752327 |
| FAVOR+ m64 | 5.902979 | 0.750986 | 0.245490 | 1.652875 | 1.278360 | 10.644294 |

TV=0.5Σ|p−p_hat|，系数L2²=Σ(p−p_hat)²；二者和KL均先按查询平均。输出NMSE先对每个文档/head汇总SSE/教师输出能量，再平均文档、heads和种子；W_O后NMSE对每个文档/层汇总后平均，包含跨head相加。所有方法分母和选定查询一致，但不同加权口径仍可能改变排序。

原模型PPL为8.664560/9.066279。AD纯删除的PPL仍高于原模型，不代表已实现无损转换。8k相对HH-exp/softmax的PPL降低约0.62%，相对FAVOR降低约8.95%。

**为什么KL与PPL可以反向**

对δ=p_hat−p，attention输出误差满足δV，平方误差为δVVᵀδᵀ；完整多head注意力子层继续做concat(δ_h V_h)W_Oᵀ，再进入残差和后续层。KL不包含V、W_O或下游模型，因此它不能确定这些误差的排序。KV数量大于value维度时，许多不同系数分布可产生相同或近似相同的输出。小KL可提供上界控制，但较大的KL不推出较大的实际输出误差。

一个精确示例：p=(0.5,0.25,0.25)，V=(0,1,1)。预测A=(0.5,0.49,0.01)的输出仍为0.5，尽管KL约0.6365；预测B=(0.45,0.275,0.275)的输出为0.55，KL仅约0.00503。此例仅解释数学上为何不存在单调关系，不代替真实数据诊断。

真实8k诊断：AD相对HH-exp在7/24heads的KL更好，但在21/24heads的输出NMSE更好；相对HH-softmax对应13/24与15/24。AD的L14H8、L14H9、L27H10三个最大KL heads解释了它相对HH-exp平均KL差距的94.08%；这是观察后的误差定位，不是用于调参的选择规则。

**KL的概率比惩罚已经在真实数据中定位**

| 方法 | 被低估超过1万倍的教师概率质量 | 这些位置的正KL贡献 | 全部正KL贡献 | 全部负KL贡献 |
| --- | --- | --- | --- | --- |
| AD纯删除，KL训练 | 12.226% | 1.776189 | 3.535038 | -0.112164 |
| Hedgehog-exp | 2.373% | 0.248866 | 2.662083 | -0.110342 |
| Hedgehog-softmax | 0.861% | 0.087814 | 2.695511 | -0.124567 |
| FAVOR+ m64 | 25.360% | 3.493970 | 6.025463 | -0.122484 |

低估集合定义为p_hat<p/10000，概率质量为Σ集合p，再平均heads、查询、文档和种子。AD的12.23%是教师概率质量，不是key数量比例。这一真实系数失配需要改进，不能因PPL较好就将其视为无关误差。它同时允许TV/L2较低，因为KL强烈惩罚概率比，而TV/L2衡量绝对概率差。正贡献与负贡献相加才等于总KL。

**检查输出指标的加权敏感性**

| 方法 | 全局能量加权输出NMSE | 全局能量加权W_O后NMSE | L14 W_O后NMSE | L27 W_O后NMSE |
| --- | --- | --- | --- | --- |
| AD纯删除，KL训练 | 0.471712 | 0.173465 | 0.975966 | 0.159429 |
| Hedgehog-exp | 0.514515 | 0.188792 | 1.220789 | 0.170831 |
| Hedgehog-softmax | 0.448660 | 0.185173 | 1.051057 | 0.169731 |
| FAVOR+ m64 | 1.401634 | 0.707747 | 1.862008 | 0.694711 |

全局能量加权采用Σ所有SSE/Σ所有教师能量。未经过W_O时，HH-softmax的0.44866优于AD的0.47171，说明此前平均head NMSE优势不等于任何加权都占优。经过真实W_O后，两种汇总下AD都较低。这支持错误方向及head/层加权会影响最终质量的解释，但还不是每个错误位置对PPL贡献的因果归因。

**跨种子、文档的可靠性**

| 对照 | 指标 | AD−对照 | 文档bootstrap95%区间 | 胜出文档数 | 各种子差值 |
| --- | --- | --- | --- | --- | --- |
| Hedgehog-exp | kl | 0.871133 | [0.8115, 0.934921] | 0/24 | [0.947366, 0.899358, 0.766677] |
| Hedgehog-exp | nll | -0.006250 | [-0.00883, -0.003834] | 21/24 | [-0.008224, -0.004268, -0.006258] |
| Hedgehog-softmax | kl | 0.851930 | [0.807422, 0.896223] | 0/24 | [0.91598, 0.881709, 0.758101] |
| Hedgehog-softmax | nll | -0.006247 | [-0.007822, -0.004685] | 23/24 | [-0.008551, -0.00567, -0.004519] |
| FAVOR+ m64 | kl | -2.480105 | [-2.622345, -2.345673] | 24/24 | [-2.464501, -2.488812, -2.487003] |
| FAVOR+ m64 | nll | -0.093727 | [-0.104299, -0.083572] | 24/24 | [-0.106264, -0.093374, -0.081542] |
| Hedgehog-exp | layer_projected_nmse | -0.128112 | [-0.152754, -0.104875] | 24/24 | [-0.114451, -0.1128, -0.157085] |
| Hedgehog-softmax | layer_projected_nmse | -0.042696 | [-0.053082, -0.03272] | 23/24 | [-0.04238, -0.038343, -0.047366] |
| FAVOR+ m64 | layer_projected_nmse | -0.710662 | [-0.746677, -0.674383] | 24/24 | [-0.816204, -0.667821, -0.64796] |

文档胜出数先对3种子取均值。所有区间只重采样文档，条件于这3个已训练模型；不覆盖未知训练随机性、其他模型/head或任意语料分布，多项诊断未统一多重校正。8k有24篇文档，不能按token数夸大独立样本量。

**位置外推诊断**

| 方法 | 8k文档内查询位置 | KL | TV |
| --- | --- | --- | --- |
| AD纯删除，KL训练 | 0–1023 | 0.347568 | 0.264302 |
| AD纯删除，KL训练 | 1024–2047 | 2.808584 | 0.582138 |
| AD纯删除，KL训练 | 2048–4095 | 3.721366 | 0.663979 |
| AD纯删除，KL训练 | 4096–8191 | 4.166038 | 0.704314 |
| Hedgehog-exp | 0–1023 | 0.384594 | 0.277936 |
| Hedgehog-exp | 1024–2047 | 1.741841 | 0.545241 |
| Hedgehog-exp | 2048–4095 | 2.479231 | 0.701518 |
| Hedgehog-exp | 4096–8191 | 3.307069 | 0.768964 |
| Hedgehog-softmax | 0–1023 | 0.404710 | 0.292007 |
| Hedgehog-softmax | 1024–2047 | 1.672596 | 0.606142 |
| Hedgehog-softmax | 2048–4095 | 2.635621 | 0.726224 |
| Hedgehog-softmax | 4096–8191 | 3.282673 | 0.765637 |
| FAVOR+ m64 | 0–1023 | 4.174994 | 0.653573 |
| FAVOR+ m64 | 1024–2047 | 5.347515 | 0.718938 |
| FAVOR+ m64 | 2048–4095 | 5.876663 | 0.752714 |
| FAVOR+ m64 | 4096–8191 | 6.467166 | 0.781436 |

在同一批8k文档的前1024位置，AD的KL约0.348，仍优于两个HH；之后明显恶化。这与1k训练范围外的外推困难一致，但不能仅凭分桶判定是RoPE、内容、距离或某一结构因素单独导致。

**计算成本**

| 方法 | m | 仅特征µs | 特征+状态µs | 状态MiB/层 |
| --- | --- | --- | --- | --- |
| AD纯删除，KL训练 | 64 | 32.848 | 64.431 | 0.380859 |
| Hedgehog-exp | 576 | 28.384 | 98.016 | 3.427734 |
| Hedgehog-softmax | 576 | 33.312 | 103.392 | 3.427734 |
| FAVOR+ m64 | 64 | 27.904 | 59.264 | 0.380859 |

复用上一轮统一RTX3090/FP32/batch1/单层12heads/CUDA graph参考计时：AD比HH-exp少34.3%耗时、比HH-softmax少37.7%，状态约1/9；AD比FAVOR慢8.7%，状态相同。HH的特征计算本身更快，总成本差异主要来自m576状态。不是全模型decode收益，也不是最优融合算子比较。

**结论的准确范围**

若目标是本实验中的局部替换语言模型质量/状态成本，AD纯删除比两个HH适配版有证据支持的优势；若目标是8k系数KL，两个HH更好；若目标是TV/L2系数近似，AD更好。相对固定FAVOR+ m64，AD在已测1k/8k质量指标上更好，但需拟合且单步更慢。尚不能声称优于整个Hedgehog或FAVOR方法族、相同m下最优方法、全部heads转换、从头预训练或未知总体理论下界。

FAVOR本轮用相同Replacement、算子、dtype、模型revision与文档复测，6个PPL与旧值完全相同。全部12个模型的FP64缓存指标重算与父结果最大差异小于1e−9。文件来源和hash见results/manifest.json，检查见checks/final_audit.json。复现：driver.py → diagnose.py → tail_diagnose.py → aggregate.py → write_report.py → audit.py。



---

原文件：normalized_kernel_novelty_20260908/REPORT.zh.md

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



---

原文件：novelty_review_20260908/REPORT.zh.md

本次为2026-09-08对已有同参结果、计算口径和相关工作的核对，没有新增训练或推理计时。实验来源为hedgehog_matched/results/{summary_product,summary,benchmark}.json。论文结论以原文为准；Hugging Face markdown可能缓存旧版本，尤其2506.21137的标题与公式应核对arXiv v3。

**已有结果与预算。** 冻结Qwen2.5-1.5B，层14/27全部24个Qheads，4096训练文档、每个拟合单遍、3种子。原始目标为exp(qᵀk/√128)，主比较使用跨文档配对的未平衡原始I-divergence，每head固定训练尺度只用于数值计算。AD与HH-exp各73729训练参数，独立Q/K网络、无投影偏置，并各有一个训练head标度。HH是加宽至128→288、拼接exp(z)/exp(-z)的同参适配，不是论文默认配置或完整训练系统。

| 项目 | AD | 同参HH-exp |
| --- | ---: | ---: |
| 两侧网络 | 128→192→64，SiLU | 128→288，exp正负拼接 |
| 特征维度m | 64 | 576 |
| 参数/head | 73729 | 73729 |
| A→B原始I相对风险 | .272450 | .469970 |
| B→A原始I相对风险 | .266388 | .470631 |
| A→B原始L2相对误差 | .365211 | .370372 |
| B→A原始L2相对误差 | .299519 | .363239 |
| 乘积训练后1k全模型PPL | 9.025792 | 9.063196 |
| 乘积训练后8k全模型PPL | 9.803906 | 9.863059 |

I风险下降42.03%/43.40%；按3种子均值，每方向24/24head更低。A/B是128篇既有留出文档拆成的两个64文档集合；每方向1024个Q×4096个K的完整所选向量乘积。不是未知总体，也不是每个pair独立。A→B的L2差值95%条件区间[-.04962,.07082]跨零。PPL收益仅约.41%/.60%，且都劣于原模型8.664560/9.066279。测试集合此前使用过；不是首次盲测。

按Hedgehog惯用的attention方向KL，另一次同因果采样目标匹配中，AD/HH-exp的1k PPL为8.85819/8.87029，8k为9.69226/9.75236；HH-softmax为8.88021/9.75233。AD只略优。此处各登记73728参数，但AD的192个query幅度输出权重不被纯KL识别，不能说有效自由度严格相同。原始I主比较不受该问题影响。乘积raw训练的AD不应因胜过同目标HH，就被称为胜过所有采用原生目标训练的Hedgehog。

**计算与速度。** FLOPs采用一次乘加计2的标准主导项，不计原模型QKV投影、FFN等公共计算，也不完整计入激活、softmax归约、输入标准化或动态稳定重缩放。对一对新增q/k及value，双侧投影AD为4h(d+m)，HH为4dp；状态更新与读取约4md_v。本实验d=d_v=128,h=192,p=288。

| 主导FLOPs/head/token | AD | HH-exp | FAVOR+ m64 |
| --- | ---: | ---: | ---: |
| 双侧投影 | 147456 | 147456 | 32768 |
| 状态更新与读取 | 32768 | 294912 | 32768 |
| 合计 | 180224 | 442368 | 65536 |

AD主导FLOPs较同参HH少59.26%，状态为其1/9；相同参数不等于相同m、状态或FLOPs。若同m64，单层HH所需参数远少于当前AD，因此速度胜利不可推广为全部HH预算。

计时RTX3090、FP32、batch1、单层12heads、真实QKV，CUDA graph热缓存，同一未充分融合的PyTorch参考实现：

| 方法 | 仅特征µs | 特征+状态更新/读取µs | 状态MiB/层 |
| --- | ---: | ---: | ---: |
| AD | 42.688 | 74.204 | .380859 |
| HH-exp | 28.288 | 98.240 | 3.427734 |
| FAVOR+ m64 | 27.520 | 59.040 | .380859 |

AD特征生成耗时为HH的1.51倍，完整微测耗时少24.47%（1.324倍加速）。与FAVOR相比，AD完整微测耗时多25.68%。状态包含FP32的s,z,g；这些数字不是全模型decode。计时来自首阶段权重；乘积阶段网络形状、算子相同，但权重没有单独重测。不能把旧Triton融合5–10µs数据与本表拼接，也没有据此证明AD算子对GPU内在更友好。

**原文相关工作与重合范围。**

- [Hedgehog，2024](https://arxiv.org/html/2402.04347v1#S4.SS2)：学习正feature map，以归一化attention交叉熵/蒸馏匹配教师。正文与附录的共享描述不完全一致，本地采用附录转换路径的独立Q/K模块。MLP正特征、冻结教师后的特征拟合不是当前工作的独有创新。
- [STILL，2026，§3.2](https://arxiv.org/html/2602.02180v1#S3.SS2)：NP-Map先计算u=f(x)/||f(x)||·||x||，然后concat(softmax(u),softmax(-u))。这是LLM转换中明确的方向/幅度分离与原始输入范数重注入。其范数进入softmax内部，改变方向；不是学习原始kernel分布积分Z(q)。完整STILL还包含混合路由，不能把其全系统效果归给NP-Map。
- [Norm×Direction / NaLaFormer，v3](https://arxiv.org/html/2506.21137v3)：显式分解范数和方向，query范数进入逐元素幂的指数，结合三角方向特征。它并非简单标量乘法，不能误称其norm分支在分母中完全消去。其研究目标偏向注意力尖锐性与模型质量；不是当前原始I风险学习方案。
- [MALA，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Fan_Rectifying_Magnitude_Neglect_in_Linear_Attention_ICCV_2025_paper.pdf)：已分析feature query范数的消去，并以缩放与平移改变attention权重，见[§3.3](https://arxiv.org/html/2507.00698v3#S3.SS3)。含减法、上下文相关系数，理论上可能出现非正权重，与固定正可分离原始kernel不同。不能据某些齐次feature的结论断言所有非线性feature都忽略原始q范数。
- [Degrees of Freedom for Linear Attention，NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c98ef086dc70d528e1c1aa1e66893365-Abstract-Conference.html)：使用分布相关积分算子/有效维度确定feature预算，并逐层学习PRF；原文§3.3同时研究原始kernel L2和softmax交叉熵。分布相关谱理论、原始kernel拟合及逐层学习不能被笼统宣称为首次。
- [FAVOR++ / Chefs’ Random Tables，NeurIPS 2022](https://research.google/pubs/chefs-random-tables-non-trigonometric-random-features/)和[FAVOR#，2023](https://arxiv.org/abs/2302.00787)：已有参数化正随机特征、利用输入统计降低估计方差的理论与构造。当前AD是学习有限正函数族，未承诺随机特征的无偏性；仅超过基础FAVOR+不是充分现代基线。
- [Efficient Attention，WACV 2021](https://openaccess.thecvf.com/content/WACV2021/papers/Shen_Efficient_Attention_Attention_With_Linear_Complexities_WACV_2021_paper.pdf)：已有Q/K分别softmax归一化并以共享上下文向量混合。其K沿序列维归一化，与当前两侧按feature维归一化、学习独立幅度不同；仍说明“softmax feature + 正混合”并非空白。

检索原论文未发现与当前“m−1个方向logit＋1个可学习log幅度、双侧两层网络、在真实PQ×PK上以raw I拟合exp(qᵀk/√d)”全部一致的配方。这不是首创证明。未对新发现的NP-Map、NaLaFormer、FAVOR++/#做同协议训练比较。

**创新性判断。** 单看公式与幅度/方向拆分，创新性偏弱，已有明确近邻；共同函数类层面，任何严格正向量f都能写成(sum f)·(f/sum f)，所以这是一种坐标表示，而非独有表达能力定理。原始I质量分解也是经典恒等式的应用。有限网络的参数化会影响表达与优化，需同深度、同m、同损失的exp/softplus MLP控制来检验，不能把两层AD对一层HH的差异全部归给幅度分支。

可继续聚焦的贡献是：在固定正特征状态与网络预算下，用分布相关原始kernel风险解释并控制幅度/方向两项；得到跨文档、长文及第二模型可复现的性能—状态—速度曲线。当前42–43% raw I下降是有价值的初步信号，非总体最优性或充分顶会证据。若能进一步给出有限m正方向混合的非平凡逼近上界，连接正性代价、网络容量及估计误差，并在NP-Map和同深度MLP等近邻对照下维持优势，贡献会更清楚。无需改变用户的原始kernel目标或转向混合窗口研究。



---

原文件：orbit_direction/PROTOCOL.zh.md

这是一轮在causal_direction旋转诊断之后明确启动的后续实验。前一轮确认集只用于发现问题，不再充当本轮新方法的盲确认集。

唯一训练变动是共同RoPE旋转增强：每篇训练文档所有Q/K使用同一个δ，以0.5概率δ=0，否则整数均匀取0..32768。κ(Rδq,Rδk)=κ(q,k)，标签保留原始未旋转点积，避免旋转浮点误差改变标签。固定m64、两侧128→192→64、73856参数/head、24heads、4096训练文档、64Q/文档、134360517合法配对/head、原均值/方差归一化、同初始化/文档顺序/AdamW与lr、4096单遍更新。每个候选只访问每个选中配对一次用于更新，不为增强多跑epoch。

预设两项：质量权重λ=.1、同架构λ=0的KL控制，种子11/29/47全部保留。增强δ序列在同种子的两种损失间完全相同。与上一轮未增强的相同架构两项、softplus-KL、FAVOR+比较；不选择性隐藏增强对KL的影响。新Wiki64篇1024tokens和新长文16篇8192tokens在任何本轮方法评价前确定，排除所有历史训练/验证/确认文本和token哈希。文档数量较小，属于初步确认，不冒充大规模benchmark。

检查原始kernel误差、方向KL、attention输出；完整模型PPL包含原模型、同持久缓存预算SWA448+4以及两层完整GQA替换的纯/64窗口混合。没有重新训练LLM，不扩展更多层。核函数类和算术复杂度未变，因此该试验首先判断对称性增强是否解决泛化缺口，不能仅凭增强训练将其称为新的kernel数学形式或新的顶会方法。只有胜过同样增强的KL控制才可能支持原始质量损失的额外价值。



---

原文件：orbit_direction/REPORT.zh.md

本轮在上一轮暴露共同RoPE旋转敏感性后，完成了一个新的、预先固定的增强实验：6个同预算拟合，另取64篇1024-token Wiki和16篇8192-token长文，与所有前序数据文本／token哈希不重叠。**旋转增强确实改善了长文泛化，但短文拟合变差；同样增强的KL也获得改善。** 它验证了一个可操作的缺口，未建立原始质量目标独有的优势。

**构造与预算。** 数学上的kernel类仍是连续rank64正MLP，非固定分区。训练输入改为同文档Q/K共同Rδ旋转，δ以.5概率为0，否则整数均匀0..32768；原始exp(qᵀk/√d)标签不变。每head73856参数、4096文档、262144个Q、134360517合法配对、4096更新、同种子初始化／文档顺序／归一化与AdamW设置，且两个目标的δ序列完全相同。目标λ=.1仍保留原始质量；λ=0是匹配强控制。LLM所有权重仍冻结，只替换同样两层全部24/336 heads。固定推理状态、FLOPs和算子结构与上一轮相同。

这是对“谱相同、固定feature却未泛化”的训练侧处理，尚未改变feature函数类，也没有把普通增强包装成新的解析kernel形式。完整协议见 [PROTOCOL.zh.md](PROTOCOL.zh.md)。

**新文档上的kernel确认。** 三种子、24heads均值如下；raw NMSE仅作测试诊断，训练未用MSE：

| 方法 | Wiki输出NMSE | 长文输出NMSE | 长文原始kernel NMSE均值 | 长文原始NMSE按head中位数 |
|---|---:|---:|---:|---:|
| split_01 | 0.228648 | 0.639424 | 1.5739e+16 | 1.2091 |
| split_kl | 0.220912 | 0.659319 | 7.0339e+25 | 9.3392e+07 |
| softplus_kl | 0.222917 | 0.672475 | 1.0141 | 0.99902 |
| favor | 1.822248 | 1.663831 | 1.0162 | 1 |
| orbit_split_01 | 0.265791 | 0.524710 | 34.521 | 0.99299 |
| orbit_split_kl | 0.257695 | 0.520259 | 1.4085e+21 | 1.8826e+07 |


主方法长文方向KL约3.38→1.73、输出NMSE约.639→.525，说明增强具有作用。增强KL输出NMSE约.520，说明主要改善并非质量项所独有。主方案长文原始NMSE从约1.6e16降至34.5，仍不能称原始kernel已拟合良好；均值和head中位数一起报告以暴露尾部失真。短文输出误差约.229→.266，存在明确取舍。

对本轮前16篇新Wiki作相同目标的受控旋转确认（三种子）：offset1024下原主方案方向KL为1.959，增强后.682；offset8192下为1.452→.625。相同增强的KL控制也达到.674和.606。目标logit共同旋转前后最大差为5e-14，核验了“目标与谱不变，固定feature的泛化改善”这一机制，而非目标被换掉。原始kernel不变性是数学性质；当前训练仅部分减弱敏感性，未构造严格不变的有限维正kernel。数据见 [rotation_confirmation.json](results/rotation_confirmation.json)。

**完整模型确认。** 每篇所有下一token位置计算PPL，种子均值如下。不同集合的绝对PPL不可与上一轮不同文档直接相减，下面全部使用本轮相同文档：

| 方法 | 新Wiki1024 PPL | 新长文8192 PPL |
|---|---:|---:|
| 原模型 | 9.299281 | 9.333546 |
| SWA448+4 | 9.364104 | 9.604680 |
| FAVOR+ | 10.591060 | 10.851284 |
| softplus-KL | 9.511564 | 9.889428 |
| 原始质量+KL | 9.528233 | 9.946173 |
| 原始质量+KL+旋转增强 | 9.553555 | 9.838766 |
| 同架构KL | 9.515930 | 9.944847 |
| 同架构KL+旋转增强 | 9.538880 | 9.811059 |
| 原始质量+KL+窗口64 | 9.437533 | 9.838060 |
| 原始质量+KL+增强+窗口64 | 9.453923 | 9.748869 |
| softplus-KL+窗口64 | 9.607410 | 9.827074 |
| 增强KL+窗口64（未校准标度） | 9.596563 | 9.824128 |


增强KL混合版本未做query质量或门控校准，其原始标度具有任意性；它不能单独充当公平的强混合质量对照。上一轮已做的匹配query读出／门控KL比较没有显示原始质量独特优势；本轮混合方案的实用结论主要应与softplus-KL窗口和缓存匹配SWA比较。不能把两轮不同文档、不同父模型的校准成绩直接拼成一次胜利。

| A−B | 集合 | ΔNLL/token | 文档配对95%区间 |
|---|---|---:|---|
| pure/orbit_split_01 − pure/split_01 | orbit_wiki | 0.002654 | [0.001638, 0.003711] |
| pure/orbit_split_kl − pure/split_kl | orbit_wiki | 0.002409 | [0.001561, 0.003233] |
| pure/orbit_split_01 − pure/orbit_split_kl | orbit_wiki | 0.001538 | [0.000720, 0.002399] |
| hybrid/orbit_split_01 − hybrid/split_01 | orbit_wiki | 0.001735 | [0.000999, 0.002487] |
| hybrid/orbit_split_01 − hybrid/softplus_kl | orbit_wiki | -0.016105 | [-0.018310, -0.013942] |
| hybrid/orbit_split_01 − window/swa448_sink4 | orbit_wiki | 0.009546 | [0.006919, 0.012421] |
| pure/orbit_split_01 − pure/split_01 | orbit_long | -0.010858 | [-0.012345, -0.009472] |
| pure/orbit_split_kl − pure/split_kl | orbit_long | -0.013543 | [-0.015579, -0.011550] |
| pure/orbit_split_01 − pure/orbit_split_kl | orbit_long | 0.002820 | [0.002026, 0.003649] |
| hybrid/orbit_split_01 − hybrid/split_01 | orbit_long | -0.009107 | [-0.010842, -0.007439] |
| hybrid/orbit_split_01 − hybrid/softplus_kl | orbit_long | -0.007990 | [-0.012349, -0.004052] |
| hybrid/orbit_split_01 − window/swa448_sink4 | orbit_long | 0.014901 | [0.011474, 0.018270] |


区间为条件于三个已训练模型的配对文档bootstrap，另保留每个种子差值；新长文只有16篇，仍属初步确认。没有第二个LLM、广泛下游任务或全模型线性化的证据，也没有已证明接近未知总体谱下界。主线理论和缓存／延迟实测见 [前一轮完整报告](../causal_direction/REPORT.zh.md)。增强未增加部署算术或状态大小，但本轮没有按新权重重新做完整速度测试；不把同一算子结构表述为已经独立实测出相同延迟。

**方向评价。** 已得到可复查的机制现象及一次新数据确认：原始目标谱不变的共同旋转，会破坏现有固定特征；训练中补足该变换可改善长度泛化。适合继续研究如何以固定m保留位置变换下的结构，并检验其对原始质量与归一化输出的不同影响。单独“MLP＋谱框架＋旋转增强”的方法新颖性和当前质量／效率优势，仍不足以支持顶会方法论文。下一步的建设性kernel、相同增强／校准的强KL控制、同缓存SWA与更紧分布下界都不可省略。

当前证据宜表述为“发现并部分缓解泛化障碍”，不能表述为“已找到接近总体理论极限的新kernel”。图：[augmentation_generalization.pdf](figures/augmentation_generalization.pdf)。原始数据：[summary.json](results/summary.json)。复现依次运行collect.py select、train.py、collect.py extract、assess.py kernels、assess.py ppl、verify.py；均使用既定本地模型／数据缓存。



---

原文件：positive_generalization/REPORT.zh.md

本轮判定：**当前“covariance 适配的正指数 quadrature＋单遍训练”实现 No-Go；没有兑现测试矩阵 NMF 展示出的精度空间。** 这个判定针对本轮函数构造、初始化、损失和优化设置的组合，不是对所有正特征、KAN 或 linear attention 的不可能性结论。

冻结真实 Qwen2.5-1.5B 后，共完成 **24 组主实验＋6 组损失消融**，每组包含4个独立的head模型。训练和评估目标始终是

$$\kappa(q,k)=\exp(q^\top k/\sqrt{128}),$$

没有把标签换成 softmax 概率。只有附加 attention 输出评估才通过分母归一化。所有实验在本地完成，没有使用子 agent。

**这一步验证什么**

上一轮得到的是：为每个测试矩阵直接求解 NMF，可以得到远低于 FAVOR+ 的误差。本轮要求 feature map 在见到测试文档之前就固定，检验这一空间是否能转化成可泛化的函数。

这个区分必须保留：测试矩阵上的 SVD、NMF 可以根据全部测试矩阵元素重新确定因子；本轮学习型特征和随机特征只能使用训练统计量。有限矩阵 NMF 是可行参照，不是已达到的总体正特征最优值。

**数据和协议**

- 模型：Qwen/Qwen2.5-1.5B，revision `8faed761d45a263340a0528343f099c05c9a4323`；使用真实投影和 RoPE 后缓存的 Q/K，正确处理 GQA。
- Heads：L14H0、L14H6、L27H0、L27H6。它们是此前大数据实验已经选定的4个head。本轮是上一轮16-head初筛后的增量验证，不是新的16-head全面评估。
- 大数据：4096篇训练、128篇验证、256篇内部测试文档，每篇1024 tokens。来源是 WikiText103 的源 train split，按文档划分，因此这里的“内部测试”不是官方benchmark test。
- 另用此前固定的20篇官方 WikiText2 test 文章前缀复核；这些文档与4096篇训练文档的token哈希无交集。这同样只是固定子集，不是完整benchmark。
- Q取位置512–1023，K取0–511。训练中对全部2,097,152个K向量做一次全局随机排列，与Q一一配对，目标是经验乘积分布。每个Q、K以及配对在每次运行中都恰好进入一次梯度更新。
- 每步512个新配对，共4096步；主实验 m=16/32/64/128、种子11/29/47、两种节点形式。固定 Adam 学习率0.003→0.0003、逐head梯度裁剪、最终checkpoint，没有依据测试分数选checkpoint。
- 训练前的均值、covariance和初始化幅度校准会读取训练数据；“单遍”指梯度训练配对不重复，不是声称整个算法只有一次数据读取。
- 两套测试文档分别取4次独立Q/K边缘池抽样，构造512×512乘积矩阵；另外各取4篇同上下文的合法矩形块。跨文档矩阵用于分布核诊断，同上下文块用于部署相关对照，二者分开汇总。

主实验做24组四head训练，随后依据过高预测诊断，增加6组 m=64 的 β=1.5 损失消融。后者是复用测试集的探索性结果，不能当作新独立测试集上的确认性结论。

**正特征的具体构造**

先令 $x=d^{-1/4}(q-\mu_Q)$、$y=d^{-1/4}(k-\mu_K)$。通过训练 covariance 的矩阵分解得到可逆变换

$$z_Q=B_Qx,\qquad z_K=B_Ky,\qquad B_Q^\top B_K=I.$$

因此 $z_Q^\top z_K=x^\top y$。小量ridge只用于选择稳定的可逆坐标，不截断 bilinear kernel，也没有改变温度。再定义

$$c_Q(q)=\frac{(q-\mu_Q)^\top\mu_K+\mu_Q^\top\mu_K/2}{\sqrt d},$$
$$c_K(k)=\frac{(k-\mu_K)^\top\mu_Q+\mu_Q^\top\mu_K/2}{\sqrt d}.$$

两侧特征为

$$f_r(q)=\sqrt{w_r}\exp\left(c_Q(q)-\frac{\|z_Q\|^2}{2}+\omega_{Q,r}^\top z_Q\right),$$
$$g_r(k)=\sqrt{w_r}\exp\left(c_K(k)-\frac{\|z_K\|^2}{2}+\omega_{K,r}^\top z_K\right),\quad w_r>0,$$

$$\hat\kappa(q,k)=\sum_{r=1}^{m} f_r(q)g_r(k).$$

shared版本约束 $\omega_{Q,r}=\omega_{K,r}$；untied版本允许两侧节点独立。这里的“共享节点”发生在变换后的空间，不表示原始q、k上完全相同的函数。两种形式都能放入标准linear attention公式。

初始化来自训练统计量适配的SDERF、经高斯逆CDF映射的scrambled Sobol节点，以及正的importance权重。再用固定的65,536个训练配对匹配整个head的平均kernel幅度。这个校准是一个head常数，不是逐行归一化。随后学习节点和log权重；学习完成后不再具有随机特征的无偏性保证。

m=64时，shared有8,256个可训练参数/head，untied有16,448个；可逆变换矩阵是训练统计量决定的固定参数。两种模型的参数量不同，此处比较的是解除节点共享的效果，不能作为等参数架构优势结论。本轮没有新增KAN/MLP架构对比；此前两层KAN、mulKAN与MLP的等参数实验仍应单独解读。

**基线与训练损失**

基线包括标准FAVOR+、FAVOR#论文中的SDERF和ADERF、covariance平衡后的FAVOR/SDERF，以及Sobol节点和训练幅度校准消融。每种固定特征基线10个随机种子、4种m。SDERF/ADERF是依据原论文公式实现的特征基线，不是对FAVOR#完整LLM系统的复现。[DERF / FAVOR#，NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/02dec8877fb7c6aa9a79f81661baca7c-Paper-Conference.pdf)

主实验最小化原始kernel的广义KL：

$$D_I(\kappa\|\hat\kappa)=\kappa\log\frac{\kappa}{\hat\kappa}-\kappa+\hat\kappa.$$

补充消融使用Bregman形式的β-divergence，本文采用β=1对应广义KL、β=2对应半平方误差的参数化：

$$D_\beta(\kappa\|\hat\kappa)=\frac{\kappa^\beta+(\beta-1)\hat\kappa^\beta-\beta\kappa\hat\kappa^{\beta-1}}{\beta(\beta-1)},\qquad\beta=1.5.$$

它对过高预测的增长量级是 $\hat\kappa^{1.5}$，比广义KL更强，又不是MSE。两种损失的数值缩放均是训练数据确定的head常数，原始目标在评估时完整恢复。[β-divergence的Bregman背景](https://www.mdpi.com/1099-4300/12/6/1532)

SVD与测试矩阵NMF继续使用原始kernel的平方误差，作为上一轮协议要求的谱诊断；这不代表神经特征训练改回了MSE。

**主要结果**

下面的误差是 $\sum(\hat K-K)^2/\sum K^2$。先按真实平方能量合并4次矩阵抽样，再对种子求均值；FAVOR为10种子，学习型为3种子。预测全零的误差等于1。

原官方测试子集，m=64：

| head | 测试矩阵NMF | FAVOR+ | shared，KL | untied，KL | shared，β=1.5 |
|---|---|---|---|---|---|
| L14H0 | 1.556e-05 | 1 | 220 | 338.5 | 1130 |
| L14H6 | 0.0005385 | 3.605 | 306.4 | 202.9 | 576.5 |
| L27H0 | 0.003806 | 1 | 75.82 | 63.1 | 4.15 |
| L27H6 | 0.00178 | 1.448 | 40.91 | 52.47 | 5.539 |

新增内部测试文档，m=64：

| head | 测试矩阵NMF | FAVOR+ | shared，KL | untied，KL | shared，β=1.5 |
|---|---|---|---|---|---|
| L14H0 | 1.486e-05 | 0.9999 | 4.339e+04 | 3.645e+04 | 8.722e+04 |
| L14H6 | 0.0004069 | 1.352 | 4536 | 1.472e+04 | 1392 |
| L27H0 | 0.0008073 | 1.006 | 11.74 | 9.655 | 2.625 |
| L27H6 | 0.00371 | 5.072 | 139.8 | 172.4 | 16.09 |

均值受到极端种子影响，但改看3个学习种子的中位数仍不通过。主实验shared在官方测试上的逐head中位数是272.28、247.40、69.44、6.50；内部测试为831.44、4092.49、7.18、36.09。这不是单个异常种子就能解释的失败。

本轮工程判据为：m=64时，在至少3/4个head上，比各固定RF基线的种子均值低25%以上，并且在至少3/4个head上达到测试矩阵NMF误差的10倍以内。学习型用种子中位数判定。**主实验在两套乘积分布、以及两套同上下文对照上均为0/4通过。** β=1.5消融也没有head达到NMF的10倍以内。

β=1.5在官方测试L27H0把shared的误差75.82降到4.15，在L27H6从40.91降到5.54，但L14H0和L14H6反而变差。因此不能把换损失本身视为完成了可泛化近似。

所有m的原始曲线、各RF单独结果、种子均值/中位数/极值和I-divergence均保存在curves.csv。m增大没有稳定解决学习型误差；不能根据某一个head或某一个种子选择性地宣称成功。

![独立内部测试文档](figures/internal_generalization.png)

![原官方测试子集](figures/official_generalization.png)

![损失消融](figures/beta15_ablation.png)

**失败原因：目前能确认与不能确认的部分**

第一，训练目标确实学到了某些有用的东西。以官方测试乘积分布的L27H0为例，FAVOR+的广义KL/真实质量为14.30，shared为1.85；对应同上下文块的attention输出误差均值，FAVOR+为1.159，shared为0.383。这里的输出比较是归一化后的附加指标，不是训练目标，且只覆盖4篇文档。它说明“KL/attention改善”和“接近原始kernel的L2谱尾”是不同的要求。

第二，本轮主要平方误差来自**过高预测**。在内部乘积测试上，合并4次抽样和3个种子的总SSE，shared四个head由过高预测贡献的比例依次为99.998%、99.987%、94.13%、99.51%；untied为99.998%、99.996%、93.01%、99.59%。在L14H0上，真实kernel最大0.1%的位置只占shared总误差约4.75%；大量错误来自模型在其它位置制造的大值。这与此前MLP主要低估真实极大值的情况不同。

第三，损失下降不能推出L2误差下降。令 $r=\log\hat\kappa-\log\kappa$，则

$$D_I=\kappa(e^r-1-r),\qquad(\hat\kappa-\kappa)^2=\kappa^2(e^r-1)^2.$$

过高预测时，前者对预测值近似线性增长，后者平方增长；这正是少量假峰值会主导平方误差的机制。β=1.5加强了这一侧的惩罚，但本轮仍不足以取得所需精度。

第四，不能把当前误差当成函数族的理论极限。同一正权重函数族让所有权重趋近零，L2相对误差就会趋近1；本次许多解远大于1，因此至少存在损失、有限数据、初始化或优化所得解与目标L2风险不一致的问题。单凭这组训练不能证明“正指数特征的最优误差就是这么高”，也不能排除更好的优化或有界特征构造。

第五，没有充分证据把问题归为单纯的数据不足导致过拟合。每个head已经使用约210万唯一配对；m=64时，多数head的完整训练KL与测试KL接近，训练平方误差经常比测试还大。固定验证集上，shared的L14H6从初始5.64降到4.25、L27H6从3.19降到1.98，但L14H0从5.24升到6.87。末尾512步有的改善、有的恶化，不能宣称整体已收敛，也不能只用一条下降曲线认定统一的欠拟合原因。

![固定验证集上的训练轨迹](figures/validation_convergence.png)

第六，之前“最大0.1%贡献91%–99.7%平方能量”是特定配对测试上的统计，不是所有真实LLM矩阵的分布定律。本轮512×512乘积矩阵的逐块对应比例覆盖约31.8%–99.99%；极端能量的估计强烈依赖head和抽样单位。原始kernel矩阵谱、配对风险、同上下文attention输出应分别报告。

**理论主线应如何据此修正**

仍保留Schmidt下界，但需要明确不同集合：

$$E_m^{\rm signed}\le E_m^{+}\le E_m^{\rm chosen\ family}\le R(\hat\kappa_{\rm trained}).$$

这些量必须基于同一个总体分布和同一种风险定义。本轮NMF是有限测试矩阵上 $E_{m,\mathrm{emp}}^{+}$ 的一个可行上界，不能直接代入总体不等式当作可泛化最优值。即使误差非常小，也可能利用了该矩阵中特定极值的位置和幅度。

covariance适配的正特征也不能直接称为新颖贡献。FAVOR#已经包含矩阵参数、二阶统计量和闭式的平均log-variance优化。本轮独立检查显示，SDERF在四个head上的训练平均log二阶矩从FAVOR的48.10/42.98/50.17/56.04降到2.47/5.85/16.14/21.25，但它的测试平方误差并未随之改善。平均log量降低，不保证非log的平均平方误差降低。

一个更贴近当前失败的推导如下。令 $x,y$ 是去均值后的缩放向量，并完整恢复前述 $c_Q,c_K$。对可逆 $T$，使用变换 $T^\top x,T^{-1}y$；记 $H=TT^\top\succ0$。对独立高斯正随机特征，有

$$\operatorname{Var}_{\omega}[f_\omega(q)g_\omega(k)]
=\kappa(q,k)^2\left[\exp\left(x^\top Hx+y^\top H^{-1}y+2x^\top y\right)-1\right].$$

所以，对 **iid** 节点，平均随机特征估计器的总体MSE为

$$\frac1m\mathbb E_{P_QP_K}\left[\kappa^2\left(e^{x^\top Hx+y^\top H^{-1}y+2x^\top y}-1\right)\right],$$

前提是这个期望存在。该公式不能直接作为ORF或Sobol的方差公式，因为它们的节点存在依赖。

这说明真正的L2风险涉及 $\kappa^2$ 加权后的极端区域，普通covariance或平均log-variance不够。对固定样本与固定中心，定义

$$J(H)=\mathbb E\!\left[\kappa^2\exp\left(x^\top Hx+y^\top H^{-1}y+2x^\top y\right)\right].$$

$x^\top Hx$对H线性，$y^\top H^{-1}y$对正定H凸，指数凸且单调，因此有限样本的J对H凸；总体版本在适当定义域上也保留凸性。这是可继续推导的受限预条件问题，尚未在本轮实施优化，也不主张其文献新颖性。凸性依据为标准matrix-fractional函数性质。[Boyd–Vandenberghe，例3.4](https://www.stanford.edu/~boyd/cvxbook/bv_cvxbook.pdf)

然而，更高阶加权风险本身会更难估计：必须先验证跨文档稳定性、极值覆盖和样本量敏感性，不能再用小矩阵的低谱尾替代这些问题。若坚持用KL类训练目标，也需要额外建立其与L2风险的比较条件；例如在真实值和预测值都有统一上界B时，可用Bregman曲率得到 $(\kappa-\hat\kappa)^2\le2B D_I(\kappa\|\hat\kappa)$，而极大或无法控制的B会使该界失去实际意义。

**对课题投入和顶会潜力的评价**

目前没有证据支持“只要沿covariance谱做正quadrature，就能在真实LLM上接近rank-m极限”，也没有新增证据支持KAN的独占理论优势。建议暂缓扩展更多KAN规模、更多LLM替换和性能宣传，先解决分布风险与可泛化特征之间的缺口。

可继续保留的研究问题是：在真实有界Q/K分布和极端权重下，如何可靠估计可泛化低秩风险，并构造控制假峰值的正特征。若能把风险分解、尾部稳定性或可解的受限最优问题做成理论，再在新模型、新文档分布上稳定验证，仍有论文空间。当前这一个模型、四个head及失败的构造还不足以支撑顶会级方法结论。

**核验与复现**

原始和变换后内积在约210万训练配对/head上最大误差小于1e−12；高斯正特征使用独立一维Gauss–Hermite积分核验；直接特征乘积与log计算一致。主实验完成19,456个feature矩阵rank下界检查，损失消融384个，另对新NMF做128个下界检查，均零违反。新NMF历史也未发现误差上升。所有矩阵指标采用Float64计算，没有裁剪目标kernel、丢失逐行缩放或添加会改变kernel的epsilon。

训练文件是experiment.py；主评估evaluate.py；β消融评估evaluate_beta15.py；数学核验verify.py；尾部归因diagnostics.py；汇总summarize.py。数据审计、配置、完整曲线和checkpoint均保留在本目录。图提供PNG，主要四曲线另有PDF。

使用已有环境复现：

```bash
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python kan_attention_theory/positive_generalization/experiment.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python kan_attention_theory/positive_generalization/evaluate.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python kan_attention_theory/positive_generalization/experiment.py --loss beta15
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python kan_attention_theory/positive_generalization/evaluate_beta15.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python kan_attention_theory/positive_generalization/diagnostics.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python kan_attention_theory/positive_generalization/verify.py
python kan_attention_theory/positive_generalization/summarize.py
python kan_attention_theory/positive_generalization/write_report.py
```

已有训练和主矩阵结果会被复用；重新从头运行时应先把本目录的结果和checkpoint另存，再使用空输出目录。数据集下载、Q/K抽取过程见上一阶段single_pass_mulkan；本轮没有训练或微调LLM，也未进行新的完整LLM困惑度或速度测试。



---

原文件：query_amplitude_ablation/PROTOCOL.zh.md

本轮在新模型训练与测试前固定，目标是检验移除查询幅度与整体标度后的正kernel。冻结Qwen2.5-1.5B，沿用层14/27全部24个Qheads、4096篇训练文档，数据与头范围不改变，不训练LLM权重，不使用子agent。

模型定义：h_minus(q,k)=exp(s_K(k))*softmax([z_Q(q),0])^T*softmax([z_K(k),0])，m64。没有查询幅度输出，没有可学习或固定的外部乘数C，也没有sqrt(m)特征倍率。Q网络128→192→63，K网络128→192→64（63方向+1幅度），隐藏SiLU。原训练head尺度c只用于将预测和目标共同除以exp(c)进行数值计算；raw_log_feature接口返回真正无C的kernel特征，必须核验尺度恢复恒等式，不把c乘回模型定义。

新模型包括两个预先固定的版本：
- reduced_plain：直接删除查询幅度输出行与head标度，73536参数/head。
- reduced_matched：将原query幅度的192参数回投为Q隐藏层的192个偏置，零初始化。KL版本73728参数/head。两个I版本另增加第一个Q方向logit的一个偏置（不是共同幅度），总73729，与原AD和HH原始损失版本严格同参。所有新增参数影响相对方向，不填充无效softmax冗余参数。额外偏置意味着这是参数再分配消融，纯删除版本用于区分两种效应。

Q/K权重从同种子原AD随机初始化规则生成，Q最后幅度行删除，K权重不改。不通过训练数据预拟合幅度来修改初始化。移除C也移除旧版m与训练尺度对应的外部倍率，因此初始化的raw kernel尺度会不同；必须报告，不把一遍训练比较当作充分优化后的函数类最优值。

训练共三种口径：product_i为先前跨文档乘积原始未平衡I；causal_i为先前因果配对balanced I；causal_kl为先前因果attention方向KL。每口径两个新结构、seed11搜索lr0.002/0.0005，仅以原有固定validation上对应目标选择；随后所选lr训练seed29/47。总24次单遍新拟合，最终18个选中检查点。沿用AdamW、wd1e-4、余弦末lr=lr/10、每head梯度clip10、相同文档顺序和选定pairs。

product_i沿用OFFSETS=[127,251,509,761,1019,1279,1531,1789,2053,2309,2557,2819,3067,3323,3581,3833]，每Q文档16个其他文档×64keys，共1024keys；4096×64×1024pairs/head。causal训练64Q/文档、全部合法prefixK，共134360517pairs/head。每个选定Q和pair只反向传播一次，向量会复用，不冒充独立样本。

父模型AD/HH复用同协议已训练的3种子检查点和评价结果，来源哈希另存；不改变父方法超参数，也不重拟合测试集。causal_kl额外展示已有HH-softmax对照。没有给任何方法增加LLM微调、窗口或旋转增强。

评价：confirm_wiki128篇1k、confirm_long24篇8k，真实因果kernel I/L2、方向KL、幅度误差、输出NMSE，以及完整冻结模型两层局部替换PPL。另评估A/B不相交文档的1024Q×4096K完整所选经验乘积，所有3种子。测试集合以前使用过，不称新盲测；经验乘积不是未知总体下界。纯KL未校准raw幅度的风险仅诊断，不作为raw kernel胜利主证据。

在评价前核验参数预算、原始尺度恢复、全部新参数的梯度连接、原AD取消query幅度/C后的attention恒等性。选中模型再做FP64显式矩阵/scan/递推一致性与实际8k FP32数值检查。PPL逐文档所有下一token位置，teacher复测作为环境核对。

计时统一RTX3090、FP32、batch1、单层12heads、CUDA graph热缓存、相同PyTorch参考状态算子。新模型使用无C的raw特征接口；原AD同时计时完整特征及推理中精确消去query幅度/C的实现。这样不把已可用于原AD的代数省略当作新训练结构独有的速度收益。与HH、FAVOR使用同一轮硬件和状态更新读取条件。计时不是完整模型decode，不与旧融合Triton数据混合。

若某种结构不稳定或泛化恶化，保留结果并进行FP64复核；不能通过测试选择scale、epoch或裁剪阈值。原始I/L2、方向/PPL、模型结构与推理实现分别下结论。



---

原文件：query_amplitude_ablation/REPORT.zh.md

本轮结果不支持在原始kernel的I-divergence训练中删除查询幅度/C：同参数新版本的乘积分布原始I风险比原AD高约74%–76%，两个方向均24/24 heads更差，与Hedgehog-exp没有明确I风险优势。KL训练下删除版与原AD的attention和PPL基本持平。原AD训练后直接在推理中消去查询幅度/C，既保留原有attention结果，也获得此次删除的计算收益；无需为此改用拟合更差的I训练结构。

所有新实验均冻结Qwen2.5-1.5B、两层24/336Qheads，4096训练文档、单遍、3种子。主配对参数量相同，另有少参数的纯删除控制；不是从头预训练或全层替换。

**实际新kernel**

$$\hat\kappa_-(q,k)=e^{s_K(k)}\operatorname{softmax}([z_Q(q),0])^\top\operatorname{softmax}([z_K(k),0]).$$

Q网络128→192→63，K网络128→192→64（63方向+1幅度），m64。没有s_Q、head标度参数、外部固定倍率或sqrt(m)。数值计算时预测和目标共同除以固定训练exp(c)，并在原始kernel评价恢复共同单位；c不是模型额外乘数。raw_log_feature返回未缩放的新kernel特征。

**参数与可比性**

I版本同参数73729/head：删除192个查询幅度权重和1个head标度后，增加192个Q隐藏偏置和1个方向logit偏置。KL版本同参数73728：增加192个Q隐藏偏置。新增参数均影响相对方向。纯删除两目标均73536参数。父AD/HH使用完全相同数据、单遍、3种子与每方法2个LR的既有检查点，原结果保留复用。原KL AD登记参数中192个查询幅度权重不被KL识别；原始I主比较没有这个失效分支问题。

新模型沿用同种子的Q/K权重初始化、移除Q幅度输出行；新增偏置零初始化。移除外部倍率改变初始raw尺度，尤其大kernel均值head；本轮没有额外预校准初始化。因此一遍训练效果不代表充分优化后的表达上限。

**乘积分布原始I训练：最直接对应原始kernel目标**

| 方法 | A→B原始I | B→A原始I | A→B原始L2 | B→A原始L2 |
| --- | --- | --- | --- | --- |
| 原AD | 0.27245 | 0.266388 | 0.365211 | 0.299519 |
| 去Q幅度/C，同参数 | 0.474177 | 0.469776 | 0.465984 | 0.429494 |
| 纯删除，少参数 | 0.491535 | 0.482388 | 0.494692 | 0.43942 |
| Hedgehog-exp | 0.46997 | 0.470631 | 0.370372 | 0.363239 |

I为E[d_I]/Eκ，L2为E[(κ−h)²]/Eκ²，按head先求比再汇总3种子/24heads。A/B为既有128篇留出1k文档分为不相交两组，每方向1024Q×4096K完整所选向量乘积，不是未知总体，也没有测试矩阵重拟合。

I训练使用d_I(x,y)=x log(x/y)−x+y，拟合对象是未归一化κ=exp(qᵀk/√128)。product_i不作逐查询质量平衡；causal_i按查询计算Σ_k d_I(κ,h)/Σ_k κ，再对查询取平均；causal_kl比较教师与预测kernel各自归一化后的因果attention方向。L2仅评估，不用于训练。三者的风险口径不能混为同一目标。

**乘积原始I训练后的因果迁移**

| 方法 | 1k方向KL | 8k方向KL | 1k输出NMSE | 8k输出NMSE | 1k PPL | 8k PPL |
| --- | --- | --- | --- | --- | --- | --- |
| 原AD | 0.802646 | 2.63221 | 0.446128 | 0.785691 | 9.02579 | 9.80391 |
| 去Q幅度/C，同参数 | 1.08181 | 2.67077 | 0.542807 | 0.745284 | 9.08771 | 9.86907 |
| 纯删除，少参数 | 1.06284 | 2.64931 | 0.518127 | 0.733888 | 9.07204 | 9.82497 |
| Hedgehog-exp | 1.24317 | 2.91787 | 0.655511 | 0.936142 | 9.0632 | 9.86306 |

**因果balanced I训练**

| 方法 | 1k方向KL | 8k方向KL | 1k输出NMSE | 8k输出NMSE | 1k PPL | 8k PPL |
| --- | --- | --- | --- | --- | --- | --- |
| 原AD | 0.430018 | 3.21871 | 0.261325 | 0.609218 | 8.90693 | 9.70089 |
| 去Q幅度/C，同参数 | 0.554452 | 3.14739 | 0.307714 | 0.68844 | 8.93389 | 9.77719 |
| 纯删除，少参数 | 0.561554 | 3.15071 | 0.313596 | 0.698594 | 8.93103 | 9.77585 |
| Hedgehog-exp | 1.46285 | 2.85763 | 0.752239 | 0.865445 | 9.1607 | 9.85453 |

**因果方向KL训练**

| 方法 | 1k方向KL | 8k方向KL | 1k输出NMSE | 8k输出NMSE | 1k PPL | 8k PPL |
| --- | --- | --- | --- | --- | --- | --- |
| 原AD | 0.347297 | 3.42322 | 0.220439 | 0.641392 | 8.85819 | 9.69226 |
| 去Q幅度/C，同参数 | 0.347538 | 3.46452 | 0.220502 | 0.639829 | 8.85846 | 9.68828 |
| 纯删除，少参数 | 0.347341 | 3.42287 | 0.220416 | 0.641975 | 8.85805 | 9.69161 |
| Hedgehog-exp | 0.377019 | 2.55174 | 0.231022 | 0.894505 | 8.87029 | 9.75236 |
| Hedgehog-softmax | 0.402539 | 2.57094 | 0.264257 | 0.745977 | 8.88021 | 9.75233 |

原模型PPL为8.664560/9.066279。所有PPL均完整冻结模型，仅两层局部替换；这些测试文档此前用过，不能称首次盲测。KL目标不识别完整raw标度，相关raw风险仅作诊断，不能拿它证明raw目标胜利。

新同参数KL版本相对原AD的两组ΔNLL文档bootstrap区间都跨0，不能宣称超越原AD；相对本轮Hedgehog-exp适配版则PPL和输出NMSE均较好。8k方向KL反而高于Hedgehog-exp，说明这里没有逐指标全面占优。HH用独立Q/K网络、无偏置、同参数扩宽到576正特征，是本研究的参数匹配适配版，不是论文默认配置或完整转换系统。

**I训练下的实际因果raw风险**

| 训练 | 方法 | 1k原始I | 8k原始I | 1k原始L2 | 8k原始L2 |
| --- | --- | --- | --- | --- | --- |
| product_i | 原AD | 2.02422 | 5.24689 | 0.822823 | 0.957457 |
| product_i | 去Q幅度/C，同参数 | 2.45905 | 8.45054 | 0.88713 | 984.771 |
| product_i | 纯删除，少参数 | 2.48675 | 8.03366 | 0.891817 | 686.264 |
| product_i | Hedgehog-exp | 2.37177 | 19.1412 | 0.859789 | 146.027 |
| causal_i | 原AD | 1.07359 | 1442.39 | 0.766658 | 1.65429e+08 |
| causal_i | 去Q幅度/C，同参数 | 1.36454 | 45449.6 | 0.847343 | 2.46854e+11 |
| causal_i | 纯删除，少参数 | 1.43213 | 74758.9 | 0.856449 | 1.24299e+11 |
| causal_i | Hedgehog-exp | 3.00708 | 8.71949 | 0.910333 | 13.0194 |

8k原始风险的异常值必须保留：乘积I新同参模型的原始L2约984.77，原AD为0.95746；因果balanced I两种AD也都有严重长上下文幅度失配，新版本更严重。独立CPU FP64重算最坏文档的SSE，相对误差分别7.96e−15和3.36e−14，确认不是缩放还原或GPU精度错误。一个乘积I错误pair的logκ为3.187、预测log h为20.131，单pair贡献该文档平方误差约24%。这说明当前模型会对长上下文中的少数pair严重高估，并不具备稳健的8k原始kernel泛化保证。详细定位见checks/raw_cases.json。

**主要配对区间：新同参版本减父方法，负值有利新版本**

| 训练 | 比较对象 | 集合 | ΔNLL/token | 配对文档95%区间 |
| --- | --- | --- | --- | --- |
| product_i | 原AD | confirm_wiki | 0.00683333 | [0.005588, 0.008186] |
| product_i | 原AD | confirm_long | 0.00661009 | [0.004295, 0.009422] |
| product_i | Hedgehog-exp | confirm_wiki | 0.00269695 | [0.001011, 0.004467] |
| product_i | Hedgehog-exp | confirm_long | 0.000594357 | [-0.002882, 0.004863] |
| causal_i | 原AD | confirm_wiki | 0.00302226 | [0.002162, 0.003977] |
| causal_i | 原AD | confirm_long | 0.00783403 | [0.005451, 0.010195] |
| causal_i | Hedgehog-exp | confirm_wiki | -0.0250706 | [-0.027143, -0.023077] |
| causal_i | Hedgehog-exp | confirm_long | -0.00787933 | [-0.010526, -0.005478] |
| causal_kl | 原AD | confirm_wiki | 3.0394e-05 | [-0.000303, 0.000362] |
| causal_kl | 原AD | confirm_long | -0.000409706 | [-0.000903, 9.1e-05] |
| causal_kl | Hedgehog-exp | confirm_wiki | -0.00133512 | [-0.002141, -0.000522] |
| causal_kl | Hedgehog-exp | confirm_long | -0.0065932 | [-0.009197, -0.004195] |

| 比较对象 | 方向 | 指标 | 差值 | 更好head数 | Q文档95%区间 |
| --- | --- | --- | --- | --- | --- |
| 原AD | ab | relative_raw_i | 0.201727 | 0/24 | [0.189295, 0.215951] |
| 原AD | ab | raw_nmse | 0.100773 | 4/24 | [0.028778, 0.157402] |
| 原AD | ba | relative_raw_i | 0.203388 | 0/24 | [0.190262, 0.218692] |
| 原AD | ba | raw_nmse | 0.129976 | 1/24 | [0.100795, 0.163286] |
| Hedgehog-exp | ab | relative_raw_i | 0.00420655 | 13/24 | [-0.019146, 0.026312] |
| Hedgehog-exp | ab | raw_nmse | 0.095612 | 7/24 | [0.069457, 0.157932] |
| Hedgehog-exp | ba | relative_raw_i | -0.000854614 | 14/24 | [-0.020921, 0.020756] |
| Hedgehog-exp | ba | raw_nmse | 0.0662557 | 6/24 | [0.041412, 0.10748] |

区间条件于三个已训练模型；乘积区间还固定测试K bank，只重采样Q文档。不覆盖未知PK、全部训练随机性或未观测尾部，多项探索比较未统一多重校正。

**本轮重新计时**

RTX3090、FP32、batch1、单层12heads、真实QKV、CUDA graph热缓存，统一PyTorch参考状态更新/读取，正序和逆序各一轮。原AD另有推理时直接消去query幅度/C的版本，仍使用原AD训练权重。

| 训练权重 | 实现 | m | 仅特征µs | 特征+状态µs | 状态MiB/层 |
| --- | --- | --- | --- | --- | --- |
| product_i | 纯删除，少参数 | 64 | 32.816 | 64.416 | 0.380859 |
| product_i | 去Q幅度/C，同参数 | 64 | 37.024 | 68.672 | 0.380859 |
| product_i | 原AD完整计算 | 64 | 42.912 | 74.56 | 0.380859 |
| product_i | Hedgehog-exp | 576 | 28.352 | 98 | 3.42773 |
| product_i | 原AD推理约掉Q幅度/C | 64 | 32.72 | 64.16 | 0.380859 |
| causal_kl | 纯删除，少参数 | 64 | 32.848 | 64.431 | 0.380859 |
| causal_kl | 去Q幅度/C，同参数 | 64 | 34.016 | 65.504 | 0.380859 |
| causal_kl | 原AD完整计算 | 64 | 42.912 | 74.56 | 0.380859 |
| causal_kl | Hedgehog-exp | 576 | 28.384 | 98.016 | 3.42773 |
| causal_kl | 原AD推理约掉Q幅度/C | 64 | 32.704 | 64.192 | 0.380859 |
| causal_kl | Hedgehog-softmax | 576 | 33.312 | 103.392 | 3.42773 |
| reference | FAVOR+ m64 | 64 | 27.904 | 59.264 | 0.380859 |

新同参I总耗时68.672µs，比原AD完整74.560µs少7.90%，但比原AD推理约消64.160µs多7.03%；新同参KL为65.504µs，比原AD推理约消64.192µs多2.04%。新同参比HH-exp总耗时少约30%–33%，主要受m64对m576的状态成本差异影响；只计算特征，HH-exp约28.4µs更快。新同参比FAVOR+ m64的59.264µs仍慢约10.5%–15.9%。同参并不等于同m或同FLOPs，当前未融合实现的launch/归约开销也影响结果。

标准主导乘加：原AD特征147456 FLOPs/head/token，纯删除及原AD推理约消为147072；同参新I加偏置约147265、KL约147264。m64状态更新读取约32768，HH m576约294912。非线性、归约、输入标准化、稳定重缩放未完整计入这些主导FLOPs，计时包含实际参考实现操作。删除分支只减少很小一部分计算；相同m64的状态大小不变。没有新测全模型decode/prefill速度，PPL评价耗时不作推理benchmark。

**机制与数值核对**

在相同权重上移除Q幅度和head倍率，归一化attention不变；FP64检查最大差5.55112e-17。纯删除与原AD在KL下共享参数的梯度也一致到FP64精度。因此KL纯删除是同一个attention函数的紧凑实现；重新训练时FP32舍入、Adam状态和学习率选择可造成小差异。补入Q方向偏置则改变了函数族。

原始I训练不同：原模型可直接用s_Q校准每个查询的幅度，新模型只能通过方向与K幅度共同调整。无Q幅度不等于每行质量相同：E_K h_minus=π_Q(q)^T E_K[e^{s_K(K)}π_K(K)]，仍随q改变；只是缺少独立查询缩放自由度。C能否被K网络吸收还取决于有限网络参数化，不能仅因符号删除就断言一般函数类容量减少。

本轮同时改变查询幅度、C、参数分配和raw初始化尺度，不能单独归因某一个因素。当前K网络无偏置，在标准化k=0处必有h_minus(q,μ_K)=1/64；已数值核验。μ_K不一定属于真实数据支持集，此恒等式只是实现约束，不是总体风险下界。未来若专门辨析C的影响，应另做保持初始raw质量可比的对照，不以本次测试结果调参。

18个新模型通过FP64矩阵/scan/递推核对；6个seed11配置的真实8k FP32最大相对输出误差为2.87361e-06，无零分母。总新拟合24次，选中18个，记录训练循环合计12.548分钟，不含数据载入和评价。

训练曲线按512篇文档分块，展示3种子均值和种子范围：[PNG](figures/training_curves.png)、[PDF](figures/training_curves.pdf)。I下原AD优势持续至单遍训练末尾；KL下AD与删除版曲线基本重合。单遍末尾仍有下降不构成充分优化或排除欠拟合的证据。

复现：proof_checks.py → train_run.py → driver.py → check_raw_cases.py → summarize_run.py → plot_training.py → write_report.py → audit_run.py。结果与父基线来源、参数预算、数据顺序、查询位置、训练尺度及文件哈希见checks/final_audit.json和results/manifest.json。研究结论应以本轮实际结果限定，不能声称总体近最优或完整Hedgehog系统复现。



---

原文件：real_llm_pilot/AUXILIARY_REPORT.zh.md

**辅助对照归档：归一化目标及早期 raw 实验（2026-09-05）。** 用户已明确主问题是直接拟合指数核。本文件保留早期辅助实验的完整结果，其中 row-target 结果不能用于判断用户方案是否成立。主要结论请看 `RAW_REPORT.zh.md`。本报告的数值来自本机冻结 Qwen2.5-1.5B 的实际实验，没有使用子 agent。

**当前证据支持分 head 研究，不支持“KAN 普遍优于现有线性 kernel”。** 在相同参数量和 2,000 步训练下，softplus-KAN 在两个被测中层 head 上优于 softplus-MLP，但首层、末层的四个 head 均落后。原始指数核拟合、去除 sink 后的谱、10,000 步训练和完整模型局部替换分别检验了目标尺度、谱结构、优化预算和实际预测质量。所有模型仅训练特征映射，LLM 权重冻结。

实验使用 [Qwen2.5-1.5B 官方权重](https://huggingface.co/Qwen/Qwen2.5-1.5B)，revision `8faed761d45a263340a0528343f099c05c9a4323`。Q/K 在投影和 RoPE **之后**截取，维度 128，缩放为 $1/\sqrt{128}$；使用正确的 GQA query-head → KV-head 映射。保存原始 BF16 张量。五个采样层的 Float32 手算注意力输出与原 SDPA 输出的相对 L2 差异为 0.094%–0.198%，符合不同精度计算的差异。

| 用途 | 文档数 | 每篇 token 数 | 来源 |
| --- | --- | --- | --- |
| 训练 | 64 | 1024 | WikiText 文档级 train |
| 验证 | 12 | 1024 | WikiText 文档级 validation |
| 测试 | 20 | 1024 | WikiText 文档级 test |
| 较长上下文测试 | 8 | 2048 | 另外的 WikiText test 文档 |
| 文体变化测试 | 8 | 1024 | 本地 LEval narrative QA 的正文 |

总计 112 篇、122,880 个 token 位置。文档文本哈希和 token 哈希均跨上述分组去重。模型预训练是否见过这些正文未知；“held-out”指没有进入本次特征映射训练。长文本组同时改变了文档和长度，不能把误差变化全部因果归于位置外推。验证时只用前 6 篇选择每个 head 的 checkpoint；全部 12 篇验证文档均有最终指标。

分布与谱覆盖层 0/7/14/21/27、每层 head 0/3/6/9，共 20 个 head；训练覆盖层 0/14/27 的 head 0/6，共 6 个 head。所有层、head 编号均从 0 开始。每个 head 独立训练 Q 和 K 两个网络，三随机种子为 11、29、47。

**真实分布的重要发现。** 以最后 512 个 query 和最前 512 个 key 形成完全合法的矩形注意力块，在 12 篇测试文档上做 SVD。raw kernel 先减去整个矩阵的最大 logit 再指数化，保留相对 Frobenius 误差；row kernel 在这个 512-key 块内逐行归一化。下表的 rank-64 floor 是平方奇异值尾和除以总能量。

| head | 高斯耦合最大奇异值 | raw floor | 归一化 floor | 删首 key 后归一化 floor |
| --- | --- | --- | --- | --- |
| L0-H0 | 0.579 | 3.65e-34 | 0.01111 | 0.01109 |
| L0-H6 | 4.051 | 8.51e-36 | 0.45522 | 0.45525 |
| L14-H0 | 0.824 | 0.000252 | 0.00055 | 0.16389 |
| L14-H6 | 0.838 | 0.000208 | 0.00049 | 0.06390 |
| L27-H0 | 0.610 | 0.00153 | 0.01153 | 0.01136 |
| L27-H6 | 0.624 | 0.00185 | 0.02357 | 0.02346 |

L14-H0 与 L14-H6 的第一 key 平均分别吸收该块约 89.9% 和 73.5% 的注意力质量。去掉第一 key 后，剩余注意力重新归一化，其谱尾大幅增加；这衡量剩余内容的条件拟合难度，并不等于原注意力矩阵的总体误差增长。删去前 4 或 16 个位置也给出相近现象，详见 `extra/sink_spectra.json`。

首层部分 head 的真实 logits 极大。例如采样因果 Q/K 对中，L0-H3 最大值约 24,245；L0-H0 平均值约 475。raw 平方误差因此可能几乎只奖励拟合少数最大元素，而归一化矩阵仍有明显谱尾。raw floor 在数值上接近 0 不能解释成精确有限秩，更不能推导注意力容易替换。

用训练样本均值和完整协方差拟合独立高斯时，20 个 head 中有 16 个不满足指数双线性核的 Hilbert–Schmidt 条件 $\|\beta\Sigma_q^{1/2}\Sigma_k^{1/2}\|_2<1/2$。此外，相同协方差并不能恢复真实归一化谱：L14-H0 的经验乘积分布 rank-64 floor 约 $1.63\times10^{-5}$，匹配高斯约 0.0321。这里的高斯只是替代分布，不能作为真实 Q/K 分布已被验证的依据。

**特征映射与拟合目标。** KAN 是两层三次 B-spline edge 网络，128→16→m，grid=8，带 SiLU base 分支；输入采用仅由训练集估计的逐通道均值/标准差。MLP 为 128→191→64（m=64 时），同样使用 SiLU，Q/K 网络彼此独立。每对 Q/K 网络分别有 73,888 与 73,854 个参数，差异小于 0.05%。KAN 的节点宽度较窄，固定 spline grid 不一定适配所有 head 的分布；这是一种具体实现的初验，不能排除其它 KAN 实现更好。

正值版本是 $f(q)=\operatorname{softplus}(F(q))$、$g(k)=\operatorname{softplus}(G(k))$，含等价的常数尺度；signed 版本使用 $1+F(q)$ 与 $1+G(k)$，输出不受符号限制。后者仍属于普通 KAN/MLP 输出函数类，常数平移只是初始化。优化为 AdamW，学习率 0.002 余弦降至 0.0002，每步选一篇训练文档、32 个因果 query、该文档全部 key，按 head 裁剪梯度。

训练了两个不同目标，不能混称：

1. **raw**：拟合 $\exp(q^\top k/\sqrt d-c_h)$，$c_h$ 是训练集核 RMS 对应的 head 常数；这是原始指数核的等价常数重缩放。训练损失为未加权的合法 Q/K 对平方误差。首层极端动态范围会造成数值退化，因此 raw 实验仅报告层 14/27 的四个 head。
2. **row**：拟合 $n_i\operatorname{softmax}(QK^\top/\sqrt d)_{ij}$，再用预测行和归一化。训练中按每个 minibatch/head 的目标能量归一化损失。这是注意力蒸馏的目标代表元，不是原始指数核的无权 L2 拟合；它还依赖文档与可见 key 集。

测试分别报告每篇文档的 kernel NMSE、attention NMSE 和 $AV$ 输出 NMSE，先在文档内计算能量比，再对文档取均值。表中“±”是三个随机种子各自测试均值的标准差，不是置信区间。全部因果指标在最后 512 个 query、所有合法前缀 key 上计算；SVD 对照则单独采用 first-512-key 矩形块，二者不直接相减。

**m=64、row 目标、2,000 步。** 数值越小越好。

| head | MLP attention | KAN attention | MLP output | KAN output |
| --- | --- | --- | --- | --- |
| L0-H0 | 0.3851 ± 0.0100 | 0.5473 ± 0.0177 | 0.0095 ± 0.0006 | 0.0168 ± 0.0005 |
| L0-H6 | 0.8833 ± 0.0009 | 0.9512 ± 0.0019 | 0.7265 ± 0.0021 | 0.8303 ± 0.0028 |
| L14-H0 | 0.2186 ± 0.0666 | 0.1543 ± 0.0028 | 0.8278 ± 0.1443 | 0.6977 ± 0.0026 |
| L14-H6 | 0.2415 ± 0.0029 | 0.1223 ± 0.0017 | 0.7264 ± 0.0017 | 0.5326 ± 0.0043 |
| L27-H0 | 0.5083 ± 0.0023 | 0.6141 ± 0.0051 | 0.1119 ± 0.0010 | 0.1442 ± 0.0017 |
| L27-H6 | 0.5105 ± 0.0016 | 0.6209 ± 0.0010 | 0.2212 ± 0.0022 | 0.2979 ± 0.0033 |

**m=64、row 目标、10,000 步。** 数值越小越好。

| head | MLP attention | KAN attention | MLP output | KAN output |
| --- | --- | --- | --- | --- |
| L0-H0 | 0.2293 ± 0.0143 | 0.3372 ± 0.0081 | 0.0061 ± 0.0005 | 0.0097 ± 0.0007 |
| L0-H6 | 0.7441 ± 0.0052 | 0.8728 ± 0.0025 | 0.6031 ± 0.0013 | 0.7597 ± 0.0028 |
| L14-H0 | 0.1759 ± 0.0124 | 0.1484 ± 0.0017 | 0.6985 ± 0.0276 | 0.6472 ± 0.0023 |
| L14-H6 | 0.1213 ± 0.0009 | 0.1041 ± 0.0006 | 0.5168 ± 0.0008 | 0.4880 ± 0.0021 |
| L27-H0 | 0.4563 ± 0.0036 | 0.5233 ± 0.0030 | 0.0990 ± 0.0007 | 0.1174 ± 0.0004 |
| L27-H6 | 0.4545 ± 0.0013 | 0.5269 ± 0.0040 | 0.1850 ± 0.0024 | 0.2319 ± 0.0015 |

10,000 步实验延长了优化并重新设置余弦学习率计划，验证集仍负责选择 checkpoint；它检查训练预算的敏感性，不构成“已找到各网络最优解”的证明。相同参数量/训练步数不等于相同 FLOPs 或训练时间。

**直接拟合 raw 指数核，2,000 步。**

| head | MLP kernel | KAN kernel | MLP attention | KAN attention |
| --- | --- | --- | --- | --- |
| L14-H0 | 0.9145 ± 0.0074 | 0.9208 ± 0.0040 | 0.2309 ± 0.0100 | 0.2671 ± 0.0549 |
| L14-H6 | 0.8163 ± 0.0014 | 0.8144 ± 0.0025 | 0.2103 ± 0.0136 | 0.2582 ± 0.0587 |
| L27-H0 | 0.9491 ± 0.0351 | 0.8978 ± 0.0027 | 0.9423 ± 0.0151 | 0.8615 ± 0.0183 |
| L27-H6 | 0.9198 ± 0.0125 | 0.8939 ± 0.0132 | 0.9020 ± 0.0189 | 0.8716 ± 0.0154 |

这些 raw 结果没有显示已经接近 rank-m 理论极限，也没有显示普遍优于 row 目标。需要区分原始尺度、训练优化、网络表达、正值限制和共享映射的跨文档泛化，不能把全部误差归因于 m 太小。

**维度扫描，2,000 步。** 每个 m 内匹配 KAN 与 MLP 参数量；跨 m 参数量变化，因此不是纯粹固定参数预算实验。

| head | MLP m=32/64/128 | KAN m=32/64/128 |
| --- | --- | --- |
| L0-H0 | 0.379 / 0.385 / 0.362 | 0.559 / 0.547 / 0.542 |
| L0-H6 | 0.889 / 0.883 / 0.888 | 0.951 / 0.951 / 0.955 |
| L14-H0 | 0.266 / 0.219 / 0.208 | 0.155 / 0.154 / 0.155 |
| L14-H6 | 0.360 / 0.241 / 0.177 | 0.122 / 0.122 / 0.121 |
| L27-H0 | 0.521 / 0.508 / 0.502 | 0.618 / 0.614 / 0.623 |
| L27-H6 | 0.526 / 0.510 / 0.500 | 0.615 / 0.621 / 0.620 |

KAN 在这个训练预算下随 m 增大的收益很小，提示不能单靠扩大输出维度解决当前瓶颈；可能的原因包括较窄的中间层、固定 grid、优化和分布变化，需要后续消融区分。

**稳定性与基线。** signed 特征出现了负核条目和非正分母。row-KAN 在 L0-H6 的 kernel NMSE 约 0.943，但归一化 attention NMSE 约 46.75；raw-signed 中若干 head 更严重。因此，“去掉 softplus 后不受非负秩限制”不能自动转化为更好的线性注意力。用 per-entry ReLU 修补 signed 核通常会破坏可分离低秩结构，不是免费的修复。

| 基线，m=64（如适用） | L0-H0 | L0-H6 | L14-H0 | L14-H6 | L27-H0 | L27-H6 |
| --- | --- | --- | --- | --- | --- | --- |
| uniform | 0.964 | 0.999 | 0.998 | 0.997 | 0.933 | 0.912 |
| first_token | 27.773 | 2.003 | 0.397 | 0.794 | 50.362 | 65.821 |
| positive_orthogonal_random_features | 24.566 | 1.880 | 0.402 | 0.661 | 7.589 | 10.028 |
| Hedgehog-style | 0.678 | 0.951 | 0.568 | 0.710 | 0.640 | 0.620 |

Hedgehog-style 使用拼接 softmax(Wx) 与 softmax(-Wx) 的可学习特征，只有 8,256 个参数/对，未匹配 KAN 参数量，也没有复现原论文完整训练配方；它是特征形式对照。正交随机特征使用同一个 Gaussian ORF 投影构造 Q/K 指数特征，在 log-space 稳定求值，未训练且没有额外温度调参，是公式基线而非优化后的 FAVOR+ 系统。对应原始方法见 [Hedgehog](https://arxiv.org/abs/2402.04347) 与 [Performer](https://arxiv.org/abs/2009.14794)。本实验不能声称击败这些论文的完整结果。

**较长上下文与文体变化。** 下表为 2,000 步正值模型的 attention NMSE。

| head | MLP 2k-token | KAN 2k-token | MLP narrative | KAN narrative |
| --- | --- | --- | --- | --- |
| L0-H0 | 0.958 | 0.962 | 0.537 | 0.673 |
| L0-H6 | 1.001 | 1.004 | 0.897 | 0.952 |
| L14-H0 | 0.967 | 0.521 | 0.205 | 0.145 |
| L14-H6 | 0.847 | 0.428 | 0.266 | 0.147 |
| L27-H0 | 0.880 | 0.913 | 0.535 | 0.623 |
| L27-H6 | 0.896 | 0.987 | 0.529 | 0.618 |

三种种子均未参与这些测试集上的选参。较长上下文误差仍然很大，目前没有长上下文质量保持的证据。

**完整 LLM 的局部替换。** 将 2,000 步得到的正值网络插回 Qwen，自回归因果状态使用 $S_t=\sum_{j\le t}g(k_j)v_j^\top$、$z_t=\sum_{j\le t}g(k_j)$，输出 $f(q_t)^\top S_t/[f(q_t)^\top z_t]$。Q/K/V 与后续层的隐藏状态随替换重新计算。累加实现与稠密特征核计算的相对 L2 误差为 $9.78\times10^{-7}$。

使用同一批 12 篇测试文档；原模型完整 1023 个预测位置的 PPL 为 **9.0480**，最后 512 个预测位置为 **8.7535**。这是本报告指定的小样本评估协议，不是标准全量 WikiText PPL。

| 替换范围，row 目标 | MLP 全位置 PPL | KAN 全位置 PPL | MLP 后512 PPL | KAN 后512 PPL |
| --- | --- | --- | --- | --- |
| middle | 9.1276 | 9.1012 | 8.8477 | 8.8141 |
| late | 9.0703 | 9.0794 | 8.7823 | 8.7955 |
| four_heads | 9.1531 | 9.1334 | 8.8797 | 8.8593 |
| six_heads | 9.8242 | 9.7503 | 9.3983 | 9.5050 |

middle/late 各替换 2 个 head；four_heads 替换中层和末层共 4 个；six_heads 再加首层 2 个。总模型有 336 个 query head，最多只替换其中 6 个。所有 query 位置都替换，包括未在训练 query 区间内的早期位置。完整位置和后半位置的结果可能排序不同，不能挑其中一种宣称普遍优越。

替换实验仍计算其余 head 的原始 SDPA，用于质量验证；没有测得生产级端到端加速。不能从少数 head 的小幅 PPL 差异推断整个模型能够无损线性化。

**计算成本。** RTX 3090，eager PyTorch Float32；每次特征计时包含 6 个独立 head 的 512 个 Q 与 512 个 K，不包括 attention scan。

| 方法 | 参数/每对 QK 网络 | 2,000 步训练秒数 | 特征映射毫秒 |
| --- | --- | --- | --- |
| mlp_positive | 73854 | 10.05 | 0.438 |
| kan_positive | 73888 | 22.71 | 2.855 |
| hedgehog | 8256 | 8.45 | 0.369 |

这些计时不是融合 kernel、BF16 推理或端到端吞吐基准。KAN 即使在局部 head 有误差优势，也必须面对特征计算开销；独立的 per-query-head K 映射还没有利用 GQA 共享来优化状态与计算。

**可以建立在真实模型上的数学框架。** 以下为推导，而非从拟合曲线猜测结论。

对 Qwen 这类 attention 前使用 RMSNorm 的冻结模型，令隐藏维度为 $D$，$z=\gamma\odot h/\sqrt{\|h\|_2^2/D+\epsilon}$，则 $\|z\|_2\le\sqrt D\|\gamma\|_\infty$。标准 RoPE 是正交变换，因此每个 head 满足 $\|q\|_2\le R_q:=\|W_q\|_2\sqrt D\|\gamma\|_\infty+\|b_q\|_2$，K 同理。于是 $\kappa(q,k)^2\le\exp(2\beta R_qR_k)$，真实乘积分布上的核必属于 $L^2(P_q\times P_k)$。常数可能很松且极大，但存在性成立。高斯替代模型发生发散，不意味着真实模型的算子发散。此结论针对这里核实的 RMSNorm 与标准 RoPE 架构，不能无条件推广到任意 Q/K 生成方式。

在明确指定的真实边缘乘积分布上，定义 $Tf(q)=\int\kappa(q,k)f(k)\,dP_k(k)$。Schmidt 分解 $\kappa=\sum_{r\ge1}\sigma_r u_rv_r$ 存在，并有 $E_m=\inf_{f_r,g_r}\mathbb E[(\kappa-\sum_{r=1}^m f_rg_r)^2]=\sum_{r>m}\sigma_r^2$。离散经验分布有权重时，应分解 $\operatorname{diag}(\sqrt w_q)K\operatorname{diag}(\sqrt w_k)$；等权情形即普通 SVD 的相应常数缩放。

实际训练的 Q/K 来自同一文档，且受因果位置约束，通常不是整个数据池中独立抽取的 $P_q\times P_k$。因此必须区分：跨文档乘积分布理论、条件于文档的算子，以及真实因果配对损失。任意非可分离配对权重下，普通 SVD 尾和不再自动等于全局最优风险。本实验额外计算经验乘积分布的谱，同时只对完全合法的矩形块使用无条件正确的矩阵秩下界。

对每篇文档矩形块 $A_C$，有 $\|A_C-\widehat A_C\|_F^2\ge\sum_{r>m}\sigma_r(A_C)^2$。对文档平均后仍给出共享特征映射的下界，但它允许每篇文档单独选择 SVD 因子，可能很松。逐行归一化不增加一个已拟合矩形低秩核的秩；因果三角 mask 则可能增加整个矩阵的秩，所以不能直接把 masked attention 的 SVD 当作线性状态维度下界。

正值特征约束给出 $E_m\le E_m^+\le\inf_{\theta_q,\theta_k}R(\operatorname{softplus}(\mathrm{KAN}_{\theta_q}),\operatorname{softplus}(\mathrm{KAN}_{\theta_k}))$。$E_m^+$ 是非负可分离逼近的 infimum，不由普通 SVD 完全决定。此次 signed 训练的结果既不是 $E_m$ 的估计，也不是非负约束代价的严格测量：网络容量、训练优化和归一化稳定性同时变化了。

KAN 的实例化应针对最优因子 $F_*=(\sqrt{\sigma_r}u_r)_{r\le m}$、$G_*=(\sqrt{\sigma_r}v_r)_{r\le m}$ 的可逼近性。若对应 $L^2$ 向量误差分别为 $\epsilon_q,\epsilon_k$，则一个可用的保守界是 $R(\widehat F,\widehat G)^{1/2}\le\sqrt{E_m}+\sqrt{\sigma_1}(\epsilon_q+\epsilon_k)+\epsilon_q\epsilon_k$。若这些奇异函数具有平滑、低宽度的可组合结构，KAN spline 逼近可控制后面两项；必须明确覆盖的输入域、平滑度、宽度、网格和尾部能量。不能仅引用 Kolmogorov–Arnold 表示定理就声称避免维数灾难。

对本次固定网格 KAN，还不能实证断言某个 spline 渐近阶数；目前只扫描了输出维度，没有扫描 grid 并验证最优奇异函数的可组合性。MLP、可学习随机特征与其它自适应方法也可能逼近相同因子。任何固定 m 方法都受同一无约束谱尾下界限制，KAN 的合理优势只能是特定因子类中的参数效率、优化或泛化，而不是突破这个极限。

真实数据提示更值得推进的对象是：保留小规模 sink/局部精确分支，对剩余内容构建按层和 head 的条件算子谱，再学习满足输出稳定性的特征。这样的结构本身不能算 KAN 独有创新；应证明并验证 KAN 在哪些实际观测到的奇异函数结构上更省参数或更易训练。

**顶会潜力的当前判断。** “两个 KAN 拟合 exp(QK) + 常规 SVD 下界”目前不足以形成有说服力的顶会主张。本实验没有发现全面效果优势，而且发现了 sink 对谱估计的影响、raw 目标失衡、signed 分母不稳和较长上下文误差。它们可以帮助形成更扎实的问题定义，但 attention sink、可学习特征映射和谱截断本身都有既有研究，不能包装成新发现。

若继续投入，优先做三项有判别力的工作：第一，在第二个模型家族及更多 head 上验证谱与 KAN 相对收益的可预测关系；第二，采用合法的正值分解或显式稳定约束，配合 sink 分支与 attention-output 蒸馏，并把 MLP/原始学习特征方法调到相近计算预算；第三，证明一个可在真实 Q/K 上检验的结构性假设对应的 KAN 逼近优势，同时给出全模型、长上下文任务和融合实现的质量—成本曲线。得到这些证据后再评估顶会竞争力，比现在预设 KAN 会更好可靠。

**复现与完整性。** 环境为 `/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python`，PyTorch 2.13.0+cu130、Transformers 5.16.1；RTX 3090。绘图脚本使用 `/root/miniconda3/bin/python` 的 NumPy/Matplotlib。所有随机种子、训练曲线、各文档指标、选中 checkpoint 与模型 revision 均已保存。

已完成 45 次 feature-pair 配置/种子训练；每次含 4 或 6 个彼此独立 head。已检查 2952 个“同一矩形块学生误差 ≥ SVD 下界”的比较，违反数为 0（容差 $10^{-5}$）。文档哈希和 token 哈希去重均通过。相关比较不构成人口分布上的有限样本置信证书。

复现入口与产物：`extract_qkv.py` 保存真实 QKV；`analyze_distribution.py` 分析分布与谱；`fit_features.py` 训练及评估；`extra_diagnostics.py` 去 sink 与基线；`model_replacement.py` 做完整模型局部替换；`run_followups.py` 顺序运行后续 GPU 实验；`summarize_results.py` 与 `write_report.py` 生成汇总。`data_qwen25_1p5b/manifest.json` 记录全部文档来源和哈希，`results/metrics.csv` 是完整对比表，`results/summary.json` 包括按文档配对 bootstrap（条件于三个已训练种子，探索性未做多重比较修正）。

![误差与谱概览](results/pilot_overview.png)

![维度扫描](results/dimension_sweep.png)

![模型局部替换](results/model_replacement.png)



---

原文件：real_llm_pilot/RAW_REPORT.zh.md

**主报告：在真实 Qwen2.5-1.5B 的 Q/K 上直接拟合指数核。** 本次主实验严格采用用户指定的目标：两个特征网络的内积拟合 $e^{q^\top k}$，归一化只交给 Linear Attention 的分母。此前 row-normalized-target 实验归档在 `AUXILIARY_REPORT.zh.md`，不作为本方案成立与否的证据。

**本轮没有验证出 KAN 的普遍优势。** 普通采样 10,000 步后，正值 KAN 在部分 head 的 raw 测试误差低于参数匹配的 MLP，但四个 head 的误差仍约 0.83–0.99。重要性采样改善了部分训练风险，却没有带来一致的测试改善。m 扫描、原始核 SVD、完整模型局部替换和数值精度对照都已完成；这些结果支持继续研究分布和泛化瓶颈，尚不足以支持优于现有论文的主张。

实际模型的注意力 logit 为 $q^\top k/\sqrt{128}$。等价地定义 $\widetilde q=128^{-1/4}q,\widetilde k=128^{-1/4}k$，本实验拟合的正是 $\kappa=e^{\widetilde q^\top\widetilde k}$。若把未缩放的原始 Q/K 直接代入 $e^{q^\top k}$，那会是不同于该 LLM 的温度，本报告没有将两者混淆。

数值上训练 $\kappa_c=e^{q^\top k/\sqrt d-c_h}$，其中 $c_h$ 是整个 head 共用、由训练集固定的常数，不依赖 query、key、文档或行和。最终函数为 $\widehat\kappa=e^{c_h}f(q)^\top g(k)$，所以损失只是原始核平方误差乘以固定常数；Linear Attention 的分子和分母都含 $e^{c_h}$，它会精确抵消。训练采用原始核平方误差；报告中的相对能量比仅用于评估和选择 checkpoint。

**数据与实现。** 模型为 [官方 Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)，revision `8faed761d45a263340a0528343f099c05c9a4323`，冻结全部 LLM 权重。截取真实投影后、RoPE 后的 BF16 Q/K/V，使用正确 GQA 映射。覆盖 20 个 head 的分布；主训练选择 L14-H0、L14-H6、L27-H0、L27-H6 四个 head，编号从 0 开始。首层部分 head 的原始 logit 超过 2 万，普通浮点数下直接 raw-L2 训练会严重退化，本轮未把首层纳入主训练，也不声称解决了这个问题。

| 分组 | 文档 | 长度 | 用途 |
| --- | --- | --- | --- |
| WikiText train | 64 | 1024 | 仅特征网络训练 |
| WikiText validation | 12 | 1024 | 前6篇选择 checkpoint |
| WikiText test | 20 | 1024 | 独立测试 |
| 另选 WikiText test | 8 | 2048 | 较长上下文 |
| LEval narrative 正文 | 8 | 1024 | 文体变化 |

共 112 篇、122,880 token 位置，文本与 token 哈希跨分组去重。这是本次特征训练的 held-out 划分，不保证 LLM 预训练从未见过正文。主风险使用真实同文档因果 Q/K 对；高斯和跨文档独立乘积分布只用于理论诊断。

KAN：两层三次 B-spline edge 网络，128→16→m，grid=8，带 SiLU base 分支。MLP 使用 SiLU，m=64 时为 128→191→64。两支网络彼此独立；Q/K 输入采用训练集估计的逐通道均值/标准差，目标仍由原始 Q/K 计算。正值版本在输出使用 softplus，signed 版本不限制符号。m=64 时，每对网络参数分别为 KAN 73,888、MLP 73,854。固定网格、较窄中间层和单一学习率均属于此初验的限制。

**评估口径。** 主指标 $\mathrm{NMSE}_{raw}=\sum_{C,i,j}(\widehat\kappa_{Cij}-\kappa_{Cij})^2/\sum_{C,i,j}\kappa_{Cij}^2$，所有纳入评估的文档共用同一个 $c_h$。先汇总能量再取比值；逐文档比值的平均只作为辅助列。测试 query 是最后 512 个位置，key 为全部合法前缀。验证也按 pooled raw 平方误差选择 checkpoint。训练不用 Linear Attention 分母；attention、$AV$ 和完整模型 PPL 仅在训练完成后计算。表中 ± 为三个种子的标准差。
最终离线表格统一用 Float64 计算已训练特征与真实指数核，Linear Attention 直接除以实际行和，未加 epsilon 或截断分母。参数不变；checkpoint 仍由原来的 Float32 训练验证选出。早期含小分母保护的离线评估另存于 `legacy_float32_evaluation_backup/`，本报告不使用那些 attention/AV 数值。完整 LLM 的 Float32 与 Float64 替换试验分别报告。

**普通采样，2,000 步：原始核拟合误差。**

| head | MLP pooled raw NMSE | KAN pooled raw NMSE | MLP逐文档均值 | KAN逐文档均值 |
| --- | --- | --- | --- | --- |
| L14-H0 | 0.9448 ± 0.0036 | 0.9500 ± 0.0021 | 0.9149 | 0.9206 |
| L14-H6 | 0.9958 ± 0.0002 | 0.9961 ± 0.0002 | 0.8170 | 0.8252 |
| L27-H0 | 0.9458 ± 0.0385 | 0.9011 ± 0.0110 | 1.0447 | 0.9101 |
| L27-H6 | 0.8826 ± 0.0020 | 0.8662 ± 0.0203 | 1.0199 | 0.9201 |

**普通采样，10,000 步：原始核拟合误差。**

| head | MLP pooled raw NMSE | KAN pooled raw NMSE | MLP逐文档均值 | KAN逐文档均值 |
| --- | --- | --- | --- | --- |
| L14-H0 | 0.9450 ± 0.0022 | 0.9531 ± 0.0061 | 0.9113 | 0.9229 |
| L14-H6 | 0.9966 ± 0.0034 | 0.9942 ± 0.0022 | 0.8841 | 0.8430 |
| L27-H0 | 0.9791 ± 0.0274 | 0.9592 ± 0.0358 | 1.1172 | 0.9413 |
| L27-H6 | 0.8819 ± 0.0320 | 0.8296 ± 0.0035 | 0.9593 | 1.0126 |

**重要性采样，10,000 步：原始核拟合误差。**

| head | MLP pooled raw NMSE | KAN pooled raw NMSE | MLP逐文档均值 | KAN逐文档均值 |
| --- | --- | --- | --- | --- |
| L14-H0 | 0.9602 ± 0.0039 | 0.9596 ± 0.0049 | 0.9337 | 0.9363 |
| L14-H6 | 1.0286 ± 0.0195 | 0.9987 ± 0.0017 | 3.0965 | 0.9184 |
| L27-H0 | 1.0760 ± 0.1099 | 0.9856 ± 0.0289 | 1.0615 | 0.9128 |
| L27-H6 | 0.8506 ± 0.0066 | 0.8462 ± 0.0377 | 0.9184 | 0.9068 |

重要性采样没有更改拟合函数或目标风险。它以训练核平方能量与均匀分布的 50:50 混合分布抽取文档，再用类似的条件分布抽取 query；每个被抽 query 仍与该文档全部合法 key 配对，并乘以目标分布/提议分布的精确比值。四个 head 各自采样。普通采样使用同一篇随机文档和均匀 query，重要性版本另用完整训练对计算 head 常数，数值条件有所不同。两者都优化 raw L2；不能把所有变化纯因果归于某一个实现细节。

设 $\pi_h(C,i)=\pi_h(C)\pi_h(i\mid C)$，每文档合法训练对数为 $N_p$，文档数为 $D$，则采样损失为 $\frac1B\sum_{b=1}^B\frac{\sum_{j\le i_b}(\widehat\kappa_c-\kappa_c)^2}{D N_p\pi_h(C_b,i_b)}$，其期望等于原始合法对的均匀平方误差。采样实现已校验常数被积函数的精确期望，以及核平方能量的 Monte Carlo 期望；具体误差保存于每次训练 JSON 的 `sampler_verification`。

**训练集误差检查。** 用同一套最终 checkpoint，在训练文档最后 512 个 query 上重新评估；它帮助区分优化/表示困难和跨文档泛化，不能单独区分网络表达与优化。

| head | 普通 MLP | 普通 KAN | 重要性 MLP | 重要性 KAN |
| --- | --- | --- | --- | --- |
| L14-H0 | 0.9051 | 0.8931 | 0.8528 | 0.8552 |
| L14-H6 | 0.9632 | 0.9485 | 0.2202 | 0.1965 |
| L27-H0 | 0.5401 | 0.6272 | 0.4159 | 0.2870 |
| L27-H6 | 0.6442 | 0.7740 | 0.2298 | 0.4439 |

**m=32/64/128 的原始核实验，2,000 步。**

| head | MLP 32/64/128 | KAN 32/64/128 |
| --- | --- | --- |
| L14-H0 | 0.9444 / 0.9448 / 0.9438 | 0.9497 / 0.9500 / 0.9501 |
| L14-H6 | 0.9959 / 0.9958 / 0.9960 | 0.9960 / 0.9961 / 0.9959 |
| L27-H0 | 0.9429 / 0.9458 / 0.9219 | 0.8997 / 0.9011 / 0.9009 |
| L27-H6 | 0.8787 / 0.8826 / 0.8834 | 0.8686 / 0.8662 / 0.8656 |

每个 m 内参数匹配，跨 m 参数数量变化。不能把这个小规模扫描解释为已经测到 KAN 的渐近最优逼近率。

**同一原始核目标的基线。** 常数为训练集核均值；正交正值随机特征包含 Q 和 K 两侧的完整范数项，m=64，三种子。mean-centered 版本使用训练均值中心化，并精确补回两个可分离边缘指数因子，仍估计同一个原始核，没有更改目标。

| 方法 | L14-H0 | L14-H6 | L27-H0 | L27-H6 |
| --- | --- | --- | --- | --- |
| train_mean_constant | 0.9996 | 0.9999 | 0.9983 | 0.9987 |
| positive_orthogonal_random_features | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| mean_centered_positive_ORF | 1.0202 | 3.3858 | 4.3295 | 409.7412 |

随机特征只做公式级原始核对照，并非复现某篇论文的完整调优系统。三次投影不能准确估计高方差随机特征的总体风险。

**原始核的 SVD 下界。** 对完全合法的 512×512 矩形块做 SVD；这里只使用 raw 指数矩阵，归一化谱不作为替代。下表的 oracle 和学生误差使用相同 12 篇测试文档、相同块和相同 head 常数，按原始能量加权汇总。oracle 允许每篇文档单独挑选自由 rank-64 因子，学生需要共享 KAN/MLP 函数，因此 oracle 下界可能很松。

| head | 经验自由 rank-64 下界 | 重要性 MLP块误差 | 重要性 KAN块误差 |
| --- | --- | --- | --- |
| L14-H0 | 0.000334 | 0.669629 | 0.700718 |
| L14-H6 | 0.000005 | 1.004463 | 0.998453 |
| L27-H0 | 0.000164 | 0.852336 | 0.831469 |
| L27-H6 | 0.000070 | 0.919933 | 0.901773 |

已检查 1728 个对应的 raw 矩形块误差与 SVD floor，违反数为 0（容差 1e-5）。这只是经验矩阵下界，不是无限总体分布最小误差的置信证书，也没有证明训练后的网络达到了其类内最优。

**拟合完成后，才使用 Linear Attention 分母。** 下表采用 10,000 步重要性采样的 raw 模型，attention 与 output NMSE 为逐文档能量比的平均。

| head | MLP attention | KAN attention | MLP AV | KAN AV |
| --- | --- | --- | --- | --- |
| L14-H0 | 0.2766 | 0.2393 | 0.9558 | 0.8817 |
| L14-H6 | 0.6749 | 0.2431 | 1.7033 | 0.7773 |
| L27-H0 | 1.2367 | 0.8788 | 0.3368 | 0.2703 |
| L27-H6 | 2.7000 | 0.8706 | 0.7989 | 0.4733 |

**KAN 输出是否使用 softplus，2,000 步 raw 目标。** signed 版本放宽了非负约束，却可能发生核条目为负、分母接近零或非正。以下均用严格分母的 Float64 离线评估。

| head | signed KAN raw误差 | softplus KAN raw误差 | signed KAN attention误差 | softplus KAN attention误差 |
| --- | --- | --- | --- | --- |
| L14-H0 | 0.95171 | 0.94997 | 15221 | 0.24957 |
| L14-H6 | 0.99636 | 0.99609 | 17952 | 0.23298 |
| L27-H0 | 0.90784 | 0.90112 | 3506.8 | 0.86131 |
| L27-H6 | 0.89234 | 0.86619 | 585 | 0.87225 |

raw L2 好坏与归一化后的稳定性是两个不同问题。负核条目、非正分母比例等完整指标见 `raw_results/metrics.csv`。

**完整模型局部替换（replacement_raw_longer，特征精度 float32）。** 替换 336 个 query head 中的 4 个，其余保持原始注意力，冻结 LLM，不做微调。12 篇指定测试文档的原模型全位置 PPL 为 9.0480，后 512 预测位置 PPL 为 8.7535。

| 方法 | 全位置 PPL均值±种子标准差 | 后512 PPL均值±种子标准差 |
| --- | --- | --- |
| mlp_positive | 仅 1/3 种子完成；不汇总比较 | 仅 1/3 种子完成；不汇总比较 |
| kan_positive | 9.1580 ± 0.0018 | 8.9021 ± 0.0024 |

以下配置在 Float32 正值特征累加中出现非正分母，已记录为失败；表中仅汇总三个种子都完成的情况：raw_mlp_positive_s29_four_heads：Nonpositive denominator in positive feature scan; zero_query_feature_rows=2; zero_key_feature_rows=3；raw_mlp_positive_s47_four_heads：Nonpositive denominator in positive feature scan; zero_query_feature_rows=0; zero_key_feature_rows=0。

**完整模型局部替换（replacement_raw_importance，特征精度 float32）。** 替换 336 个 query head 中的 4 个，其余保持原始注意力，冻结 LLM，不做微调。12 篇指定测试文档的原模型全位置 PPL 为 9.0480，后 512 预测位置 PPL 为 8.7535。

| 方法 | 全位置 PPL均值±种子标准差 | 后512 PPL均值±种子标准差 |
| --- | --- | --- |
| mlp_positive | 仅 0/3 种子完成；不汇总比较 | 仅 0/3 种子完成；不汇总比较 |
| kan_positive | 9.1634 ± 0.0045 | 8.9060 ± 0.0056 |

以下配置在 Float32 正值特征累加中出现非正分母，已记录为失败；表中仅汇总三个种子都完成的情况：raw_mlp_positive_s11_four_heads：Nonpositive denominator in positive feature scan; zero_query_feature_rows=0; zero_key_feature_rows=1；raw_mlp_positive_s29_four_heads：Nonpositive denominator in positive feature scan; zero_query_feature_rows=1; zero_key_feature_rows=0；raw_mlp_positive_s47_four_heads：Nonpositive denominator in positive feature scan; zero_query_feature_rows=1; zero_key_feature_rows=2。

**完整模型局部替换（replacement_raw_longer_fp64，特征精度 float64）。** 替换 336 个 query head 中的 4 个，其余保持原始注意力，冻结 LLM，不做微调。12 篇指定测试文档的原模型全位置 PPL 为 9.0480，后 512 预测位置 PPL 为 8.7535。

| 方法 | 全位置 PPL均值±种子标准差 | 后512 PPL均值±种子标准差 |
| --- | --- | --- |
| mlp_positive | 9.1872 ± 0.0145 | 8.9351 ± 0.0065 |
| kan_positive | 9.1574 ± 0.0013 | 8.8979 ± 0.0036 |

**完整模型局部替换（replacement_raw_importance_fp64，特征精度 float64）。** 替换 336 个 query head 中的 4 个，其余保持原始注意力，冻结 LLM，不做微调。12 篇指定测试文档的原模型全位置 PPL 为 9.0480，后 512 预测位置 PPL 为 8.7535。

| 方法 | 全位置 PPL均值±种子标准差 | 后512 PPL均值±种子标准差 |
| --- | --- | --- |
| mlp_positive | 9.1919 ± 0.0085 | 8.9499 ± 0.0056 |
| kan_positive | 9.1628 ± 0.0048 | 8.9043 ± 0.0044 |

累加状态的 Linear Attention 与稠密特征核输出已做数值对照；替换后重新计算后续隐藏状态。这里不是标准全量 WikiText PPL，也不是全模型线性化，更不是端到端速度测试。
Float64 复核只提高已训练特征网络和累加状态的计算精度，Qwen 主体仍为 BF16，没有重训或改变目标函数。它用于确认 Float32 下溢，不能当作高效推理方案。

**泛化与成本。**

| head | MLP 2048-token raw误差 | KAN 2048-token raw误差 | MLP narrative raw误差 | KAN narrative raw误差 |
| --- | --- | --- | --- | --- |
| L14-H0 | 1.7030 | 1.0377 | 0.9777 | 0.9766 |
| L14-H6 | 6.8814 | 1.0010 | 1.0352 | 0.9602 |
| L27-H0 | 1.0695 | 0.9985 | 0.9398 | 0.9073 |
| L27-H6 | 0.9642 | 1.0523 | 0.8675 | 1.0089 |

| 方法 | 参数/每对网络 | 普通10k训练秒 | 重要性10k训练秒 | 512 Q+512 K特征毫秒（4个head） |
| --- | --- | --- | --- | --- |
| mlp_positive | 73854 | 49.69 | 54.45 | 0.435 |
| kan_positive | 73888 | 120.99 | 115.93 | 2.079 |

RTX 3090、eager PyTorch Float32、未融合。相同参数量与训练步数不等于相同计算预算。较长测试组还更换了文档，不能把变化完全归因于长度。

**理论应如何接到真实分布。**

Qwen attention 前的 RMSNorm 给出 $\|z\|_2\le\sqrt D\|\gamma\|_\infty$，标准 RoPE 保持范数。因此 $\|q\|_2\le R_q:=\|W_q\|_2\sqrt D\|\gamma\|_\infty+\|b_q\|_2$，K 同理，$\kappa^2\le e^{2\beta R_qR_k}$。对该真实冻结模型的边缘乘积分布，Hilbert–Schmidt 算子与 Schmidt 分解确实存在，虽然这个范数上界可能极松。匹配高斯的 20 个 head 中 16 个违反指数核可积性条件，只说明高斯替代不合适，不能推断真实算子发散。

对明确指定的 $P_q\times P_k$，$\kappa=\sum_r\sigma_r u_rv_r$，自由 m 维双特征的最小平方误差为 $E_m=\sum_{r>m}\sigma_r^2$。softplus 特征再受非负可分离因子的限制，故 $E_m\le E_m^+\le\inf_{\mathrm{KAN}}R$。真实同文档因果 Q/K 配对通常不是这个全局乘积分布，不能直接套用同一 SVD 最优式；本报告使用每文档合法矩形块的下界，并把经验乘积分布谱单列在分布结果中。

KAN 的理论任务是逼近 $F_*=(\sqrt{\sigma_r}u_r)_{r\le m}$ 与 $G_*=(\sqrt{\sigma_r}v_r)_{r\le m}$。若两者的 $L^2$ 向量逼近误差为 $\epsilon_q,\epsilon_k$，则 $\sqrt{R}\le\sqrt{E_m}+\sqrt{\sigma_1}(\epsilon_q+\epsilon_k)+\epsilon_q\epsilon_k$。要进一步得到 spline 的收敛阶，必须证明这些奇异函数在真实支持域上有适合该 KAN 宽度/深度的平滑组合结构，并控制尾部能量、正值约束和优化误差。通用表示定理不能单独给出维数无关优势。
上述直接逼近 SVD 因子的界适用于允许符号的特征。softplus-KAN 不能直接逼近带负值的奇异函数；应先选取非负可分解近似，再分析它的非负因子。若通过 softplus 逆函数转换因子，还需要控制因子接近 0 时的正则性和常数，不能把 signed 的逼近率原样搬过去。本次没有求出 $E_m^+$，因此尚未分离出纯粹的非负约束代价。

对于原始指数核，均值、范数和极端 Q/K 对决定了 raw 能量的分布；只匹配协方差或归一化后的谱不够。可以进一步研究精确的可分离边缘因子、倾斜分布与重要性采样，这些必须保持原始风险的定义。它们也不是 KAN 独有优势，仍需与调优的 MLP、可学习特征及正值随机特征比较。

**课题评价。** 当前实验只验证了一种有限宽度固定网格 KAN 在一个真实模型四个 head 上的行为，尚不能支持“效果会比现有研究更好”。常规 SVD 下界加替换网络也不足以形成顶会新颖性。值得继续的主张应是：在可由真实 Q/K 检验的分布/奇异函数结构条件下，KAN 能以更小的参数或计算预算逼近原始指数核，并在稳定的 Linear Attention 分母下保留 LLM 质量。若优势只来自更长训练或改进采样，它就属于优化方法的贡献，不能直接归为 KAN 的理论优势。

后续最有判别力的实验是扩展第二个模型家族和更多 head，扫描 KAN 网格/宽度与 MLP 预算，测量训练与测试 raw 风险及尾部能量，并进行更多 head 的真实替换。已有归一化目标实验仅留作辅助，不用于替代这些证据。没有把本次小样本实验称为对 [Hedgehog](https://arxiv.org/abs/2402.04347) 或 [Performer](https://arxiv.org/abs/2009.14794) 整套方法的复现或超越。

**复现。** 全流程入口 `reproduce_raw.py`；主训练入口 `run_raw_followups.py`，重要性采样及审计入口 `continue_raw_pipeline.py`，单次训练入口 `fit_features.py --objective raw`；`raw_importance.py` 实现风险保持的采样器，`exact_raw_evaluation.py` 重算严格分母的 Float64 离线指标，`audit_raw.py` 验证增添其它 key 不改变既有 raw 目标、常数缩放可恢复原始指数核，并评估训练风险。`raw_results/summary.json`、`metrics.csv` 保留完整数值、按文档配对 bootstrap 与 SVD 检查。bootstrap 条件于三个已训练种子，为探索性分析，未做多重比较修正。模型、原始 QKV、所有 checkpoint、曲线和文档哈希都保留在本目录。

本主报告汇总 36 次配置/种子训练，每次同时训练四个独立 head。运行环境为 `/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python`（torch 2.13.0+cu130、transformers 5.16.1）；绘图使用 `/root/miniconda3/bin/python summarize_raw.py`。

![原始核误差](raw_results/raw_kernel_overview.png)

![维度与分母评估](raw_results/raw_dimension_and_attention.png)



---

原文件：single_pass_mulkan/REPORT.zh.md

这轮实验已完成两层 KAN、两层 mulKAN 与两层 MLP 的严格等参数、单遍训练比较。结果没有支持“加入 KAN 或乘法就会全面超过 MLP”。数据增多有帮助，损失函数会改变结构排序；在更直接控制原始 kernel 质量的 I-divergence 对照中，应以表中的误差及完整模型替换结果判断。

**实验对象与公平性**

冻结 Qwen2.5-1.5B，使用真实投影、RoPE 后的 Q/K，head dimension=128，正确按 GQA 将 query head 映射到 KV head。比较第 14/27 层的 0/6 号 head。目标始终是

$$K(q,k)=\exp(q^\top k/\sqrt{128}),\qquad \widehat K=e^{c_h}\sum_{r=1}^{64}f_r(q)g_r(k),$$

其中两支网络独立，$f,g$ 为 softplus 正特征；$1/\sqrt{128}$ 是该 LLM 原有 attention scale，可吸收到 q、k 中。$c_h$ 是只用训练集计算的每 head 公共常数，所有 query 共用，未做逐行标签归一化。特征的固定缩放可同样吸收到网络中，不改变表达式的可分离结构。本轮按 softplus 正特征方案比较，未重复 signed-feature 方案。

| 模型 | 单支结构（小预算） | Q/K 两支合计有效参数 | 双倍预算 |
|---|---|---|---|
| MLP | 128→192→64，中间 SiLU | 73,856 | 147,584 |
| KAN | 128→16→64，两层 cubic B-spline+SiLU base | 73,856 | 147,584 |
| mulKAN | 128→(9加法+3二元乘法)→(32加法+32二元乘法) | 73,856 | 147,584 |

mulKAN 第一层先产生 15 个子节点，再组合为 12 个隐藏节点；第二层产生 96 个子节点，再组合为 64 个输出。两层都有乘法。每条 spline edge 含 11 个三次 B-spline 系数及 1 个 SiLU base 权重。三种结构统一无中间 bias、最终每支有 64 个 bias，所有计入的参数都参加前向计算，没有补零或闲置参数。双倍预算将 MLP/KAN 隐藏宽度加倍，mulKAN 第一层加法/乘法节点数加倍，最终 m=64 保持不变。

实现遵循 [MultKAN 原论文](https://arxiv.org/abs/2408.10205)和[官方 MultKAN 代码](https://github.com/KindXiaoming/pykan/blob/master/kan/MultKAN.py)的子节点乘积机制，是用于该实验的固定网格版本，没有引入 symbolic 分支。等参数并不等 FLOPs、吞吐或优化难度。

**数据量与“每条数据只训练一次”**

训练集为 4,096 篇文章的各 1,024 token 前缀，共 4,194,304 token，比旧 64 篇实验扩大 64 倍。前 64 篇保留旧训练文章，新增 4,032 篇；另有 128 篇验证、256 篇测试文章。新划分来自 [WikiText document-level 数据集](https://huggingface.co/datasets/EleutherAI/wikitext_document_level)的 WikiText-103 TRAIN split，是内部留出集，**不是官方 WikiText 测试集**。所有文档的文本和 token 哈希均去重；官方验证/测试及旧评估文档不进入新训练样本。

每篇取后半段 512 个 Q，与前半段 512 个 K 的固定随机排列一一配对，每个 Q、K 各使用一次，全部满足 causal 条件。因此每个 head 共 2,097,152 个唯一训练对，而非遍历每篇全部 512² 个组合。每篇只做一次梯度更新，每次运行只有 1 epoch，没有 replay、早停或测试集选 checkpoint。校准均值/方差、计算公共幅度及训练集误差会额外读取数据，但不做重复梯度训练。不同随机种子是独立重复实验，单遍约束逐次运行成立。

每一组使用同样的数据对和文档顺序，种子为 11/29/47；AdamW，lr 0.002→0.0002，weight decay=1e-4，按 head 梯度裁剪，使用最终 checkpoint。共 45 个配置运行，每次并行训练 4 个相互独立的 head 特征映射。KAN 网格固定为 8 段，没有按模型单独调学习率、初始化或网格；这些选择可能影响排序。

**更换后的损失：始终拟合未归一化 kernel**

令 $y=e^{q^\top k/\sqrt{128}-c_h}$，$\hat y=\sum_r f_r(q)g_r(k)>0$。程序通过 logsumexp 稳定计算 $\log\hat y$；不是先让特征内积拟合 logits 再在外面取 exp，后一种做法一般不再是线性 attention kernel。

1. 乘性误差 log-cosh：$L_{lc}=\log\cosh(\log\hat y-\log y)$。对 log-ratio 的梯度为 tanh，有界，避免原始平方误差被极大 kernel 值主导；代价是普通样本的相对误差改善，未必充分改善极少数大质量样本。
2. 未归一化广义 KL / I-divergence：$L_I=\hat y-y+y\log(y/\hat y)$。代码名为 `poisson`，仅借用其损失形式，不假设 kernel 是整数计数。训练省略只依赖 y 的常数，因此日志中的训练 loss 可以为负；评估报告的是恢复完整常数后的非负 divergence。

本轮**没有用原始 kernel 的 MSE 做训练**。NMSE 仅作为评估量，以保留和 L2/SVD 理论之间的联系。公共缩放 $e^{c_h}$ 不改变 log-cosh；对 I-divergence 只产生每 head 的固定权重，四个 head 的网络参数互不共享。

I-divergence 适合本问题的关键理由可以直接推导。对一行 kernel，记 $Z=\sum_jK_j$、$\hat Z=\sum_j\hat K_j$，$p=K/Z$、$\hat p=\hat K/\hat Z$，则

$$D_I(K\Vert\hat K)=Z\,D_{KL}(p\Vert\hat p)+Z\log(Z/\hat Z)-Z+\hat Z.$$

它同时惩罚行质量失配与归一化 attention 形状失配，但训练标签仍是原始 K。质量误差项 $Z\log(Z/\hat Z)-Z+\hat Z$ 非负，所以 $D_{KL}(p\Vert\hat p)\le D_I(K\Vert\hat K)/Z$；若每个 value 的范数不超过 V，则由 Pinsker 得 $\|\hat o-o\|\le V\sqrt{2D_I(K\Vert\hat K)/Z}$。这是逐行界；当前随机配对损失估计的是文档中合法矩形块的平均 entry risk，并不自动保证每一行都小。

**主要结果：4,096 篇、小参数预算**

| 训练损失 | 结构 | 测试 log-cosh↓ | 原始 kernel NMSE↓ | I-divergence/目标质量↓ | attention 输出 NMSE↓ |
|---|---|---|---|---|---|
| logcosh | MLP | 0.5858 ± 0.0033 | 0.9892 ± 0.0009 | 3.8381 ± 0.0762 | 0.5868 ± 0.1113 |
| logcosh | KAN | 0.8360 ± 0.0021 | 0.9861 ± 0.0007 | 4.3835 ± 0.0414 | 0.4821 ± 0.0010 |
| logcosh | mulKAN | 0.7794 ± 0.0137 | 0.9458 ± 0.0207 | 3.8151 ± 0.1332 | 0.5205 ± 0.0172 |
| 原始 I-divergence | MLP | 1.8450 ± 0.0409 | 0.8837 ± 0.0038 | 1.7227 ± 0.0197 | 0.3961 ± 0.0050 |
| 原始 I-divergence | KAN | 2.1746 ± 0.1064 | 0.8920 ± 0.0010 | 1.8355 ± 0.0146 | 0.4371 ± 0.0063 |
| 原始 I-divergence | mulKAN | 2.1453 ± 0.1354 | 0.8845 ± 0.0020 | 1.8028 ± 0.0191 | 0.4196 ± 0.0035 |

表中先在每个 head 内聚合，再对固定 4 个 head 求算术平均，± 为 3 个 seed 的样本标准差，不是置信区间。前三个指标使用 256 篇测试文档的 131,072 个唯一配对样本/head；输出 NMSE 使用前 32 篇测试文档的完整合法 512×512 矩形块。原始 kernel NMSE 为 pooled SSE/真实 kernel 平方和，I-divergence 除以真实 kernel 质量；输出 NMSE 先逐文档计算再平均。不同量的数值不可直接互相比较。

log-cosh 下 MLP 更好地拟合多数样本的乘性误差；KAN/mulKAN 对原始平方风险或输出误差的表现不完全沿用这一排序。I-divergence 在三种结构上都改变了这种权衡。原始 kernel NMSE 仍然很高，不能把改进描述为已经准确替代 softmax kernel。

![损失对照](results/loss_comparison.png)

**数据是否不足、是否过拟合**

| 训练文档数 | 结构 | 训练 log-cosh | 测试 log-cosh | 测试−训练 | 输出 NMSE |
|---|---|---|---|---|---|
| 64 | MLP | 1.2608 | 1.2790 | 0.0182 | 2.4917 |
| 64 | KAN | 1.5027 | 1.5108 | 0.0080 | 2.6557 |
| 64 | mulKAN | 1.4676 | 1.4764 | 0.0089 | 2.6557 |
| 512 | MLP | 0.9271 | 0.9363 | 0.0092 | 1.9677 |
| 512 | KAN | 1.0359 | 1.0362 | 0.0003 | 2.2486 |
| 512 | mulKAN | 1.0020 | 1.0040 | 0.0019 | 1.3082 |
| 4096 | MLP | 0.5760 | 0.5858 | 0.0098 | 0.5868 |
| 4096 | KAN | 0.8337 | 0.8360 | 0.0023 | 0.4821 |
| 4096 | mulKAN | 0.7755 | 0.7794 | 0.0039 | 0.5205 |

以 log-cosh 评价时，训练和测试误差都随规模增加而下降，差距较小，未出现“大幅压低训练误差、测试误差很差”的典型整体过拟合图景。这说明不能把问题简单归因于反复训练少量数据，数据覆盖、优化步数及结构/目标偏差仍需区分。

但是，对大值更敏感的风险仍有训练—测试差距。下面是 4,096 篇、I-divergence 训练的结果：

| 结构 | 训练 raw NMSE | 测试 raw NMSE | 训练 I-div/质量 | 测试 I-div/质量 |
|---|---|---|---|---|
| MLP | 0.8495 | 0.8837 | 1.2333 | 1.7227 |
| KAN | 0.8743 | 0.8920 | 1.4271 | 1.8355 |
| mulKAN | 0.8021 | 0.8845 | 1.3397 | 1.8028 |

mulKAN 的训练 raw NMSE 比 MLP 低，但测试时这一优势没有保留；这提示需要关注尾部泛化。该差距可能同时反映稀有样本的拟合偏差与有限测试集的尾部波动，不能仅凭小的平均 log-cosh gap 宣布没有过拟合，也不能仅凭两个集合的加权风险之差证明其原因一定是过拟合。

必须保留的混杂因素是：训练文档增加时，optimizer steps 也从 64 增至 4,096，学习率退火随总步数变化，训练集校准统计也变化。因此本实验回答的是“更多新数据、单遍训练是否有益”，没有分离纯数据量效应与优化预算效应；单遍训练既不保证不发生过拟合，也不保证已经收敛。旧 MSE 实验的采样与训练步数不同，不能与本表作只更换损失的严格消融。

![数据规模](results/data_scaling.png)

**增大参数预算**

| Q/K 参数 | 结构 | 测试 log-cosh↓ | 原始 kernel NMSE↓ | 输出 NMSE↓ | 512对特征计算 ms↓ |
|---|---|---|---|---|---|
| 73856 | MLP | 0.5858 ± 0.0033 | 0.9892 ± 0.0009 | 0.5868 ± 0.1113 | 0.6457 |
| 73856 | KAN | 0.8360 ± 0.0021 | 0.9861 ± 0.0007 | 0.4821 ± 0.0010 | 2.2708 |
| 73856 | mulKAN | 0.7794 ± 0.0137 | 0.9458 ± 0.0207 | 0.5205 ± 0.0172 | 2.5452 |
| 147584 | MLP | 0.5348 ± 0.0044 | 0.9855 ± 0.0027 | 0.4390 ± 0.0432 | 0.6526 |
| 147584 | KAN | 0.7698 ± 0.0030 | 0.9773 ± 0.0029 | 0.4767 ± 0.0184 | 2.3135 |
| 147584 | mulKAN | 0.6992 ± 0.0121 | 0.9350 ± 0.0123 | 0.4964 ± 0.0051 | 2.5171 |

这里全部使用 log-cosh、4,096 篇、1 epoch，m 始终为 64。时间为 RTX 3090 上一次同时处理 4 个 head、每 head 512 对输入的 eager 特征网络+logsumexp 实测；不包括完整线性 attention，也不是 fused kernel 或端到端加速比较。参数匹配后，当前 spline 实现仍有显著计算开销。

**真实 kernel 的尾部与 SVD 对照**

| head (layer,index) | 最大 logit | 最大0.1%样本的质量占比 | 最大0.1%样本的平方能量占比 |
|---|---|---|---|
| [14, 0] | 6.00 | 58.97% | 90.74% |
| [14, 6] | 7.17 | 62.47% | 99.69% |
| [27, 0] | 12.72 | 34.30% | 98.56% |
| [27, 6] | 14.64 | 33.58% | 97.49% |

这里的分位数只描述本次真实 Q/K 样本，不假定高斯，也不将经验大值集中等同于严格意义的幂律分布。高平方能量集中解释了为什么 log-cosh、raw NMSE 和归一化后的输出会给出不同评价，以及为什么有限测试样本的 raw NMSE 会敏感。

| head | signed rank-64 SVD相对下界 | I-div MLP实际NMSE | I-div KAN实际NMSE | I-div mulKAN实际NMSE |
|---|---|---|---|---|
| [14, 0] | 2.954e-03 | 0.7595 | 0.7683 | 0.7629 |
| [14, 6] | 9.787e-04 | 0.9892 | 0.9900 | 0.9891 |
| [27, 0] | 2.676e-04 | 0.8684 | 0.8869 | 0.8287 |
| [27, 6] | 3.399e-04 | 0.8706 | 0.8822 | 0.8526 |

上述下界与实际误差使用同一组 32 个 512×512 测试块，按真实 kernel 平方能量聚合。SVD 可使用该测试块本身任意构造有符号因子，是逐块 oracle；我们的网络必须从训练数据学习、在不同文档共享函数，并具有非负因子。因此接近零的 oracle 下界不说明当前正特征模型能够达到它，也不能把实际误差与该下界之差全部归因于优化失败。它更不是全局 $P_q\otimes P_k$ 的总体最优风险证明。

对任意 m 维可分离表示，矩形预测矩阵的秩仍然至多为 m。mulKAN 的乘法发生在单侧特征网络内部，可以更直接构造坐标交互，却不会突破该秩限制。I-divergence 和 log-cosh 改变了优化目标，Schmidt 奇异值尾和只给 L2 评估的最优秩下界，不能原封不动称为这两个新损失的最小误差。要证明 mulKAN 特有优势，需要针对真实分布的主导特征函数给出相同参数或计算预算下更好的逼近率，再控制采样和优化误差；通用逼近定理本身不够。

**完整 LLM 的四个 head 替换验证**

| 损失 | 模型 | 困惑度 ± seed SD | 相对原模型 ΔNLL | 后512位置困惑度 |
|---|---|---|---|---|
| 原始模型 | — | 8.2371 | 0.0000 | 7.5847 |
| logcosh | MLP | 8.2858 ± 0.0048 | 0.00590 | 7.6425 |
| logcosh | KAN | 8.2854 ± 0.0025 | 0.00585 | 7.6438 |
| logcosh | mulKAN | 8.2821 ± 0.0013 | 0.00546 | 7.6421 |
| poisson | MLP | 8.2769 ± 0.0014 | 0.00483 | 7.6384 |
| poisson | KAN | 8.2837 ± 0.0011 | 0.00565 | 7.6433 |
| poisson | mulKAN | 8.2829 ± 0.0013 | 0.00555 | 7.6429 |

这是 32 篇内部测试文章、每篇 1024 token 的局部替换质量实验，只替换第 14/27 层的 0/6 号 head，共 4/336 个 query head；其余保持原模型。不微调，后续隐藏状态和 Q/K/V 重新计算。所有 causal 位置均替换，包含训练未覆盖的前半段 query 和后半段 key，因此包含位置分布外推。困惑度不能当作官方 WikiText benchmark，也不能据此推断全模型线性化后的结果。

使用真实 causal prefix-sum 公式；特征网络及累加用 Float64。逐特征的正对角重标定与逐 query 的公共缩放在分子分母精确抵消，不剪裁分母、不改变训练目标。为保留其他 head，原始 SDPA 仍会计算，因此此实验只评估质量，不评估加速。

**按 head 的结果**

| 损失 | head | 结构 | 原始 kernel NMSE | I-divergence/质量 | 输出 NMSE |
|---|---|---|---|---|---|
| logcosh | [14, 0] | MLP | 0.9905 | 5.2019 | 0.9534 |
| logcosh | [14, 0] | KAN | 0.9686 | 5.4323 | 0.8718 |
| logcosh | [14, 0] | mulKAN | 0.8472 | 4.5068 | 0.9555 |
| logcosh | [14, 6] | MLP | 0.9997 | 6.5139 | 0.9641 |
| logcosh | [14, 6] | KAN | 0.9979 | 7.4394 | 0.4736 |
| logcosh | [14, 6] | mulKAN | 0.9955 | 7.3955 | 0.6294 |
| logcosh | [27, 0] | MLP | 0.9891 | 1.9112 | 0.1739 |
| logcosh | [27, 0] | KAN | 0.9921 | 2.3760 | 0.2229 |
| logcosh | [27, 0] | mulKAN | 0.9821 | 1.7331 | 0.1779 |
| logcosh | [27, 6] | MLP | 0.9774 | 1.7254 | 0.2556 |
| logcosh | [27, 6] | KAN | 0.9859 | 2.2862 | 0.3601 |
| logcosh | [27, 6] | mulKAN | 0.9586 | 1.6249 | 0.3194 |
| poisson | [14, 0] | MLP | 0.7622 | 1.5642 | 0.6731 |
| poisson | [14, 0] | KAN | 0.7672 | 1.7317 | 0.6890 |
| poisson | [14, 0] | mulKAN | 0.7613 | 1.6268 | 0.6469 |
| poisson | [14, 6] | MLP | 0.9928 | 3.2446 | 0.3544 |
| poisson | [14, 6] | KAN | 0.9929 | 3.3175 | 0.3799 |
| poisson | [14, 6] | mulKAN | 0.9931 | 3.3677 | 0.3688 |
| poisson | [27, 0] | MLP | 0.8962 | 1.0436 | 0.2147 |
| poisson | [27, 0] | KAN | 0.9212 | 1.1709 | 0.2586 |
| poisson | [27, 0] | mulKAN | 0.9014 | 1.1119 | 0.2520 |
| poisson | [27, 6] | MLP | 0.8835 | 1.0381 | 0.3423 |
| poisson | [27, 6] | KAN | 0.8868 | 1.1219 | 0.4211 |
| poisson | [27, 6] | mulKAN | 0.8820 | 1.1049 | 0.4106 |

**配对不确定性检查**

| 损失 | 结构−MLP | 指标 | 差值 | 探索性95%区间 |
|---|---|---|---|---|
| logcosh | KAN | logcosh | 0.2502 | [0.2439, 0.2572] |
| logcosh | KAN | raw_nmse | -0.0031 | [-0.0319, 0.0040] |
| logcosh | KAN | normalized_idiv | 0.5454 | [0.1324, 0.8595] |
| logcosh | KAN | output_nmse | -0.1047 | [-0.2247, -0.0058] |
| logcosh | mulKAN | logcosh | 0.1937 | [0.1762, 0.2084] |
| logcosh | mulKAN | raw_nmse | -0.0433 | [-0.1331, -0.0225] |
| logcosh | mulKAN | normalized_idiv | -0.0230 | [-0.5916, 0.4433] |
| logcosh | mulKAN | output_nmse | -0.0662 | [-0.1850, 0.0405] |
| poisson | KAN | logcosh | 0.3296 | [0.1960, 0.4145] |
| poisson | KAN | raw_nmse | 0.0084 | [-0.0069, 0.0172] |
| poisson | KAN | normalized_idiv | 0.1129 | [0.0648, 0.1691] |
| poisson | KAN | output_nmse | 0.0410 | [0.0328, 0.0492] |
| poisson | mulKAN | logcosh | 0.3003 | [0.1354, 0.3961] |
| poisson | mulKAN | raw_nmse | 0.0008 | [-0.0263, 0.0140] |
| poisson | mulKAN | normalized_idiv | 0.0802 | [0.0127, 0.1449] |
| poisson | mulKAN | output_nmse | 0.0234 | [0.0143, 0.0327] |

负差值表示该结构误差更低。按整篇文档联合重采样，再对共享的 3 个 seed 重采样，2,000 次；head 固定。这保留文档内部相关性，但 seed 只有 3 个、测试语料单一，区间只能作探索性判断，不代表跨模型、跨领域显著性。

**验证与复现**

数据唯一性、每篇 512 个 key 的完整无重复排列、每个 run 的单 epoch/步数、跨结构顺序哈希、45 个 checkpoint 的有效参数量均已核验。Float64 小块检查验证原始 kernel 的缩放恢复、分母归一化不变性和 I-divergence 逐行分解；完整替换额外检查 causal scan 与 dense masked feature kernel 相等。数值结果记录在 `audit.json` 与 `replacement.json`。

主要文件：`extract_large.py`（真实 Q/K 提取）、`models.py`（两层网络及损失）、`train.py`（45 个单遍实验）、`validate.py`（审计）、`distribution.py`（尾部与 SVD）、`replacement.py`（完整模型替换）、`summarize.py`（报告与图）。`fits/` 保存每个模型的 checkpoint、逐 seed/head 结果及逐测试文档指标，`results/metrics.csv` 和 `results/summary.json` 可直接分析。

模型 revision：`8faed761d45a263340a0528343f099c05c9a4323`；数据 revision：`647234772b9554e208af6c826f23b99e3cac88c8`。运行环境 torch 2.13.0+cu130、transformers 5.16.1、RTX 3090。提取脚本依赖已下载的数据 parquet 和前轮 64 篇训练缓存；manifest 记录文档来源和哈希。

```bash
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python kan_attention_theory/single_pass_mulkan/models.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python kan_attention_theory/single_pass_mulkan/train.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python kan_attention_theory/single_pass_mulkan/validate.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python kan_attention_theory/single_pass_mulkan/distribution.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python kan_attention_theory/single_pass_mulkan/replacement.py
/root/miniconda3/bin/python kan_attention_theory/single_pass_mulkan/summarize.py
```

`train.py` 遇到已有结果会跳过；要从头重训使用 `--out` 指定新的目录，并在后处理时改为读取相应路径。

**课题判断**

这组数据支持继续研究“真实 Q/K 分布上的正可分离 kernel 学习及目标函数选择”，目前没有支持“mulKAN 在等预算下必然更好”的强结论。仅凭把 MLP 换成 KAN/mulKAN，顶会论证仍不足。更有价值的下一阶段是：明确正秩/总体算子与局部 SVD oracle 的差别，为未归一化 I-divergence 提供分布加权理论；在多个 LLM、更多 head、全 causal 位置和跨长度数据上验证；进行训练算力匹配及公平超参数搜索，并比较真实端到端质量—速度曲线。当前结果应作为可复现的初步验证和反证约束，不能提前宣称优于现有最强线性注意力研究。



---

原文件：single_pass_mulkan/diagnostics/REPORT.zh.md

有训练尚不充分的证据，但现有日志不能把优化不足与表达能力不足完全分开。另一方面，“0.1% 贡献大部分平方能量”揭示的是目标风险的极端权重集中，不等于 attention 概率集中，也不能直接推断幂律或无穷方差。本次只读取已有数据和 checkpoint 做诊断，没有追加梯度训练或变更原实验。

**末段损失是否仍在下降**

必须比较同一个集合上的风险。旧日志的 `loss` 是当前一篇文档的一次 loss，不是固定训练集或移动平均；I-divergence 训练还省略了随目标变化的常数，跨文档的负数变化更不能作为收敛证据。下面使用全程固定的前 32 篇验证文档，模型均为 N=4096、小参数预算，先平均固定四个 head，再平均三个 seed：

| 训练目标 | 模型 | 2048步 | 4096步 | 下降比例 |
|---|---|---|---|---|
| logcosh | MLP | 0.6468 | 0.5880 | 9.1% |
| logcosh | KAN | 0.9053 | 0.8446 | 6.7% |
| logcosh | mulKAN | 0.8435 | 0.7855 | 6.9% |
| poisson | MLP | 1.2137 | 1.0887 | 10.3% |
| poisson | KAN | 1.2892 | 1.2236 | 5.1% |
| poisson | mulKAN | 1.2547 | 1.1657 | 7.1% |

后半程仍有 5.1%—10.3% 改善，不能把当前 checkpoint 当作已达到该结构的最佳逼近性能。准确说法是：存在优化尚未充分完成的迹象；已有稀疏记录没有证明最终几十步还在下降，也没有证明延长训练一定继续改善。学习率已按 4096 步从 0.002 退火至 0.0002，曲线变缓可能包含学习率效应。

不同 head 并不一致。例如 KAN 的 (14,0) 验证 I-divergence/质量从 0.8031 升至 0.8198，而其他 head 带动整体下降。两倍参数的 log-cosh 实验也降低了训练和测试风险：MLP 测试 0.5858→0.5348，KAN 0.8360→0.7698，mulKAN 0.7794→0.6992。这说明容量或参数化可能有影响，但优化轨迹也改变了，不能单独据此识别纯容量瓶颈。

当前结论只能是固定数据、步数、学习率、参数预算下的比较，不能写成充分收敛后的结构上限比较。训练不足与尾部泛化差距可以同时存在；前轮 mulKAN 的 I-divergence 训练 raw NMSE=0.8021、测试=0.8845，不因整体曲线仍下降就消失。

![末段验证曲线](late_convergence.png)

**0.1% 平方能量的精确定义**

每个 head 在 256 篇内部测试文档上有 N=131,072 个配对样本。将 K 从大到小排序，取最大的 ceil(0.001N)=132 个，记集合 S。所报数字是

$$C_2=\frac{\sum_{i\in S}K_i^2}{\sum_{i=1}^{N}K_i^2}.$$

“能量”只是平方和的线性代数术语。C2 不是概率质量，不是模型误差，不表示 99.9% 的 token 没有作用。与之并列的 C1=Σ_SK/ΣK 是整个样本池中的未归一化 kernel 质量占比，也不等于逐 query 归一化后再平均的 attention 概率。

| head | 最大0.1%的平方和占比 C2 | 同一集合的质量占比 C1 | 单个最大样本的平方和占比 | 平方权重集中度数量 |
|---|---|---|---|---|
| [14, 0] | 90.74% | 58.97% | 33.69% | 6.49 |
| [14, 6] | 99.69% | 62.47% | 62.84% | 2.15 |
| [27, 0] | 98.56% | 34.30% | 53.93% | 3.01 |
| [27, 6] | 97.49% | 33.58% | 46.63% | 4.11 |

最后一列是 (ΣK²)²/ΣK⁴，即归一化平方权重的 inverse participation ratio；它量化权重集中度，**不是统计上独立样本的数量**。例如 (14,6) 单个样本就占 62.84% 的平方和，前三个占约 98.11%。因此该经验平方风险很容易被少量文档中的极端对主导，测试排名应同时看文档 bootstrap、其他语料及分层误差。

一个简单例子：若 99.9% 的 K=1，0.1% 的 K=100，那么 C2=10/(10+0.999)=90.92%；大值改成 600，C2=360/(360+0.999)=99.72%。指数函数只需将 logits 拉开 log(100)=4.61 或 log(600)=6.40，就能产生这种比例。这个现象不要求数值溢出，也不要求幂律分布或无穷二阶矩。真实有限权重 LLM 的 Q/K 有界结论仍适用。

**平方风险、乘性误差、attention 形状在强调什么**

令 δ_i=Khat_i/K_i−1，则

$$\operatorname{NMSE}=\sum_i w_i\delta_i^2,\qquad w_i=\frac{K_i^2}{\sum_jK_j^2}.$$

这是乘性误差的极不均匀加权平均。若普通 99.9% 样本全部拟合正确、仅将尾部预测成接近零，则 NMSE≈C2，可以达到 0.91—0.997。反过来只拟合尾部、把其余样本预测成接近零，也可能得到 NMSE≈1−C2。故 raw NMSE≈1 不意味着所有样本都没学会，较低 raw NMSE 也不意味着多数样本拟合良好。目标能量与误差占比理论上不同；本次追加推断确认，各模型确实主要在尾部留下平方误差，见下表。

令 r=log(Khat/K)，三个目标可以统一写成

$$L_{\rm square}=K^2(e^r-1)^2,\quad L_I=K(e^r-1-r),\quad L_{\rm lc}=\log\cosh r.$$

在小误差附近，它们分别近似 K²r²、Kr²/2、r²/2。对 log-prediction 的梯度分别为 2K²e^r(e^r−1)、K(e^r−1)、tanh r。尤其严重低估时 r→−∞，平方损失对 log-prediction 的梯度趋零，而 I-divergence 的梯度趋 −K。该公式说明 I-divergence 对严重低估的大值具有不同的纠正机制；实际参数梯度还乘以网络 Jacobian，不能据此保证优化必然成功。

训练仍直接拟合原始正 kernel；使用 log-ratio 计算损失或对公共幅度校准，不会把训练标签改成归一化 attention。给每行 K 乘一个仅依赖 query 的公共因子，会改变原始 kernel 风险，而在 linear attention 分母中抵消。这说明两个目标本来就不完全相同，而不是要求放弃用户指定的原始 kernel 拟合目标。

**大值来自哪里：位置、范数和方向**

在 (14,0)/(14,6) 的 132 个最大配对样本中，分别有 107/91 个来自 key 位置 0；但这两个 head 最大的三个平方能量样本均来自非开头位置。例如 (14,6) 最大样本来自内部测试文档 234，q位置1022、k位置260，logit=7.1698。其 q范数42.42（全体均值23.42）、k范数18.48（均值18.41）、余弦0.103（全体均值−0.227）。不能只检查 key 范数，也不能只检查最初几个 token。

第 27 层两个 head 的大值位置更分散，其尾部平均余弦分别约 0.362/0.461，全体约 0.045/0.161。不同 head 同时涉及范数和方向变化，观察到相关性并不等于已经识别因果机制。

在前 32 篇文档的完整合法 512×512 块上，比较前四个 key：

| head | 前4个key的全局平方和占比 | 前4个key的平均逐行attention质量 |
|---|---|---|
| [14, 0] | 23.89% | 87.68% |
| [14, 6] | 0.61% | 68.77% |
| [27, 0] | 0.26% | 2.25% |
| [27, 6] | 0.72% | 1.71% |

第14层的前几个 key 吸收了大量平均 attention 概率，这与 [StreamingLLM 的 attention sink 观察](https://arxiv.org/abs/2309.17453)相符；但该局部证据不是完整的机制证明，不能把全部 raw-kernel 尾部等同于 sink。上述概率只对前512个 key 归一化、query为后512位置，属于本实验的矩形块诊断，不是完整 causal 上下文的概率。

![平方能量、kernel质量与attention的差异](tail_vs_attention.png)

**已训练模型如何拟合尾部**

下表都是三个 seed 的平均。中位预测比例是先在各 seed 的尾部取 Khat/K 中位数，再平均；尾部总质量比例是 Σ_SKhat/Σ_SK。

| 损失 | 模型 | head | 尾部占总平方误差 | 尾部中位预测比例 | 尾部总质量预测比例 |
|---|---|---|---|---|---|
| logcosh | MLP | [14, 0] | 91.06% | 0.0163 | 0.0108 |
| logcosh | MLP | [14, 6] | 99.71% | 0.0153 | 0.0041 |
| logcosh | MLP | [27, 0] | 99.02% | 0.0377 | 0.0302 |
| logcosh | MLP | [27, 6] | 98.27% | 0.0458 | 0.0315 |
| logcosh | KAN | [14, 0] | 91.95% | 0.0598 | 0.0351 |
| logcosh | KAN | [14, 6] | 99.81% | 0.1861 | 0.0385 |
| logcosh | KAN | [27, 0] | 98.93% | 0.0242 | 0.0224 |
| logcosh | KAN | [27, 6] | 98.05% | 0.0271 | 0.0193 |
| logcosh | mulKAN | [14, 0] | 95.93% | 0.5578 | 0.3769 |
| logcosh | mulKAN | [14, 6] | 99.90% | 0.7152 | 0.1620 |
| logcosh | mulKAN | [27, 0] | 99.11% | 0.0664 | 0.0583 |
| logcosh | mulKAN | [27, 6] | 98.40% | 0.0759 | 0.0667 |
| poisson | MLP | [14, 0] | 98.18% | 0.9204 | 0.5750 |
| poisson | MLP | [14, 6] | 99.89% | 0.8228 | 0.1880 |
| poisson | MLP | [27, 0] | 98.09% | 0.2190 | 0.2030 |
| poisson | MLP | [27, 6] | 96.42% | 0.2330 | 0.2384 |
| poisson | KAN | [14, 0] | 98.40% | 0.9109 | 0.5518 |
| poisson | KAN | [14, 6] | 99.89% | 0.7413 | 0.1567 |
| poisson | KAN | [27, 0] | 98.23% | 0.1935 | 0.1836 |
| poisson | KAN | [27, 6] | 96.51% | 0.2287 | 0.2355 |
| poisson | mulKAN | [14, 0] | 98.46% | 0.8970 | 0.5589 |
| poisson | mulKAN | [14, 6] | 99.89% | 0.8041 | 0.1683 |
| poisson | mulKAN | [27, 0] | 97.71% | 0.1874 | 0.2292 |
| poisson | mulKAN | [27, 6] | 95.68% | 0.2370 | 0.2774 |

例如 MLP/log-cosh 的尾部中位预测只达到目标的约1.5%—4.6%，普通样本的稳健拟合确实伴随大值低估。MLP/I-divergence 在 (14,0)/(14,6) 的中位比例提升到约92%/82%，但 (14,6) 的尾部总质量只恢复约19%，意味着最极端的少数点仍明显低估。不能仅靠一个中位数判断尾部已学好。

**对下一轮训练和理论的启发**

1. 先测收敛再判断结构。保留原始 kernel 标签和每个训练对只用一次的约束，使用更长的新数据流，增加记录频率，固定验证/训练探针，保存中间 checkpoint；学习率先保持可学习区间，再退火。用共同的更长学习率计划比较4096/8192/16384步，而不是分别在每个终点提前退火。另做固定唯一数据对数、改变batch大小的步数对照，可帮助区分数据覆盖与更新步数，但batch变化本身仍有混杂。单遍约束下无法把固定数据重复优化作为完全等价的消融。
2. 关注尾部覆盖，而不只计 token 总数。按训练数据中确定的 logit 桶、key位置、q/k范数和方向模式做分层监控，比较普通样本、sink样本和非sink极端样本。若训练时改变采样分布，应记录采样概率；目标保持原风险时使用正确的逆概率权重，并保留均匀探索。直接按每行完整 QK 找top-k会增加训练成本，不能宣称推断时免费得到该信息。
3. I-divergence 可作为主要损失，辅以较小的 log-cosh 项来兼顾普通样本。权重使用训练集公共尺度校准、验证集选择，不进行逐行标签归一化。这个组合尚未在本次运行中测试，不能宣称已经优于现方案。不要把删去极端值后的好指标当作解决原始 kernel 拟合问题。
4. 正特征可考虑单侧幅度与形状分解：Khat=a(q)b(k)Σ_r f_r(q)g_r(k)，a,b>0。这仍可吸收到m维特征中，保持 linear attention 结构；query幅度在分母抵消，但若目标是原始K，训练时仍需恢复正确幅度。该分解可能改善优化条件，不构成 KAN 特有的表达优势，也未在此次诊断中验证。
5. 理论应说明风险对数据分布的重新加权：raw NMSE 对相对误差使用 dP2/dP=K²/E[K²]；质量归一化 I-divergence 使用 dP1/dP=K/E[K]。在普通样本、sink和非sink极端样本的混合分布上分析谱与逼近，比无条件套用一个高斯更贴近当前证据。经验权重高度集中会使矩估计与经验谱对稀有事件敏感；需要跨文档稳定性和总体泛化分析。

这些结果不证明 m=64 已达到秩瓶颈，也不证明没有秩瓶颈。逐测试块的有符号 SVD oracle 不受正因子、跨文档共享函数、训练估计和优化的限制，不能用其低残差宣布当前网络只缺训练步数。对 KAN/mulKAN 的有价值主张，应是相同预算下更快或更稳地学会这些结构，而不是仅凭通用逼近或乘法节点断言更强。

诊断输入为 `../fits/`、`../data/`；代码为 `../diagnose_convergence_tail.py`、`../summarize_diagnostics.py`；所有数值保存在 `convergence_tail.json`。原始45个训练 checkpoint 未修改。

