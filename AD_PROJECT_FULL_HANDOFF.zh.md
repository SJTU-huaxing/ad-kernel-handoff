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
