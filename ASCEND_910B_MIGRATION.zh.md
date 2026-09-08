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
