# AD / Hedgehog 速度复核与 FLA NPU 可用范围

本次按用户要求补做固定权重的算子、feature与单层attention微基准；没有恢复预训练，没有优化器更新，也没有修改17组实验的模型权重、冻结训练代码或协议。训练队列的停止标记继续保留。

**直接使用FLA的结论需要按具体路径区分。** 当前环境有可直接运行的FLA模块和模型路径：官方HedgehogFeatureMap通过前向/梯度CPU FP64对照，官方LinearAttentionForCausalLM在`fused_recurrent`模式完成BF16小模型前向和反向。此前仅凭`chunk_linear_attn`的失败，不能推断所有纯线性路径或全部FLA模型不可用。之前未检查recurrent这一替代路径，验证范围不完整，本次已补上。

**AD的速度结论也必须区分feature生成和状态计算。** 原始GPU实验和此次NPU复测都不支持“AD的feature map在所有Hedgehog对照上更快”。更小的状态可让AD在部分完整attention计算中取得优势，但优势依赖对照宽度、执行模式、序列/批量和是否包含反向。

**1. 当前环境的FLA直接运行检查**

环境：Ascend910B3、CANN9.0.0、torch2.7.1、torch_npu2.7.1.post4、triton-ascend3.2.1、FLA/fla-core0.5.2。FLA先于自定义Triton首次编译导入；本次没有更换环境版本。

| 路径 | 本次/已有证据 | 结论范围 |
|---|---|---|
| FLA `HedgehogFeatureMap` 原模块，FP32 | 本次最大输出误差9.27e-8，输入/参数梯度最大相对RMS约1.45e-7 | 本机该feature模块可直接运行且小样本前后向正确 |
| `chunk_linear_attn`，FP32，B1/T65/H2/m64/dV64 | 本次重新执行，跨块处最大输出误差0.364241，Q/K梯度相对RMS约0.060567/0.014518 | 该路径未通过正确性检查；不能用于当前正式训练 |
| `chunk_linear_attn`，BF16 | 之前本机记录generic chunk_h的MLIR编译失败 | 所测配置失败，不推断所有dtype/尺寸永久不可用 |
| `fused_chunk_linear_attn`，FP32，B1/T65/H2/m64/dV64 | 本次BiShengHIR编译失败，`func.call`的cc/cbuf地址空间类型不匹配 | 该替代路径在所测配置不可用 |
| `fused_recurrent_linear_attn`，FP32，B1/T65/H2/m64/dV64 | 本次最大输出误差2.38e-7，Q/K/V梯度最大相对RMS1.91e-7，最终state相对RMS1.48e-7 | 此路径通过独立CPU FP64对照 |
| 官方 `LinearAttentionForCausalLM`，BF16 recurrent模式 | 本次1层、hidden128、heads2、FFN256、词表512、B1/T65，loss=6.220703125，所有已有梯度有限 | 证明该小模型执行路径可以运行；无优化器更新，不是完整预训练或全模型数值正确性证明 |
| FLA融合线性层交叉熵 | 已用于前述17组正式训练 | 当前工程确实复用了FLA NPU模块 |

小模型保留FLA默认融合RMSNorm/SwiGLU，使用普通PyTorch交叉熵，`use_cache=False`，`feature_map="hedgehog"`，`attn_mode="fused_recurrent"`；没有加载预训练权重。更完整的FP32 recurrent训练尺寸验证及计时见后表。

官方[NPU路线图](https://github.com/fla-org/flash-linear-attention/issues/942)明确按modules、ops、layers、models逐步推进；[v0.5.2 Ascend CI](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/.github/workflows/ascend-a2-ci.yml)覆盖modules、ops/utils、GDN、KDA、attnres，不能把这些测试等同于所有linear_attn模式和完整模型组合都已验证。某些通用Triton实现也可以经Triton-Ascend正确运行；有无独立NPU专用文件不是判定可用性的充分条件。

失败和通过结果均保留：[重新检查chunk日志](../work/logs/fla-recheck-after-stop-fp32.log)、[fused_chunk结果](../work/reproduction/feature_speed_npu_v1/fla_fused_chunk_float32.json)、[recurrent结果](../work/reproduction/feature_speed_npu_v1/fla_fused_recurrent_float32.json)、[官方小模型结果](../work/reproduction/feature_speed_npu_v1/fla_model_recurrent_bfloat16.json)。

**2. FLA自带模型/feature与本轮实验的定义差异**

FLA模型框架可以复用，但当前实验模型不能无修改地替换成默认FLA模型并沿用原来的实验名称和检查点含义。主要差异来自实际源码：

| 项目 | FLA0.5.2默认纯LinearAttention路径 | 本轮统一预训练模型 |
|---|---|---|
| Hedgehog函数 | `softmax(concat(2u,-2u))`，合在一起做一次softmax | HH-softmax为`concat(softmax(u),softmax(-u))`；HH-exp为正负指数拼接 |
| feature权重 | 一个HedgehogFeatureMap中的Linear权重在head维共享；Q/K默认两个独立模块 | 每head分别有独立可训练Q/K映射；原m576与当前m128/m384预算分别记录 |
| AD支持 | 默认feature_map选项没有本项目的纯AD双MLP/K幅度参数化 | AD Q64→96→63、K64→96→64，K幅度保留 |
| 归一化 | 模型默认`norm_feature_map=False`，attention输出另有head内RMSNorm | 显式因果分母归一化，共有骨干按当前配置RMSNorm |
| 位置 | 该纯LinearAttention分支未做当前模型的RoPE | 线性Q/K先RoPE再各乘d^-1/4 |
| 精度/数值状态 | 由具体FLA算子决定；`normalize=True`的分母加1e-10 | 线性attention/状态FP32，极小分母时执行数值细分，完整续接状态/梯度，未加epsilon |

FLA函数依据：[官方feature_map.py](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/modules/feature_map.py)，本机安装代码与该函数一致。本次未把FLA的原模块偷偷替换到已训练HH中，也未把不同函数的质量成绩混合。

对AD而言，可以考虑使用已验证的FLA recurrent**分子算子**，外部保留当前log-feature缩放、精确分母、S/z/scale续接与数值分段；这仍需要实际AD权重、极端数值输入、全部层、缓存和恢复的集成验证。当前补测的正特征随机输入不证明其已经解决v2针对真实训练出现的极小分母问题。

**3. 原始RTX3090实验究竟验证了什么**

原始交接文件中，最初含Q幅度的AD约为feature42.7μs、feature+state74.2μs，HH-exp约28.3/98.2μs。与当前研究对象对应的**删除Q幅度/C之后的纯AD**，原始新鲜双顺序复测如下：

| 原GPU方法 | 特征宽度 | feature Q/K pair μs | feature+state μs | 每层state bytes |
|---|---:|---:|---:|---:|
| 纯AD reduced_plain | 64 | 32.848 | 64.431 | 399,360 |
| HH-exp | 576 | 28.384 | 98.016 | 3,594,240 |
| HH-softmax | 576 | 33.312 | 103.392 | 3,594,240 |

协议为FP32、B1、单token、单层12heads、d128、真实QKV、CUDA Graph热缓存、统一未专门融合的PyTorch状态更新。纯AD相对HH-exp的feature生成略慢，而feature+state约1.52×快；相对HH-softmax的合计约1.60×快。HH的状态宽度为9倍，状态读写/计算成本是核心差异之一。

原始证据：[query_amplitude_ablation/results/benchmark.json](../snapshot/kan_attention_theory/query_amplitude_ablation/results/benchmark.json)。该RTX3090结果保留为历史观测，未冒称是在当前910B上重跑的CUDA结果。

**4. 910B上的原实验尺寸复测**

使用保存的原始seed11纯AD、HH-exp和HH-softmax权重，取layer14的12个head：AD d128/hidden192/m64，HH d128/m576、无偏置。Q/K/V来自本机重新提取的公开WikiText/Qwen数据，原GPU原始QKV没有交接，因此不是原输入逐字节复现。Q/K pair计时包含原协议的K按GQA重复；状态核复用原始未优化PyTorch分支的相同操作。

同卡顺序执行，两轮反转方法顺序；每次8次调用预热、3次执行预热，15个计时样本；decode每个样本包含32次调用。表格中位数合并两轮共30个样本，每项共计960次调用。Graph执行与eager分别报告，计时中排除图捕获和编译。

| 模式 | 方法 | feature μs | state-only μs | feature+state μs |
|---|---|---|---|---|
| NPU Graph | AD m64 | 54.867 | 76.626 | 146.600 |
| NPU Graph | HH-exp m576 | 40.460 | 100.267 | 156.122 |
| NPU Graph | HH-softmax m576 | 63.214 | 100.342 | 191.659 |
| Eager | AD m64 | 650.846 | 510.457 | 1204.873 |
| Eager | HH-exp m576 | 570.735 | 518.021 | 1153.127 |
| Eager | HH-softmax m576 | 629.001 | 540.266 | 1201.460 |

NPU Graph下，AD的feature+state相对HH-exp约1.065×快、相对HH-softmax约1.307×快；其feature单独计算慢于HH-exp、快于HH-softmax。Eager下AD合计慢于HH-exp，与HH-softmax接近。这说明原GPU较明显的合计优势在NPU上缩小，并且依赖Graph消除部分主机发射开销。

单token微基准的S/z初始为0并显式传入state；历史核按原协议原位累计重放，当前函数式递推返回的新state不串接到下次调用。这里测固定热缓存操作，不是长prefill后的整段生成。

feature、state-only和两者合计分别独立计时，缓存状态/分配和调度不同，因此三列不要求严格满足加法关系。热缓存重复同一token的微基准不是端到端生成延迟，也不是最优融合实现的上限。

**5. 当前预训练尺寸：单层feature与attention**

当前seed11 step1600的layer0特征权重；四个方法使用完全相同的Q/K/V输入，输入由AD checkpoint的第一层骨干投影从固定test tokens生成，投影采用BF16、RoPE和温度缩放FP32。投影、数据读取及权重加载不计入时间；attention的feature与状态均为FP32。

训练形状B16、T1024、12heads、d64如下，单位ms。“前向+反向”计算输入和所有feature参数的梯度，没有优化器或完整语言模型的其余部分。每项两轮反转方法顺序，各15个样本。

| 方法 | feature前向 ms | feature前后向 ms | feature+attention前向 ms | feature+attention前后向 ms |
|---|---|---|---|---|
| AD m64 | 1.8608 | 4.7828 | 5.6917 | 17.6333 |
| HH-softmax m128 | 2.1279 | 4.0959 | 7.8124 | 25.1498 |
| HH-exp m128 | 1.1044 | 2.4571 | 6.7888 | 23.4007 |
| HH-exp m384 | 4.4637 | 8.0793 | 16.9743 | 66.9329 |

对HH-exp m128，AD的feature前向/前后向都更慢，但完整feature+attention前后向约1.33×快。对HH-softmax m128，AD feature前向略快、前后向更慢，完整feature+attention前后向约1.43×快。对HH-exp m384，AD feature前后向约1.69×快，完整feature+attention前后向约3.80×快。m384的参数量接近AD，状态为AD的6倍；m128同时有较少参数和较大状态，不能笼统称所有比较都精确等预算。

当前尺寸B1/T1的单token执行结果：

| 模式 | 方法 | feature μs | feature+state μs |
|---|---|---|---|
| NPU Graph | AD m64 | 45.072 | 166.010 |
| NPU Graph | HH-softmax m128 | 50.240 | 186.557 |
| NPU Graph | HH-exp m128 | 30.441 | 158.737 |
| NPU Graph | HH-exp m384 | 32.699 | 180.858 |
| Eager | AD m64 | 664.640 | 1609.052 |
| Eager | HH-softmax m128 | 698.116 | 1617.122 |
| Eager | HH-exp m128 | 650.692 | 1573.088 |
| Eager | HH-exp m384 | 650.896 | 1572.854 |

特别是Graph下，当前AD m64的feature+state约166.0μs，HH-exp m128约158.7μs，**这个decode场景AD没有优势**。B1/T1024和B1/T8192的前后向样本也全部保存在原始JSON/CSV。

本次单层、共同输入的计算优势不能替代整模型训练观测。此前完整1600步实际训练循环中，AD seed11约33.51分钟，HH-softmax m128约27.09分钟、HH-exp m128约27.75分钟、HH-exp m384约53.95分钟。其他层的输入分布、数值分段、MLP/词表损失、DDP/优化器等不在这个微基准里，因此当前全模型训练并没有表现出AD对两个m128对照更快。

**6. FLA recurrent在实际训练尺寸的补充计时**

本次已对m64/m128/m384的B1/T65/H2执行独立CPU FP64前向、梯度与最终state检查；对B16/T1024/H12/dV64，再与已经过CPU验证的CANN实现核对完整尺寸输出及梯度。六项检查均通过，随后计时。

计时输入为固定正随机特征对应的**同一log-feature**，两个实现都包括当前控制器相同的数值缩放、精确因果分母、极小分母检查、最终S/z/scale构造。FLA接入还包括当前BHTM布局到其BTHM连续输入的转换。外部不启用FLA的epsilon归一化。两轮反转后端顺序，各7个样本，单位ms；不含feature MLP、QKV投影、全模型或优化器。

| m | 实现 | 前向 ms | 前向+反向 ms |
|---|---|---|---|
| 64 | fla_recurrent | 4.4153 | 17.0489 |
| 64 | cann_chunk | 3.9241 | 12.9542 |
| 128 | fla_recurrent | 7.9970 | 32.8158 |
| 128 | cann_chunk | 5.8055 | 21.2734 |
| 384 | fla_recurrent | 20.6151 | 95.3084 |
| 384 | cann_chunk | 12.5675 | 58.8531 |

在这个接入口径下，FLA recurrent前后向分别比CANN慢约1.316×、1.543×、1.619×；因此确认了可用替代算子，但没有获得切换后端的速度收益。m64的原始正特征分子路径加分母约13.10ms、与CANN约13.11ms接近；那次初测的两侧输入表示与外层操作不同，不能代替上表同一log-feature输入的比较。初测及其源码另存`recurrent_training_shapes_raw_positive.json`与`sources/`，保留这一区别。

本适配计时只处理零初始状态、数值安全的随机特征；如果分母触发细分条件，会明确报错。还没有把这个候选接入冻结模型的递归数值分段、非零恢复state或所有真实AD层。因此“这些尺寸前后向通过”与“已经可以无验证地恢复整模型训练”是不同层次的结论。官方小模型测试通过也不证明本项目全部训练设置都适用。

**7. 验证与产物**

七种历史/当前feature模型的小样本FP32 NPU前向和输入/参数梯度，均与CPU FP64密集归一化attention对照通过，所测梯度最大相对RMS约2.38e-6。历史状态更新移植还对照了当前独立递推实现，最大相对RMS约1.28e-7。FLA原Hedgehog模块另行检查，没有混入同函数速度对比。

AD/HH速度部分共有164个计时轮次、2460个原始样本、82个合并条件。每轮顺序、全部样本、参数数目、输入和权重SHA256、源码SHA256及验证误差均保留。

- [AD/HH汇总CSV](../work/reproduction/feature_speed_npu_v1/summary.csv)
- [全部计时样本CSV](../work/reproduction/feature_speed_npu_v1/samples.csv)
- [历史尺寸完整JSON](../work/reproduction/feature_speed_npu_v1/historical.json)
- [当前尺寸完整JSON](../work/reproduction/feature_speed_npu_v1/current.json)
- [FLA recurrent训练尺寸JSON](../work/reproduction/feature_speed_npu_v1/recurrent_training_shapes.json)
- [AD/HH基准源码](../experiments/benchmark_feature_maps_npu.py)
- [FLA模式/小模型检查源码](../experiments/probe_fla_alternative_modes.py)
- [FLA recurrent训练尺寸基准源码](../experiments/benchmark_fla_recurrent_npu.py)

没有修改原始交接文件、已完成实验的冻结训练源码、配置或检查点。先前的停止报告作为停止时快照保留；这份报告单独记录用户后来授权的算子与速度验证。
