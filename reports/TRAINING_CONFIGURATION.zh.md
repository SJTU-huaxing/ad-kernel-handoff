# 从零预训练配置

> 当前状态：用户已于2026-09-09 04:04 UTC明确停止训练及全部自动队列，未经重新授权不恢复。已完成seed11十组、seed29七组，均为1600步/104,857,600 tokens；ELU seed29未完成。下文较早阶段的“继续运行”等表述仅为历史记录。停止快照和完整结果见 [STOPPED_EXPERIMENTS.zh.md](STOPPED_EXPERIMENTS.zh.md)。

当前正式协议见 `configs/pretraining_protocol_v2.json`，数据清单见 `work/pretraining/data/fineweb_edu_v1/manifest.json`。主种子 11 在各方法上训练 1,073,741,824 tokens；种子 29/47 各训练 104,857,600 tokens，且使用相同的完整预算学习率日程。早期三种子指标不能当作已收敛的完整预算结果。

第一版 AD 在第 348 步因极小缩放分母发生非有限梯度，旧源码和轨迹已归档。修复版保留相同数学核和超参数；遇到不安全的计算尺度时细分执行段，完整传递因果状态及其梯度，不截断历史、不钳位幅度、不添加 epsilon。所有方法重新从相同 seed 初始化开始。诊断、限制和证据见 `reports/NUMERICAL_STABILITY.zh.md`。

共有骨干：12 层、768 hidden、12 heads（d64）、2048 SwiGLU intermediate、RMSNorm、RoPE、无 dropout、共享 50,257 词表 embedding。共有权重逐元素相同；feature 初始化在独立 RNG 范围内进行，不改变后续骨干的随机初始化。AD/EXP 的两层 feature 网络参数完全相同，Q 幅度/C 均不存在，K 幅度保留。

精度需按实际执行路径理解：骨干为 BF16 autocast、FP32 主参数；**线性方法的 feature 和归一化 attention 为 FP32，softmax 对照使用 BF16 Q/K/V 的原生 SDPA**。融合词表线性层交叉熵共用于训练，验证只计算前向 NLL。`attention_dtype` 配置控制线性分支；softmax 分支按原生 SDPA 的执行方式处理，不能把它也描述为完整 FP32 attention。

正式线性训练采用 CANN 原生 batched GEMM/cumsum 分块实现。在独占同一张 910B 上，以相同权重、16×1024 输入、3 次预热及 7 次计时，并反转两种后端的顺序复测：

| 模型 | CANN 原生分块前向+反向 | 混合 Triton 路径 | 原生路径加速 |
|---|---:|---:|---:|
| AD m64，随机初始化 | 约 0.381 s | 约 0.538 s | 约 1.41× |
| HH-exp m384，随机初始化 | 约 0.973 s | 约 1.360 s | 约 1.40× |
| AD m64，真实故障前权重/输入，含数值分段 | 约 0.415 s | 约 0.578 s | 约 1.39× |

该计时不含 AdamW、数据载入及 HCCL。原生路径多用约 1–3 GiB 显存，仍可容纳每卡 micro-batch16。实际训练每卡累积两次，global batch 为 64×1024=65,536 tokens。修复版原始计时在 `work/reproduction/backend_benchmark_stable_v2.json` 和 `backend_benchmark_stable_trained_v2.json`；第一版计时另行保留。

FAVOR 随机矩阵固定，只训练骨干 Q/K 投影；Hedgehog 的独立 Q/K 权重及偏置可训练。m128 的 HH、m384 的更大 HH、m64/m256 的 FAVOR 分别报告状态与参数成本，不冒称所有预算相等。传统 ELU+1、softplus 和原 softmax 对照共用同一数据流、优化器和日程。

训练块从固定 EOS 分隔语料中取非重叠目标区间，再按 seed 对块做无放回排列；块内允许跨文档 attention，块之间重置状态。这是明确的固定长度预训练约定，不是隐藏的分块截断梯度或局部注意力窗口。每个训练样本内部的线性状态梯度贯穿完整 1024 tokens。

检查点包含模型、AdamW、各 rank RNG、学习率所需步数、无放回排列哈希、数据游标及有效 token 数。合成恢复检查与正式语料分目录，合成 tokens 不计为真实预训练进度。任何训练失败会停止队列，保留已写入的检查点；不会自动切换模型函数或放宽数值阈值继续训练。

每个预定里程碑后，先用两张 NPU 分片完成固定测试流、64 篇长文、62 篇 WikiText 和 216 个关联回忆提示的评测，再启动下一训练任务。评测输入及协议已在读取这些模型成绩前冻结。训练日志记录 rank0 当前进程累计的数值分段次数；重启会重置该诊断计数，不影响模型或优化器状态。
