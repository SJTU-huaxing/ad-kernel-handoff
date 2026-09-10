约 100M 参数、每方法每 seed 恰好 2B tokens 的双 910B 预训练方案
================================================================

本方案已编写为独立训练与评测入口。**正式训练尚未启动**。此前被冻结的 `src/ad_kernel`、seed 11/29 检查点及旧停止标记不作修改。短时性能和正确性检查的产物位于 `work/pretrain_100m_2b/probes`，带有 `probe=true`，不能作为正式 2B 结果。

**启动方式**

在当前机器运行：

```bash
cd /cache/huaxing/ad-kernel-handoff
bash scripts/launch_100m_2b.sh
```

这条命令只列出计划。实际以前台方式启动全部默认任务：

```bash
bash scripts/launch_100m_2b.sh --execute
```

脚本自动激活 `nonlinear-qk`，每个任务使用两个进程、HCCL 数据并行和两张 NPU。默认顺序为 seed 11 的 AD、softmax、FAVOR+、Hedgehog，再执行 seed 29 的同四种方法；**共 8 次完整预训练，每次 2B，合计 16B 训练目标 tokens**。每次训练完成后自动评测再启动下一方法。若只执行 seed 11：

```bash
bash scripts/launch_100m_2b.sh --execute --seeds 11
```

启动器将训练及评测的 stdout/stderr 实时显示在当前终端，同时追加保存到 `work/pretrain_100m_2b/logs/<method>_s11.train.log` / `.eval.log`。无需 `nohup` 或后台运行。训练启动时打印总步数与模型参数，之后每 10 个 optimizer steps 打印 step、累计 tokens、loss、学习率、梯度范数、每步耗时及 tokens/s；验证与检查点事件也会显示。每一步的完整指标仍写入对应 run 的 `train.jsonl`。仅 seed 11 的四种方法各训练 2B tokens，共 8B，连同评测预计约 25–33 小时。

实时损失曲线已接入 TensorBoard，启动查看服务使用 `bash scripts/tensorboard_100m.sh`，与训练分别在两个 Terminal 中前台运行。每个训练更新步都有曲线数据，浏览器通过 JupyterLab 的 `/proxy/6006/` 入口访问。详见 [TensorBoard 使用说明](TENSORBOARD.zh.md)；也可打开 `notebooks/Training_TensorBoard.ipynb`。

也可用 `--methods ad64 softmax` 选择方法子集；默认仍为全部四种方法。再次运行相同启动命令会从已有完整检查点恢复，已完成且评测齐全的任务跳过。更换模型、数据、种子、优化器、代码或批量后不能混用旧检查点。

**模型来源和框架选择**

已核对 [FLA 官方模型仓库](https://github.com/fla-org/flash-linear-attention) 及 fla-hub 的公开模型目录；本次检索没有找到可直接照搬的约 100M 配置。采用 [FLAME 的 transformer_340M.json](https://github.com/fla-org/flame/blob/24dea9133e2166da89a563afbdcefb7c55615646/configs/transformer_340M.json) 所用的 LLaMA 式解码器结构，缩小宽度与层数，并适配当前已备好的 GPT-2 词表及训练数据。它是依据 FLA/FLAME 写出的新约 100M 配置，不是声称官方发布过同名 100M 权重；所有正式任务从随机初始化开始。

FLAME 源码固定在 `24dea9133e2166da89a563afbdcefb7c55615646`；其 README 指定的 TorchTitan 固定在 `0b44d4c437c424b6bf719661c0eb4283dc4068bc`，均保存在 `work/reference`。补齐 Python 依赖后，原版 `flame.train` 在当前 datasets 5.0.1 环境仍因导入已移除的 `ShufflingConfig` 而失败，日志为 `work/logs/flame-import-probe.log`。这说明当前组合不能直接运行，并不表示 FLAME 原理上不能适配 NPU；本次没有完成其整套数据加载、检查点和优化器的 Ascend 移植。

按照你允许的备用方案，实际 pipeline 是 `src/pretrain_100m/train.py`：原生 PyTorch/torch_npu HCCL DDP，参考 [FLAME 的训练配方](https://github.com/fla-org/flame/tree/24dea9133e2166da89a563afbdcefb7c55615646)。复用了安装版 FLA 的 `RMSNorm`、`GatedMLP`/融合 SwiGLU、融合线性输出层交叉熵、官方 Hedgehog 模块；没有把本地训练器标为“原版 FLAME”。

**统一骨干与精确参数量**

| 项目 | 本方案 |
|---|---|
| 层数 / 隐藏宽度 | 12 / 640 |
| 注意力头 / 每头维度 | 10 / 64，无 GQA |
| FFN | SwiGLU，中间维度 2048，线性层无 bias |
| 归一化 | Pre-RMSNorm，epsilon=1e-6；每块注意力与 FFN 前归一化，末端另有 RMSNorm |
| 位置编码 | 四种方法统一 RoPE，theta=10000 |
| 训练长度 / 评测长度上限 | 2048 / 8192；没有声称在 8192 长度训练过 |
| 词表 | GPT-2，50257 tokens；EOS=50256 |
| 输入嵌入 / 输出头 | 权重共享 |
| 初始化 | 普通骨干权重 N(0,0.02²)，残差输出投影按 1/sqrt(2L) 缩放 |
| Dropout / 额外辅助损失 | 0 / 无 |
| 骨干与优化器状态 | FP32 主参数，骨干计算 BF16 autocast |
| 线性 feature map / 状态 / 归一化分母 | FP32 |

| 方法 | 总可训练参数 | feature map 可训练参数 | 每头特征维度 |
|---|---:|---:|---:|
| AD m64 | 101,964,800 | 2,937,600 | 64 |
| 标准 softmax | 99,027,200 | 0 | 不适用 |
| FAVOR+ m64 | 99,027,200 | 0 | 64 |
| FLA Hedgehog | 99,127,040 | 99,840 | 128 |

四种方法共有 99,027,200 个骨干参数。相同 seed 下骨干初始化逐字节相同；feature map 初始化使用隔离的随机数状态，不扰动骨干。AD 的额外参数约占骨干的 3%，本方案不宣称各方法严格等参数。

与参考配置相比，宽度从 1024 缩至 640、层数从 24 缩至 12；FFN 明确设置为 2048，以获得合适参数规模和矩阵形状。因当前 GPT-2 词表比参考的 32000 更大，采用共享输入/输出嵌入。所有模型统一使用同一种骨干和 RoPE，因而这里比较的是 feature map/注意力方式，不是逐个复刻官方各模型的所有默认设置。

**四种方法的准确定义**

每头 Q/K 在 RoPE 后各乘以 d^(-1/4)，使 softmax 与随机特征使用一致的注意力温度。

* **AD m64**：沿用原始纯 AD 定义。每层每头独立的 Q/K 两层 SiLU MLP，64→96；Q 输出 63 维方向，补 0 后 softmax；K 输出 63 维方向和 1 维 log-amplitude。`phi_q=softmax([f_q(q),0])`，`phi_k=exp(s_k)·softmax([f_k(k),0])`。没有 Q amplitude、外部 C、额外门控或振幅裁剪。
* **标准 softmax**：完整因果 `softmax(QKᵀ/sqrt(d))V`，使用 torch_npu 支持的 SDPA；没有线性近似、局部窗口或丢弃历史。
* **FAVOR+ m64**：正的正交随机特征 `phi(x)=exp(Ωx−||x||²/2)/sqrt(64)`，Ω 由正交方向与 Gaussian/chi 范数构造，每层每头固定，Q/K 使用相同 Ω。初始化后不重采样、不训练 Ω。当前 FLA 0.5.2 没有 FAVOR+/Performer feature-map 模块，因此这项由本项目按 [Google Research 的 Performer/FAVOR+ 定义](https://github.com/google-research/google-research/blob/master/performer/fast_attention/tensorflow/fast_attention.py) 实现，不能称为 FLA 原生 FAVOR+ 模型。源码参考副本及 SHA 位于 `work/reference/performer_source.json`。
* **FLA Hedgehog**：使用安装版 `fla.modules.feature_map.HedgehogFeatureMap` 的投影、bias 和 identity 初始化；Q/K 为独立模块，每个模块在一层的各头之间共享。公式是 `softmax(concat(2·Linear(x), −2·Linear(x)))`，输出 128 维。训练中的 log-space 适配直接对官方投影结果做等价的 `log_softmax`，避免先 softmax 下溢再取 log；它的值和所有输入/参数梯度已与官方 `forward` 对照。**不是以前实验里“两个分支分别 softmax”且每头独立参数的本地 HH 变体。**

三种线性方法都计算带完整因果历史的归一化正核注意力。执行时采用可逆的逐特征缩放，分母极小时细分执行区间并传递完整、可微的状态，保持相同核函数。没有增加分母 epsilon、截断上下文或对 AD 振幅做截断。

**算子和双卡执行**

FLA 的部分 Ascend 算子可用；不能把 CANN 适配理解成所有 FLA 路径均已通过验证。此前当前环境中纯 `chunk_linear_attn` 未通过跨块数值检查，`fused_chunk_linear_attn` 编译失败；`fused_recurrent_linear_attn` 的前后向检查通过，但同输入、同精度的注意力基准慢于当前 CANN 路径。详见 `reports/FEATURE_SPEED_AND_FLA.zh.md`。

默认线性注意力使用 CANN 原生矩阵乘、cumsum 等算子实现的 chunk 算法，chunk size=64，保留可靠的 FP32 状态；代码同时提供已验证语义的 FLA recurrent 备选后端。已直接采用 FLA 融合 RMSNorm、残差加归一化、SwiGLU/线性层及融合 linear cross entropy，优化器采用 `NpuFusedAdamW` 和融合梯度裁剪。

不使用 FSDP、张量并行或激活重计算：约 100M 模型及优化器能容纳在单卡内，两卡数据并行使通信集中在梯度同步，避免额外逐层参数通信。梯度累积中仅最后一个 microbatch 同步。CPU 线程预取已 tokenized 的内存映射数据、使用 pinned memory 和异步设备拷贝；训练中不在线下载或 tokenization。设备编号沿用平台已有的 `ASCEND_VISIBLE_DEVICES` 映射；逻辑 npu:0/1 对应当前分配的物理 3/2。

**完整 2B 训练配方**

唯一正式配置为 `configs/pretrain_100m_2b.json`，启动脚本读取它。各方法微批量和实测时间见下方生成的性能表。

| 参数 | 值 |
|---|---|
| 每方法每 seed 训练预算 | 恰好 2,000,000,000 个参与损失的目标 token |
| 全局批量 | 192 条 × 2048 = 393,216 tokens/常规更新 |
| 总更新数 | 5087；包含部分源数据块与最后不满全局批量的更新 |
| 优化器 | NpuFusedAdamW，beta=(0.9,0.95)，epsilon=1e-8 |
| 峰值学习率 / 最终学习率 | 1e-3 / 1e-4 |
| 学习率计划 | 前 100M tokens 线性 warmup，约 255 步；其余预算 cosine 衰减 |
| Weight decay | ndim≥2 参数为 0.1，norm 与 bias 等 ndim<2 参数为 0 |
| 梯度裁剪 | 全局 L2 norm=1 |
| 验证与存盘 | 每 512 更新及最终更新；保留最后两个完整优化器检查点和最终模型权重 |
| seed | 默认 11、29，所有方法每个 seed 都跑满 2B |
| 非有限 loss/gradient | 停止并保留上次完整检查点，不把跳过的更新算进完成预算 |

学习率峰值、AdamW betas、weight decay、cosine 及最小 LR 比例参考 FLAME 配方。将 warmup 比例保持在约 5%，按本次 2B 预算重新计算；epsilon 从其示例的 1e-15 调整为 1e-8，使用 FP32 主参数和 NPU 融合 AdamW。**没有沿用之前短实验的学习率进度、权重或 optimizer state。**

**训练数据、顺序与评测**

重用已经准备好的 `HuggingFaceFW/fineweb-edu` 英文数据，固定 revision `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`、sample/10BT 的前四个 parquet。训练池 2,000,100,000 tokens、1,934,765 个文档；本次只使用前 2,000,000,001 个 token 组成恰好 2B 个下一词预测目标，实际涉及 1,934,665 个文档（最后文档可能截断）。分词器为 `openai-community/gpt2`，revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`。完整来源、质量过滤、去重与文件 SHA 见 `work/pretraining/data/fineweb_edu_v1/manifest.json`。

使用英文分数≥0.8、教育质量 int_score≥3、最低文本长度过滤，及精确文本哈希与近似 5-gram 去重；以规范化 hostname 做训练/验证/测试划分。近似去重不等于证明零残留重复，hostname 也不等于注册域名。训练数据是 EOS 分隔的连续文本流，每块内允许跨文档注意力，四种方法完全一致。

将这 2B 个目标分为不重叠的 2048-token 块，保留最后的部分块，按 `85000+seed` 生成无放回排列。同 seed 各方法相同排列和每步目标集合；microbatch 切分可以不同。样本边界可共享一个输入 token，但目标位置不重复。最后不足的样本/标签置 ignore_index=-100，以全局实际有效标签数归一化梯度；这已经覆盖到“一张卡只有被忽略标签”的末批场景测试。

启动后自动对最终固定 2B 检查点进行以下评测，结果写入各 run 的 `evaluation`：

| 评测 | 数据/规模与输出 |
|---|---|
| 域隔离 validation / test | 两者各 2,000,000 源 tokens，分别评分 1,999,999 个下一词目标；token 加权 NLL/PPL，保留部分末块 |
| WikiText 新域 | 62 个文档、286,096 个目标；文档和 2048 块边界重置，保留末块；NLL/PPL |
| 长度扩展 | 固定 64 个隔离文档，2048/4096/8192 长度；NLL/PPL 与 8192 的位置区间损失 |
| 关联召回 | 固定 72 张键值表、216 个 prompt，8/32/64 pairs，1024/4096/8192 长度；候选准确率、全词表准确率及答案 log probability |

测试集、长文和召回不用于选 LR、seed、方法或最佳 checkpoint；只报告固定预算结果。训练期间 validation 用于监控，不做 early stopping。每次评测由两张 NPU 分片，记录模型与评测数据 SHA。这里没有冒称运行了 lm-evaluation-harness 的其他任务。

`src/pretrain_100m/summarize.py` 自动生成全部已观察到的 loss 原始 CSV、训练曲线 PNG/PDF、各 seed 的指标 CSV，以及 seed 均值/标准差。两 seed 的标准差仅作描述，不作为显著性证明。

**监控、停止、恢复**

```bash
tail -f work/pretrain_100m_2b/queue.log
tail -f work/pretrain_100m_2b/logs/ad64_s11.train.log
```

平稳停止新队列：

```bash
touch work/pretrain_100m_2b/STOP
```

当前更新完成后保存检查点，队列不再启动下一个任务。需要恢复时，移除这个新队列的 STOP 文件，再运行同一启动命令；旧 `work/pretraining/stopped_by_user.json` 继续保留。

每个 run 的 `run.json` 记录配置、数据、代码、数据顺序、种子和版本绑定；`checkpoint_*.pt` 保存模型、优化器、进度及每卡 RNG；`last.json` 原子更新指向完整检查点。预计全部 8 次运行保留的检查点约占 22–24 GB，另有已存在的训练数据及日志。

<!-- MEASUREMENTS_AND_VALIDATION -->

**实测吞吐与训练时间估计**

下表来自上述约 100M 模型、2048 长度和真实训练数据的双卡检查。每个候选运行 8 次更新，排除前 3 次预热，以后 5 次完整前向、反向、梯度累积、HCCL、裁剪与优化器更新计时。

| 方法 | 每卡 microbatch / 累积 | 双卡 tokens/s | 单卡峰值分配显存 | 2B 纯计算外推 | 实际排期预留 |
|---|---:|---:|---:|---:|---:|
| AD m64 | 8 / 12 | 92,697 | 16.86 GiB | 5.99 h | 7–10 h |
| Softmax | 32 / 3 | 153,469 | 26.31 GiB | 3.62 h | 4–5 h |
| FAVOR+ m64 | 8 / 12 | 100,185 | 13.05 GiB | 5.55 h | 6–8 h |
| FLA Hedgehog | 8 / 12 | 80,717 | 17.73 GiB | 6.88 h | 8–10 h |

四种方法单 seed 纯计算外推合计约 22.0 小时；考虑评测、检查点、运行波动及后期数值细分，建议按 **25–33 小时/seed，默认两个 seed 合计 50–66 小时（约 2–3 天）** 排期。计时没有覆盖完整 2B 的后期训练状态，尤其 AD 的数值细分次数可能增加，因此这不是保证完成时间。

已比较 AD 每卡 4/8/16/24/32、FAVOR+ 8/16/32、Hedgehog 8/16/24、softmax 16/32/48；AD 32 出现 OOM，其余所列候选通过。64-token chunk 也优于所测 128-token chunk。正式配置选用实测吞吐最佳的候选，不以占满显存为目标；保留的 HBM 有利于后期数值细分。两卡利用率的逐次采样随各 throughput 日志保存；AICore 的高活跃度不等于达到理论 Cube FLOPS。

softmax 调度检查确认实际调用 `npu::npu_fusion_attention_v3` / `aclnnFlashAttentionScore` 及其反向算子，而非只根据函数名推断已加速。

**AD 与本次官方 FLA Hedgehog 的速度结论**

同一真实数据骨干产生的 Q/K，B=16、H=10、T=2048、d=64、FP32，排除 QKV/RoPE 准备时间，两轮反转测试顺序：

| 仅 feature map | AD m64 | FLA Hedgehog m128 |
|---|---:|---:|
| Q/K 前向 | 3.083 ms | 1.614 ms |
| Q/K 前向+输入及参数梯度 | 9.248 ms | 6.960 ms |

**AD 对这个官方 Hedgehog 没有单独 feature-map 计算速度优势**，本次 forward/backward 反而慢约 1.33 倍。完整训练中 AD 吞吐高约 14.8%，是更小特征/状态维度等整条计算路径共同产生的结果，不能归因于 feature map 本身更快，也不能从初期短测保证整个 2B 阶段一直保持相同比例。

**完成的验证与产物位置**

* `validation/contracts.json`：15 组独立目标覆盖检查；两个 seed 的完整 2B 顺序/预算；四方法共同骨干逐字节一致；Hedgehog 官方函数的值和输入/参数梯度一致；三种线性方法对 FP64 稠密因果参考的输出与梯度检查通过。
* `validation/npu_components.json`：四种小模型的实际 NPU BF16 融合路径与 CPU FP32 参考相比，全参数梯度相对 RMS 最大为 0.525%；全忽略标签微批量的 loss 和梯度均为 0。融合 AdamW 的 3 步更新与 CPU 参考最大参数误差 5.96e-08。
* `validation/resume.json`：真实双卡运行，4097 目标的末批场景，连续 3 步与 2 步保存后恢复 1 步的模型和优化器通过严格 FP32 容差检查；共检查 82 个张量，最大绝对差 9.31e-10。不是逐位确定性保证；同一次运行的两卡模型逐字节相同，步数、目标计数和学习率严格一致。
* `validation/end_to_end.json`：最终正式结构的短时保存/加载和全部五类评测入口检查通过。这里只证明代码流程可运行，不是 2B 训练质量结果。
* `validation/time_estimate.json` / `selected_throughput.csv`：所选吞吐、候选来源、时间估计与最终代码/配置绑定；`throughput_and_time.png` / `.pdf` 为静态图。

正式产物启动后写入 `work/pretrain_100m_2b/runs/{method}_s{seed}`，汇总到 `work/pretrain_100m_2b/reports`。目前没有正式 2B 质量指标。
