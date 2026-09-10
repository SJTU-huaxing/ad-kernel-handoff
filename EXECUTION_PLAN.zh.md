# 双 910B 实施计划与证据

> 当前状态：用户已于2026-09-09 04:04 UTC明确停止训练及全部自动队列，未经重新授权不恢复。已完成seed11十组、seed29七组，均为1600步/104,857,600 tokens；ELU seed29未完成。下文较早阶段的“继续运行”等表述仅为历史记录。停止快照和完整结果见 [STOPPED_EXPERIMENTS.zh.md](reports/STOPPED_EXPERIMENTS.zh.md)。

目标：理解并构建 AD 项目，复现重要历史实验，随后在 100M–2B tokens 范围完成全层线性语言模型预训练，与 Hedgehog、FAVOR+、传统 feature function 和同深度 EXP 等强控制进行训练、质量及效率对比。小型烟测不能代替该目标。

## 已冻结的研究对象

`h(q,k)=exp(s_K(k))*softmax([z_Q(q),0])ᵀsoftmax([z_K(k),0])`。
删除 Q 幅度和 C，保留 K 幅度；无隐式窗口、遗忘门、旋转增强或幅度截断。历史复现保持 d128、hidden192、m64、无偏置、73536 参数/head、24 个独立 head。

## 阶段与验收证据

1. 交接核验和阅读：1613 个清单文件哈希；主交接、迁移、当前账本及相关历史原文；原 `snapshot/` 与 `checkpoints/` 保持原始证据。
2. 构建：独立 `src/ad_kernel` 包；CPU FP64 参考、源检查点严格加载和相同初始化；NPU FP32/BF16 前向与梯度；两卡 HCCL。
3. 重要复现：Q 幅度约消和 K 幅度梯度；同深度 AD/EXP 两种初始化及 3 seeds；Hedgehog/FAVOR 历史适配；1k/8k 静态质量、局部替换 PPL；因果性、状态续接、缓存、稳定性。历史数据和新抽取数据分开报告。
4. 预训练协议：常规约 100M–150M 参数骨干作为首个规模；全层归一化线性 attention；相同 tokenizer、文本流、tokens/batch、优化器与 seed；AD/EXP、Hedgehog、FAVOR+、ELU+1 等传统特征及 softmax 参考。记录参数、m、状态、计算各预算，不能冒称所有预算同时相等。
5. 数据：选择可审计的高质量公开语料，固定 revision、文本去重和独立验证划分；100M tokens 起步并按实测吞吐/质量继续扩大到本目标内的更高 token 预算。另设新域、长度外推及检索/关联回忆评测。
6. 训练：两卡 HCCL DDP；先验证与单卡同 global batch 的等价性；checkpoint 保存优化器、调度器、RNG、数据游标和实际有效 token 数；不将重复消费冒称单遍。
7. 结果：逐文档 NLL、有效 token 聚合 PPL、配对统计和多 seed；独立计时和显存；源码/配置/数据/checkpoint 哈希；负结果完整保存。所有声明必须有当前运行产物支持。

## 当前状态（2026-09-09）

- 仓库清单检查通过：1613 文件。
- 已复制可修改历史工作区：`work/kan_attention_theory/`；不将其已有 JSON 当作新结果。
- 当前运行软件栈见 `work/target_environment.json`；两张 64 GiB Ascend 910B3，CANN 9.0.0，torch 2.7.1、torch_npu 2.7.1.post4、FLA 0.5.2。
- 原 8.744 GB QKV 数据和原始 token/划分 manifest 不在仓库；已询问可用位置，同时推进不依赖该数据的核验和公开数据重建。缺该资产时不会声称原表格的逐样本精确复现。
- 固定 revision Qwen2.5-1.5B 已下载并通过原 10 文件 SHA256 清单。两卡完成新公开数据 137 个 post-RoPE 分片（约 6.6 GB），各分片绑定数据清单并通过哈希、形状、有限值和 token/位置一致性检查。
- CPU FP64 attention 8 项检查通过；12 个 AD/EXP 检查点的源实现/新实现特征和参数梯度精确一致；9 个 HH/FAVOR 检查点 log 矩阵精确一致。新证据在 `work/reproduction/`。
- 新公开数据 AD/EXP 两种初始化、两档 seed 11 学习率及选定学习率的 3 seeds 已完成（16 次拟合），四组均由前 32 验证文档选出 lr=0.002。尚未访问确认集指标；HH-exp/softmax 及固定 FAVOR 复现正在运行。
- FLA 0.5.2 原生纯线性 chunk_h 在本机出现中间状态错误（FP32）和编译失败（BF16），不可用于正式训练；3.2.2 + torch_npu post8 独立候选也编译失败，主环境保留 3.2.1 + post4。项目 `ascend-hybrid` 使用 FP32 GEMM/前缀状态及 FLA NPU 输出/反向分块核，正在扩大输出、梯度、初末状态和不同 m 的验证范围。
- FineWeb-Edu 固定 revision 的四个 parquet 已下载；GPT-2 tokenizer 已固定 revision 并验证 Git blob。按冻结数据协议做质量过滤、精确及近似去重、按 hostname 隔离划分，目标准备约 2B 训练 tokens 和各 2M 验证/测试 tokens。数据准备尚在运行，不能把 partial 文件作为完成的数据集。
- 预训练尚未开始，最终模型、完整预算及所有对比尚未完成。

### 后续进展

- 21 个选定复现模型（AD/EXP 两种初始化、HH-exp/softmax、固定 FAVOR，各 3 seeds）已完成。冻结清单 `work/reproduction/public_wikitext_v1/frozen_evaluation.json`；正式 1k/8k CPU FP64 静态指标和 NPU 局部替换 PPL 正在两卡运行。最初接口对显式因果 mask 的处理已修正，旧尝试产物保留在父目录，当前有效结果按评测源码哈希分目录保存。
- `ascend-hybrid` 在长度 63/64/65/129/1024、m64/128/256/384、padding 与初末状态梯度上通过 CPU FP64 对照。FP32 梯度相对 RMS 误差最高约 2.2e-6；BF16 最高约 3.7%。正式预训练将特征和 attention 保持 FP32，骨干使用 BF16 autocast。
- FineWeb-Edu 数据处理完成：训练 2,000,100,000 tokens（1,934,765 文档），验证/测试各 2,000,000 tokens。规范化 hostname 集合在划分之间完全分离；精确重复、近似重复和高频桶过滤分别记账。清单 SHA256 为 `9976b25b1d3e54c50d7e20b926be6cd1c16fd52aae28e999629639cacd4d0d96`。
- 统一骨干已构建：12 层、width768、12 heads、FFN2048、RMSNorm、RoPE、SwiGLU、共享输入/输出 embedding。AD 总参数 127,076,352，其中 feature 参数 3,525,120；所有方法的共有权重初始化逐元素一致。CPU 模型因果性、反向和固定 FAVOR 参数检查 9 项通过。
- 两卡 HCCL 与同 global batch 单卡参考的梯度相对 RMS 误差约 1.78e-7，AdamW 更新最大差约 2.47e-6。
- NPU 融合 LM loss 已验证含忽略标签、非单位上游梯度、真实 50,257 词表；loss 差 <1e-6、梯度差约 0.25%–0.31%。AD 和 HH m384 的 16×1024 真实尺寸训练步骤通过，单进程峰值分别约 20/37 GiB；并发质量任务期间的耗时只作容量诊断。
- 预训练协议在 `configs/pretraining_protocol_v1.json`：10 方法，seed11 到 1.074B tokens，另两 seeds 到统一 104.858M 里程碑，全部使用同一完整预算学习率日程。该设计明确区分三种子的早期结果与单种子的最终预算。当前只在独立 synthetic 目录验证中断恢复，尚未开始正式语料训练。

### 正式训练启动

- 21 模型的确认评测已全部完成，报告 `reports/ASCEND_REPRODUCTION.zh.md`。两个初始化下 AD 的 8k KL 相对 EXP 降低约 12%–17%，PPL 降低约 0.34%–0.39%；3 seeds 的长文方向一致，1k PPL 没有稳定优势。Hedgehog 的 8k KL 更低，但 PPL 仍较 AD 高；原 softmax 教师仍最好。
- 独占卡、相同权重和输入、反转顺序的后端基准发现：CANN 原生分块比混合 Triton 路径快约 1.36–1.37×。正式协议在训练开始前据此采用 `torch` 分块后端，仍使用 FLA Ascend 融合词表交叉熵。精度和成本详见 `reports/TRAINING_CONFIGURATION.zh.md`；原后端协议草案保留在 `work/protocol_drafts/`。
- 原生分块的两卡恢复检查通过：连续 4 步与 2 步后恢复至 4 步的所有参数、AdamW、RNG 和数据游标精确相同，证据 `work/reproduction/resume_check_torch.json`。合成检查不计入正式 tokens。
- 已启动 `experiments/run_pretraining_queue.py`，队列日志 `work/logs/pretraining-queue.log` / `work/pretraining/queue.jsonl`。先完成各方法 seed11 的共同早期里程碑，再做其他种子，然后继续主种子到完整预算。当前从 softmax 参考开始；总体目标仍未完成。

后续持续以实际进程、日志和检查点更新本文件及 `work/reports/`；不因阶段完成而标记整个目标完成。

### 数值故障修复与第二版正式队列

- 第一版 softmax 完成 1,600 步，验证 PPL 57.8074；随后 AD 在第 348 步触发非有限梯度检查。出错步骤未更新权重，队列停止。初版源码、配置、日志和两条训练轨迹已归档到 `work/pretraining/archive/protocol_v1_global_scaling/`，不混入第二版对比。
- 故障已从保存状态重现并提取单 head 回归输入。因全序列最大 K 缩放导致早期分母约 `1.48e-44`，CPU/NPU FP32 均失败；不是仅换 NPU 精度配置即可解决。数值分段、完整状态及梯度续接修复后，真实 head 的输出/梯度通过 FP64 对照，完整故障 batch 的所有参数梯度有限。详见 `reports/NUMERICAL_STABILITY.zh.md`。
- 修复版 36 项 CPU 测试、实际 NPU 回归、全模型 batch、缓存及双卡断点恢复检查通过。恢复比较的参数和 AdamW 最大差均为 0，证据 `work/reproduction/resume_check_stable_v2.json`。该证据是所测四步的结果，不承诺所有长运行逐位确定性。
- 修复版固定权重和独占卡基准仍支持原生 CANN 后端：随机 AD 约 0.381 秒、HH m384 约 0.973 秒；实际故障前 AD 权重和输入含数值分段约 0.415 秒，均快于混合 Triton 路径。
- 正式协议更新为 `configs/pretraining_protocol_v2.json`（SHA256 `8d22d8fa06dbbfad4a71fcb886cac322ea3fabcdb8231e4ba7a5eb6f315aebe4`）。数学核、模型、数据、初始化、优化器和 token 预算不变；各方法统一重启。队列先运行 AD，再运行 EXP 和其他控制；每个固定里程碑后进行双卡完整质量评测。
- 测试输入已准备完成：主测试近 2M tokens、64 篇连贯长文（同篇 1k/4k/8k 前缀）、62 篇 WikiText raw 测试页（286,158 tokens）及 216 个固定关联回忆提示。协议及限制见 `configs/pretrained_evaluation_v1.json`。当前阶段汇总在 `reports/PRETRAINING_RESULTS.zh.md`，整体目标仍在进行。
- 第二版 AD seed11 已完成 1,600 步、104,857,600 tokens，验证 NLL 4.40472078、PPL 81.8363。末段训练吞吐约 41k tokens/s（两卡，包含数值分段），峰值分配约 22.5 GiB/rank。当前正进行该固定检查点的完整测试；尚不能据此宣称对其他线性基线有优势。主种子之后仍需继续到 1.074B。
- AD 的首个完整固定预算测试已完成（当前 1/40 个预定评测检查点）。主测试 PPL 68.4765，WikiText 256.8189，同篇长文 1k/4k/8k 为 86.7531/335.9628/447.5376，长度外推明显退化。关联回忆没有显示可靠优势；缓存续接通过，实际缓存 2,433,024 字节。结果在 `work/pretraining/results/ad64_s11_step001600/02ae37d5258c/`，汇总报告 `reports/PRETRAINING_RESULTS.zh.md`。EXP seed11 已接续训练；尚无同预算线性基线成绩可据此作优劣结论。
- 最终推理性能队列 `experiments/run_inference_benchmarks.py` 已启动等待，只有全部训练及质量评测完成后才会独占两卡测量最终模型的 B1/B4、1k/4k/8k 预填充与 128-token 解码。该队列仍未执行正式计时，不能把排队等同于效率评测完成。

### 相同目标文本的补充诊断

- 原长文表比较同篇文档的不同长度前缀，目标文本也不同，不能把 PPL 差值全部归因于上下文长度。为区分目标难度、上下文与绝对位置影响，另冻结 `configs/context_control_v1.json`：全部条件预测原 64 篇长文完全相同的 token 索引 7169..8192，即固定 8k 评测前缀的最后 1,024 个目标 tokens，并非整篇原文的结尾。
- 诊断包含 1k/4k/8k context 的位置重置条件，以及 1k/4k context 保留原文 RoPE 位置的条件。只对最终 1.074B、seed11 的十种方法运行，不参与任何训练或模型选择。该协议在观察到 AD 的早期前缀结果后制定，明确作为探索性补充，不能冒称最初预注册的主指标。
- 目标索引、零位置偏移与冻结前向的一致性、softmax RoPE 共同位置偏移、末段 NLL 对齐等 8 项 CPU 检查通过；配对区间方向、常数效应和 hostname 整组重采样 3 项检查通过。NPU 最终模型诊断尚未运行。
- 补充队列 `experiments/run_context_controls.py` 将等待正式推理计时完成后再使用两卡，结果汇总到 `reports/CONTEXT_CONTROLS.zh.md`。整体预训练/对比目标仍未完成。

### 首个同预算 AD/EXP 比较

- 第二版 EXP seed11 也已完成 1,600 步及全部九个质量产物，当前完整质量检查点为 2/40。EXP 的训练耗时 2,021.0 秒（训练进程内记录，含验证和保存；启动队列记录为 2,067.9 秒），验证 PPL 82.0685。其最后十步吞吐约 39.7k tokens/s，单 rank 峰值分配约 22.1 GiB。双卡已接续 softmax 对照训练。
- AD/EXP 主测试 PPL 为 68.4765/68.8534，AD 在当前 seed11 的相对 PPL 降低约 0.547%；WikiText 为 256.8189/264.1524。WikiText 的 EXP−AD NLL 差约 +0.028155，条件于本种子的配对文档 bootstrap 95% 区间为 [+0.018364, +0.038147]。
- 同篇长文的 1k/4k 两者几乎相同；8k PPL 为 AD 447.5376、EXP 451.6553，EXP−AD NLL 差约 +0.009159，但 hostname 整组区间为 [−0.011867, +0.030971]，仍跨过零。当前不能确认 AD 的稳定长文优势，且这不是三个种子或最终预算的结论。两者均通过缓存续接检查，B1 全 12 层缓存均为 2,433,024 字节。
- `experiments/summarize_pretrained.py` 现自动生成 `reports/AD_COMPARISONS.zh.md`，只使用双方都完成的相同种子，分别列主测试、迁移和各长文前缀的比较。增加了正式检查点集合、实际 token 数及训练数据清单绑定核对。主测试相对 PPL 已独立对照检查点 PPL 比率确认。训练及评测源码保持冻结，阶段图已更新；整个目标仍未完成。

### Softmax 早期强对照与最终汇总核验

- 第二版 softmax seed11 完成同一 104,857,600-token 里程碑及全部九项质量产物，当前完整质量检查点 3/40。验证 PPL 58.1927；主测试 49.3610、WikiText 120.9612、同篇长文 1k/4k/8k 为 59.5816/139.0283/213.5236。当前这个种子上明显优于 AD/EXP，结果完整保留，不据此改变原定训练或评测预算。
- softmax 的训练进程内耗时 729.2 秒（含验证和保存），队列含启动记录为 775.0 秒。末段双卡训练吞吐约 155.9k tokens/s，峰值分配约 12.0 GiB/rank。它的 attention 为 BF16 原生 SDPA，线性方法为 FP32；这些训练观测不是最终独占推理基准。缓存续接通过，B1 的 1024-token KV cache 为 37,748,736 字节，随长度增加。质量评测耗时约 44.0 秒，不作为模型吞吐。
- 双卡已继续 Hedgehog-softmax m128 训练。最终汇总进程 `experiments/run_final_reports.py` 已启动，等待上下文诊断队列完成后再运行产物审查和图表导出；实际监督 PID 与进程创建时间均核对，前序失败时停止。
- 新增 `experiments/audit_pretraining_artifacts.py`，核对原始 1,613 文件、训练/评测数据实际字节、公开复现来源及权重、正式固定预算检查点和质量/效率/上下文同一权重绑定，以及完整优化器/RNG/数据游标。当前 AD/EXP 的已保存早期状态通过核对，但仍不计为主种子的完整预算。产物覆盖审查与科学结论/整个任务完成审查分开。
- 核验及相关统计、计时测试共 12 项通过，包含缺失项、重复 seed 和错误预算不能冒充完整结果的检查。汇总脚本加进程锁，避免手动核验与运行队列同时写入。训练及冻结评测源码未修改。

### 实际训练成本记录

- 增加 `experiments/summarize_training_cost.py` 与 `reports/TRAINING_COST.zh.md`，按真实 start/resume 调用分别记录实际更新区间、训练循环总时长、阶段平均吞吐、rank0 日志峰值内存及数值分段计数。初次阶段的 AD/EXP/softmax 耗时分别为 33.508/33.684/12.154 分钟，包含中途验证及保存，不含启动前准备和独立质量评测。
- 训练循环阶段平均吞吐分别约 52.2k/51.9k/143.8k tokens/s；这是完整阶段观测，与末段稳态吞吐含义不同。AD/EXP 截至第 1600 步训练日志的 rank0 数值分段累计为 127,708/127,493，softmax 为 0。实际数值执行的开销保留在比较中，未通过截断状态或改变数学核规避。
- 每次恢复仅计算本次更新的 tokens；被中断或未结束的调用单独标记，重放区间不冒充不同训练数据。4 项专门测试通过，包括 1600→16384 只计增量、重放不与前次混并、到达 target 的训练日志不等于完成、错误完成标记拒绝。最终产物审查会自动重新生成并绑定训练成本报告。Hedgehog-softmax 的双卡训练继续正常推进。

### 十种方法的首轮完整对比

- 队列已完成全部十种方法的 seed11 早期评测，以及 seed29 的 AD、EXP、softmax、HH-softmax m128、HH-exp m128、FAVOR+ m256，当前完整质量检查点 16/40。实际双卡正在训练 FAVOR+ m64 seed29；未因交互等待被打断而重启训练。
- 同一 seed11、104.858M tokens 下，主测试 PPL 从低到高为：softmax 49.3610、HH-exp m384 66.5009、FAVOR+ m256 66.9378、HH-softmax m128 68.2490、AD 68.4765、EXP 68.8534、FAVOR+ m64 69.0455、HH-exp m128 69.8839、softplus 100.3877、ELU+1 103.2330。这是一个种子的早期排序，不是最终预算排序。
- 已有的两个匹配种子上，AD 相对 EXP 的主测试几何平均 PPL 降低约 0.934%；对 HH-softmax m128 的方向随种子变化，平均差约 0.055%，非常接近。FAVOR+ m256 的主测试优于 AD；softmax 仍明显领先。AD 的 8k PPL 优于 EXP，但落后于其他已测强基线，不能据 AD/EXP 单独比较宣称整体最佳。
- 新增 `reports/EARLY_QUALITY_COST.zh.md` 和 `early_quality_cost_seed11.png/svg/json`。固定同一完整种子，将质量与实际 1k B1 缓存、参数和训练循环时间按同一 checkpoint SHA256 联结。AD 缓存为 2.320 MiB；HH-exp m384 为 13.922 MiB，FAVOR+ m256 为 9.281 MiB，softmax 为 36 MiB。该图不是最终推理基准，所有方法的状态精度与计时范围明确标注。
- 当前产物审查已核对 16 个完整质量检查点和对应权重、16 个已结束训练调用、6 个达到各自目标预算的恢复状态，以及原始/公开复现/数据来源。主种子全部仍待继续至 1.074B，剩余种子和最终效率、上下文诊断继续按既定队列执行。
