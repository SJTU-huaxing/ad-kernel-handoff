# Ascend 项目运行入口

当前状态：用户已明确要求停止训练。训练、评测及后续自动队列均已停止，记录在 `work/pretraining/stopped_by_user.json`。在用户重新授权恢复前，不启动训练。当前已完成 seed11 十种方法、seed29 七种方法的 104.858M-token 评测；详见 `reports/STOPPED_EXPERIMENTS.zh.md`。

先阅读 `reports/PROJECT_MAP.zh.md` 和 `reports/ASCEND_REPRODUCTION.zh.md`，分别了解交接项目和已完成的公开数据复现。原 `snapshot/`、`checkpoints/` 及其 1,613 文件清单保留原样；`work/kan_attention_theory/` 是可修改的历史工作区，其中原有 JSON 不能当作当前新实验结果。

## 环境与当前状态

```bash
cd /cache/huaxing/ad-kernel-handoff
source /cache/huaxing/miniforge3/etc/profile.d/conda.sh
conda activate nonlinear-qk
python experiments/status.py
```

Jupyter 选择 `nonlinear-qk (Python 3.11, Ascend NPU)`。版本、kernel 启动方式和已知兼容问题详见 `/cache/huaxing/nonlinear-qk-setup/README.md`。环境激活会加载 CANN 并设置所需库路径，保留平台给出的 NPU 物理可见性映射。

`status.py` 读取实际进程、当前运行日志及汇总文件。seed11 的目标是完整 1.074B tokens；另两个 seed 的目标是 104.858M。主种子达到早期里程碑时仍有后续训练，不能按已经完成整个预算理解。最终完成还需核对原始训练、评测和计时产物。

## 代码与数据

- `src/ad_kernel/features.py`：交接的纯 AD/同深度 EXP 定义，无 Q 幅度和 C，保留 K 幅度。
- `src/ad_kernel/baselines.py`：可学习 Hedgehog 和固定随机矩阵 FAVOR+。
- `src/ad_kernel/attention.py`：完整因果状态与梯度、原生分块、必要时的数值分段。
- `src/ad_kernel/model.py`：12 层、width768、12 heads、SwiGLU2048、RMSNorm、RoPE、共享词嵌入的全层 LM。
- `configs/pretraining_protocol_v2.json`：当前冻结训练协议。模型、数据、初始化、优化器和预算说明见 `reports/TRAINING_CONFIGURATION.zh.md`。
- `work/pretraining/data/fineweb_edu_v1/manifest.json`：约 2B 个训练 tokens 的来源、过滤、去重、hostname 划分和哈希。所有方法在相同 seed 下使用相同的无放回目标块顺序。
- `work/pretraining/evaluation/`：主测试之外的固定长文、WikiText 与关联回忆输入。

正式训练采用 CANN 原生分块和 FLA NPU 融合词表交叉熵。FLA 原生纯线性 chunk 的中间状态存在本机兼容问题。第一版训练又暴露了全局缩放的极小分母问题；修复、证据和开销见 `reports/NUMERICAL_STABILITY.zh.md`。修复后的所有方法从相同初始化重新开始，初版轨迹保存在 `work/pretraining/archive/protocol_v1_global_scaling/`。

## 运行队列与恢复

当前队列依次执行：三种子的共同早期里程碑、主种子完整预算，以及每个检查点后的两卡质量评测。独占推理计时等待全部训练与主评测完成，探索性上下文诊断再等待计时完成。

`experiments/status.py` 会列出活动 PID。训练队列有进程锁；恢复时使用同一个入口即可验证已完成步骤并续接检查点：

```bash
python experiments/run_pretraining_queue.py
```

恢复包括模型、AdamW、各 rank RNG、数据顺序哈希与游标。任一训练/质量失败会停止队列。训练代码或协议与检查点哈希不符时会拒绝续接；不要通过删除绑定检查把不同实验合并。

活动训练期间保留 `src/ad_kernel/*.py`、`experiments/pretrain.py`、训练协议及已冻结评测源码。独立诊断写在 `experiments/` 下并注明范围。`work/pretraining/frozen_v2/` 是启动时的源码快照，各运行自身保存的绑定仍是核对依据。

日志入口：`work/logs/pretraining-queue.log`、`work/pretraining/queue.jsonl`、`work/logs/inference-benchmark-queue.log`、`work/logs/context-control-queue.log`。各模型训练和各评测 shard 有单独日志。完整权重位于 `work/pretraining/runs/{method}_s{seed}/model_step*.pt`；`last.pt` 另含恢复所需优化器等状态。

## 结果与重新生成报告

```bash
python experiments/summarize_pretrained.py
python experiments/summarize_efficiency.py
python experiments/summarize_context_controls.py
python experiments/summarize_training_cost.py
python experiments/render_pretraining_figures.py
```

所有质量、独占推理计时和补充上下文诊断结束后，`experiments/run_final_reports.py` 自动重新汇总，并执行 `experiments/audit_pretraining_artifacts.py --require-complete`。其独立进程等待补充队列完成，同时核对实际被监督的 PID；不会因等待超时自行重启训练。

也可执行 `python experiments/audit_pretraining_artifacts.py` 核验当前产物快照。核验会读取实际数据及 checkpoint 字节、重新生成汇总，核对 40 个质量检查点、20 个计时任务、10 个上下文诊断和 30 个达到各自目标预算的恢复状态。早期已经保存的 seed11 状态会被验证，但尚不算达到最终预算。产物写入 `work/pretraining/reports/artifact_audit.json`，完整性标记不替代科学解释或整个任务的完成审查。

- `reports/PRETRAINING_RESULTS.zh.md`：固定预算主测试、长文、WikiText、关联回忆及配对统计。
- `reports/AD_COMPARISONS.zh.md`：AD 与各控制的同预算、相同种子集合比较；主测试相对 PPL 与文档/hostname 配对区间。
- `reports/EFFICIENCY_RESULTS.zh.md`：最终检查点的质量、预填充、解码、缓存及峰值分配；原始计时保留全部样本。
- `reports/TRAINING_COST.zh.md`：每次实际 start/resume 调用的训练成本，按本次更新区间计 tokens，单独记录被中断或尚未结束的调用。
- `reports/EARLY_QUALITY_COST.zh.md`：十种方法在 seed11、104.858M tokens 的同预算质量/参数/实测 1k 缓存/训练时间表；对应 PNG/SVG 和逐点来源在 `work/pretraining/reports/figures/early_quality_cost_seed11.*`。
- `reports/CONTEXT_CONTROLS.zh.md`：同一 8k 评测片段的固定末尾目标，在不同上下文/位置偏移下的探索性诊断。
- `work/pretraining/reports/`：机器可读汇总、来源绑定和 PNG/SVG 图。图是生成时的进度快照。

质量汇总只合并完整、来源一致的 shard；不把已有目录、排队任务或合成恢复测试当作完成的正式实验。第一版结果、当前早期三种子结果、最终单种子结果分开解释。

长文前缀 PPL 会同时受到目标文本和位置/上下文的影响；补充诊断固定目标 tokens。WikiText 风格迁移测试的精确文档过滤不能排除全部近似重叠。文档和 hostname bootstrap 区间均条件于实际列出的训练种子。不同 tokenizer 的 PPL 不直接比较，Qwen 局部替换复现不等于从零训练的全层 LM 成绩。


用户停止训练后授权的AD/Hedgehog速度与FLA可用范围复核已完成，见 [FEATURE_SPEED_AND_FLA.zh.md](reports/FEATURE_SPEED_AND_FLA.zh.md)。FLA recurrent及小型官方模型存在通过验证的路径；chunk/fused_chunk在所测配置失败，不能据此概括全部FLA。AD的feature生成与状态计算优势分别报告。此次没有恢复训练，停止标记继续生效。
