# 项目结构与接续口径

研究对象是正可分离 kernel：`exp(s_K) * softmax([z_Q,0])ᵀ softmax([z_K,0])`。Q 幅度与外部 C 已删除，K 幅度保留。当前独立 Q/K 网络不是共享特征的 Mercer kernel；因果 mask 后的整张 attention 矩阵也不能直接套用 rank≤m。

| 区域 | 用途 | 接续规则 |
|---|---|---|
| 根目录原交接文档 | 研究历史、理论、资产清单、原实验口径 | 保留原文，不能把历史表格当作新运行结果 |
| `snapshot/`、`checkpoints/` | 1613 文件哈希核验的原始证据及关键权重 | 未修改 |
| `work/kan_attention_theory/` | 原源码的可运行工作副本 | 用于逐项源实现比较，其已有 JSON 仍是历史结果 |
| `src/ad_kernel/features.py` | 当前 AD 和同深度 EXP | 保持历史初始化 RNG、无偏置布局及状态键；也支持从零模型的 d64/hidden96 |
| `src/ad_kernel/baselines.py` | 独立 Q/K Hedgehog、固定 FAVOR | HH 显式报告维度/偏置；FAVOR 随机矩阵为 buffer，原 Q/K 投影仍可学习 |
| `src/ad_kernel/attention.py` | CPU/设备通用密集、递推、分块归一化参考 | 不加入窗口、遗忘门或额外 epsilon kernel |
| `src/ad_kernel/ascend_linear.py` | 已验证的混合 NPU 兼容路径 | FP32 GEMM 前缀状态 + FLA NPU 输出/梯度核；正式 v2 训练经基准比较选择了更快的 `attention.py` 原生 CANN 路径 |
| `src/ad_kernel/model.py` | 全层 decoder 骨干 | RMSNorm、RoPE、SwiGLU、共享 embedding，所有方法共有权重初始化相同 |
| `configs/` | 冻结的数据、复现与预训练协议 | 分开记录重新抽取数据的复现和从零预训练 |
| `experiments/`、`tests/` | 构建检查、提取、训练、评测入口 | 查看实际日志和哈希绑定，不以代码存在代替实验通过 |
| `work/reproduction/` | 当前机器的新复现证据 | 包含源权重比较、算子检查、真实 teacher/cache、21 模型确认结果 |
| `work/pretraining/` | 处理后的语料、从零训练与恢复状态 | synthetic 检查目录不计入正式预训练 token 预算 |

入口顺序：环境激活 → `prepare_reproduction_data.py` → 两进程 `extract_reproduction_qkv.py` → `reproduce_features.py stats` → AD/EXP 与 HH/FAVOR 拟合 → `assess_reproduction.py freeze` → 两进程确认评测 → `summarize_reproduction.py`。完整运行状态见 `EXECUTION_PLAN.zh.md`。

实际预训练由 `experiments/run_pretraining_queue.py` 启动双进程 `experiments/pretrain.py`，具体方法和预算由 `configs/pretraining_protocol_v2.json` 固定。检查点包含模型、AdamW 状态、各 rank RNG、数据顺序哈希、数据游标和实际 token 数。恢复检查使用同样的两卡执行路径，比较中断恢复与连续执行。用户已明确停止本轮训练，未经重新授权不恢复；停止时的完整说明见 `STOPPED_EXPERIMENTS.zh.md`。

理论上已成立的是尺度约消、K 幅度有效梯度、可分离状态和有限表示/统计/优化误差的区分；没有证明 AD 普遍优于同预算 EXP、达到 Schmidt 下界或具有新的通用收敛率。历史 Gaussian 代理的 24/24 heads 不满足 HS 条件，总体置信下界为 0，仍应保留这些负结果。

原实验是固定 Qwen 两层局部替换，而不是全层线性 LM 预训练。新公开数据也不是缺失的原 GPU manifest。PPL、attention KL、输出 NMSE、原始 kernel 风险及硬件吞吐分别报告；从一个指标的优势不能推出其他指标的优势。
