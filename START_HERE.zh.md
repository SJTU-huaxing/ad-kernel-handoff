**AD幅度—方向正kernel项目交接入口，研究快照2026-09-08，打包核验2026-09-09**

用户已明确确认目标：**去除外部C和查询Q幅度，保留K幅度**。不是删除K幅度。接收设备有两张昇腾910B；本机没有NPU，本包不包含已经完成910B迁移或预训练的主张。

文档日期采用会话提供的日期，审计中的`source_host_utc`单独保留源主机时钟读数；两者存在日期差异，不用该时钟倒推实验先后。

接收方先依次阅读：

1. [完整项目交接](AD_PROJECT_FULL_HANDOFF.zh.md)：研究目标、版本、全部阶段、理论边界、实现与实验结论。
2. [双910B迁移与预训练接续](ASCEND_910B_MIGRATION.zh.md)：设备核查、后端改造、精度与反向验证、双卡预算、后续实验。
3. [当前结果账本](CURRENT_RESULTS.zh.md)：直接从审计后的JSON生成的数值表，包含不同初始化、Hedgehog/FAVOR、时延和配对统计。
4. [下一位Codex的接续说明](NEXT_CODEX_PROMPT.zh.md)：可以直接粘贴到新设备对话中。
5. [原文索引](SOURCE_INDEX.zh.md)及[历史报告/协议/理论全文](HISTORICAL_REPORTS_FULL.zh.md)：保留历史各阶段的完整表格、负结果、证明和修订。

若只携带一份供阅读的文档，使用[完整合订总册](AD_PROJECT_ALL_IN_ONE.zh.md)：已合并上述主文档、迁移方案、结果账本、接续提示及40份历史原文。接收端可依据[分段行号](READING_MAP.json)分块阅读，避免一次读取被工具截断。进行工程复现仍需整个源码包和所需资产。

当前已验证kernel：

\[
\pi_Q(q)=\operatorname{softmax}([z_Q(q),0]),\quad
\pi_K(k)=\operatorname{softmax}([z_K(k),0]),\quad
h(q,k)=e^{s_K(k)}\pi_Q(q)^\top\pi_K(k).
\]

Q网络128→192→63，K网络128→192→64，SiLU、无偏置；K前63维为方向、最后一维为log幅度。m=64，每个head共73536参数。它是连续正特征，不是分区查表，不是KAN，也没有旋转增强/窗口/遗忘门作为默认组成部分。

当前证据：冻结Qwen2.5-1.5B的两层共24/336 heads，单遍feature拟合。在同深度、同参数、同m和匹配初始attention的控制下，AD比直接EXP-MLP的8k KL低约15%–16%，局部替换PPL低0.33%–0.43%；1k PPL略差，参考feature+state步骤慢约10.7%。这些结果支持进入小规模完整预训练，不等于已经证明普遍理论优势、全模型加速或接近总体下界。

本目录的`snapshot/kan_attention_theory/`是原项目文档、源码、汇总和小型证据的只读副本。`checkpoints/kan_attention_theory/`另有21个关键检查点，随独立检查点压缩包分发。**大型QKV缓存、基础LLM权重及历史大矩阵没有包含在文档包内。** 精确范围见[文件清单](ARTIFACT_INVENTORY.json)和[外部数据指纹](EXTERNAL_DATA_MANIFEST.json)。缺数据时可以阅读全部研究结论；完整复现实验需要另行迁移清单中的资产。

两个压缩包均解出同一个`AD_kernel_handoff_20260908/`根目录，先解文档源码包，再解检查点包即可合并。文档源码快照1377个文件约73.4MB，检查点21个文件约130.7MB（均为未压缩大小）；外部最终数据8.744GB和基模型3.099GB不在这两个包中。压缩包实际大小和SHA256另见同级`AD_kernel_handoff_DELIVERY_20260908.json`与`AD_kernel_handoff_20260908.sha256`。

在目标设备的接收目录执行：

```bash
sha256sum -c AD_kernel_handoff_20260908.sha256
tar -xzf AD_kernel_handoff_documents_20260908.tar.gz
tar -xzf AD_kernel_handoff_checkpoints_20260908.tar.gz
cd AD_kernel_handoff_20260908
python verify_bundle.py --require-checkpoints
python inspect_target.py > target_environment.json
```

只解了文档包时运行`python verify_bundle.py`，将显式显示检查点和外部资产未校验；若检查点只解了一部分则校验失败。外部数据和基模型另迁移后，可以加`--project-root /新位置/kan_attention_theory --model-root /新位置/Qwen快照`验证全部数据/基模型文件。

校验脚本仅核验文件与指纹，不会训练、下载或安装；设备检查脚本只读取设备/软件信息。旧设备的[交接审计](HANDOFF_AUDIT.json)与逐文件[BUNDLE_MANIFEST.json](BUNDLE_MANIFEST.json)随包保存。原历史源码带有旧设备绝对路径、CUDA调用和“已有结果则跳过”的逻辑，不能把直接运行成功但复用了JSON当成NPU复现实验。

`build_snapshot.py`、`capture_source_environment.py`、`write_current_results.py`、`finalize_handoff.py`是旧设备制作交接包的脚本，**不属于接收端常规执行入口**；不要在新设备上重写原快照/审计。接收端先运行校验与设备检查，再按迁移文档创建新的工作目录。
