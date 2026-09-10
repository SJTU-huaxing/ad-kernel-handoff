# 固定目标 token 的上下文与位置诊断

已完成 0/10 个最终预算模型（seed11、1.074B tokens）。未完成前只作阶段报告。

此补充协议在看到 AD 的早期前缀测试后制定，属于探索性诊断。它不参与模型、训练预算或超参数选择。

全部条件都预测同一批 64 篇长文中，固定 8k 评测前缀的最后 1,024 个目标 tokens（原文 token 索引 7169..8192）。这不是整篇原文的结尾。reset 表示输入片段从 RoPE 位置 0 开始；document positions 表示保留其在原文中的位置。每项均从空状态开始，模型权重不变。

| 方法 | 1k reset PPL | 4k reset PPL | 8k full PPL | 1k document positions PPL | 4k document positions PPL |
|---|---:|---:|---:|---:|---:|

同 context 长度下 reset 与 document positions 的差异用于诊断绝对位置敏感性；保留 document positions 后与 8k full 比较，用于观察移除较早上下文的影响。相比前缀表，这里不存在目标 token 不同造成的文本难度差异。

逐文档配对差异、文档及 hostname 整组 bootstrap 区间保存在 `work/pretraining/reports/context_controls.json`。区间仅条件于这一个训练种子，不能用于宣称对所有训练随机性或模型规模普遍成立。
