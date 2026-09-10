# AD 与 Hedgehog 推理速度：当前约 100M 配置实测

2026-09-09 已完成实测。**当前实现没有表现出 AD 的普遍推理速度优势：Hedgehog 的带缓存逐 token 解码更快；AD 在 batch=8 的长输入 prefill 上耗时低约 6%，注意力缓存减半。AD 特征映射本身的速度优势没有得到支持。**

采用同一张 Ascend 910B3、当前约 100M 架构、BF16 主干和 FP32 特征/缓存。AD 为 m=64，Hedgehog 为官方 FLA 映射 m=128。两者使用 seed 11 新初始化权重，共同主干权重哈希相同。**本次是架构与实现的稳态计算性能对比，尚非完成 2B tokens 预训练后模型的测试。**

| 阶段 | Batch | 输入或已缓存 tokens | AD 耗时 | Hedgehog 耗时 | 比较 |
|---|---:|---:|---:|---:|---|
| Prefill | 1 | 2048 | 50.146 ms | 43.700 ms | Hedgehog 耗时低 12.9% |
| Prefill | 8 | 2048 | 123.475 ms | 131.660 ms | AD 耗时低 6.2% |
| Prefill | 8 | 8192 | 528.728 ms | 560.765 ms | AD 耗时低 5.7% |
| NPU 图执行 decode | 1 | 2048 | 2.775 ms/步 | 2.664 ms/步 | Hedgehog 耗时低 4.0% |
| NPU 图执行 decode | 8 | 2048 | 4.870 ms/步 | 4.049 ms/步 | Hedgehog 耗时低 16.8% |

Batch=1 的一步生成 1 个 token；batch=8 的一步为 8 个请求各生成 1 个 token。2048 上下文、batch=8 的图执行总吞吐约为 AD 1,643 tokens/s、Hedgehog 1,976 tokens/s。512/2048/4096 三种缓存长度的图执行解码时间基本相同。

普通 eager 执行也测了：2048 上下文下，batch=1 为 AD 39.792、Hedgehog 36.337 ms/步；batch=8 为 AD 38.710、Hedgehog 35.677 ms/步。大量小算子的主机调度显著影响普通执行。NPU 图执行把同样的计算及状态更新录制后重放，已有逐步数值校验；后续部署应优先采用这类减少调度开销的实现。

单独测第一层全部 10 个头的 Q/K 特征映射，在 NPU 图执行、FP32 下：

| Batch | tokens | AD | Hedgehog |
|---:|---:|---:|---:|
| 1 | 1 | 75.23 μs | 74.51 μs |
| 8 | 1 | 114.62 μs | 68.91 μs |
| 1 | 2048 | 378.44 μs | 149.30 μs |
| 8 | 2048 | 1,489.14 μs | 702.52 μs |

单请求单 token 的特征映射耗时接近，其余上述条件都是 Hedgehog 更快。因此不能把 AD 在部分 prefill 场景的优势解释为“AD feature map 计算更快”。AD 的特征维度较小，会减少状态和相关注意力计算；这与观测到的 prefill 优势一致，但各项具体贡献仍需分算子 profiling 才能精确归因。

每请求的 12 层持久注意力缓存：AD **1.934 MiB**，Hedgehog **3.867 MiB**，均不随上下文长度增长。这只是注意力缓存，不是模型总内存；AD 自身的模型参数也更多。

测试覆盖 prefill 512/2048/8192 tokens、decode 上下文 512/2048/4096 tokens、batch 1/8。两轮反转 AD/Hedgehog 顺序，每条件共 6 次采样，每次 decode 持续推进 64 步。输入来自相同的真实 FineWeb-Edu 留出集；包括完整模型、最后位置词表投影及缓存更新，排除采样、分词、网络服务和首次编译/图捕获。使用固定的真实后续 token 保证输入相同，不是生成质量评测。

缓存正确性校验覆盖两种方法、两种 batch：FP32 对原模型整段前向的最大绝对误差小于 3e-6；BF16 分步与整段相对 L2 差异最大 0.596%；NPU 图执行与普通逐步执行的 logits 完全相同。全部正式训练保持停止，训练配置及原训练源码没有改动，计时任务结束后两张 NPU 均已空闲。

完整明细、原始样本、环境与源码/输入/权重哈希保存在：

- [详细报告](../work/pretrain_100m_2b/inference_ad_vs_hedgehog/REPORT.zh.md)
- [图表 PNG](../work/pretrain_100m_2b/inference_ad_vs_hedgehog/latency.png) / [PDF](../work/pretrain_100m_2b/inference_ad_vs_hedgehog/latency.pdf)
- [完整模型原始测量](../work/pretrain_100m_2b/inference_ad_vs_hedgehog/benchmark.json)
- [特征映射原始测量](../work/pretrain_100m_2b/inference_ad_vs_hedgehog/features.json)
- [中位数、四分位区间及分轮结果](../work/pretrain_100m_2b/inference_ad_vs_hedgehog/summary.json)

复现命令，只进行推理校验与计时，输出到新的时间戳目录：

```bash
cd /cache/huaxing/ad-kernel-handoff
bash scripts/benchmark_100m_inference.sh
```
