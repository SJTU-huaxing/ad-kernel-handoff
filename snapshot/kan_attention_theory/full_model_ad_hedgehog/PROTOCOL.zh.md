本实验回答：在相同完整基础模型、相同替换范围、相同输入与推理实现下，当前 AD 与同参数预算 Hedgehog 哪个实际更省时间。

范围固定为 Qwen/Qwen2.5-1.5B，revision `8faed761d45a263340a0528343f099c05c9a4323`。只替换已经训练的第14、27层，共24/336个查询头，其余26层仍使用原模型的SDPA。不是全头线性化，也不复制某一层feature权重到未训练层。

对照包括原模型、AD纯删除版（每头73536参数、m64）、严格同参数AD（73728、m64）、Hedgehog-exp与Hedgehog-softmax（均73728、m576）。沿用此前验证集选定的causal-KL、seed11、lr0.002检查点；没有重新训练或按计时选择检查点。AD删除C和Q幅度、保留K幅度。Hedgehog为本项目独立Q/K投影、同参数预算扩宽适配版。

主测量为RTX3090、batch1、未填充输入，BF16基础模型与FP32 feature/state，TF32关闭。两类方法共享现有参考chunk scan（chunk64）与参考单步state实现。二者都不使用只支持旧m64架构的专用Triton feature/step算子。未替换层使用相同的SDPA后端。测量描述这些具体实现，不代表双方各自最佳融合算子。

输入长度1024、4096、8192。使用既有confirm_long前三篇文档；8192之后的64个固定续写token来自后续文档。固定续写用于让两种方法接收同样的token、位置和长度；时间包含整个模型前向与最后位置LM head，排除分词、随机采样、服务调度与网络传输。它是完整模型计算计时，不是自由生成的质量评估。

每个长度先对每种方法做两次prefill+8token解码预热，再做12个配对区组。每个区组使用相同文档对所有五种方法计时，方法顺序由固定随机种子77331生成；三个文档各出现四次。每次包含一次prefill和64步逐token解码。所有正式区组保留，无事后删点。区组前记录GPU温度、频率、功率、利用率与显存。

同时记录CUDA事件与同步wall-clock耗时。主指标为完整prefill墙钟、每个decode token墙钟及prefill+64步decode墙钟之和。墙钟包含HF执行、Python派发和GPU等待，不能直接理解为纯GPU算术时间。汇报各方法中位数，并对同文档/同区组方法的均值比做20000次区组bootstrap，给出95%区间。区间只描述本硬件本轮计时的重复性，不描述跨设备或训练种子的总体置信度。

缓存必须真实生效：替换层的DynamicCache条目使用StateOnlyLayer，不保存历史K/V；FP32状态包含S、z、数值gauge g。每次prefill与decode之后核对替换层KV storage为0、已见token数正确、线性状态字节不随解码增长，报告全模型的KV+state与prefill新增峰值显存。gauge只用于数值稳定，不改变kernel或引入遗忘。

计时前进行两项数值核验：真实模型Q/K/V上显式正kernel与scan比较；完整序列一次前向与prefill后逐步解码比较。首轮BF16短序列核验中，原模型logits相对L2差0.02658，AD纯删除版0.05367，后者超过原先0.03阈值。没有把阈值放宽后当作通过，而是追加FP32基础模型下272/8192位置的独立比较，门限1e-3。BF16差异原样记录，FP32核验后恢复原始BF16参数张量，不经重新量化。数值验证必须通过才能开始正式计时。

复现（本机已有精确基础模型、检查点与manifest）：

```bash
cd /root/autodl-tmp/kan_attention_theory
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python full_model_ad_hedgehog/benchmark.py benchmark
```

结果文件为`checks/correctness.json`、`checks/environment.json`、`results/protocol.json`、`results/raw.json`和`results/summary.json`。原始日志保存在`logs/benchmark.log`。可用`benchmark.py summarize`从原始样本重新汇总；默认重新benchmark会覆盖本目录结果，复现时应另存旧结果。
