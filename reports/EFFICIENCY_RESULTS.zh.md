# 最终模型的推理质量与效率

已完成 0/20 项预定独占设备基准（十种方法，各 B1/B4）。未完成时，本表仅是阶段结果。

全部为 seed11、1.074B tokens 固定检查点。骨干为 BF16 autocast、FP32 主权重；线性 feature/状态/attention 为 FP32，softmax SDPA 及 K/V cache 为 BF16。没有权重量化或图捕获。

下表展示 8k context。预填充含最后一个位置的词表投影及紧凑缓存构建；解码是固定输入 tokens 的 128 次连续单步计算，包含词表投影，计时不含预填充、采样及 tokenizer。所有数字取 3 次预热后 7 次计时的中位数。

## Batch 1

| 方法 | 主测试 PPL | 参数量 M | 预填充 ms | 解码 ms/step | 解码 tokens/s | 缓存 MiB | 解码峰值 GiB |
|---|---:|---:|---:|---:|---:|---:|---:|

## Batch 4

| 方法 | 主测试 PPL | 参数量 M | 预填充 ms | 解码 ms/step | 解码 tokens/s | 缓存 MiB | 解码峰值 GiB |
|---|---:|---:|---:|---:|---:|---:|---:|

缓存按实际保留 tensor storage 计数，含线性数值缩放；峰值是 PyTorch allocated memory，包含权重和临时张量，不能与仅缓存大小混用。reserved memory 和各次原始时长另行保存。

1k/4k/8k 全部记录、样本时长、数值分段次数、设备快照及数据/checkpoint/source 哈希见 `work/pretraining/reports/efficiency.json` 及其原始产物。测量只代表本次输入、执行实现和两张 910B；最后预算只有一个训练种子，不证明普遍最优。
