以下只基于真实模型配置、已完成的局部/算子计时和CPU算术估算，没有运行全head替换、模型前向或新GPU实验。对象是本轮连续Q/K幅度—方向kernel，m64、两侧128→192→64，每Q head独立映射；原Qwen2.5-1.5B的28层、每层12Q heads/2KV heads、d128保持。必须真正移除HF原KV cache路径，才能得到这里的固定状态结果；仅注册新的attention回调仍可能保留原cache。

**算术量。** 每次乘加算2 FLOPs，忽略激活、softmax、RoPE、分母标量操作等。原模型Q/K/V/O、FFN和最终词表投影共3.08714 GFLOPs/token。新特征双侧每head147456 FLOPs，状态更新/读取每head约32768 FLOPs，336heads共0.060555 GFLOPs/token。

原模型decode约 C_old(T)=3.08714+0.000172032 T GFLOPs/token；全线性约 C_new=3.14769 GFLOPs/token。这是保留所有原投影和FFN的比较，不是只算attention部分。

|上下文|原decode GFLOPs/token|新decode GFLOPs/token|算术量变化|原KV MiB|新状态 MiB|
|---|---|---|---|---|---|
|1024|3.2633|3.1477|−3.5%|28|10.582|
|4096|3.7918|3.1477|−17.0%|112|10.582|
|8192|4.4964|3.1477|−30.0%|224|10.582|
|32768|8.7243|3.1477|−63.9%|896|10.582|

理想线性prefix聚合下，8192-token prefill总算术量从27.240 TFLOPs降到21.963 TFLOPs，下降19.4%；当前block64聚合的矩阵额外开销后约22.030 TFLOPs。这里prefill只对末token做词表投影，和计时协议一致。并行scan的数据搬运和激活代价不在这些FLOPs内。

**状态和权重。** 每head FP32状态m(d_v+1)，全模型共10.582MiB。新特征参数共24815616，FP32权重94.664MiB。假定没有其他缓存差异，8K时这两项替代原224MiB KV，净节省约118.754MiB，而不是总显存缩小21倍。1K时额外特征权重反而使这部分总显存增加约77.246MiB。若将来共享同组K映射或换精度，预算会改变，但那不属于本估算。

简化的每token读写模型：BF16原dense权重约3.087GB读一次，原KV各元素读一次；新权重读一次，新线性状态读/写各一次。8K总字节比约0.966，即只少3.4%，远小于30%的FLOP降幅。它只是乐观流量模型，不是硬件计时预测；实际GQA复用、cache、KV拷贝及算子启动会改变流量和延迟。

**decode时间：只能给条件情景，当前数据不能唯一识别。**

T_new = T_old − T_removed_attention/cache + T_features + T_linear_state/glue。

旧原模型RTX3090、batch1、8K下约22ms/token。被移除的attention/cache路径没有独立profile，因此不能从四head替换的+2～4ms按84倍推断全模型，更不能直接套用30% FLOP减少。

一个有明确来源但仍偏乐观的工程情景：旧普通MLP（同样的两层尺寸）的四head Q/K特征在CUDA Graph中为0.047104ms。假设每层12heads合并后的时间是四head测量的1～3倍，则28层新增特征约1.32～3.96ms/token。再假设移除attention/cache省0～3ms、线性状态及衔接花0.2～1ms，就得到20.52～26.96ms/token。可粗略记为21～27ms，对比22ms约−5%～+23%。这不是置信区间，也不是全head实测；旧softplus MLP与新幅度—方向的非线性不同，1～3倍扩展和另外两项都是待验证假设。宽松预算可留20～30ms。

该情景只对新增特征路径做按层合并/CUDA Graph，保留其余模型原执行方式。如果同时优化整个模型，原模型也必须采用相同优化再比较。

直接采用eager、逐head调用当前代码不满足上述假设，可能明显更慢。作为开销示例，旧四head合并MLP特征的eager计时0.559584ms，在每层12heads为其1～3倍的情景下，仅新增特征就约15.7～47.0ms/token，尚未加其余模型。它说明实现方式足以改变结论，不能将算法线性复杂度直接等同于低延迟。

**prefill应单独估计。** 旧四head、8192tokens、同尺寸MLP特征CUDA Graph为1.1694ms；按84倍head数线性扩展，新增特征约98.2ms。原模型8192-token prefill约471ms；已有三点曲线中的二次项约50ms（只是曲线代理，没有单独profile）。按此代理移除50ms、增加98ms，得到约519ms，再加新的线性scan。因而沿用当前FP32特征实现，prefill可能仍慢，即使算术量下降19%。新的融合、混合精度或更高效scan可以改变此结果，目前没有可靠的全head毫秒值。

所以对“增加多少”的当前回答应是：合理合并/图捕获实现下，8K decode可暂按约持平至慢20%左右做情景预算，并保留略快可能；当前逐head原型不能套用这个预算。32K以上更有机会从移除长历史读取中获益，但没有实测支持具体加速倍数。所有数字均不构成全head替换的模型质量保证。

计算脚本：estimate_all_heads.py；逐项数值与假设：results/all_heads_mlp_estimate.json。计时来源：results/benchmark_gauge.json、../candidate_validation/results/feature_benchmark.json。旧的分区kernel估算参数为a1024,e64，不适用于本轮MLP kernel。
