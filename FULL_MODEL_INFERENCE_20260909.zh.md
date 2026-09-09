**2026-09-09 新增完整模型推理计时**

在冻结Qwen2.5-1.5B中替换已训练的第14/27层，共24/336 heads。RTX3090、batch1、BF16模型、FP32 feature/state，双方共享参考线性scan/step。三个真实文档、每个长度12个交错配对区组，共180次prefill+64步decode计时。

| 8k上下文 | Prefill ms | Decode ms/token | Prefill+64步 ms |
|---|---:|---:|---:|
| 原模型 | 467.142 | 22.4310 | 1902.409 |
| AD纯删除 | 471.458 | 23.7936 | 1996.439 |
| AD严格同参数 | 471.955 | 23.8924 | 2000.737 |
| Hedgehog-exp | 496.761 | 23.7721 | 2017.770 |
| Hedgehog-softmax | 496.954 | 23.8786 | 2026.484 |

表为各指标中位数。AD的prefill更快；逐token解码没有稳定优势。8k+64步总时间的配对均值相对HH-exp/softmax分别下降约0.82%/1.25%；1k相对HH-exp略慢，4k总时间差异区间跨0。严格同参数控制结论一致。两类局部替换模型在本实现中都没有超过原模型的推理速度。

替换层原KV storage实测为0，状态不随解码增长。8k完整KV+state为AD208.7617MiB、HH214.8555MiB；prefill新增峰值为AD773.20MiB、HH2440.39MiB。这些不是总显存数字。

FP32完整前向与缓存解码在272/8192位置的logits相对L2差均小于3.2e-6。BF16单次前向与解码的数值差异也原样保留，不能隐去。尚未验证全head转换、整模型图捕获服务或910B性能，本轮没有新增PPL评价。

详见[完整报告](snapshot/kan_attention_theory/full_model_ad_hedgehog/REPORT.zh.md)、[协议及复现命令](snapshot/kan_attention_theory/full_model_ad_hedgehog/PROTOCOL.zh.md)、[原始计时](snapshot/kan_attention_theory/full_model_ad_hedgehog/results/raw.json)和[最终核验](snapshot/kan_attention_theory/full_model_ad_hedgehog/checks/final_audit.json)。

