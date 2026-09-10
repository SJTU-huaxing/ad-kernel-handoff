# 用户停止时的实验报告：seed11 / seed29

**训练已于 2026-09-09 04:04:13 UTC 按用户要求停止。** 训练控制器、双卡 torchrun 及其两个工作进程、等待中的推理/上下文诊断/最终报告队列均已退出；停止记录的 `remaining_live` 为空，NPU 检查未发现计算进程。已增加训练队列入口的停止标记检查，未经用户重新授权不自动恢复。

停止证据：[stopped_by_user.json](../work/pretraining/stopped_by_user.json)。本报告只读取已存在的日志、检查点元数据和评测结果；整理报告没有启动新的训练或 NPU 评测。

**当前完整结果是 17 组：seed11 十种方法，seed29 七种方法。每组均完成 v2 协议的 1,600 步、104,857,600 个训练目标 tokens，以及该检查点的全部质量评测。** 所有数字对应同一早期训练预算。原计划 seed11 的 16,384 步、1,073,741,824 tokens 尚未达到；seed47 尚未运行。

seed29 的 ELU+1 在停止时最后一条训练日志为 step100、6,553,600 tokens，尚无保存检查点，未完成质量评测。日志间隔为十步，因此只能确认“最后记录到第100步”，不能据此断言停止瞬间恰好执行了100步。seed29 的 softplus、HH-exp m384 尚未开始。以下完整结果表与主要曲线均不包含这三组未完成实验。

完整报告包位于 `work/pretraining/stopped_report/`，压缩包为 [stopped_experiments_seed11_seed29.zip](../work/pretraining/stopped_experiments_seed11_seed29.zip)。包含全文、PNG/SVG 图、原始日志、153 个评测分片 JSON、CSV、实际配置、源码快照和 SHA256 清单；大型模型权重及语料仍保留在原路径。

**1. 已完成实验的语言模型指标**

训练损失是最后十个优化器步骤、两张卡平均的 next-token 交叉熵，单位 nats/token。它只代表末段训练 batch，不能当作全训练集损失。PPL 为 `exp(总负对数似然 / 有效目标token数)`；表内各项均越低越好，不对文档 PPL 作算术平均。

- 验证和主测试：各 1,999,872 个有效目标 tokens，固定 1024-token 块；两者的原始预处理流均为 2,000,000 tokens，未使用尾部不足一块的部分。
- Wiki：62 篇 WikiText-103 raw 测试文档，286,158 个输入语料 tokens、286,096 个预测目标；按文档和 1024 个目标重置上下文，保留不足一块的文末。
- 长文：同一批 64 篇连贯文档、38 个 hostname，分别评价每篇前 1024、4096、8192 个目标 tokens；合计 65,536、262,144、524,288 个目标。

**seed11：十种方法，均为 step1600。**

| 方法 | 末10步训练损失 | 验证 PPL | 主测试 PPL | Wiki PPL | 长文1k PPL | 长文4k PPL | 长文8k PPL |
|---|---|---|---|---|---|---|---|
| AD m64 | 4.3386 | 81.8363 | 68.4765 | 256.8189 | 86.7531 | 335.9628 | 447.5376 |
| EXP m64 | 4.3435 | 82.0685 | 68.8534 | 264.1524 | 86.6927 | 335.8312 | 451.6553 |
| Softmax | 4.0056 | 58.1927 | 49.3610 | 120.9612 | 59.5816 | 139.0283 | 213.5236 |
| HH-softmax m128 | 4.3341 | 81.3602 | 68.2490 | 259.4802 | 85.7474 | 272.7062 | 361.0793 |
| HH-exp m128 | 4.3605 | 83.3169 | 69.8839 | 260.9799 | 88.7469 | 305.9324 | 395.1988 |
| FAVOR+ m256 | 4.3159 | 79.5464 | 66.9378 | 226.3786 | 84.9241 | 228.7809 | 305.9517 |
| FAVOR+ m64 | 4.3484 | 82.2333 | 69.0455 | 243.2275 | 87.8094 | 245.7779 | 324.1119 |
| ELU+1 m64 | 4.7483 | 121.6150 | 103.2330 | 416.0784 | 127.4232 | 253.6197 | 318.3371 |
| Softplus m64 | 4.7192 | 118.0524 | 100.3877 | 389.5314 | 123.5820 | 265.1516 | 330.9531 |
| HH-exp m384 | 4.3106 | 79.4711 | 66.5009 | 253.6472 | 84.1901 | 281.0845 | 361.9100 |

**seed29：七种方法，均为 step1600。**

| 方法 | 末10步训练损失 | 验证 PPL | 主测试 PPL | Wiki PPL | 长文1k PPL | 长文4k PPL | 长文8k PPL |
|---|---|---|---|---|---|---|---|
| AD m64 | 4.3707 | 81.0683 | 67.8880 | 257.6890 | 84.0983 | 334.8204 | 446.5732 |
| EXP m64 | 4.3815 | 82.2261 | 68.7952 | 255.9939 | 85.5496 | 363.5854 | 498.7578 |
| Softmax | 4.0262 | 57.5128 | 48.8610 | 120.6502 | 57.7501 | 136.5582 | 206.7709 |
| HH-softmax m128 | 4.3684 | 81.2447 | 68.0393 | 257.7605 | 85.0651 | 290.3090 | 388.2775 |
| HH-exp m128 | 4.3847 | 82.2526 | 69.1395 | 261.5674 | 85.7734 | 320.0426 | 419.4492 |
| FAVOR+ m256 | 4.3568 | 79.8703 | 67.2699 | 242.6674 | 83.3488 | 237.0229 | 318.4990 |
| FAVOR+ m64 | 4.3777 | 81.9252 | 68.7671 | 249.0751 | 85.1761 | 255.7274 | 338.9017 |

所有未舍入的训练损失、验证/主测试/Wiki/各长度的 NLL 与 PPL、有效 token 数、参数、成本及检查点 SHA256 见 [metrics.csv](../work/pretraining/stopped_report/metrics.csv)。每个检查点的完整机器可读质量结果见 [summary.json](../work/pretraining/reports/summary.json)。

**2. 训练损失曲线**

每条完整曲线包含 160 个原始日志点，每点覆盖十步，即 655,360 个训练目标 tokens。图的横轴是累计训练目标 tokens，纵轴是交叉熵；没有额外移动平均、拟合或插值平滑。左图显示完整过程，右图放大 20M tokens 之后。不同 seed 的数据排列不同，因此跨 seed 的局部波动也包含 batch 内容差异。

![seed11 十种方法的训练损失](../work/pretraining/stopped_report/figures/training_loss_seed11.png)

![seed29 七种方法的训练损失](../work/pretraining/stopped_report/figures/training_loss_seed29.png)

![逐方法对照 seed11 与 seed29](../work/pretraining/stopped_report/figures/training_loss_by_method.png)

下载：[全部 2720 个原始损失点 CSV](../work/pretraining/stopped_report/training_loss.csv)、[seed11 SVG](../work/pretraining/stopped_report/figures/training_loss_seed11.svg)、[seed29 SVG](../work/pretraining/stopped_report/figures/training_loss_seed29.svg)、[逐方法 SVG](../work/pretraining/stopped_report/figures/training_loss_by_method.svg)。ELU seed29 的十个未完成日志点另存 [incomplete_elu64_seed29_loss.csv](../work/pretraining/stopped_report/incomplete_elu64_seed29_loss.csv)。

曲线从约 10.2 nats/token 降至表中末段数值；softmax 的训练损失最低，ELU+1 与 softplus 明显较高。AD/EXP、HH 和 FAVOR 中部分曲线非常接近，需要结合固定测试集区分。验证只在这一完整阶段末尾记录一次，当前没有多点验证损失曲线可供绘制。

**3. 每一种方法具体是什么**

所有线性方法都在全部 12 层使用正特征的归一化因果 attention。每个 head 的输入维度为 d=64，m 表示映射后的特征/状态宽度。它们共同计算：

```text
S_t = Σ(i≤t) φK(k_i) v_iᵀ
z_t = Σ(i≤t) φK(k_i)
y_t = φQ(q_t)ᵀ S_t / [φQ(q_t)ᵀ z_t]
```

Q/K 先做 RoPE；进入线性 feature 前各乘 `64^(-1/4)`。这里的 φ 是数学表达，实际代码在 log-feature 空间构造并以数值缩放状态执行。数值缩放不构成新的可学习门。

| 方法 | 实际特征函数与可训练部分 |
|---|---|
| AD m64 | 每层每 head 独立 Q/K 两层无偏置 MLP，隐藏层96、SiLU。Q 为64→96→63，拼接常数0后取64维 softmax：`φQ=softmax([zQ,0])`。K 为64→96→64，前63维是方向，末1维是幅度对数：`φK=exp(sK)·softmax([zK,0])`。保留K幅度；无Q幅度、外部C或额外门。 |
| EXP m64 | Q网络及 `φQ` 与AD完全相同；K仍是64→96→64，但直接 `φK=exp(MLP_K(k))`。与AD的可训练feature参数数目、网络深度和状态宽度完全相同，是关键消融对照。 |
| HH-softmax m128 | 每层每head各有独立可训练Q/K仿射映射 `u=W x+b`，64→64，含偏置。`φ=concat(softmax(u), softmax(-u))`；正负两半分别做softmax，拼接后m=128。 |
| HH-exp m128 | 同样的独立Q/K仿射映射64→64、含偏置；`φ=concat(exp(u),exp(-u))/sqrt(128)`。 |
| HH-exp m384 | 仿射映射扩为64→192，拼接正负指数分支后m=384；`φ=concat(exp(u),exp(-u))/sqrt(384)`。它扩大了状态预算，参数总量接近AD。 |
| FAVOR+ m64 | 每head固定正交高斯随机投影Ω，Q/K共用该head的Ω；`φ(x)=exp(Ωx−||x||²/2)/sqrt(64)`。Ω由QR正交方向和高斯半径构造，作为buffer保存，训练中不更新；骨干Q/K投影继续训练。 |
| FAVOR+ m256 | 与FAVOR+ m64同一构造，投影特征数增至256，分母为sqrt(256)。无额外可训练feature参数，但状态更大。 |
| ELU+1 m64 | 对Q/K逐元素做 `φ(x)=ELU(x)+1`，没有额外feature网络。 |
| Softplus m64 | 对Q/K逐元素做 `φ(x)=log(1+exp(x))`，没有额外feature网络；实现使用稳定的log-feature表达。 |
| Softmax | 标准完整因果 `softmax(QKᵀ/sqrt(64))V`，由PyTorch原生SDPA执行；全部12层都使用该标准attention。 |

AD/EXP的每个 head、每层 Q/K 网络彼此独立。新语言模型的输入归一化buffer为 mean=0、std=1，未加载历史Qwen拟合统计。HH的Q/K映射初始值相同，但权重和偏置各自独立可训练；m384的额外投影行以正交矩阵初始化。FAVOR随机矩阵在给定 seed 下固定。

代码依据：[features.py](../src/ad_kernel/features.py)、[baselines.py](../src/ad_kernel/baselines.py)、[model.py](../src/ad_kernel/model.py)。Hedgehog 类特征参考 [Hedgehog 论文](https://arxiv.org/abs/2402.04347)，FAVOR+ 的方法来源可查 [Performer 官方实现](https://github.com/google-research/google-research/tree/master/performer)。本轮比较使用本项目的统一骨干和本地特征实现；这些名字对应上述确切函数，不代表完整照搬原论文的训练pipeline。

**4. 参数、状态缓存与实际训练成本**

共同骨干有 **123,551,232 个可训练参数**。AD/EXP精确匹配feature参数；HH各版本显式采用不同参数和状态预算。FAVOR矩阵计为固定buffer，不计入可训练参数。

| 方法 | 总可训练参数 | feature可训练参数 | 特征宽度m | B1/1k实际cache MiB |
|---|---|---|---|---|
| AD m64 | 127,076,352 | 3,525,120 | 64 | 2.320312 |
| EXP m64 | 127,076,352 | 3,525,120 | 64 | 2.320312 |
| Softmax | 123,551,232 | 0 | 不适用 | 36.000000 |
| HH-softmax m128 | 124,749,312 | 1,198,080 | 128 | 4.640625 |
| HH-exp m128 | 124,749,312 | 1,198,080 | 128 | 4.640625 |
| FAVOR+ m256 | 123,551,232 | 0 | 256 | 9.281250 |
| FAVOR+ m64 | 123,551,232 | 0 | 64 | 2.320312 |
| ELU+1 m64 | 123,551,232 | 0 | 64 | 2.320312 |
| Softplus m64 | 123,551,232 | 0 | 64 | 2.320312 |
| HH-exp m384 | 127,145,472 | 3,594,240 | 384 | 13.921875 |

cache为B1、全12层实际保留张量storage，包含线性状态S、z及数值scale，单位MiB=2²⁰ bytes；排除权重、训练激活、运行时临时张量和内存分配器预留。线性方法状态为 `12×12×m×(64+2)` 个FP32标量，随序列长度保持常数。softmax为BF16 K/V，1k为36 MiB，按结构在4k/8k分别为144/288 MiB。缓存小不等于在当前实现下训练或推理更快。

训练循环的实测成本如下。这些是本机本实现的阶段观测；时间从训练循环开始到 `run_complete`，包含取batch、HCCL、AdamW、验证和检查点保存，排除环境/模型初始化、训练前数据哈希以及后续独立质量评测。tokens/s为全阶段实际训练tokens除以该时间。

| 方法 | seed | 训练循环分钟 | 循环平均tokens/s | rank0峰值GiB | rank0日志数值分段次数 |
|---|---|---|---|---|---|
| AD m64 | 11 | 33.508 | 52155.0 | 22.507 | 127,708 |
| EXP m64 | 11 | 33.684 | 51883.1 | 22.056 | 127,493 |
| Softmax | 11 | 12.154 | 143795.1 | 12.013 | 0 |
| HH-softmax m128 | 11 | 27.089 | 64513.5 | 24.246 | 1 |
| HH-exp m128 | 11 | 27.751 | 62974.8 | 22.159 | 23,736 |
| FAVOR+ m256 | 11 | 40.815 | 42817.7 | 33.433 | 13,234 |
| FAVOR+ m64 | 11 | 21.588 | 80954.0 | 18.170 | 18,842 |
| ELU+1 m64 | 11 | 20.140 | 86773.8 | 17.745 | 0 |
| Softplus m64 | 11 | 20.419 | 85589.5 | 18.868 | 0 |
| HH-exp m384 | 11 | 53.953 | 32391.9 | 40.026 | 0 |
| AD m64 | 29 | 32.654 | 53519.6 | 22.554 | 120,912 |
| EXP m64 | 29 | 33.534 | 52115.4 | 22.129 | 135,054 |
| Softmax | 29 | 12.174 | 143551.8 | 12.013 | 0 |
| HH-softmax m128 | 29 | 27.061 | 64580.4 | 24.239 | 0 |
| HH-exp m128 | 29 | 27.833 | 62789.3 | 22.153 | 25,019 |
| FAVOR+ m256 | 29 | 40.794 | 42839.8 | 32.310 | 12,298 |
| FAVOR+ m64 | 29 | 22.157 | 78874.6 | 18.206 | 27,306 |

峰值是rank0训练日志中的最大已分配GiB，既不是两卡合计，也不是完整系统保留显存。数值分段计数是rank0进程截至最后训练日志的累计值，可能包含此前验证；末次日志之后的最终验证不在该计数中。softmax用BF16 SDPA、线性feature/attention用FP32，因此速度/显存也体现了精度和实现路径的差异。

原计划的最终独占推理延迟/吞吐基准还没有执行，停止时其有效结果为0组。本报告没有用质量评测耗时冒充推理性能。

**5. 关联回忆、长文位置分段与缓存一致性**

关联回忆为固定seed91371的合成表格查找任务，不作为训练数据：8/32/64组键值，目标位置按开头/中间/末尾平衡，表格值是不同的英文单GPT2-token词，插入中性填充到1k/4k/8k。每个长度×键值数条件24题，总共72张表、216个提示。以下为“只在表中候选值之间选择”的准确率，单位%；列名为长度/键值数。并列最高按分数分摊，因此允许2.08%一类结果。随机水平依次为12.5%、3.125%、1.5625%。

seed11：

| 方法 | 1k/8 | 1k/32 | 1k/64 | 4k/8 | 4k/32 | 4k/64 | 8k/8 | 8k/32 | 8k/64 |
|---|---|---|---|---|---|---|---|---|---|
| AD m64 | 4.17 | 4.17 | 0.00 | 16.67 | 4.17 | 0.00 | 16.67 | 4.17 | 0.00 |
| EXP m64 | 8.33 | 8.33 | 0.00 | 4.17 | 4.17 | 0.00 | 4.17 | 4.17 | 0.00 |
| Softmax | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 12.50 | 0.00 | 2.08 |
| HH-softmax m128 | 16.67 | 4.17 | 0.00 | 8.33 | 4.17 | 0.00 | 8.33 | 4.17 | 0.00 |
| HH-exp m128 | 12.50 | 4.17 | 0.00 | 4.17 | 4.17 | 0.00 | 4.17 | 4.17 | 0.00 |
| FAVOR+ m256 | 20.83 | 8.33 | 0.00 | 8.33 | 4.17 | 0.00 | 8.33 | 4.17 | 0.00 |
| FAVOR+ m64 | 8.33 | 4.17 | 0.00 | 12.50 | 8.33 | 4.17 | 12.50 | 4.17 | 0.00 |
| ELU+1 m64 | 8.33 | 4.17 | 0.00 | 12.50 | 4.17 | 0.00 | 8.33 | 4.17 | 0.00 |
| Softplus m64 | 4.17 | 4.17 | 0.00 | 4.17 | 8.33 | 0.00 | 0.00 | 8.33 | 0.00 |
| HH-exp m384 | 8.33 | 4.17 | 0.00 | 8.33 | 4.17 | 0.00 | 4.17 | 4.17 | 8.33 |

seed29：

| 方法 | 1k/8 | 1k/32 | 1k/64 | 4k/8 | 4k/32 | 4k/64 | 8k/8 | 8k/32 | 8k/64 |
|---|---|---|---|---|---|---|---|---|---|
| AD m64 | 12.50 | 4.17 | 0.00 | 8.33 | 10.42 | 8.33 | 8.33 | 8.33 | 8.33 |
| EXP m64 | 8.33 | 4.17 | 0.00 | 4.17 | 4.17 | 0.00 | 4.17 | 4.17 | 0.00 |
| Softmax | 12.50 | 4.17 | 0.00 | 4.17 | 4.17 | 0.00 | 4.17 | 4.17 | 8.33 |
| HH-softmax m128 | 12.50 | 8.33 | 0.00 | 8.33 | 4.17 | 0.00 | 8.33 | 4.17 | 0.00 |
| HH-exp m128 | 8.33 | 4.17 | 0.00 | 4.17 | 4.17 | 0.00 | 4.17 | 4.17 | 0.00 |
| FAVOR+ m256 | 8.33 | 12.50 | 8.33 | 4.17 | 4.17 | 0.00 | 4.17 | 4.17 | 0.00 |
| FAVOR+ m64 | 4.17 | 4.17 | 0.00 | 8.33 | 4.17 | 0.00 | 8.33 | 8.33 | 0.00 |

**17组模型在全部9种条件下，不限制候选集合的 next-token top-1 正确率均为0%。** 这说明当前早期检查点没有显示可靠的零样本表格检索能力；它同时受格式理解和检索影响，不足以证明某架构无法学会关联回忆。正确答案平均log概率、候选平均rank、分数精度的原值见 [recall.csv](../work/pretraining/stopped_report/recall.csv)，共153行。

长文在同一次8k执行内，按互不重叠目标位置分段得到下表。区间采用0起始、左闭右开记法；有效目标数依次65,536、196,608、262,144。

| 方法 | seed | 8k执行：目标0–1024 PPL | 目标1024–4096 PPL | 目标4096–8192 PPL |
|---|---|---|---|---|
| AD m64 | 11 | 86.7632 | 527.7160 | 596.0393 |
| EXP m64 | 11 | 86.6927 | 527.4412 | 607.4187 |
| Softmax | 11 | 59.5783 | 184.3990 | 327.9448 |
| HH-softmax m128 | 11 | 85.7549 | 401.0363 | 478.0823 |
| HH-exp m128 | 11 | 88.7437 | 462.1627 | 510.5058 |
| FAVOR+ m256 | 11 | 84.9276 | 318.3314 | 409.1513 |
| FAVOR+ m64 | 11 | 87.8121 | 346.3636 | 427.4174 |
| ELU+1 m64 | 11 | 127.4224 | 319.0175 | 399.5785 |
| Softplus m64 | 11 | 123.5866 | 341.9631 | 413.1004 |
| HH-exp m384 | 11 | 84.1947 | 420.1135 | 465.9668 |
| AD m64 | 29 | 84.1055 | 530.6523 | 595.6233 |
| EXP m64 | 29 | 85.5504 | 588.7463 | 684.3491 |
| Softmax | 29 | 57.7468 | 181.9290 | 313.0910 |
| HH-softmax m128 | 29 | 85.0665 | 437.0646 | 519.3206 |
| HH-exp m128 | 29 | 85.7675 | 496.2976 | 549.8211 |
| FAVOR+ m256 | 29 | 83.3506 | 335.7856 | 427.9986 |
| FAVOR+ m64 | 29 | 85.1769 | 368.9196 | 449.1260 |

同篇1k/4k/8k前缀增加长度时，待预测的文本也增加，后段可能本来就更难，因此不能把全部PPL上升归因于上下文长度。相同长度下各方法预测的是同一组目标，方法间比较成立。8k执行的前1k与独立1k执行会因浮点执行路径存在微小差异。原计划“固定完全相同目标、只改变上下文与位置”的补充诊断尚未执行。

缓存续接和单token解码在全部17组均通过已冻结的BF16模型logit一致性检查。这里列相对RMS误差及平均输出分布KL；最大/99分位绝对误差、63/1024/1025时cache字节数均保留于 [cache.csv](../work/pretraining/stopped_report/cache.csv)。

| 方法 | seed | 续接129相对RMS | 续接129输出KL | 解码1025相对RMS | 解码1025输出KL | 通过 |
|---|---|---|---|---|---|---|
| AD m64 | 11 | 0.00243237 | 8.99889e-05 | 0.00509376 | 0.000109891 | 是 |
| EXP m64 | 11 | 0.0024675 | 7.49518e-05 | 0.00538566 | 7.20636e-05 | 是 |
| Softmax | 11 | 0 | 0 | 0.0043849 | 0.000140931 | 是 |
| HH-softmax m128 | 11 | 0.00237567 | 8.17242e-05 | 0.00369229 | 5.04464e-05 | 是 |
| HH-exp m128 | 11 | 0.00218297 | 5.75999e-05 | 0.00447626 | 0.000118994 | 是 |
| FAVOR+ m256 | 11 | 0.00294944 | 0.000102872 | 0.00540445 | 6.21896e-05 | 是 |
| FAVOR+ m64 | 11 | 0.00323674 | 0.000150509 | 0.00332583 | 5.03017e-05 | 是 |
| ELU+1 m64 | 11 | 0.00185734 | 3.79426e-05 | 0.00309072 | 5.0247e-05 | 是 |
| Softplus m64 | 11 | 0.00200157 | 4.02247e-05 | 0.00320274 | 3.75454e-05 | 是 |
| HH-exp m384 | 11 | 0.00222119 | 7.8628e-05 | 0.00461242 | 6.02608e-05 | 是 |
| AD m64 | 29 | 0.00233313 | 6.20543e-05 | 0.0103989 | 0.000258582 | 是 |
| EXP m64 | 29 | 0.00257806 | 9.25253e-05 | 0.00385069 | 6.5282e-05 | 是 |
| Softmax | 29 | 0 | 0 | 0.00529427 | 0.000271815 | 是 |
| HH-softmax m128 | 29 | 0.00235872 | 6.37042e-05 | 0.00309252 | 8.04387e-05 | 是 |
| HH-exp m128 | 29 | 0.00237801 | 6.4831e-05 | 0.00666051 | 0.000262348 | 是 |
| FAVOR+ m256 | 29 | 0.0029253 | 0.00010753 | 0.00368697 | 0.000118799 | 是 |
| FAVOR+ m64 | 29 | 0.00382261 | 0.000197911 | 0.00321482 | 9.99832e-05 | 是 |

通过的是协议规定的浮点容差，并非声称两种执行逐位相同；底层attention输出/梯度另有更严格的CPU FP64及NPU FP32验证证据。

**6. 当前结果支持的结论**

softmax在两个已完成seed的主测试、Wiki和全部长文长度上均领先所有本轮线性方法，同时在本机训练循环中耗时最短；代价是随长度增长的KV缓存。线性方法中，seed11主测试最好的是HH-exp m384；seed29已完成集合中最好的是FAVOR+ m256。由于seed29的三种方法尚未完成，两个seed的全部方法平均排序目前不完整。

下表只使用AD与该对照都已完成的匹配seed。正的NLL差/相对PPL降低表示AD更好；PPL降低按匹配seed的几何平均计算。

| AD对照方法 | 匹配seed | 对照−AD 主测试NLL | AD相对PPL降低(%) |
|---|---|---|---|
| EXP m64 | 11/29 | +0.009382 | +0.9338 |
| HH-exp m128 | 11/29 | +0.019306 | +1.9121 |
| HH-softmax m128 | 11/29 | -0.000551 | -0.0551 |
| HH-exp m384 | 11 | -0.029275 | -2.9708 |
| FAVOR+ m64 | 11/29 | +0.010570 | +1.0514 |
| FAVOR+ m256 | 11/29 | -0.015937 | -1.6064 |
| ELU+1 m64 | 11 | +0.410498 | +33.6680 |
| Softplus m64 | 11 | +0.382549 | +31.7880 |
| Softmax | 11/29 | -0.328105 | -38.8334 |

AD相对EXP的主测试PPL在两个seed都改善，几何平均降低0.934%；与HH-softmax m128几乎持平且逐seed方向相反；FAVOR+ m256主测试优于AD。AD的8k PPL优于EXP，但落后于其他列出的线性基线，不能据AD/EXP单项比较宣称整体最佳。

AD/EXP的Wiki结论也随seed变化：seed11 AD更低，seed29 EXP更低。对8k，EXP−AD平均NLL差约+0.059838，两个seed分别+0.009159/+0.110517；hostname整组bootstrap的条件95%区间约[+0.040344,+0.077426]。此区间条件于这两个已训练模型，只描述文档/hostname抽样不确定性，不代表一般训练seed的不确定性。完整配对统计及范围在 [AD_COMPARISONS.zh.md](AD_COMPARISONS.zh.md)。

当前每个模型只训练104.858M tokens，而且仍沿用1.074B预算的长学习率日程，不能称为已收敛的100M训练或最终模型排名。停止后不补跑缺失实验来改变这个快照。

**7. 模型与优化器的实际配置**

本轮是从随机权重初始化的自回归decoder语言模型，目标是普通next-token交叉熵；没有额外蒸馏、检索或辅助损失。全部12层都采用对应的方法。

| 项目 | 实际设置 |
|---|---|
| 层数 / hidden / heads / head dim | 12 / 768 / 12 / 64 |
| QKV / attention output | 无偏置线性768→2304，输出768→768 |
| FFN | SwiGLU，中间维2048；上投影768→4096分为gate/value，SiLU(gate)×value，下投影2048→768 |
| 归一化 | attention与FFN前置RMSNorm，最终RMSNorm，eps=1e-6 |
| 位置编码 | RoPE，theta=10000，Q/K旋转计算FP32 |
| embedding / LM head | 词表50,257，宽768，输入embedding与输出head共享权重 |
| Dropout | 无 |
| 骨干初始化 | linear/embedding正态std=.02；attention输出及FFN下投影std=.02/sqrt(24) |
| 精度 | FP32主参数，BF16骨干autocast；线性feature、attention、状态FP32；softmax Q/K/V与KV cache BF16 |
| 训练目标长度 | 1024 tokens/sequence |
| 并行与batch | 两卡DDP；每卡micro-batch16，梯度累积2；global batch64 sequences |
| 每优化器步tokens | 64×1024=65,536 |
| 优化器 | AdamW，峰值LR=6e-4，betas=(.9,.95)，eps=1e-8 |
| weight decay | ndim≥2参数0.1；一维参数0 |
| 梯度裁剪 | 全模型L2 norm上限1.0；非有限梯度则停止，不应用该步更新 |
| 学习率日程 | 128步线性warmup，此后cosine；到16,384步降至峰值的0.1 |
| 第1600步实际LR | 0.0005891631097564793 |
| 当前完成预算 | 每个完整run 1600步，104,857,600目标tokens |
| 原计划预算 | seed11为16,384步、1,073,741,824 tokens；seed29/47为1600步；当前均服从停止指令 |
| 数据排列 | 不重叠1024目标块，`torch.randperm`，种子85000+model_seed，无放回 |
| 保存 / 验证 | 每256步保存恢复状态；里程碑保存模型；阶段1600步验证并质量评测 |

同一seed下各方法共有骨干的初始权重逐元素相同，feature初始化在独立RNG范围，避免扰动后续骨干初始化；训练块排列也完全相同。不同seed会改变模型初始化、FAVOR随机特征和数据顺序。一次1024目标块内保留完整因果历史/梯度，EOS分隔文档可以在块内互相attention，块间重置状态。

正式冻结配置：[pretraining_protocol_v2.json](../configs/pretraining_protocol_v2.json)；17组实际运行配置（包含微批量、累积、数据顺序哈希及源码绑定）：[run_configurations.json](../work/pretraining/stopped_report/run_configurations.json)。`ModelConfig`代码默认值并非本次最终运行设置，以v2协议和每个 `run.json` 为准，实际 `backend="torch"`。

**8. 训练数据来源、数量与处理**

训练语料来自官方 [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)，使用其 `sample/10BT` 的四个parquet文件 `000_00000` 至 `003_00000`；固定revision为 `87f09149ef4734204d70ed1d046ddc9ca3f2b8f9`。本地处理扫描2,063,512篇源文档。

| 划分 | 已准备tokens | 保留文档数 | 规范化hostname数 |
|---|---:|---:|---:|
| Train | 2,000,100,000 | 1,934,765 | 709,073 |
| Validation | 2,000,000 | 1,672 | 913 |
| Test | 2,000,000 | 1,966 | 1,001 |

这是可采样语料池的规模。**每个已完成run实际用于优化器更新的目标tokens只有104,857,600**，不是已经把约2B的语料全部训练一遍。训练按token块取样，当前没有逐run独立文档数统计；不能把训练池的1,934,765篇当作每个run已完整读过的文档数。

17个完整run的更新tokens相加为1,782,579,200，其中seed11十组1,048,576,000、seed29七组734,003,200；相同seed的方法使用相同数据块，不同seed也可能重叠，因此总和不是独立语料量。ELU29未完成日志的6,553,600 tokens另记，不混入完整17组。

分词器是 `openai-community/gpt2`，固定revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`，词表50,257、EOS ID50256。每篇后加EOS，存为little-endian uint16二进制；保留每篇文档偏移、原始来源与文本hash。GPT2在本轮提供分词器，模型权重从零初始化。

处理规则：英文、language_score≥0.8、int_score≥3、至少200字符及50个词；NFKC/小写/空白归一化后做SHA256精确去重；word 5-gram bottom-32 sketch，共享最小四个hash之一召回候选，估计Jaccard≥0.8判为近重复；候选桶达到256时单独按高重复处理。质量、短文本、精确重复、近重复、高频桶拒绝数分别37,619、486、16,257、7,937、46,635；验证/测试/训练超预算拒绝分别8,285、7,800、90。

划分依据是小写并去除开头 `www.` 的hostname，BLAKE2b(person=`ad-split-v1`)取模1000，0–4验证、5–9测试，其余训练。三个保留hostname集合完全分离；这里是hostname粒度，非可注册主域粒度，近似去重也不能证明所有语义重叠为零。

数据文件位于 `work/pretraining/data/fineweb_edu_v1/`。数据清单：[manifest.json](../work/pretraining/data/fineweb_edu_v1/manifest.json)，SHA256为 `9976b25b1d3e54c50d7e20b926be6cd1c16fd52aae28e999629639cacd4d0d96`；四个源parquet的hash及过滤规则见 [pretraining_data_v1.json](../configs/pretraining_data_v1.json)。

长文从同一固定FineWeb-Edu来源的test-hostname桶选取64篇，每篇至少8193个GPT2 tokens，排除与训练精确匹配并对该长文集合做近重复过滤；选择在观察模型成绩前冻结。[逐文档来源与hash](../work/pretraining/evaluation/long_v1/manifest.json)。Wiki来自 `EleutherAI/wikitext_document_level` 的 `wikitext-103-raw-v1 test`，固定revision `647234772b9554e208af6c826f23b99e3cac88c8`。检测到的训练精确文档匹配数为0；网页引用或近重复仍可能存在，因此Wiki只作为语料/风格迁移检查。[Wiki与回忆清单](../work/pretraining/evaluation/transfer_recall_v1/manifest.json)。

**9. 训练框架来源与训练pipeline来源**

本轮训练框架是 **PyTorch 2.7.1 + torch_npu 2.7.1.post4 + torch.distributed/DDP + HCCL**，语言模型和训练循环由本项目在本地编写，包名 `ad-kernel-ascend==0.1.0`。[Ascend/PyTorch官方仓库](https://github.com/Ascend/pytorch)提供NPU适配；[PyTorch DDP](https://docs.pytorch.org/docs/stable/generated/torch.nn.parallel.DistributedDataParallel.html)提供数据并行能力。Hugging Face相关依赖用于读取公开资产、数据和tokenizer。

正式attention执行为 [attention.py](../src/ad_kernel/attention.py) 的CANN原生batched GEMM/cumsum与数值稳定分段实现。训练loss使用 [FLA 0.5.2](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/pyproject.toml) 的Ascend融合词表线性层交叉熵 `FusedLinearCrossEntropyLoss(num_chunks=8, reduction="mean")`。评测前向以256个位置为块计算词表CE，保留完整有效目标。

FLA的原始 `chunk_linear_attn` 在本机测试曾出现FP32跨块结果错误和BF16编译失败，候选Triton-Ascend3.2.2/torch_npu post8也未解决所测编译故障。项目另实现并验证了 `ascend_linear.py` 混合兼容路径，但独占同卡完整模型前后向基准中，CANN原生路径更快：随机AD约0.381秒对0.538秒、HH m384约0.973秒对1.360秒；因而v2选用了原生路径。当前正式训练应准确描述为“自定义线性attention + FLA融合loss”。

| Pipeline环节 | 本项目实际入口/来源 | 内容 |
|---|---|---|
| 数据准备 | [prepare_pretraining_data.py](../experiments/prepare_pretraining_data.py) + data_v1配置 | 固定数据revision和文件hash、质量/去重/hostname划分、tokenize/EOS、二进制及manifest |
| 模型 | [model.py](../src/ad_kernel/model.py) + features/baselines | 从随机初始化构建12层统一骨干，各方法作用于全部attention层 |
| 训练循环 | [pretrain.py](../experiments/pretrain.py) + protocol_v2 | torchrun启动两个rank，DDP梯度累积、AdamW、学习率、非有限检查、日志、验证、checkpoint |
| 调度 | [run_pretraining_queue.py](../experiments/run_pretraining_queue.py) | 逐方法/seed训练里程碑后调用双分片质量评测；现已停止且增加停止标记检查 |
| 质量评测 | [evaluate_pretrained.py](../experiments/evaluate_pretrained.py) + [pretrained_evaluation_v1.json](../configs/pretrained_evaluation_v1.json) | packed、Wiki、长文、关联回忆、cache；两个手动分片，输出绑定检查点/代码hash |
| 缓存执行 | [inference_runtime.py](../experiments/inference_runtime.py) | 实现完整forward、prefill续接、逐token decode，记录实际cache张量 |
| 汇总与统计 | [summarize_pretrained.py](../experiments/summarize_pretrained.py) | token加权NLL/PPL、匹配seed比较、文档/hostname条件bootstrap |
| 成本/完整性 | summarize_training_cost.py / audit_pretraining_artifacts.py | 按真实调用计时和进度，核对数据、原交接文件、源码、权重与评测绑定 |
| 本次停止快照 | [export_stopped_report.py](../experiments/export_stopped_report.py) | CPU读取已有日志/结果，导出曲线、CSV和源码快照 |

主流程为：固定数据资产 → 清洗/去重/划分 → GPT2/EOS打包 → 1024目标块的seed排列 → 两卡DDP从零训练 → 保存固定预算检查点 → 固定质量评测 → 逐token汇总和条件统计。恢复状态含模型、AdamW、各rank RNG、数据游标、排列hash、实际tokens、数据/协议/训练源码hash；`model_step001600.pt`用于权重评测，`last.pt`保留完整恢复状态。

每个run的配置和源代码hash随结果保存，可在报告包 `sources/` 与 `source_sha256.json` 中逐项核验。该自定义pipeline的来源就是本项目这些脚本，不能为它虚构一个上游官方训练仓库或提交号。

历史交接的AD/EXP研究资产与原代码在 `snapshot/`、`checkpoints/` 和 `work/kan_attention_theory/`；已核验原始1613个文件未被修改。早先复现使用Qwen2.5-1.5B的局部层替换，另见 [ASCEND_REPRODUCTION.zh.md](ASCEND_REPRODUCTION.zh.md)。本报告的17组是其后单独开展的全层从零预训练，指标与数据预算独立记账。

**10. 环境、数值版本与已保留资产**

| 组件 | 本次实际版本/设置 |
|---|---|
| 硬件 / 系统 | 两张Ascend 910B3 64GB，Linux aarch64 / Ubuntu22.04 |
| CANN / 驱动 | 9.0.0 / 25.5.1 |
| conda / Python | Miniforge conda26.5.3 / Python3.11.16 |
| torch / torch_npu | 2.7.1（运行字符串2.7.1+cpu）/ 2.7.1.post4 |
| torchvision | 0.22.1 |
| triton-ascend | 3.2.1（内部triton运行字符串3.2.0） |
| flash-linear-attention / fla-core | 0.5.2 / 0.5.2 |
| transformers / datasets | 4.57.6 / 5.0.1 |
| numpy / pyarrow / tokenizers | 1.26.4 / 25.0.1 / 0.22.2 |
| 环境 / Jupyter kernel | `/cache/huaxing/miniforge3/envs/nonlinear-qk` / `nonlinear-qk` |

ARM64基础torch wheel的`+cpu`标记不代表本轮在CPU训练，NPU功能由torch_npu提供，训练日志有两卡HCCL与NPU指标。Triton发行包与运行字符串差异也已单独记录。详细环境及锁定文件见 [环境说明](../../nonlinear-qk-setup/README.md)；报告包 `environment.json` 是读取本环境包元数据所得。

正式v2来自一次明确的数值修复：v1的AD在第348步因全序列K缩放令早期分母进入FP32次正规极小范围而触发非有限梯度停止。v1的已完成softmax和失败AD轨迹均已归档，未混入以上17组。

v2在缩放后的因果分母小于 `sqrt(finfo(dtype).tiny)` 时细分执行段，完整传递S/z/scale状态及梯度；不增加分母epsilon、不截断历史、不钳位幅度、不加入遗忘门。所有方法在v2中从相同seed重新初始化。修复后36项CPU测试、真实故障head和全模型NPU回归通过；所测两卡连续4步与2步后恢复到4步的权重、AdamW、RNG和游标完全相同。该短程恢复证据不承诺任意长训练逐位确定性。详见 [NUMERICAL_STABILITY.zh.md](NUMERICAL_STABILITY.zh.md)。

停止后完整性审查已核对17个质量检查点及各自9个评测分片、数据实际字节与绑定、原始交接文件，审查时无活动worker。审查中的“quality 17/40、最终推理0/20、上下文0/10、目标预算恢复状态7/30”采用原完整计划分母；其中seed11的10组虽有早期恢复状态，但尚未达到原定最终预算，因此不计入7个目标预算完成项。这些缺项表示用户停止时的真实覆盖，不是报告导出失败。

保留位置：模型与优化器在 `work/pretraining/runs/{method}_s{seed}/`；质量结果在 `work/pretraining/results/{method}_s{seed}_step001600/02ae37d5258c/`；原始训练轨迹为每个run的 `train.jsonl`。停止快照包附完整质量和日志副本、配置、曲线及hash，后续无需启动训练即可检查本报告的全部已完成结果。
