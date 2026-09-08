本轮复现依赖工作区中已经保存的冻结模型、Q/K数据及各轮manifest。完整重新提取/训练会覆盖或跳过已有结果，请先在独立目录保留当前产物。版本与源码校验值见 checks/final_audit.json 和 results/artifact_manifest.json。

训练、评估环境：/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python。绘图环境：/root/miniconda3/bin/python（前一环境没有matplotlib，因此报告使用后一环境；报告只读取JSON，不参与GPU实验）。

已完成的数据和配置记录：PROTOCOL.zh.md；数学假设、推导与边界：THEORY.zh.md；最终解读：REPORT.zh.md。

核心代码职责：

- core.py：两层正特征、原始I损失与条件矩形评价。
- train.py：单遍训练；fits/ 保存全部38组检查点/标度对照及元数据。
- fresh_data.py：三个独立确认轮次，results/fresh_manifest*.json 固定文档/token/hash。
- calibrate.py：16384个训练keys上的log-MGF与既有读出的单遍校准；生成对应全局标度对照。
- product.py：全部新经验Q/K边缘乘积，FP64分块统计原始kernel风险，固定网络seed11/FAVOR seed1009。
- baselines.py：既有FAVOR+/centered FAVOR+/SDERF/ADERF的5随机种子。
- runtime_new.py：各head独立的线性状态；校准kernel在线执行代数等价的父特征。
- gauge_assess.py：第三轮条件kernel与完整冻结模型四头替换PPL。
- benchmark_gauge.py：1024/4096/8192 prompt下的完整模型四头替换计时。
- verify.py、finalaudit.py：代数/递归/数据/参数/聚合检查。

已有GPU实验结果可直接重建表、文档bootstrap和图：

```bash
/root/miniconda3/bin/python /root/autodl-tmp/kan_attention_theory/mlp_direction/report.py
```

无需重新训练的检查：

```bash
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python /root/autodl-tmp/kan_attention_theory/mlp_direction/verify.py
/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python /root/autodl-tmp/kan_attention_theory/mlp_direction/finalaudit.py
```

阶段日志为 results/stages.json、stages2.json、stages3.json、stages_final.json。初次原始μ折叠FP32部署结果单独保存在 ppl_gauge_folded_fp32.json；最终PPL在ppl_gauge.json。所有结果均为指定四个head，不包含全head替换或模型微调。
