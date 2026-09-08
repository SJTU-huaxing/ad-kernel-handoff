**AD 幅度—方向 kernel：研究交接与双昇腾 910B 接续**

本仓库将原来的文档源码包和21个检查点合并。目标版本已经用户确认：**删除 C 和查询 Q 幅度，保留 K 幅度**。

\[
h(q,k)=e^{s_K(k)}\operatorname{softmax}([z_Q(q),0])^\top\operatorname{softmax}([z_K(k),0]).
\]

包含完整研究文档、40份历史原文、183份项目Python源码、已审计结果及21个AD/EXP/Hedgehog/FAVOR检查点。**按用户要求，不上传原始数据、QKV缓存或基础LLM权重，只保留获取方式、版本与逐文件校验信息。**

已补齐旧精简包未带的186份较大指标文件：当前仓库包括原项目各阶段`results/`目录中的**全部911份JSON/CSV结果文件**，包括全部17份`summary.json`及逐文档指标。精确范围、大小与SHA256见[RESULTS_COVERAGE.json](RESULTS_COVERAGE.json)。原`ARTIFACT_INVENTORY.json`描述的是补齐前的交接包；判断当前结果覆盖范围请使用新清单。

这不表示全部运行产物都已上传：大型中间特征/矩阵、所有历史检查点及完整终端日志仍未全部收录。报告、结果指标与原始激活数据是不同的材料。

**下载与阅读**

登录有访问权限的GitHub账号后，在仓库页面选择 **Code → Download ZIP**，一次下载整个仓库；也可以使用Git克隆。

```bash
git clone https://github.com/SJTU-huaxing/ad-kernel-handoff.git
cd ad-kernel-handoff
python verify_repository.py
```

新设备上的Codex先读：

1. [完整研究交接总册](AD_PROJECT_ALL_IN_ONE.zh.md)：当前公式、理论、实验、结果、负结果、全部研究发展及历史原文。
2. [双910B迁移步骤](ASCEND_910B_MIGRATION.zh.md)：后端适配、前向/梯度核验、双卡训练与预训练计划。
3. [数据与基模型获取方式](ACQUIRE_ASSETS.zh.md)：精确复现和重新抽取数据的区别。
4. [接续提示](NEXT_CODEX_PROMPT.zh.md)及[当前数值账本](CURRENT_RESULTS.zh.md)。

原`START_HERE.zh.md`和历史交接文档仍保留此前“两包解压”的记录；**使用本仓库时不需要再找那两个包，文件已经合并。** 旧文档的源码/检查点相对路径保持有效，旧SHA证据原样保留。原完整实验尚未在910B复现，也未完成从零预训练。

**建立可修改的工作目录**

```bash
python prepare_workspace.py
python inspect_target.py > work/target_environment.json
```

这会复制源码和检查点到`work/kan_attention_theory/`，保留`snapshot/`作为原始证据。`work/`不进入Git。旧代码存在CUDA和绝对路径依赖，须按迁移文档改造后运行；仅复制文件不代表NPU移植已完成。

**当前证据范围**

冻结Qwen2.5-1.5B、局部替换两层24/336 heads、单遍feature KL拟合。同深度、同参数、同m的EXP控制下，AD的8k KL低约15%–16%、局部PPL低约0.33%–0.43%；1k PPL略差，参考feature+state步骤慢约10.7%。这支持小规模完整预训练研究，尚不能声称普遍占优、全模型加速或达到总体理论下界。

文件校验只证明交接内容一致，不代替模型复现或统计泛化验证。
