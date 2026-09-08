你正在接续一个已有真实实验和理论记录的研究项目。先阅读START_HERE.zh.md、AD_PROJECT_FULL_HANDOFF.zh.md、ASCEND_910B_MIGRATION.zh.md、CURRENT_RESULTS.zh.md；需要原始细节时读取SOURCE_INDEX及snapshot对应文件。不要仅凭本段摘要开始改模型。

用户明确不允许使用子agent。目标设备有两张昇腾910B，但环境尚未检查。当前已确认目标kernel为：

h(q,k)=exp(s_K(k))*softmax([z_Q(q),0])^T*softmax([z_K(k),0])。

删除的是C和Q幅度，保留K幅度。当前pure版本无bias、Q128→192→63、K128→192→64、SiLU、m64、73536参数/head；真实post-RoPE输入。不是固定分区、不是KAN、没有旋转增强或GDN门作为默认组件。

主结果是冻结Qwen2.5-1.5B两层24/336heads的feature单遍KL拟合和局部替换PPL，尚未从零预训练。最新同深度EXP控制在两种初始化、三个种子下支持8k收益，但1k PPL略差、当前参考step约慢10.7%。理论仅确认结构/梯度与适用域明确的风险关系，没有证明普遍优越或总体达界。历史Gaussian与某些kernel构造失败、GLA/GDN的strong residual控制更好，必须保存。

现在先校验交接包，读取目标设备、CANN/torch_npu/驱动及显存，整理明确的NPU移植清单。不要安装源CUDA wheel。源码中的模块级Triton CUDA导入、inference_mode scan、原地state和HF cache更新都需处理；不能把换.cuda()为.npu()当成已支持预训练。

保持snapshot和历史结果只读，在新目录进行设备无关CPU FP64 oracle、NPU FP32前向/梯度及BF16稳定性检查。先确认源检查点函数和GQA/RoPE正确，再做小规模可微训练scan、单卡和HCCL双卡烟测。不得仅复用已有JSON就报告NPU复现。

之后围绕100M–150M完整LM预训练制定具体预算和协议：AD与同深度EXP是主归因，同数据、m、参数与计算记录；加入HH/FAVOR/softmax参考。主损失为LM交叉熵，K幅度可由任务梯度直接训练，不需要Q幅度/C辅助loss。是否混合窗口、换归一化、clip幅度等都是新设计，须显式消融。目标是跨seed、新数据、长上下文与质量—计算曲线的可靠结果，稳定后再考虑350M–1B。

不需要重新问用户已明确的“删除哪一侧幅度”、是否保留原始负结果、是否允许只做文档读取和可逆迁移核查。缺失的预训练总预算或系统级操作条件，应在实际设备证据和具体可审核方案形成后再处理；不要凭猜测宣称已经授权任何付费云服务。所有新结果保存协议、源码/数据/检查点哈希、逐文档指标与数值精度记录。
