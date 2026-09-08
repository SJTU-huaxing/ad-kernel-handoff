本轮不训练、不选择检查点，不改变既有heldout数据。主对象为KL训练的纯删除AD，73536参数/head，m64；对照为既有同预算Hedgehog-exp/softmax，73728参数/head，m576，以及固定随机FAVOR+ m64。AD较HH少192参数/head（约0.26%）；FAVOR没有数据拟合，不能称同训练参数比较。

以1k128文档、8k24文档和种子11/29/47评价。缓存kernel指标基于原教师QKV的64个选定Q/文档；全模型PPL覆盖全部下一token，并让两层24/336个head同时替换后真实传播。头均值、token加权风险、输出误差和PPL分别报告，不能互换。

FAVOR使用与本轮AD/HH完全相同的Replacement、状态算子、精度、模型revision和文档重新运行verify、8k precision、kernel和PPL。原AD/HH文件只读复用；保存父数据来源和hash。

诊断全部4方法×3种子在两种split上的attention KL、TV、系数L2、输出平方误差、按head和位置分解，以及通过冻结W_O后的层输出误差。位置分桶只是既有8k文档中的观察，不能分离长度、内容、RoPE和训练分布等因素。诊断不用于调参。配对文档bootstrap条件于3个固定模型，另报告全部种子差值及胜出文档比例；不把tokens当独立样本，不做未知总体或顶会结论。

运行同一硬件参考计时复用上一轮新测结果。不是端到端解码benchmark。没有全层转换或LLM微调。所有结论限定本次模型、head、语料和训练预算。
