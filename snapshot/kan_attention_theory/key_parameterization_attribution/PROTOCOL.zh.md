2026-09-08，本轮训练前固定方案。目标是归因AD中K幅度—方向参数化的作用，主比较为AD vs直接exp，两者同m64、同Q结构128→192→63+末尾0、同K结构128→192→64、SiLU、无偏置、73536有效参数/head。Q输入和K输入沿用相同训练标准化。Q网络结构/初始化相同，均允许正常训练；不把“相同Q网络”理解为训练过程中强制权重相等。没有查询幅度/C，没有raw I或其他辅助损失。

AD：phi_Q=softmax([z_Q,0])，phi_K=exp(s_K)softmax([z_K,0])。K网络的前63输出为方向logits，末尾输出为s_K。
EXP：相同phi_Q，phi_K=exp(u_K)，u_K为K网络全部64输出。仅此输出变换不同，没有缩减/增添hidden width或state维度。

两种初始化：
1. standard：相同seed、相同原AD初始化代码产生完全相同Q/K权重；不同K输出变换会产生不同初始attention。AD复用上一轮已固定的纯删除KL检查点；EXP新训练。
2. matched_zero：Q和K第一层仍按同seed初始化，把双方K第二层所有权重置0。AD的K特征为1/64，EXP为1；这只是所有keys共享的常量比例，归一化attention都为合法因果前缀上的均匀分布，对任意Q/K完全相同。此对照排除初始attention函数差异，但无法证明跨所有初始化/优化器的优势。两边Q可正常训练，第一步某些梯度为0是此初始化的性质，不视为失效参数。

冻结Qwen2.5-1.5B，层14/27全部24/336Qheads，训练只更新新feature网络。4096篇1k文档，64个选定Q/文档、全部合法prefixK，共134360517训练pairs/head。每个选定pair单遍反向传播，向量可复用。与父实验严格相同文档顺序seed85000+seed、AdamW wd1e-4、每head梯度clip10、余弦lr至lr/10、纯方向KL。

新增配置standard_EXP、matched_zero_AD、matched_zero_EXP。每个配置seed11搜索lr0.002/0.0005，只按同一validation前32文档的KL选择；选择后训练seed29/47。共12新拟合、9新选中检查点，加复用standard_AD的3种子，共4组12个被比较模型。不会根据heldout结果改epoch、初始化、lr、数据或裁剪。所有方法的2个LR候选validation结果都展示。

评估沿用confirm_wiki128篇1k、confirm_long24篇8k；系数KL为主拟合指标，同时TV、系数L2、输出NMSE、W_O后NMSE、概率低估质量与完整冻结模型两层局部替换PPL。L2/MSE只评价，不用于训练。原先的8k数据已使用过，不能称新盲测。所有下一token的PPL与缓存64Q的静态误差不同，分别报告。

先核验参数数目、标准初始化逐参数相同、zero初始化attention恒等、计算/梯度连接；选定模型做FP64矩阵/scan/递推及真实8kFP32精度检查。所有新模型PPL采用与父AD相同Replacement和算子，teacher重测。禁止用test matrix重拟合来解释泛化。

推理微基准重新在同一RTX3090、FP32、batch1、单层12heads、CUDA graph热缓存、同状态算子下测AD和EXP；包含标准及函数匹配初始化训练权重。两者状态均m64，主导投影FLOPs均147072/head/token，状态计算约32768。不是全模型decodebenchmark。训练时长只列运行记录，不将跨历史运行的壁钟差异解释为算法训练加速。

归因以EXP−AD配对比较为准，分别报告两个初始化下各种子、head/文档差异和条件文档bootstrap区间。只有在同深度、同m、同参数的对照下仍有优势，才支持AD参数化的增益。若结果依赖初始化或评价口径，按条件结论报告；不能据一次单遍实验声称总体最优、渐近容量优势、或把所有与Hedgehog的差距单独归因网络深度。
