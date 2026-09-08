本次为2026-09-08对已有同参结果、计算口径和相关工作的核对，没有新增训练或推理计时。实验来源为hedgehog_matched/results/{summary_product,summary,benchmark}.json。论文结论以原文为准；Hugging Face markdown可能缓存旧版本，尤其2506.21137的标题与公式应核对arXiv v3。

**已有结果与预算。** 冻结Qwen2.5-1.5B，层14/27全部24个Qheads，4096训练文档、每个拟合单遍、3种子。原始目标为exp(qᵀk/√128)，主比较使用跨文档配对的未平衡原始I-divergence，每head固定训练尺度只用于数值计算。AD与HH-exp各73729训练参数，独立Q/K网络、无投影偏置，并各有一个训练head标度。HH是加宽至128→288、拼接exp(z)/exp(-z)的同参适配，不是论文默认配置或完整训练系统。

| 项目 | AD | 同参HH-exp |
| --- | ---: | ---: |
| 两侧网络 | 128→192→64，SiLU | 128→288，exp正负拼接 |
| 特征维度m | 64 | 576 |
| 参数/head | 73729 | 73729 |
| A→B原始I相对风险 | .272450 | .469970 |
| B→A原始I相对风险 | .266388 | .470631 |
| A→B原始L2相对误差 | .365211 | .370372 |
| B→A原始L2相对误差 | .299519 | .363239 |
| 乘积训练后1k全模型PPL | 9.025792 | 9.063196 |
| 乘积训练后8k全模型PPL | 9.803906 | 9.863059 |

I风险下降42.03%/43.40%；按3种子均值，每方向24/24head更低。A/B是128篇既有留出文档拆成的两个64文档集合；每方向1024个Q×4096个K的完整所选向量乘积。不是未知总体，也不是每个pair独立。A→B的L2差值95%条件区间[-.04962,.07082]跨零。PPL收益仅约.41%/.60%，且都劣于原模型8.664560/9.066279。测试集合此前使用过；不是首次盲测。

按Hedgehog惯用的attention方向KL，另一次同因果采样目标匹配中，AD/HH-exp的1k PPL为8.85819/8.87029，8k为9.69226/9.75236；HH-softmax为8.88021/9.75233。AD只略优。此处各登记73728参数，但AD的192个query幅度输出权重不被纯KL识别，不能说有效自由度严格相同。原始I主比较不受该问题影响。乘积raw训练的AD不应因胜过同目标HH，就被称为胜过所有采用原生目标训练的Hedgehog。

**计算与速度。** FLOPs采用一次乘加计2的标准主导项，不计原模型QKV投影、FFN等公共计算，也不完整计入激活、softmax归约、输入标准化或动态稳定重缩放。对一对新增q/k及value，双侧投影AD为4h(d+m)，HH为4dp；状态更新与读取约4md_v。本实验d=d_v=128,h=192,p=288。

| 主导FLOPs/head/token | AD | HH-exp | FAVOR+ m64 |
| --- | ---: | ---: | ---: |
| 双侧投影 | 147456 | 147456 | 32768 |
| 状态更新与读取 | 32768 | 294912 | 32768 |
| 合计 | 180224 | 442368 | 65536 |

AD主导FLOPs较同参HH少59.26%，状态为其1/9；相同参数不等于相同m、状态或FLOPs。若同m64，单层HH所需参数远少于当前AD，因此速度胜利不可推广为全部HH预算。

计时RTX3090、FP32、batch1、单层12heads、真实QKV，CUDA graph热缓存，同一未充分融合的PyTorch参考实现：

| 方法 | 仅特征µs | 特征+状态更新/读取µs | 状态MiB/层 |
| --- | ---: | ---: | ---: |
| AD | 42.688 | 74.204 | .380859 |
| HH-exp | 28.288 | 98.240 | 3.427734 |
| FAVOR+ m64 | 27.520 | 59.040 | .380859 |

AD特征生成耗时为HH的1.51倍，完整微测耗时少24.47%（1.324倍加速）。与FAVOR相比，AD完整微测耗时多25.68%。状态包含FP32的s,z,g；这些数字不是全模型decode。计时来自首阶段权重；乘积阶段网络形状、算子相同，但权重没有单独重测。不能把旧Triton融合5–10µs数据与本表拼接，也没有据此证明AD算子对GPU内在更友好。

**原文相关工作与重合范围。**

- [Hedgehog，2024](https://arxiv.org/html/2402.04347v1#S4.SS2)：学习正feature map，以归一化attention交叉熵/蒸馏匹配教师。正文与附录的共享描述不完全一致，本地采用附录转换路径的独立Q/K模块。MLP正特征、冻结教师后的特征拟合不是当前工作的独有创新。
- [STILL，2026，§3.2](https://arxiv.org/html/2602.02180v1#S3.SS2)：NP-Map先计算u=f(x)/||f(x)||·||x||，然后concat(softmax(u),softmax(-u))。这是LLM转换中明确的方向/幅度分离与原始输入范数重注入。其范数进入softmax内部，改变方向；不是学习原始kernel分布积分Z(q)。完整STILL还包含混合路由，不能把其全系统效果归给NP-Map。
- [Norm×Direction / NaLaFormer，v3](https://arxiv.org/html/2506.21137v3)：显式分解范数和方向，query范数进入逐元素幂的指数，结合三角方向特征。它并非简单标量乘法，不能误称其norm分支在分母中完全消去。其研究目标偏向注意力尖锐性与模型质量；不是当前原始I风险学习方案。
- [MALA，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/papers/Fan_Rectifying_Magnitude_Neglect_in_Linear_Attention_ICCV_2025_paper.pdf)：已分析feature query范数的消去，并以缩放与平移改变attention权重，见[§3.3](https://arxiv.org/html/2507.00698v3#S3.SS3)。含减法、上下文相关系数，理论上可能出现非正权重，与固定正可分离原始kernel不同。不能据某些齐次feature的结论断言所有非线性feature都忽略原始q范数。
- [Degrees of Freedom for Linear Attention，NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/c98ef086dc70d528e1c1aa1e66893365-Abstract-Conference.html)：使用分布相关积分算子/有效维度确定feature预算，并逐层学习PRF；原文§3.3同时研究原始kernel L2和softmax交叉熵。分布相关谱理论、原始kernel拟合及逐层学习不能被笼统宣称为首次。
- [FAVOR++ / Chefs’ Random Tables，NeurIPS 2022](https://research.google/pubs/chefs-random-tables-non-trigonometric-random-features/)和[FAVOR#，2023](https://arxiv.org/abs/2302.00787)：已有参数化正随机特征、利用输入统计降低估计方差的理论与构造。当前AD是学习有限正函数族，未承诺随机特征的无偏性；仅超过基础FAVOR+不是充分现代基线。
- [Efficient Attention，WACV 2021](https://openaccess.thecvf.com/content/WACV2021/papers/Shen_Efficient_Attention_Attention_With_Linear_Complexities_WACV_2021_paper.pdf)：已有Q/K分别softmax归一化并以共享上下文向量混合。其K沿序列维归一化，与当前两侧按feature维归一化、学习独立幅度不同；仍说明“softmax feature + 正混合”并非空白。

检索原论文未发现与当前“m−1个方向logit＋1个可学习log幅度、双侧两层网络、在真实PQ×PK上以raw I拟合exp(qᵀk/√d)”全部一致的配方。这不是首创证明。未对新发现的NP-Map、NaLaFormer、FAVOR++/#做同协议训练比较。

**创新性判断。** 单看公式与幅度/方向拆分，创新性偏弱，已有明确近邻；共同函数类层面，任何严格正向量f都能写成(sum f)·(f/sum f)，所以这是一种坐标表示，而非独有表达能力定理。原始I质量分解也是经典恒等式的应用。有限网络的参数化会影响表达与优化，需同深度、同m、同损失的exp/softplus MLP控制来检验，不能把两层AD对一层HH的差异全部归给幅度分支。

可继续聚焦的贡献是：在固定正特征状态与网络预算下，用分布相关原始kernel风险解释并控制幅度/方向两项；得到跨文档、长文及第二模型可复现的性能—状态—速度曲线。当前42–43% raw I下降是有价值的初步信号，非总体最优性或充分顶会证据。若能进一步给出有限m正方向混合的非平凡逼近上界，连接正性代价、网络容量及估计误差，并在NP-Map和同深度MLP等近邻对照下维持优势，贡献会更清楚。无需改变用户的原始kernel目标或转向混合窗口研究。
