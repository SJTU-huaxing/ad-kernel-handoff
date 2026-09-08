import json,pathlib,math
import numpy as np
P=pathlib.Path(__file__).resolve().parent;R=P/'results'
def read(f):return json.loads(f.read_text())
def main():
 a=read(R/'summary.json');bm=read(R/'benchmark.json');mi=read(R/'microbenchmark.json');gauss=read(P/'checks/gaussian_moment_domain.json');cw=read(R/'causal_witness.json');rot=read(R/'rotation_diagnostic.json');audit=read(P/'checks/final_audit.json')
 def pr(mode,k,split):return next(r['ppl_mean'] for r in a['ppl'] if r['mode']==mode and r['kind']==k and r['split']==split)
 def kr(k,split):return next(r for r in a['kernel'] if r['kind']==k and r['split']==split)
 choices=[('pure','teacher','原模型'),('pure','favor','FAVOR+'),('pure','hedgehog','Hedgehog公式控制'),('pure','learned_prf','可学习PRF控制'),('pure','split_kl','同架构KL'),('pure','exp_kl','exp-MLP KL'),('pure','softplus_kl','softplus-MLP KL'),('pure','split_01','原始质量+KL，λ=.1'),('hybrid','split_01','λ=.1 + 精确窗口64'),('hybrid','softplus_kl','softplus-KL + 精确窗口64'),('hybrid','calibrated','原始质量校准 + 窗口64'),('hybrid','gate','匹配门控KL + 窗口64'),('window','swa448_sink4','SWA448 + 4 sinks')]
 tab='| 方法 | Wiki1024 PPL | Wiki8192 PPL | 文学64 PPL |\n|---|---:|---:|---:|\n'
 for mode,k,label in choices:tab+=f"| {label} | {pr(mode,k,'confirm_wiki'):.6f} | {pr(mode,k,'confirm_long'):.6f} | {pr(mode,k,'confirm_prose'):.6f} |\n"
 kt='| 方法 | Wiki输出NMSE | 长文输出NMSE | Wiki原始kernel NMSE | 长文原始kernel NMSE |\n|---|---:|---:|---:|---:|\n'
 for k in ['split_1','split_01','split_kl','exp_kl','softplus_kl','calibrated','gate','favor']:
  s=kr(k,'confirm_wiki');l=kr(k,'confirm_long');kt+=f"| {k} | {s['output_nmse']:.6f} | {l['output_nmse']:.6f} | {s['raw_nmse']:.4g} | {l['raw_nmse']:.4g} |\n"
 bt='| 模式／方法 | prefill ms | decode ms/token | KV＋状态＋窗口 MiB |\n|---|---:|---:|---:|\n'
 for r in bm['results']:
  if r['prompt_tokens']==8192:bt+=f"| {r['mode']}/{r['name']} | {r['prefill_ms']:.3f} | {r['decode_ms_per_token']:.3f} | {r['total_bytes']/2**20:.5f} |\n"
 ct='| 比较A−B，负值有利A | 集合 | ΔNLL/token | 配对文档95%区间 |\n|---|---|---:|---|\n'
 for r in a['paired_comparisons']:
  if r['split']=='confirm_long':ct+=f"| {r['a']} − {r['b']} | 8192 | {r['mean']:.6f} | [{r['paired_document_95_ci'][0]:.6f}, {r['paired_document_95_ci'][1]:.6f}] |\n"
 lb=np.array(cw['summary'])[:,2];md={r['name']:r['microseconds_per_layer'] for r in mi['results'] if r['component']=='features_and_state'};ratio=md['split_01_s11']/md['favor_s11']
 text=fr'''本轮把三项要求分别落实为有条件的理论、严格匹配控制和真实状态部署。结果是：**固定缓存节省已得到验证；主方法没有稳定胜过强KL控制；接近未知总体下界的主张仍缺乏证据。** 共同RoPE旋转揭示了一个明确的表示泛化缺口，后续增强实验独立保存在 [orbit_direction](../orbit_direction/PROTOCOL.zh.md)。

**范围与公平性。** 冻结Qwen2.5-1.5B，revision `{audit['model_revision']}`。为了释放GQA缓存，将原来L14/L27的少数head扩展到这两层全部24个Q heads、4个KV组。其余26层保留原始attention；完整模型PPL指整个冻结模型输出，不代表已经全模型线性化。训练4096文档，各64个不同Q、全部合法前缀K，共每head {audit['training_pairs_per_head']:,} 个配对，4096次单遍更新。比此前只用后半Q与前半K，覆盖了自注意力、近邻和所有因果位置。

主MLP两侧128→192→64，73856参数/head；λ=1、.1、0及exp/softplus-KL均匹配参数、数据、顺序和优化预算。λ=.1在验证集输出误差上预先选为主方案，所有其他结果保留。FAVOR+、Hedgehog公式、learned PRF都使用m64，后两者参数分别4128、8256/head，属于论文公式控制与共同单遍预算实验，未完整复现或充分调优原论文训练流程，不能将胜过这些控制等同于胜过论文SOTA。校准额外使用同4096文档中不重叠的64个Q，仅优化193参数的幅度读出并折叠回原参数预算；数据配对不重复，文档被再次使用，额外训练计算已明示。

**一、理论收紧后的可用结论。**

真实模型在投影前使用RMSNorm，投影权重有限，默认RoPE为正交变换。因此真实Q/K范数有界，原始κ的L²存在，Schmidt谱尾定理有严格适用基础。它不要求真实分布是Gaussian。

相反，以本轮全部缓存训练Q/K协方差构造的Gaussian替代模型，**{gauss['valid_heads']}/{gauss['total_heads']} heads**满足a_max<1/2；a_max范围{min(gauss['maximum_a']):.3f}～{max(gauss['maximum_a']):.3f}。该最终检查使用每head全部262144个Q、每KV组全部4194304个K，初始按位置抽样的结果已另外归档。这否定了直接在这些head上使用有限Gaussian二阶谱公式的前提，不能解释为真实kernel二阶矩发散。RMSNorm给出的保守logκ上界仍达2.25万～6.30万，数学上的有限性不会自动提供有用的统计界。

对固定乘积参考分布，令C为原始kernel诱导概率的可测粗粒化矩阵。数据处理、Pinsker与最佳核范数rank-m逼近给出原始I风险的合法下界：

$$ R_I(h)/\mathbb E\kappa\ge\tfrac12\big(\sum_{{r>m}}\sigma_r(C)\big)^2. $$

对query-balanced目标也有相同形式。它用整个选定经验边缘的完整乘积积分，实际MLP没有查表分区；但固定key bank与未知P_K之间的差异仍未被置信区间覆盖。

因果场景另用合法区域的互不重叠矩形分解。令w_b为矩形真实概率质量，C_b为其中条件概率矩阵，则所有m维正可分离kernel满足

$$\mathbb E_C\mathcal L_\lambda(C)\ge\mathbb E_C B_m(C),\qquad B_m(C)=\tfrac12\sum_b w_b\big(\sum_{{r>m}}\sigma_r(C_b)\big)^2.$$

这修正了把三角mask矩阵直接当rank-m矩阵的错误。完整证明、原始风险与方向／质量分解、有限样本条件见 [THEORY.zh.md](THEORY.zh.md)。

实际128篇新Wiki的前512位置，m64因果下界按head平均后范围{lb.min():.3g}～{lb.max():.3g}，中位数{np.median(lb):.3g}。乘积分布下的粗粒化界也很小。**两类界加入所列保守iid文档置信修正后均为0。** 因此不能声称未知总体最小误差已数值确定，也不能声称新kernel接近该最小误差；一个很松的下界不允许反过来证明方法离真实最优很远。原始无约束rank-m、正特征rank-m、固定宽度MLP三种最优值仍需区分。

**二、拟合的仍是原始kernel，长上下文质量尚未守住。**

主方法数学形式保持为连续正特征：

$$h_{{orig}}(q,k)=e^{{s_h}}64e^{{s_Q(q)+s_K(k)}}\pi_Q(q)^\top\pi_K(k),\quad\pi_X=\operatorname{{softmax}}([z_X,0]).$$

s_h是训练数据确定的固定head数值标度，可吸收到两侧feature中。它不是每行attention归一化。λ>0的目标为方向KL加λ倍原始质量I项，零损失目标仍是κ；λ=0明确是强KL控制。MSE只在测试中作为诊断，未用于神经训练。

三种子、24heads平均；原始NMSE先按head汇总该集合平方误差／平方能量，再平均head与种子：

{kt}

原始质量失真和归一化输出失真不能混为一谈。特别是长上下文中的巨大原始误差，一部分来自幅度外推，attention分母会消去query共同标度，所以归一化结果看起来没有同样严重。查询校准保持纯线性方向不变；它相对于原始未校准KL有明显幅度作用，但不能以此主张更好的纯线性attention。完整kernel表和逐文档数据见 [summary.json](results/summary.json)，图见 [kernel_generalization.pdf](figures/kernel_generalization.pdf)。

**三、完整模型PPL与真正的强控制。**

确认集为128篇1024-token Wiki、24篇8192-token新长文、128条64-token文学前缀，排除既有训练/验证/确认文本和token哈希。下表为三训练种子PPL均值，原模型与SWA无训练种子：

{tab}

文学64在混合版本中完全位于精确窗口内，不能用它证明学到了远处泛化。与原模型的细小差异来自BF16／FP32 attention实现的积累次序；所有混合feature在这个集合给出相同结果。它也不是标准LAMBADA最后词准确率。

同架构KL、softplus-KL和同读出预算的门控KL不可省略。原始质量校准与门控KL在长文的差异区间跨0，在短文则略逊于门控KL；不支持“原始质量校准独有的输出收益”。近邻混合优于某些纯线性方案，但SWA448+4的持久缓存与混合方案只差4096 bytes（约0.44%选中层预算），且本轮PPL更好。

{ct}

区间对文档做配对bootstrap，条件于这三个固定训练种子；另外保留每种子的ΔNLL。不能把24个head或重复token当独立模型样本，也不能由三个种子外推完整训练随机性分布。混合softplus-KL的完整模型结果是在其已冻结kernel的首批指标后追加的强控制，未改其训练；因果矩形与旋转测试也明确标记为事后机制诊断。

[LoLCATs](https://arxiv.org/html/2410.10254v2)已使用精确窗口＋远处线性、共同分母和相关算子优化，混合结构本身不构成新颖贡献。[2026-08-28的SWA预印本](https://arxiv.org/html/2608.28444v1)进一步强调了无需训练的滑窗＋sink比较。本轮加入了该类对照；当前结论仅限本模型、两层和这些PPL数据，未完整复现该论文广泛任务结果。

**四、部署收紧：缓存是真的省了，速度没有显著赢。**

RTX3090、BF16模型、FP32 feature/state、TF32关闭，batch1。3次prefill预热、6轮prefill+32固定后续token解码，首轮丢弃；不存在并发GPU工作。FAVOR与MLP都采用融合特征投影和同一融合状态更新，8k实际结果为：

{bt}

旧实验的缓存问题已修复：被替换两层的DynamicCache真实存储为0，另存24×64×(128+2)个FP32数，即798720 bytes。n8192时全模型224MiB降至208.7617MiB，节省约6.80%；64近邻窗口另加128KiB。状态与窗口在继续decode时保持固定大小，其余26层KV仍随长度增长。

纯线性主方案21.229ms/token相对原模型21.320ms/token仅约0.43%的中位数差别，处于本次重复波动范围，不能宣布稳健加速；prefill稍慢，混合参考实现也更慢。当前是HF eager局部替换，不是已优化的全模型线性服务系统。短上下文低于约390tokens时，本配置FP32状态还可能比被移除的BF16 GQA KV更大。MLP额外约6.76MiB权重也应计入总显存，不能只报KV而冒充总显存同幅下降。8k的prefill新增峰值显存实测从原模型745.1MiB升至纯MLP773.2MiB，校准混合版本为819.0MiB；向量化前缀状态与局部窗口临时张量尚未优化，持久缓存减少并不等于prefill峰值减少。

剥离Python逐算子派发、以真实Q/K做CUDA graph热缓存微测：FAVOR投影＋状态为{md['favor_s11']:.3f}µs/层，主MLP为{md['split_01_s11']:.3f}µs/层，MLP约{ratio:.2f}倍。主导FLOPs比为2.75倍，单看feature投影为4.5倍。这说明融合和固定开销可以缩小耗时倍率，**不会减少数学FLOPs，也不能推出MLP比FAVOR天生更适合GPU**。FAVOR微测仍更快。完整模型差距小还因为仅替换2/28层，其他计算与权重读取占主要开销。图见 [inference.pdf](figures/inference.pdf)。

FP64显式矩阵／递推、FP32融合feature、8k完整真实Q/K的显式参考、完整模型prefill→decode和真实storage均已检查。8k数值检查最大相对输出误差1.46e-6，没有零分母，说明这批质量问题不能归咎于已测到的递推数值错误。详见 [检查](checks/long_precision_and_window.json)。

**五、下一步的证据来自不变性缺口。**

共同RoPE旋转保持所有原始kernel值和对应分布的Schmidt谱不变，验证中最大logit差为{rot['max_raw_logit_invariance_error']:.3g}。然而固定主特征在offset0→1024下，方向KL约0.364→1.973；exp-KL约0.348→2.267。FAVOR的平均KL在该诊断下变化很小，但其原始水平更差。这个受控变化说明，固定feature子空间的对齐／泛化是谱下界之外的一项实质问题。它不能证明共同平移是所有真实长上下文失败的唯一原因，RoPE不变性本身也并非新发现。

据此单独启动相同m、参数量、更新数的共同旋转增强与增强KL控制，并使用另外80篇全新文档确认，见 [后续实验协议](../orbit_direction/PROTOCOL.zh.md)。它是否值得继续，必须由新确认结果决定；单纯“MLP正特征＋一般谱框架”，当前仍不足以支持顶会方法论文。主张创新需要同时拿出超越同样增强的KL、匹配缓存的SWA的质量／效率结果，并解决证书过松的问题。

复现入口：[REPRODUCE.zh.md](REPRODUCE.zh.md)。本轮审计通过：36个feature/checkpoint构造、108组kernel评价、132组完整模型PPL、24个全模型计时条件；[final_audit.json](checks/final_audit.json)与[artifact_manifest.json](results/artifact_manifest.json)保留预算、哈希和结果完整性证据。
'''
 (P/'REPORT.zh.md').write_text(text)
 print(str(P/'REPORT.zh.md'))

if __name__=='__main__':main()
