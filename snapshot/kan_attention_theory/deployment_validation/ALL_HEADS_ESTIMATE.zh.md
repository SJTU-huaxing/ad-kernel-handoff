本文件仅做解析计算及历史计时外推，没有运行新模型或 kernel 实验。它不预测全 head 替换后的困惑度。

当前两种构造不同。正 kernel 使用谱坐标上的硬分区，再用条件均值表；Galerkin 使用连续的谱特征，没有分区。两者都有 1024 个训练 landmark，不需要访问历史 token 来计算新 token 的特征。

令 l_Q(q)_j=exp(q^T kbar_j/sqrt(d))，l_K(k)_i=exp(qbar_i^T k/sqrt(d))。以下固定数值缩放均吸收入投影矩阵、阈值和系数。

正 kernel：z_Q=W_Q^T l_Q(q)，z_K=W_K^T l_K(k)，c_Q=tree_Q(z_Q)，c_K=tree_K(z_K)，

    kappa_hat_P(q,k) = B[c_Q(q),c_K(k)],
    B_rs = E[kappa(Q,K) | c_Q(Q)=r,c_K(K)=s] >= 0.

实际 B 用独立训练文档估计。等价非负特征为 phi_Q(q)=B^T e_cQ、phi_K(k)=e_cK；维度 m=64。谱嵌入维度 e=64，W_Q,W_K 各为 1024×64。

Galerkin：投影训练积分算子、白化基函数并做 SVD 后，

    kappa_hat_G(q,k) = sum_{r<=m} sigma_tilde_r u_tilde_r(q)v_tilde_r(k)
                    = l_Q(q)^T A_Q A_K^T l_K(k),

其中 A_Q,A_K 各为 1024×64，吸收 sqrt(sigma_tilde)。这给出连续的 64 维特征，但不保证非负。这里描述的是本项目的 Galerkin-Schmidt 构造。

计算量假设：Qwen2.5-1.5B，28 层，12 Q heads/层，2 KV heads/层，D=1536，I=8960，d=d_v=128，vocab=151936。全部 336 个 Q heads 使用同规模、各自独立的新 kernel，a=1024，e=m=64。原模型 Q/K/V/O 投影、MLP 与 LM head 不变，删除所有原始 attention 的历史 KV 依赖。动态状态按每个 Q head 一份计算。

每 MAC 计 2 FLOPs，只计主导矩阵乘加；不把 exp、比较、除法等折成普通 FLOPs。原模型因果 prefill 按合法的三角区域计算，实际算子边界块可能有额外工作。Prefill 仅计算最后位置 LM logits，与此前缓存推理计时一致。

每 token 的全模型投影/MLP：

    F_base = 28 * [4D^2 + 4D*H_KV*d + 6D*I]
           = 2,620,391,424 FLOPs.

最后位置 LM head：F_lm=2D*vocab=466,747,392 FLOPs。

新 kernel 每 head、每新 token 的 Q/K 特征：

    F_phi = 4a(d+e) = 786,432 FLOPs,

另外需要 2a=2048 个 exp。全部 heads 合计每个 decode token 有 688,128 个 exp。

正 kernel 的稀疏桶更新 O(d_v)，输出读出主导成本约 2m*d_v=16,384 FLOPs/head。因此，T 为当前可见 token 数时：

    F_decode_old(T) = F_base + F_lm + 336*2T(d+d_v)
                    = 3.087138816e9 + 172032*T.
    F_decode_new    = F_base + F_lm + 336*(F_phi+2m*d_v)
                    = 3.356884992e9.

标准线性聚合的 prefill 主导成本约 4m*d_v/token/head：

    F_prefill_old(N) = N*F_base + F_lm + 336*N*(N+1)*(d+d_v).
    F_prefill_new(N) = N*F_base + F_lm + 336*N*(F_phi+4m*d_v).

当前块长 64 的分块因果实现另有约 336*N*2*64*(m+d_v) FLOPs，8k 时使整模型估计增加约 0.068 TFLOPs。结果保存在 results/all_heads_analytic_estimate.json。

| 上下文长度 | 原模型 decode GFLOPs/token | 新模型 decode GFLOPs/token | decode 变化 | 原模型 prefill TFLOPs | 新模型 prefill TFLOPs | prefill 变化 |
|---|---:|---:|---:|---:|---:|---:|
| 1024 | 3.263 | 3.357 | +2.9% | 2.774 | 2.966 | +6.9% |
| 2048 | 3.439 | 3.357 | -2.4% | 5.728 | 5.931 | +3.5% |
| 4096 | 3.792 | 3.357 | -11.5% | 12.177 | 11.861 | -2.6% |
| 8192 | 4.496 | 3.357 | -25.3% | 27.240 | 23.722 | -12.9% |
| 16384 | 5.906 | 3.357 | -43.2% | 66.024 | 47.443 | -28.1% |
| 32768 | 8.724 | 3.357 | -61.5% | 178.227 | 94.885 | -46.8% |

逻辑 FLOPs 的 crossover 约为 decode 1568 tokens、prefill 3200 tokens；它们不是实测速度 crossover。Galerkin 的主导特征成本相同，dense 状态更新使 decode 每 head 多约 2m*d_v FLOPs，整模型差异很小。

时间必须另作条件性外推：当前原模型使用 BF16，新增 landmark 特征使用 FP32，TF32 关闭；GPU 吞吐、启动和访存均不同。[NVIDIA 矩阵乘法性能说明](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html)

Prefill 8192 的历史数据：原模型 472.15 ms，新 kernel 的 4-head Q/K 特征 2.7064 ms（融合、CUDA Graph）。若全 heads 的吞吐可按 head 数近似线性外推，新特征总预算为 84*2.7064=227.34 ms。

仅作为敏感性模型，用原模型 N=1024/4096/8192 的三个历史计时拟合 t(N)=b0+b1*N+b2*N^2，得到 b0=5.907 ms，b1=0.050737 ms/token，b2=7.5408e-7 ms/token^2。8k 的非二次部分约 421.55 ms。若把二次项近似视为被删除的 attention 工作，替换后为：

    t_prefill_new(8192) ≈ 421.55 + 227.34 + t_linear_aggregation
                       ≈ 648.89 ms + t_linear_aggregation.

这不是 attention profiling：三个点无法区分算子效率变化与真正的二次工作。若给线性聚合及其他误差预留 0–100 ms，得到约 0.65–0.75 秒，较历史原模型约慢 40%–60%。这个区间是明确假设下的预算范围，不是置信区间或实际全 head 结果。

Decode 的历史原模型约 22.07 ms/token（8k）。4-head 特征的 graph 时间为 0.040416 ms，eager 为 0.366048 ms。

若 12-head 批处理花费为现有 4-head 的 1–3 倍，则全部 28 层的新增特征 GPU 预算约 1.13–3.39 ms/token。若 eager 的主机开销每层保留一次、GPU 工作按 3 倍扩展，则新增特征预算约：

    28 * [(0.366048 - 0.040416) + 3*0.040416] = 12.51 ms/token.

总时间应为 t_old - t_removed_attention_and_cache + t_features + t_linear_and_glue。被删除路径尚无独立 profile，故不能唯一确定净变化。作为显式敏感性场景：

| 场景 | 假设删除的原 attention/cache | 新线性聚合及连接开销假设 | 预计总 decode |
|---|---:|---:|---:|
| 新 attention 路径 graph/fused | 1–3 ms | 0.1–0.5 ms | 约 20–25 ms/token，约 -10%～+15% |
| 延用 eager 组织 | 1–3 ms | 0.5–2 ms | 约 32–36 ms/token，约 +45%～+65% |

表中删除成本与连接开销是敏感性假设，不是测量。不能用四个 heads 的端到端时间差直接乘 84，因为原混合适配器还包含分组/索引开销，全替换会移除它们；并行度和数据复用也会改变。

全部 heads 独立维护 FP32 状态时，动态状态为 336*64*129*4=10.58 MiB，8k 原 BF16 KV 为 224 MiB。但当前 kernel 还新增约 509.25 MiB 的主要静态参数（anchors、投影、B 表，未计少量树元数据）。这些静态参数不随上下文增长，却会影响单 token 访存。因此固定状态带来的缓存优势与实际速度优势需要分别评估。

这些推算不验证：全 heads 的精度/数值稳定性、低精度特征可用性、最优算子或生产引擎吞吐。它们说明当前形式在长上下文能减少逻辑计算和动态缓存，但现有 FP32 landmark 实现不保证更低延迟。
