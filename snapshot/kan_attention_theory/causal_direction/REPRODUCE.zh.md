工作目录 `/root/autodl-tmp`，Python `/root/autodl-tmp/conda-envs/nonlinear-qk/bin/python`，模型与数据从固定本地Hugging Face缓存读取。所有训练LLM权重冻结、TF32关闭、单卡RTX3090。

依次运行 `data.py select`、`data.py extract`、`train.py`、`calibration_data.py`、`calibrate.py`、`verify.py`、`confirm_driver.py`、`verify_more.py`、`causal_witness.py`、`rotation_diagnostic.py`、`gaussian_full_covariance.py`、`benchmark.py`（文件均在本目录）。其中后三项机制/条件诊断是在初始确认后追加，不能回写为预先的模型选择规则。已有训练和逐配置结果会跳过，重做应使用新的输出目录或先移走相应旧结果。不能在确认结果上继续调参再将同一确认集称为新测试。

报告用 `/root/miniconda3/bin/python analyze.py` 生成统计和图，再用研究Python运行 `audit.py`，最后用前者运行 `write_report.py`。`gaussian_full_covariance.py`应在`rotation_diagnostic.py`之后运行：最终条件审计使用全部262144个缓存Q/head、4194304个K/group；初始按位置抽样的检查另行归档。因果矩形的FP64小矩阵SVD使用CPU LAPACK，避免RTX3090的Jacobi分解开销，不改变矩阵或样本数。

`confirm_driver.py` 顺序执行确认激活提取、kernel评价、总体谱见证数值、完整模型PPL；不并发GPU进程。`benchmark.py` 单独运行，输出实际缓存存储与全模型时间。数据分片包含真实post-RoPE Q/K，原训练64 Q/文档与校准64 Q/文档位置严格不重叠；K可以重复作为不同Q的配对对象，同一配对不重复训练。训练统计、校准初始化/标准化并不算额外SGD epoch，但确实使用对应训练数据统计；报告不能声称每个datum只被程序读取一次。

主损失 λ>0 的零点仍为原始指数kernel。固定head log_scale仅作数值标度，精确混合分支也减同一个值。测试raw NMSE只作诊断，不作神经训练目标。谱分析固定参考key bank的完整乘积，不把mask的rank性质混用，也不把有限经验数值证书冒充未知总体下界。

完整模型PPL覆盖每篇文档所有下一token位置，Wiki128×1024、长文24×8192、文学128×64；最后一种在64精确窗口内，混合/窗口结果不能作为学到远处能力的证据。所有指标保留逐文档值以便配对统计；训练种子11、29、47，头和文档相关性须分别处理，不能将24heads当24次独立模型训练。
