# 实时训练曲线

已使用 nonlinear-qk 环境中现有的 TensorBoard 2.21.0 配置查看服务。日志读取器只使用 CPU，独立读取 rank 0 已有的 `train.jsonl`、`validation.jsonl`，不导入 torch，不修改训练源码、配置或检查点。当前已有训练也可以直接接入，无需重启训练。

## 打开页面

查看服务在远程机器的 `127.0.0.1:6006` 运行。请通过已登录的 ModelArts JupyterLab 访问：复制当前浏览器的 JupyterLab 地址到新标签页，将 `/lab` 及其后面的内容替换为 `/proxy/6006/`，保留前面的域名和实例路径。

例如：

```text
原地址：https://你的域名/实例路径/lab/tree/...
新地址：https://你的域名/实例路径/proxy/6006/
```

也可以在 JupyterLab 打开 `notebooks/Training_TensorBoard.ipynb`，选择 nonlinear-qk kernel，运行单元格，在 Notebook 内显示 TensorBoard。单元格同时提供新标签页链接。浏览器访问沿用当前 JupyterLab 登录会话。

## 查看服务与训练命令

本次配置时查看服务已启动；若机器或服务重启，在一个 Terminal 中运行：

```bash
cd /cache/huaxing/ad-kernel-handoff
bash scripts/tensorboard_100m.sh
```

这条命令以前台方式同时运行 JSONL 日志读取器与 TensorBoard，不启动训练。服务已运行时再次执行会提示 6006 端口已占用。关闭此命令只停止曲线查看，训练继续写 JSONL；重新启动服务会补齐历史曲线。

在另一个 Terminal 中启动 seed 11 的四种方法：

```bash
cd /cache/huaxing/ad-kernel-handoff
bash scripts/launch_100m_2b.sh --execute --seeds 11
```

依次为 AD、softmax、FAVOR+ m64、Hedgehog，每种 2B tokens、两张 NPU，训练后自动评测。训练终端仍每 10 个更新步输出状态；TensorBoard 读取每一步的完整 JSONL，曲线粒度是每个 optimizer step。没有正式训练日志时 TensorBoard 为空，首批训练指标写入后自动出现。短时 probe 的曲线不混入正式训练页面。

## 曲线选择

打开 Scalars，勾选需要比较的 run：

- `steps/ad64_s11`
- `steps/softmax_s11`
- `steps/favor64_s11`
- `steps/hedgehog_s11`

主要指标：

| Tag | 内容 | 横坐标 |
|---|---|---|
| `train/loss` | 全局有效 token 平均训练交叉熵损失 | optimizer step |
| `validation/loss` | 留出验证集 NLL | optimizer step |
| `validation/perplexity` | 验证集困惑度 | optimizer step |
| `train/learning_rate` | 学习率 | optimizer step |
| `train/grad_norm` | 记录的梯度范数 | optimizer step |
| `performance/tokens_per_second` | 双卡训练吞吐 | optimizer step |
| `performance/step_seconds` | 每个更新步耗时 | optimizer step |
| `progress/trained_tokens` | 累计训练 token 数量 | optimizer step |

按相同训练 token 预算比较时，选择 `tokens/<method>_s11` 的 run，并查看 `train/loss_by_tokens`、`validation/loss_by_tokens` 或 `validation/perplexity_by_tokens`。这些曲线的 TensorBoard `Step` 横坐标直接存累计训练 tokens。验证数据集的 token 数不会被误用为训练进度。

日志每 2 秒扫描一次，新增事件立即 flush，TensorBoard 服务端每 2 秒重读事件。在 TensorBoard Settings 中启用自动刷新，将刷新周期设为 5 秒；必要时点击刷新按钮。总体延迟取决于当前训练步耗时与页面刷新周期，通常是指标写出后的数秒。

建议先用 Smoothing=0 看原始损失，再用 0.6 左右观察趋势；平滑只影响显示，不修改原始数据。验证曲线在每 512 步及最终一步出现，并不逐步计算。原 JSONL 没有每步绝对时间戳，事件 wall time 采用导入时间；回填历史记录时应以 Step/tokens 横坐标比较，避免将导入时间当作原训练时间。

## 文件与验证

- 原始指标：`work/pretrain_100m_2b/runs/<method>_s11/train.jsonl`、`validation.jsonl`。
- TensorBoard 日志目录：`work/pretrain_100m_2b/tensorboard`。
- 服务状态：`work/pretrain_100m_2b/tensorboard_service.json`。
- CPU 集成验证：`work/pretrain_100m_2b/validation/tensorboard.json`。

已验证逐步追加、写到一半的行等待补齐、服务重启不重复点、恢复训练回退后的旧点清理，以及通过 TensorBoard HTTP 接口读取准确的曲线值。测试数据放在自动删除的临时目录，没有写入正式训练曲线。重启或纠正历史时使用 TensorBoard 的 SessionLog.START 清理旧显示；原始训练日志始终保留。

本地 TensorBoard HTTP 接口已验证可用。Jupyter 代理扩展已加载，未携带浏览器登录会话的本机访问会跳转登录页；请在当前已登录的 JupyterLab 浏览器中打开代理地址。
