**数据与基础模型获取方式**

本仓库按用户要求不存储原始数据、QKV缓存或基础模型权重；21个小型feature检查点已包含。阅读研究文档、检查源码不需要下载下面的资产。

**1．基础模型：从官方仓库获取同一revision**

模型为[Qwen/Qwen2.5-1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B)，使用base版本，非Instruct或其他量化版本。精确revision为：

`8faed761d45a263340a0528343f099c05c9a4323`

全部10个文件及SHA256见[EXTERNAL_MODEL_MANIFEST.json](EXTERNAL_MODEL_MANIFEST.json)，合计3098972223bytes。可以在目标设备执行：

```bash
python -m pip install huggingface_hub
python fetch_base_model.py --download
```

脚本通过官方Hugging Face下载指定revision和清单文件，默认存到`work/models/Qwen2.5-1.5B/`，然后验证大小和SHA256。首次不带`--download`只显示计划，不下载。此命令仅下载模型，不安装NPU软件栈，也不启动训练。下载之后让模型与tokenizer从该本地目录加载；旧代码的硬编码cache路径需要按迁移文档适配。[官方snapshot_download说明](https://huggingface.co/docs/huggingface_hub/guides/download)

也可以从旧设备复制完整已解引用的同revision快照：

```bash
mkdir -p work/models/Qwen2.5-1.5B
rsync -aL --info=progress2 OLD_HOST:/root/autodl-tmp/hf-cache/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323/ work/models/Qwen2.5-1.5B/
```

`OLD_HOST`由实际旧设备SSH地址替换；`-L`用于复制HF snapshot符号链接指向的真实文件。不要迁移整个账号配置目录。

**2．现有实验的精确QKV缓存：从旧设备迁移**

这些QKV是本项目提取的激活，不是Hugging Face上现成的公开数据集，也没有已发布的公网下载链接。精确复现已有表格，应从旧设备取回`causal_direction/data/`；文件指纹见[EXTERNAL_DATA_MANIFEST.json](EXTERNAL_DATA_MANIFEST.json)，270个文件、8743844018bytes。

```bash
python prepare_workspace.py
mkdir -p work/kan_attention_theory/causal_direction/data
rsync -a --info=progress2 OLD_HOST:/root/autodl-tmp/kan_attention_theory/causal_direction/data/ work/kan_attention_theory/causal_direction/data/
python verify_bundle.py --require-checkpoints --project-root work/kan_attention_theory --model-root work/models/Qwen2.5-1.5B
```

其中`manifest.json`约68.6MB，保存4504条记录的准确input_ids、Q位置和划分；它也没有上传到本GitHub仓库。这里的EXTERNAL_DATA_MANIFEST仅是文件名/大小/哈希清单，不能替代该原始实验manifest。读取确认数据前，先检查文档与缓存位置一致。

如果只缺激活、希望在NPU重新提取，可以先单独迁移原实验manifest：

```bash
rsync -a OLD_HOST:/root/autodl-tmp/kan_attention_theory/causal_direction/data/manifest.json work/kan_attention_theory/causal_direction/data/
```

然后移植`work/kan_attention_theory/causal_direction/data.py`的extract路径。它保留相同tokens和Q位置，但NPU投影/attention的舍入路径可能不同；新提取的张量须另记SHA与数值差异，不能声称逐字节复现源GPU缓存。当前脚本包含`.cuda()`、固定cache和Transformers registry依赖，不能在910B上未经适配直接执行。

**3．公开文本来源与从头重建**

主要文本来源为：

- [EleutherAI/wikitext_document_level](https://huggingface.co/datasets/EleutherAI/wikitext_document_level)：包含WikiText-2和WikiText-103文档。旧实验WikiText-103快照revision为`647234772b9554e208af6c826f23b99e3cac88c8`。
- [EleutherAI/lambada_openai](https://huggingface.co/datasets/EleutherAI/lambada_openai)：当前confirm_prose采用其test样本的64-token前缀。旧缓存的Hub commit未单独固定，不能自行填入一个版本号冒充历史记录。

下载原始文本后重新抽取的来源流程可在以下源码中查阅：

1. `snapshot/kan_attention_theory/real_llm_pilot/extract_qkv.py`：最早的文档划分、真实post-RoPE激活抽取。
2. `snapshot/kan_attention_theory/single_pass_mulkan/extract_large.py`：扩展单遍训练文档，固定WikiText-103 revision与去重。
3. `snapshot/kan_attention_theory/causal_direction/data.py`：复用训练/验证tokens，确定新Q位置，选确认文档，提取两层24heads。

最后一步的select还依赖前期排除文档记录与缓存；**单独下载公开WikiText并运行select，不足以保证得到原表格的准确划分。** 原GPU实验manifest是复用相同tokens最短、最可靠的入口。旧设备资产不可取得时，应按公开来源建立新的数据协议并清楚标注新数据，重新评估所有baseline。

**4．从零预训练另有数据协议**

旧feature拟合的约419万训练tokens不等于100M–1B完整LM预训练语料。后续预训练token预算、语料和新确认集尚未冻结。不要把上述数据获取说明当成已完成预训练数据准备，也不要把旧确认集继续用于选超参数后称作新盲测。
