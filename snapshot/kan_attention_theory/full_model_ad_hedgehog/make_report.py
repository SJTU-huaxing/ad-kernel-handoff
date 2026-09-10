"""Create readable tables and a standalone uncertainty figure from raw timing."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

P = Path(__file__).resolve().parent
NAMES = {'teacher':'原模型', 'ad_plain':'AD纯删除', 'ad_matched':'AD严格同参数',
         'hh_exp':'Hedgehog-exp', 'hh_softmax':'Hedgehog-softmax'}


def main():
    s = json.loads((P/'results/summary.json').read_text())
    raw = json.loads((P/'results/raw.json').read_text())
    checks = json.loads((P/'checks/correctness.json').read_text())
    diagnostic_path = P/'results/dispatch_diagnostic.json'
    dispatch = json.loads(diagnostic_path.read_text()) if diagnostic_path.exists() else None
    rows, comps = s['aggregates'], s['comparisons']
    lookup = {(r['prompt_tokens'],r['case']):r for r in rows}
    lines = [
        '**实测结论：AD的prefill更快；逐token解码没有稳定速度优势；8k输入加64步解码的总时间略优于Hedgehog。不能宣称AD在完整模型推理各阶段都更快。**',
        '',
        '8k时，AD纯删除版prefill中位数471.458ms，HH-exp/softmax为496.761/496.954ms，约省5.1%。解码中位数分别23.7936、23.7721、23.8786ms/token，差距很小；配对均值比较中AD对HH-exp慢约0.575%，对HH-softmax基本持平。prefill+64步总时间的配对均值下降约0.82%和1.25%；1k时AD相对HH-exp总时间反而略慢，4k总时间差异区间跨0。严格同参数AD给出同样的主要判断。',
        '',
        '原模型在本实现中的prefill和decode都更快：8k为467.142ms和22.4310ms/token。因此本轮不支持相对原模型的端到端推理加速。',
        '',
        '本轮直接测量同一个完整Qwen2.5-1.5B模型的推理计算。只替换已训练的第14/27层，共24/336 heads；其余26层相同。不是全部heads替换后的模型，也不是昇腾910B结果。',
        '',
        'RTX3090；batch1；BF16基础模型，FP32正特征与线性状态；TF32关闭。Hedgehog是本项目同参数预算扩宽适配版（m576），AD为m64。二者共享参考PyTorch线性scan/step，没有调用仅适用旧m64方法的专用融合算子。',
        '',
        '每个长度12个配对区组，三个实际文档各四次，每次prefill后逐token解码64步；区组内随机交错五种方法。固定续写token保证内容和长度相同。时间包含全部网络层与最后位置LM head，排除分词、采样、服务调度和网络传输。',
        '',
        '**完整模型墙钟中位数**', '',
        '| 上下文 | 方法 | Prefill ms | Decode ms/token | Decode token/s | Prefill+64步 ms |',
        '|---|---|---:|---:|---:|---:|',
    ]
    for r in rows:
        lines.append(f"| {r['prompt_tokens']} | {NAMES[r['case']]} | {r['prefill_wall_ms']:.3f} | "
                     f"{r['decode_wall_ms_per_token']:.4f} | {1000/r['decode_wall_ms_per_token']:.2f} | {r['request_wall_ms']:.3f} |")
    lines += ['', '请求总时间取每个样本prefill+64步decode之和后求中位数，因此不一定等于两列中位数简单相加。token/s由decode中位数取倒数。', '',
              '**配对区组不确定性：正值表示AD耗时更低，负值表示AD更慢**', '',
              '| 上下文 | AD版本 | 对照 | 阶段 | AD耗时下降% | 区组bootstrap 95%区间 | 判断 |',
              '|---|---|---|---|---:|---|---|']
    metric_names = {'prefill_wall_ms':'Prefill','decode_wall_ms_per_token':'Decode',
                    'request_wall_ms':'Prefill+64步'}
    for c in comps:
        if c['b']=='teacher':
            continue
        lo,hi = c['bootstrap_95_percent']
        decision = 'AD更快' if lo>0 else '对照更快' if hi<0 else '无法稳定区分'
        lines.append(f"| {c['prompt_tokens']} | {NAMES[c['a']]} | {NAMES[c['b']]} | {metric_names[c['metric']]} | "
                     f"{c['latency_reduction_percent']:.2f} | [{lo:.2f}, {hi:.2f}] | {decision} |")
    lines += ['', '百分比基于配对区组的均值比；表一为中位数，两者口径不同。区间来自20000次整区组重采样，不能把64个相关decode token当成64次独立重复。它只衡量本轮固定硬件/输入/检查点下的计时重复性，不是跨硬件或跨训练seed保证，也未做多重比较校正。', '',
              '**缓存实际占用与prefill峰值**', '',
              '| 上下文 | 方法 | 原KV MiB | 线性状态 MiB | KV+state MiB | Prefill新增峰值 MiB |',
              '|---|---|---:|---:|---:|---:|']
    for r in rows:
        mem=r['cache_after_prefill']
        lines.append(f"| {r['prompt_tokens']} | {NAMES[r['case']]} | {mem['kv_bytes']/2**20:.3f} | "
                     f"{mem['state_bytes']/2**20:.3f} | {mem['total_bytes']/2**20:.3f} | {r['prefill_incremental_peak_bytes']/2**20:.3f} |")
    lines += ['', '每个正式试验都验证两替换层原KV storage为0、cache已见token长度正确、线性状态在64步decode后字节数不变。状态含S、z和数值gauge g。AD两层共0.761719 MiB，Hedgehog两层共6.855469 MiB；其他26层KV相同，所以完整缓存并非相差9倍。峰值为PyTorch已分配内存相对prefill前的增量，不是总设备显存；模型权重与常驻feature权重没有计入这一增量。', '',
              '**数值核验**', '',
              '| 方法 | BF16短序列logits相对L2差 | BF16 8k差 | FP32短序列差 | FP32 8k差 |',
              '|---|---:|---:|---:|---:|']
    cc={(r['case'],r['model_dtype'],r['total_tokens']):r for r in checks['cases']}
    for case in NAMES:
        vals=[cc[case,dtype,n]['prefill_decode_logit_relative_l2'] for dtype,n in
              [('bfloat16',272),('bfloat16',8192),('float32',272),('float32',8192)]]
        lines.append('| '+NAMES[case]+' | '+' | '.join(f'{v:.7g}' for v in vals)+' |')
    lines += ['', '一次完整前向与prefill+16步decode在BF16下存在可见差异，原模型也有此现象。首次BF16 0.03门限被AD纯删除版触发；因此追加了独立FP32比较，而不是放宽门限。FP32核验全部通过，最大相对差小于3.2e-6，支持状态递推与位置/缓存接入正确。真实Q/K/V下显式正kernel与FP32 scan相对L2误差均小于6e-7。这些抽样检查不能保证所有输入的数值误差；BF16结果差异原样保留，所有所测末位置top1一致。计时使用原始BF16参数张量，未经过额外量化。', '']
    if dispatch:
        lines += ['**补充：主机逐算子派发与CUDA Graph的组件计时**', '',
                  '| 方法 | 主机逐算子派发 µs/层 | CUDA Graph µs/层 |',
                  '|---|---:|---:|']
        dl={(r['case'],r['mode']):r for r in dispatch['aggregates']}
        for case in [c for c in NAMES if c!='teacher']:
            lines.append(f"| {NAMES[case]} | {dl[case,'host']['wall_us']:.3f} | {dl[case,'graph']['wall_us']:.3f} |")
        lines += ['', '这项独立诊断在正式全模型计时结束后运行，使用真实缓存QKV、单层12heads、相同FP32特征与step。仅改变主机派发方式。差异包含派发、分配和图执行效应，不能全部归因于Python，也不能把组件CUDA Graph结果直接当完整模型加速。完整模型主实验使用HF执行，尚未采用整模型图捕获或专用服务引擎。', '']
    lines += ['当前证据应按prefill、逐token解码和指定输出长度的请求总时间分别解读。更小的m降低状态运算和存储量，但完整模型耗时还包含其余26层、QKV/O投影、FFN、权重读取和主机派发；不能从旧单层34%～38%的耗时下降推出整模型同幅加速。', '',
              '本轮没有重新测PPL；沿用此前三seed完整模型局部替换评估：AD纯删除1k/8k PPL为8.858055/9.691610，HH-exp为8.870295/9.752359，HH-softmax为8.880211/9.752327。这些是已有质量证据，不能把本轮固定token计时说成新的自由生成质量测试。', '',
              '全部原始正式计时均在[results/raw.json](results/raw.json)，均值比与区间在[results/summary.json](results/summary.json)，可读表格在[results/timing.csv](results/timing.csv)。协议和命令见[PROTOCOL.zh.md](PROTOCOL.zh.md)，数值核验见[checks/correctness.json](checks/correctness.json)。', '']
    (P/'REPORT.zh.md').write_text('\n'.join(lines))
    with (P/'results/timing.csv').open('w') as out:
        fields=['case','prompt_tokens','trials','prefill_wall_ms','decode_wall_ms_per_token','request_wall_ms',
                'prefill_cuda_ms','decode_cuda_ms_per_token','prefill_incremental_peak_bytes']
        writer=csv.DictWriter(out,fieldnames=fields)
        writer.writeheader()
        writer.writerows({k:r[k] for k in fields} for r in rows)
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),constrained_layout=True)
    lengths=[1024,4096,8192]
    for ax,metric,title in zip(axes,metric_names,['Prefill','Decode per token','Prefill + 64 decode steps']):
        for bcase,offset,color,label in [('hh_exp',-.09,'#146b9e','vs HH-exp'),
                                         ('hh_softmax',.09,'#c66b22','vs HH-softmax')]:
            points=[next(c for c in comps if c['a']=='ad_plain' and c['b']==bcase and c['metric']==metric and c['prompt_tokens']==n) for n in lengths]
            values=[c['latency_reduction_percent'] for c in points]
            errors=[[values[i]-c['bootstrap_95_percent'][0] for i,c in enumerate(points)],
                    [c['bootstrap_95_percent'][1]-values[i] for i,c in enumerate(points)]]
            ax.errorbar(values,[i+offset for i in range(3)],xerr=errors,fmt='o',capsize=3,color=color,label=label)
        ax.axvline(0,color='gray',lw=1)
        ax.set_yticks(range(3),['1k','4k','8k'])
        ax.set_title(title)
        ax.set_xlabel('AD latency reduction (%)')
        ax.grid(axis='x',alpha=.2)
    axes[0].set_ylabel('Prompt length')
    axes[0].legend(fontsize=8)
    fig.suptitle('Full Qwen2.5-1.5B, 2/28 layers replaced; paired timing 95% intervals',fontsize=11)
    (P/'figures').mkdir(exist_ok=True)
    fig.savefig(P/'figures/full_model_timing.pdf')
    fig.savefig(P/'figures/full_model_timing.png',dpi=180)
    print(json.dumps({'report':str(P/'REPORT.zh.md'),'timing_rows':len(rows),
                      'raw_trials':len(raw['results']),'paired_comparisons':len(comps)}))


if __name__=='__main__':
    main()
