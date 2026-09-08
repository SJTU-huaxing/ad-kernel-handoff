"""Render exact audited JSON numbers into the handoff, without rerunning fits."""
import json
from pathlib import Path
P=Path(__file__).resolve().parent
R=P/'snapshot/kan_attention_theory'
def read(p):return json.loads(p.read_text())
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','|'+'|'.join(['---']*len(headers))+'|']+
        ['| '+' | '.join(map(str,row))+' |' for row in rows])

def main():
    a=read(R/'key_parameterization_attribution/results/summary.json')
    n=read(R/'normalized_attention_assessment/results/summary.json')
    out=['**当前有效结果账本：由审计后JSON自动生成**',
         '没有新增训练或计算新的模型指标。本表不跨不同数据集拼排名；所有条件和限制见完整交接文档。数字来源为snapshot中的两份summary.json及query_amplitude_ablation的benchmark.json。',
         '**A．最新K参数化归因：三种子均值**']
    fields=['kl','output_nmse','ppl','head_tv','head_coeff_l2','layer_projected_nmse']
    out.append(table(['split','group']+fields,[[r['split'],r['group']]+[f'{r[f]:.9f}' for f in fields] for r in a['table']]))
    out.append('每种子PPL顺序固定为11、29、47。')
    out.append(table(['split','group','s11','s29','s47'],[[r['split'],r['group']]+[f'{v:.9f}' for v in r['seed_ppl']] for r in a['table']]))
    out.append('**B．归因配对统计：EXP−AD，正值有利AD**')
    rows=[]
    for r in a['comparisons']:
        lo,hi=r['conditional_document_ci95']
        rows.append([r['split'],r['initialization'],r['metric'],f'{r["exp_minus_ad"]:.9f}',
                     f'[{lo:.9f}, {hi:.9f}]',','.join(f'{v:.9f}' for v in r['seed_differences']),
                     f'{r["ad_better_documents"]}/{r["documents"]}',r.get('ad_better_heads','—')])
    out.append(table(['split','init','metric','Δ','文档95%CI','三种子Δ','AD胜文档','AD胜head/24'],rows))
    out.append('区间为条件于3个固定拟合的整文档bootstrap；不覆盖未知训练随机性/模型家族，无多重比较校正。')
    out.append('**C．学习率选择与训练曲线记录**')
    out.append(table(['group','lr','seed11 validation KL','selected'],[[r['group'],r['lr'],f'{r["validation_kl"]:.9f}',r['selected']] for r in a['selection']]))
    for r in a['training']:
        out.append(f'`{r["group"]}`；steps={r["steps"]}；每512文档平均KL曲线（3种子）='+json.dumps(r['seed_mean_head_curves'])+
                   '；validation KL='+json.dumps(r['validation_kl'])+'；训练循环秒数='+json.dumps(r['training_seconds'])+'。')
    out.append('训练秒数包含历史AD复用，不能作同期算法加速比较。每组全量epoch=1，最终checkpoint；验证集选择学习率，不按确认数据选择。')
    out.append('**D．最新同深度时延和状态：RTX3090参考微基准**')
    out.append(table(['group','feature μs','feature+state μs','两轮step中位μs','state bytes/12heads'],
        [[r['group'],f'{r["features_us"]:.6f}',f'{r["total_us"]:.6f}',','.join(f'{x:.6f}' for x in r['repeat_total_us']),r['state_bytes_per_layer']] for r in a['benchmark']]))
    out.append('FP32、batch1、单层12heads、CUDA Graph热缓存；不是完整模型时间或910B测量。')
    out.append('**E．当前pure AD对Hedgehog/FAVOR：相同确认数据，历史匹配对照**')
    out.append(table(['split','group','KL','output NMSE','PPL'],[[r['split'],r['group'],f'{r["kl"]:.9f}',f'{r["output_nmse"]:.9f}',f'{r["ppl"]:.9f}'] for r in n['table']]))
    out.append('AD73536参数/m64；HH-exp与HH-softmax73728参数/m576；FAVOR固定随机feature/m64。HH是本项目参数匹配适配版；不存在所有方案同时同参数、同state、同计算的单一比较。')
    ff=['head_tv','head_coeff_l2','layer_projected_nmse','head_output_global_nmse','layer_projected_global_nmse']
    out.append(table(['split','group']+ff,[[r['split'],r['group']]+[f'{r[f]:.9f}' for f in ff] for r in n['diagnostics']]))
    out.append('**F．严重低估尾部：8k**')
    out.append(table(['group','predicted < teacher/10000的教师质量','该部分正KL'],
        [[r['group'],f'{r["severe_underestimate_mass"]:.9f}',f'{r["severe_positive_kl"]:.9f}'] for r in a['table'] if r['split']=='confirm_long']))
    out.append('注意这是教师概率质量，不是token占比；正KL项不加负项不能还原完整KL。Hedgehog/FAVOR完整阈值表保存在normalized_attention_assessment的summary.tail。')
    out.append('**G．教师和实现审计**')
    out.append(table(['split','重新测得PPL','历史PPL','差'],[[r['split'],r['new'],r['old'],r['difference']] for r in a['teacher_check']]))
    checks=read(R/'key_parameterization_attribution/checks/audit.json')
    out.append('最新审计passed='+str(checks['passed'])+'，新增拟合='+str(checks['new_fit_count'])+'，新选定='+str(checks['new_selected_count'])+'，复用='+str(checks['reused_selected_count'])+'。')
    out.append('所有源文件SHA及结果SHA见对应checks/audit.json；它记录的是旧设备绝对路径，迁移后用相对目录定位，不应修改旧审计文件中的历史路径。')
    (P/'CURRENT_RESULTS.zh.md').write_text('\n\n'.join(out)+'\n')

if __name__=='__main__':main()
