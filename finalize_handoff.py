"""Author-side handoff audit and packaging. Requires the original sibling project."""
import hashlib
import json
import py_compile
import re
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit

from verify_bundle import sha256

P = Path(__file__).resolve().parent
R = P.parent / 'kan_attention_theory'
S = P / 'snapshot' / 'kan_attention_theory'


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def combined_document():
    ordered = [
        'AD_PROJECT_FULL_HANDOFF.zh.md', 'ASCEND_910B_MIGRATION.zh.md',
        'CURRENT_RESULTS.zh.md', 'NEXT_CODEX_PROMPT.zh.md', 'SOURCE_INDEX.zh.md',
        'HISTORICAL_REPORTS_FULL.zh.md',
    ]
    output = [
        '**AD项目完整交接总册：当前版本、理论、实验、历史原文与双910B接续**', '',
        '目标已确认：删除C和Q幅度，保留K幅度。研究快照2026-09-08，整理校验2026-09-09。', '',
        '本文件合并六份交接材料，其中历史全文包含40份原项目文档。它可以独立阅读；'
        '源码、JSON、图和检查点仍在独立文件中，不嵌入本册。分段行号见READING_MAP.json。', '',
        '历史原文保留当时结论，后续纠正以本册前部当前结论为准。'
        '历史相对链接以所标注的原文件目录为基准；请用SOURCE_INDEX打开原目录副本。'
        '外部大数据/基模型须按资产清单另行迁移。', '',
    ]
    ranges = []
    history_ranges = []
    for name in ordered:
        output.extend(['---', '', f'**合订来源：{name}**', ''])
        start = len(output) + 1
        lines = (P / name).read_text().splitlines()
        output.extend(lines)
        ranges.append({'source': name, 'start_line': start, 'end_line': len(output)})
        if name == 'HISTORICAL_REPORTS_FULL.zh.md':
            markers = [(start + i, line.removeprefix('原文件：'))
                       for i, line in enumerate(lines) if line.startswith('原文件：')]
            for i, (line, source) in enumerate(markers):
                end = markers[i + 1][0] - 1 if i + 1 < len(markers) else len(output)
                history_ranges.append({'source': source, 'start_line': line, 'end_line': end})
        output.append('')
    assert len(history_ranges) == 40
    (P / 'AD_PROJECT_ALL_IN_ONE.zh.md').write_text('\n'.join(output) + '\n')
    write(P / 'READING_MAP.json', {'document': 'AD_PROJECT_ALL_IN_ONE.zh.md',
                                 'sections': ranges, 'historical_documents': history_ranges,
                                 'line_numbering': '1-based, inclusive'})


def audit():
    inv = read(P / 'ARTIFACT_INVENTORY.json')['files']
    included = [r for r in inv if r['in_document_bundle']]
    weights = [r for r in inv if r['in_checkpoint_bundle']]
    for row in included + weights:
        origin = R / row['path']
        assert origin.stat().st_size == row['bytes'], str(origin)
        assert sha256(origin) == row['sha256'], str(origin)
        copy = (S if row['in_document_bundle'] else P / 'checkpoints' / 'kan_attention_theory') / row['path']
        assert sha256(copy) == row['sha256'], str(copy)
    docs = [r for r in included if r['path'].endswith('.md')]
    scripts = [r for r in included if r['path'].endswith('.py')]
    assert len(docs) == 40 and len(scripts) == 183 and len(weights) == 21
    assert len(included) == 1377
    assert sum(r['bytes'] for r in weights) == 130695675
    history = (P / 'HISTORICAL_REPORTS_FULL.zh.md').read_text()
    for row in docs:
        assert (R / row['path']).read_text() in history, row['path']

    # Reconcile the final experiment's earlier recorded hashes, including external evidence.
    earlier = read(S / 'key_parameterization_attribution/checks/audit.json')
    assert earlier['passed'] and earlier['new_fit_count'] == 12
    assert earlier['new_selected_count'] == 9 and earlier['reused_selected_count'] == 3
    provenance_count = 0
    for key in ['source_sha256', 'result_sha256']:
        for filename, digest in earlier[key].items():
            assert sha256(Path(filename)) == digest, f'Historical hash drift: {filename}'
            provenance_count += 1
    for row in earlier['metadata_and_parameter_checks']:
        assert sha256(R / 'key_parameterization_attribution/fits' / (row['name'] + '.pt')) == row['checkpoint_sha256']

    # Verify authored key tables against the preserved summary; no new model metrics computed.
    summary = read(S / 'key_parameterization_attribution/results/summary.json')
    expected_ppl = {
        ('confirm_wiki', 'standard_ad'): 8.85805483249895,
        ('confirm_wiki', 'standard_exp'): 8.854965644571442,
        ('confirm_wiki', 'matched_zero_ad'): 8.862187379765931,
        ('confirm_wiki', 'matched_zero_exp'): 8.855540686111347,
        ('confirm_long', 'standard_ad'): 9.691610484166858,
        ('confirm_long', 'standard_exp'): 9.733331121199464,
        ('confirm_long', 'matched_zero_ad'): 9.694544312247054,
        ('confirm_long', 'matched_zero_exp'): 9.726657820208986,
    }
    main_text = (P / 'AD_PROJECT_FULL_HANDOFF.zh.md').read_text()
    ledger = (P / 'CURRENT_RESULTS.zh.md').read_text()
    for row in summary['table']:
        assert abs(row['ppl'] - expected_ppl[row['split'], row['group']]) < 1e-12
        assert f'{row["ppl"]:.6f}' in main_text
        for field in ['kl', 'output_nmse', 'ppl', 'head_tv', 'head_coeff_l2', 'layer_projected_nmse']:
            assert f'{row[field]:.9f}' in ledger
    assert 128 * 192 * 2 + 192 * (63 + 64) == 73536
    assert 12 * 64 * (128 + 2) * 4 == 399360
    assert read(P / 'EXTERNAL_DATA_MANIFEST.json')['bytes'] == 8743844018
    assert read(P / 'EXTERNAL_MODEL_MANIFEST.json')['bytes'] == 3098972223

    links = 0
    for name in ['START_HERE.zh.md', 'AD_PROJECT_FULL_HANDOFF.zh.md',
                 'ASCEND_910B_MIGRATION.zh.md', 'CURRENT_RESULTS.zh.md',
                 'NEXT_CODEX_PROMPT.zh.md', 'SOURCE_INDEX.zh.md']:
        for target in re.findall(r'\]\(([^)]+)\)', (P / name).read_text()):
            parts = urlsplit(target.strip('<>'))
            if parts.scheme or not parts.path:
                continue
            if parts.path in ['HANDOFF_AUDIT.json', 'BUNDLE_MANIFEST.json']:
                continue  # Generated immediately after this audit; checked before archiving.
            assert (P / unquote(parts.path)).exists(), f'{name}: missing {target}'
            links += 1
    helpers = sorted(P.glob('*.py'))
    for helper in helpers:
        py_compile.compile(str(helper), doraise=True)
    result = {'passed': True, 'session_date': '2026-09-09',
              'source_host_utc': datetime.now(timezone.utc).isoformat(),
              'date_note': 'Document dates follow conversation context; source host clock is recorded separately and may differ.',
              'source_documents_byte_identical': len(docs),
              'source_python_scripts_byte_identical': len(scripts),
              'source_snapshot_files_verified': len(included),
              'checkpoint_files_verified': len(weights),
              'historical_experiment_hashes_rechecked': provenance_count,
              'authored_local_links_checked': links, 'handoff_helpers_compiled': len(helpers),
              'current_summary_tables_checked': True, 'complete_history_text_checked': True,
              'external_assets': 'Fingerprints captured; not included in either archive.',
              'limitations': ['No NPU execution or access', 'No new training/evaluation',
                             'File/numerical transcription audit, not statistical or theoretical certification',
                             'Historical report links can refer to intentionally omitted external assets']}
    write(P / 'HANDOFF_AUDIT.json', result)
    return result


def manifest_and_archives():
    records = []
    for path in sorted(P.rglob('*')):
        if not path.is_file():
            continue
        rel = path.relative_to(P)
        if '__pycache__' in rel.parts or path.suffix == '.log' or rel.as_posix() == 'BUNDLE_MANIFEST.json':
            continue
        assert not path.is_symlink(), str(path)
        records.append({'path': rel.as_posix(), 'bytes': path.stat().st_size,
                        'sha256': sha256(path),
                        'group': 'checkpoints' if rel.parts[0] == 'checkpoints' else 'documents'})
    write(P / 'BUNDLE_MANIFEST.json', {'format': 1, 'files': records,
                                      'excluded': ['BUNDLE_MANIFEST.json itself', '__pycache__', '*.log'],
                                      'optional_group': 'checkpoints (all required when any are present)'})
    assert (P / 'HANDOFF_AUDIT.json').is_file() and (P / 'BUNDLE_MANIFEST.json').is_file()
    deliveries = []
    for group in ['documents', 'checkpoints']:
        target = P.parent / f'AD_kernel_handoff_{group}_20260908.tar.gz'
        chosen = [r for r in records if r['group'] == group]
        if group == 'documents':
            manifest = P / 'BUNDLE_MANIFEST.json'
            chosen.append({'path': manifest.name, 'bytes': manifest.stat().st_size, 'sha256': sha256(manifest)})
        with tarfile.open(target, 'w:gz', compresslevel=6) as archive:
            for row in chosen:
                archive.add(P / row['path'], arcname=f'{P.name}/{row["path"]}', recursive=False)
        expected = {f'{P.name}/{r["path"]}': r for r in chosen}
        with tarfile.open(target, 'r:gz') as archive:
            seen = set()
            for member in archive:
                assert member.isfile() and member.name in expected
                assert member.name not in seen
                row = expected[member.name]
                stream = archive.extractfile(member)
                h = hashlib.sha256()
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    h.update(block)
                assert member.size == row['bytes'] and h.hexdigest() == row['sha256']
                seen.add(member.name)
            assert seen == set(expected)
        deliveries.append({'archive': target.name, 'bytes': target.stat().st_size,
                           'sha256': sha256(target), 'files': len(chosen),
                           'all_members_decompressed_and_hash_verified': True})
    write(P.parent / 'AD_kernel_handoff_DELIVERY_20260908.json',
          {'handoff_root': P.name, 'archives': deliveries,
           'external_data_bytes': 8743844018, 'external_model_bytes': 3098972223})
    (P.parent / 'AD_kernel_handoff_20260908.sha256').write_text(
        ''.join(f'{r["sha256"]}  {r["archive"]}\n' for r in deliveries))
    return deliveries


def main():
    combined_document()
    result = audit()
    print(json.dumps({'audit': result}, ensure_ascii=False), flush=True)
    deliveries = manifest_and_archives()
    print(json.dumps({'archives': deliveries}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
