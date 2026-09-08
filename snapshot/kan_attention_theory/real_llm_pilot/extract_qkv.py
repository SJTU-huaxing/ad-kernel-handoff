"""Capture actual post-RoPE Q/K/V from a frozen Qwen2 checkpoint."""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2 import modeling_qwen2


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--model', default='Qwen/Qwen2.5-1.5B')
    parser.add_argument('--train', type=int, default=64)
    parser.add_argument('--validation', type=int, default=12)
    parser.add_argument('--test', type=int, default=20)
    parser.add_argument('--long', type=int, default=8)
    parser.add_argument('--ood', type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    random.seed(20260905)
    torch.manual_seed(20260905)
    cache = '/root/autodl-tmp/hf-cache'
    tokenizer = AutoTokenizer.from_pretrained(args.model, cache_dir=cache, local_files_only=True)
    root = Path(cache) / 'datasets/EleutherAI___wikitext_document_level'
    documents = []
    used_hashes = set()
    for split, count, length, source_split in [
        ('train', args.train, 1024, 'train'),
        ('validation', args.validation, 1024, 'validation'),
        ('test', args.test, 1024, 'test'),
        ('test_long', args.long, 2048, 'test'),
    ]:
        data = Dataset.from_file(str(next(root.rglob(f'*-{source_split}.arrow'))))
        order = list(range(len(data)))
        random.Random(20260905).shuffle(order)
        selected = 0
        for index in order:
            text = data[index]['page']
            text_hash = digest(text)
            if text_hash in used_hashes:
                continue
            tokens = tokenizer(text, add_special_tokens=False)['input_ids']
            if len(tokens) < length:
                continue
            used_hashes.add(text_hash)
            documents.append(dict(split=split, source='EleutherAI/wikitext_document_level',
                                  source_split=source_split, document_index=index,
                                  text_sha256=text_hash, input_ids=tokens[:length]))
            selected += 1
            if selected == count:
                break
        if selected < count:
            raise RuntimeError(f'Only {selected} eligible documents for {split}, requested {count}')
    ood_path = Path('/root/autodl-tmp/memory-gla/eval-data/leval_narrative_qa.jsonl')
    with ood_path.open() as handle:
        ood_documents = [json.loads(line) for line in handle if line.strip()]
    selected = 0
    for index, item in enumerate(ood_documents):
        text = item['input']
        text_hash = digest(text)
        if text_hash in used_hashes:
            continue
        tokens = tokenizer(text, add_special_tokens=False)['input_ids']
        if len(tokens) < 1024:
            continue
        used_hashes.add(text_hash)
        documents.append(dict(split='ood', source='cached LEval narrative_qa', source_split='evaluation',
                              document_index=index, text_sha256=text_hash, input_ids=tokens[:1024]))
        selected += 1
        if selected == args.ood:
            break

    layers = [0, 7, 14, 21, 27]
    heads = [0, 3, 6, 9]
    head_labels = [(layer, head) for layer in layers for head in heads]
    captured = {}
    numerical_checks = []
    original_sdpa = modeling_qwen2.ALL_ATTENTION_FUNCTIONS['sdpa']

    def capture_sdpa(module, query, key, value, attention_mask, **kwargs):
        output = original_sdpa(module, query, key, value, attention_mask, **kwargs)
        if module.layer_idx in layers:
            kv_heads = [head // module.num_key_value_groups for head in heads]
            captured[module.layer_idx] = (
                query[0, heads].detach().cpu(),
                key[0, kv_heads].detach().cpu(),
                value[0, kv_heads].detach().cpu(),
            )
            if len(numerical_checks) < len(layers):
                q = query[0, heads].float()
                k = key[0, kv_heads].float()
                v = value[0, kv_heads].float()
                logits = q @ k.transpose(-1, -2) * kwargs['scaling']
                causal = torch.ones(logits.shape[-2:], device=logits.device, dtype=torch.bool).tril()
                manual = logits.masked_fill(~causal, -torch.inf).softmax(-1) @ v
                actual = output[0][0, :, heads].transpose(0, 1).float()
                relative = ((manual - actual).square().sum() / actual.square().sum()).sqrt()
                numerical_checks.append(dict(layer=module.layer_idx,
                                             relative_output_l2=float(relative),
                                             note='float32 recomputation versus bfloat16 SDPA'))
        return output

    modeling_qwen2.ALL_ATTENTION_FUNCTIONS.register('sdpa', capture_sdpa)
    print(json.dumps(dict(event='loading_model', model=args.model, documents=len(documents))), flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, cache_dir=cache, local_files_only=True,
        dtype=torch.bfloat16, attn_implementation='sdpa',
    ).to('cuda').eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    manifest = dict(model=args.model, revision=getattr(model.config, '_commit_hash', None),
                    config=model.config.to_dict(), seed=20260905,
                    head_labels=head_labels, head_dimension=model.config.hidden_size // model.config.num_attention_heads,
                    qk_location='after Q/K projection and RoPE; original SDPA inputs; BF16 stored without conversion',
                    selection='first fixed-length prefix of shuffled eligible unique documents',
                    documents=[], torch_version=torch.__version__)
    start = time.perf_counter()
    for number, doc in enumerate(documents):
        name = f"{doc['split']}_{doc['document_index']:04d}.pt"
        record = {key: value for key, value in doc.items() if key != 'input_ids'}
        record.update(file=name, length=len(doc['input_ids']), token_sha256=digest(json.dumps(doc['input_ids'])))
        if not (args.out / name).exists():
            captured.clear()
            ids = torch.tensor(doc['input_ids'], device='cuda').unsqueeze(0)
            with torch.inference_mode():
                model.model(input_ids=ids, use_cache=False)
            if set(captured) != set(layers):
                raise RuntimeError(f'Expected layers {layers}, captured {list(captured)}')
            tensors = [torch.cat([captured[layer][j] for layer in layers], dim=0) for j in range(3)]
            torch.save(dict(q=tensors[0], k=tensors[1], v=tensors[2],
                            input_ids=ids[0].cpu(), head_labels=head_labels), args.out / name)
        manifest['documents'].append(record)
        if number % 4 == 0 or number + 1 == len(documents):
            print(json.dumps(dict(event='captured', completed=number+1, total=len(documents),
                                  split=doc['split'], seconds=round(time.perf_counter()-start, 1))), flush=True)
    manifest['numerical_checks'] = numerical_checks
    manifest['seconds'] = time.perf_counter() - start
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps(dict(event='complete', checks=numerical_checks, seconds=manifest['seconds'])), flush=True)


if __name__ == '__main__':
    main()
