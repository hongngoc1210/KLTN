"""Group-stratified 80/10/10 split; keeps complete documents together.

Greedy initialization followed by seeded local moves/swaps minimizes weighted
relative distribution error. This is a heuristic, not a guarantee of exact
stratification or a globally optimal solution. Gold annotations are preserved.
"""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


def split_dataset(source, output, seed=42, iterations=100000):
    records = json.loads(source.read_text(encoding='utf-8'))
    if not isinstance(records, list) or not records:
        raise ValueError('Input must be a nonempty JSON array.')
    for r in records:
        if not isinstance(r.get('doc_id'), str) or not r['doc_id'].strip():
            raise ValueError('Each QA requires a nonempty doc_id.')
    if len({r['qa_id'] for r in records}) != len(records):
        raise ValueError('qa_id is not unique.')
    chunk_docs = {}
    for r in records:
        previous = chunk_docs.setdefault(r['chunk_id'], r['doc_id'])
        if previous != r['doc_id']:
            raise ValueError('A chunk_id belongs to multiple doc_id values.')
    ratios = np.array([.8, .1, .1]); names = ['train', 'dev', 'test']
    grouped = {}
    for r in records:
        grouped.setdefault(r['doc_id'], []).append(r)
    doc_ids = sorted(grouped)
    if len(doc_ids) < 3:
        raise ValueError('At least three documents are required.')
    features = []
    def feature_counts(rows):
        c = Counter({'qa_count': len(rows), 'doc_count': 1})
        for r in rows:
            for field in ['domain', 'label', 'question_type']:
                c[f'{field}:{r[field]}'] += 1
            c[f"domain_label:{r['domain']}|{r['label']}"] += 1
            c[f"label_question:{r['label']}|{r['question_type']}"] += 1
            c[f"span_count:{len(r['span_label'])}"] += 1
            for s in r['span_label']:
                c[f"diagnosis:{s['diagnosis']}"] += 1
                c[f"span_label:{s['label']}"] += 1
        return c
    counters = [feature_counts(grouped[d]) for d in doc_ids]
    features = sorted(set().union(*(c.keys() for c in counters)))
    x = np.array([[c[f] for f in features] for c in counters], dtype=float)
    totals = x.sum(axis=0); target = ratios[:, None] * totals
    # Normalize category blocks so the number of categories alone does not
    # determine their influence. The floor prevents tiny classes dominating.
    block_weights = {'qa_count': 12, 'doc_count': .2, 'domain': 2, 'label': 3,
        'question_type': 2, 'domain_label': 2, 'label_question': 2,
        'diagnosis': 1, 'span_label': 1, 'span_count': 1}
    blocks = [f.split(':', 1)[0] for f in features]
    block_totals = {b: sum(totals[j] for j, block in enumerate(blocks) if block == b) for b in set(blocks)}
    weight = np.array([block_weights[b] / block_totals[b] for b in blocks])[None, :] / (target + 5)
    rng = np.random.default_rng(seed)
    current = np.zeros_like(target); assignment = np.full(len(doc_ids), -1, dtype=int)
    rarity = (x / totals).max(axis=1)
    order = np.lexsort((rng.random(len(doc_ids)), -rarity))
    for i in order:
        delta = ((2 * (current-target) * x[i] + x[i] ** 2) * weight).sum(axis=1)
        dest = int(delta.argmin()); current[dest] += x[i]; assignment[i] = dest
    # Required allocation: one implicit_subject_drift span in train and one
    # in test, moving entire documents and preserving all annotations.
    rare_docs = [i for i, d in enumerate(doc_ids) if any(
        span['diagnosis'] == 'implicit_subject_drift'
        for row in grouped[d] for span in row['span_label'])]
    rare_span_counts = [sum(span['diagnosis'] == 'implicit_subject_drift'
        for row in grouped[doc_ids[i]] for span in row['span_label']) for i in rare_docs]
    if len(rare_docs) != 2 or rare_span_counts != [1, 1]:
        raise ValueError('The requested constraint requires two distinct documents with one implicit_subject_drift span each.')
    fixed = {rare_docs[0]: 2, rare_docs[1]: 0}
    for i, dest in fixed.items():
        old = assignment[i]
        current[old] -= x[i]; current[dest] += x[i]; assignment[i] = dest
    group_counts = np.bincount(assignment, minlength=3)
    accepted = 0
    for step in range(iterations):
        i = int(rng.integers(len(doc_ids))); a = assignment[i]
        if i in fixed:
            continue
        if step % 2 == 0:
            b = int(rng.integers(3))
            if a == b or group_counts[a] <= 1:
                continue
            change = x[i]
            delta = (((-2*(current[a]-target[a])*change + change**2)*weight[a]).sum()
                     + ((2*(current[b]-target[b])*change + change**2)*weight[b]).sum())
            if delta < -1e-14:
                current[a] -= change; current[b] += change
                assignment[i] = b; group_counts[a] -= 1; group_counts[b] += 1; accepted += 1
        else:
            j = int(rng.integers(len(doc_ids))); b = assignment[j]
            if a == b or j in fixed:
                continue
            change = x[j] - x[i]
            delta = (((2*(current[a]-target[a])*change + change**2)*weight[a]).sum()
                     + ((-2*(current[b]-target[b])*change + change**2)*weight[b]).sum())
            if delta < -1e-14:
                current[a] += change; current[b] -= change
                assignment[i], assignment[j] = b, a; accepted += 1
    mapping = {d: names[int(assignment[i])] for i, d in enumerate(doc_ids)}
    subsets = {name: [r for r in records if mapping[r['doc_id']] == name] for name in names}
    assert [sum(s['diagnosis'] == 'implicit_subject_drift' for r in subsets[name] for s in r['span_label']) for name in names] == [1, 0, 1]
    # Verify exact preservation, unique IDs, disjoint documents and chunks.
    assert sum(map(len, subsets.values())) == len(records)
    assert Counter(r['qa_id'] for rows in subsets.values() for r in rows) == Counter(r['qa_id'] for r in records)
    for field in ['doc_id', 'chunk_id', 'qa_id']:
        sets = [{r[field] for r in subsets[name]} for name in names]
        for a in range(3):
            for b in range(a+1, 3):
                assert not sets[a] & sets[b], f'{field} overlap'
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in subsets.items():
        (output / f'{name}.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    pd.DataFrame([{'doc_id': d, 'split': mapping[d], 'qa_count': len(grouped[d])} for d in doc_ids]).to_csv(output/'document_split.csv', index=False, encoding='utf-8-sig')
    balance_rows = []
    for j, f in enumerate(features):
        item = {'feature': f, 'total': int(totals[j])}
        for k, name in enumerate(names):
            item[name] = int(current[k, j])
            item[f'{name}_allocation_pct'] = round(current[k, j]/totals[j]*100, 3)
        balance_rows.append(item)
    pd.DataFrame(balance_rows).to_csv(output/'balance_report.csv', index=False, encoding='utf-8-sig')
    summary = []
    for name in names:
        rows = subsets[name]
        summary.append({'split': name, 'qa_count': len(rows), 'qa_percent': round(len(rows)/len(records)*100, 3),
            'doc_count': len({r['doc_id'] for r in rows}), 'chunk_count': len({r['chunk_id'] for r in rows}),
            'span_count': sum(len(r['span_label']) for r in rows)})
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(output/'split_summary.csv', index=False, encoding='utf-8-sig')
    distribution_tables = {}
    for field in ['domain', 'label', 'question_type']:
        counts = pd.DataFrame({name: Counter(r[field] for r in subsets[name]) for name in names}).fillna(0).astype(int).sort_index()
        percentages = counts.div([len(subsets[name]) for name in names], axis=1)*100
        counts.to_csv(output/f'{field}_counts.csv', encoding='utf-8-sig')
        percentages.round(3).to_csv(output/f'{field}_percentages.csv', encoding='utf-8-sig')
        distribution_tables[field] = percentages
    metadata = {'source': str(source), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'seed': seed, 'target_ratios': dict(zip(names, ratios.tolist())), 'group_by': 'doc_id',
        'method': 'weighted group stratification: greedy initialization and local moves/swaps',
        'iterations': iterations, 'accepted_improvements': accepted, 'objective': float(((current-target)**2*weight).sum()),
        'fixed_documents': {doc_ids[i]: names[dest] for i, dest in fixed.items()},
        'implicit_subject_drift_counts': {'train': 1, 'dev': 0, 'test': 1},
        'block_weights': block_weights, 'summary': summary,
        'validation': {'all_records_preserved': True, 'unchanged_annotations': True,
            'disjoint_doc_ids': True, 'disjoint_chunk_ids': True, 'disjoint_qa_ids': True}}
    (output/'split_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['# Chia dataset theo tài liệu', '', f'Tỷ lệ mục tiêu QA: train 80%, dev 10%, test 10%; seed {seed}.',
        'Toàn bộ QA của cùng doc_id nằm trong một tập. Không nhân bản hoặc loại mẫu, giữ nguyên annotation.', '',
        '## Phương pháp', '', 'Phân tầng theo nhóm: khởi tạo greedy, sau đó tối ưu bằng chuyển/hoán đổi toàn bộ tài liệu. Hàm mục tiêu cân đối số QA, domain, label, question_type, domain×label, label×question_type, số span/QA, span label và diagnosis. Đây là heuristic; không bảo đảm cân bằng chính xác hoặc nghiệm tối ưu toàn cục.', '',
        '## Quy mô', '', summary_df.to_string(index=False), '', '## Phân phối trong từng tập (%)', '']
    for field, percentages in distribution_tables.items():
        lines += [f'### {field}', '', percentages.round(3).to_string(), '']
    rare = [r for r in balance_rows if r['feature'].startswith('diagnosis:') and r['total'] < 100]
    lines += ['## Ràng buộc nhãn hiếm', '', 'implicit_subject_drift: train 1 span, dev 0 span, test 1 span. Cố định hai tài liệu chứa nhãn này rồi tối ưu các tài liệu còn lại.', '', '## Diagnosis ít mẫu', '', pd.DataFrame(rare).to_string(index=False), '',
        'Lớp hiếm không thể bảo đảm có đủ mẫu độc lập trong cả ba tập. Không gộp implicit_subject_drift trong bước chia này; taxonomy giữ nguyên.', '',
        '## Kiểm tra', '', 'Không giao nhau doc_id, chunk_id, qa_id; mỗi QA xuất hiện đúng một lần. SHA-256 của nguồn được lưu trong split_metadata.json.',
        'Kiểm tra nhóm không chứng minh không có trùng gần đúng về nội dung giữa các tài liệu. Các tập được xáo theo nhóm bằng seed trong bước chọn, nhưng thứ tự QA trong mỗi file giữ theo nguồn.', '',
        'Dùng dev để chọn tham số/ngưỡng; few-shot chỉ lấy từ train; giữ test cố định cho tất cả mô hình.']
    (output/'README.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(summary_df.to_string(index=False))
    print('Validation passed: records preserved; document, chunk and QA IDs disjoint.')
    print('Rare diagnoses:', rare)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    base = Path(__file__).resolve().parents[1]
    parser.add_argument('--input', type=Path, default=base/'output/data_final.json')
    parser.add_argument('--output', type=Path, default=base/'output/splits')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--iterations', type=int, default=100000)
    args = parser.parse_args()
    split_dataset(args.input, args.output, args.seed, args.iterations)
