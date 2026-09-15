"""Validate every recorded pair and summarize the fixed cold-prefill experiment."""
import json
import pathlib
import statistics

folder = pathlib.Path(__file__).resolve().parent

def read_report(name):
    rows = [json.loads(line) for line in (folder / name).read_text().splitlines()]
    assert rows[-1]['type'] == 'verification' and rows[-1]['passed'] is True, name
    runs = [row for row in rows if row['type'] == 'run']
    assert len(runs) == 1, name
    return rows[0], runs[0]

def output_signature(report):
    return [(request['id'], [(seq['tokens'], seq['finish_reason'])
            for seq in request['sequences']]) for request in report['requests']]

profile_manifest, profile_run = read_report('diagnostic.jsonl')
assert profile_manifest['profiling_enabled'] and not profile_manifest['performance_gate_eligible']
main = [row for row in profile_run['profile'] if row['scope'] == 'model_main']
main_total = sum(row['elapsed_ms'] for row in main)
profile = [{**row, 'percent_of_model_main': 100 * row['elapsed_ms'] / main_total} for row in main]
protocol = json.loads((folder / 'protocol.json').read_text())
pairs = []
identity = None
source = None
last_started = 0
expected = output_signature(profile_run['report'])
for pair, order in enumerate(protocol['ordinary_comparison']['order'], 1):
    observations = {}
    for tile in order:
        name = f'pair{pair}-tile{tile}.jsonl'
        manifest, run = read_report(name)
        assert manifest['started_unix_ms'] > last_started, name
        last_started = manifest['started_unix_ms']
        assert not manifest['profiling_enabled'] and manifest['performance_gate_eligible'], name
        assert run['profile'] == [], name
        assert manifest['environment']['matmul_tile_env'] == str(tile), name
        assert manifest['environment']['rayon_threads'] == 4, name
        if identity is None:
            identity = manifest['identity']
            source = manifest['environment']['build']['source_sha256']
        assert manifest['identity'] == identity == profile_manifest['identity'], name
        assert manifest['environment']['build']['source_sha256'] == source, name
        report = run['report']
        assert output_signature(report) == expected, name
        assert report['engine']['prompt_tokens_prefilled'] == 2049, name
        assert report['engine']['prefill_chunks'] == 65, name
        assert report['engine']['shared_attention_layer_calls'] == 0, name
        assert report['summary']['successful_requests'] == 1, name
        assert report['summary']['output_tokens'] == 1, name
        observations[tile] = {
            'elapsed_ms': report['elapsed_ms'],
            'prefill_ms': report['engine']['prefill_ms'],
            'ttft_ms': report['summary']['ttft_ms']['p50'],
            'prompt_tokens_per_second': 2049 * 1000 / report['elapsed_ms'],
        }
    ratio = observations[4]['elapsed_ms'] / observations[6]['elapsed_ms']
    pairs.append({'pair': pair, 'order': order, 'tile4': observations[4], 'tile6': observations[6],
                  'elapsed_ratio_4_over_6': ratio,
                  'latency_reduction_percent': 100 * (1 - 1 / ratio)})
ratios = [row['elapsed_ratio_4_over_6'] for row in pairs]
result = {
    'diagnostic_source_sha256': profile_manifest['environment']['build']['source_sha256'],
    'ordinary_source_sha256': source,
    'diagnostic_elapsed_ms': profile_run['report']['elapsed_ms'],
    'diagnostic_model_main_ms': main_total,
    'diagnostic_profile': profile,
    'pairs': pairs,
    'paired_median_elapsed_ratio_4_over_6': statistics.median(ratios),
    'observed_ratio_range': [min(ratios), max(ratios)],
    'exact_output_signature': expected,
    'all_three_pairs_improve': all(ratio > 1 for ratio in ratios),
    'conclusion': 'Retain opt-in candidate only' if all(ratio > 1 for ratio in ratios) else 'Reject and revert candidate',
    'limitations': 'Three serial pairs on an uncontrolled desktop; no reliable tail or confidence-bound estimate; cold prefill only; defaults remain unchanged; not shared-attention gate evidence.',
}
print(json.dumps(result, indent=2))
