#!/usr/bin/env python3
"""Validate the fixed PR16 measurement protocol; never infer CI/merge readiness.

Uses only the standard library. Run after all measurements, without overlapping
inference: python3 analyze_readiness.py [--directory DIR] [--output NEW_FILE].
Missing or malformed evidence is a failed gate. Output files are never replaced.
"""
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import re
import sys

BOOTSTRAP_SEED = 0x7061697265644349
BOOTSTRAP_RESAMPLES = 10000
MASK64 = (1 << 64) - 1
INPUT_FILES = (
    'compiled-build.json',
    'mixed-workload.json', 'mixed-baseline.json', 'mixed-shared.json',
    'pressure-workload.json', 'pressure-roomy.json',
    'pressure-baseline.json', 'pressure-shared.json',
)
INVALID_EVIDENCE = (KeyError, IndexError, TypeError, ValueError, OSError,
                    ZeroDivisionError, OverflowError, AttributeError, StopIteration)


def no_duplicates(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError('duplicate JSON key: ' + key)
        value[key] = item
    return value


def parse_json(text):
    def invalid(value):
        raise ValueError('non-finite JSON number: ' + value)
    def real(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            invalid(value)
        return parsed
    return json.loads(text, object_pairs_hook=no_duplicates, parse_constant=invalid,
                      parse_float=real)


def read_json(path):
    return parse_json(path.read_text())


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(value):
    # serde_json::Value uses sorted object keys in these manifests. Inputs have
    # only integral numbers, strings and arrays, avoiding float spelling issues.
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def is_digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def finite(value, positive=False):
    return (type(value) in (int, float) and math.isfinite(value)
            and (value > 0 if positive else value >= 0))


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def median(values):
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    lower, upper = ordered[middle - 1:middle + 1]
    return lower + (upper - lower) * 0.5


def distribution(values):
    ordered = sorted(values)
    return dict(count=len(ordered), min=ordered[0], max=ordered[-1],
                **{name: ordered[math.ceil(len(ordered) * p) - 1]
                   for name, p in [('p50', .5), ('p95', .95), ('p99', .99)]})


def bootstrap_interval(ratios):
    state = BOOTSTRAP_SEED
    size = len(ratios)
    rejection = ((-size) & MASK64) % size
    medians = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sample = []
        for _ in range(size):
            while True:
                state = (state + 0x9e3779b97f4a7c15) & MASK64
                value = state
                value = ((value ^ (value >> 30)) * 0xbf58476d1ce4e5b9) & MASK64
                value = ((value ^ (value >> 27)) * 0x94d049bb133111eb) & MASK64
                value ^= value >> 31
                if value >= rejection:
                    sample.append(ratios[value % size])
                    break
        medians.append(median(sample))
    medians.sort()
    return {'lower': medians[BOOTSTRAP_RESAMPLES // 40 - 1],
            'upper': medians[BOOTSTRAP_RESAMPLES * 39 // 40 - 1]}


class Check:
    def __init__(self, name):
        self.name = name
        self.reasons = []
        self.data = {}

    def require(self, condition, message):
        if not condition:
            self.reasons.append(message)

    def equal(self, actual, expected, message):
        def brief(value):
            text = repr(value)
            return text if len(text) <= 400 else text[:397] + '...'
        self.require(type(actual) is type(expected) and actual == expected,
                     f'{message}: expected {brief(expected)}, got {brief(actual)}')

    def close(self, actual, expected, message):
        self.require(finite(actual) and math.isclose(actual, expected, rel_tol=1e-10,
                                                   abs_tol=1e-8),
                     f'{message}: expected {expected!r}, got {actual!r}')

    def attempt(self, label, function):
        try:
            return function()
        except INVALID_EVIDENCE as error:
            self.reasons.append(f'{label}: {type(error).__name__}: {error}')
            return None

    def result(self):
        return dict(name=self.name, passed=not self.reasons,
                    failure_reasons=self.reasons, **self.data)


def check_distribution(check, actual, expected, label):
    check.equal(actual['count'], expected['count'], label + '.count')
    for key in ('min', 'max', 'p50', 'p95', 'p99'):
        check.close(actual[key], expected[key], label + '.' + key)


def load_report(folder, name, check):
    path = folder / (name + '.jsonl')
    lines = path.read_text().splitlines()
    check.require(bool(lines), 'report is empty')
    rows = [parse_json(line) for line in lines]
    manifest = rows[0]
    verification = rows[-1]
    check.equal(manifest['type'], 'manifest', 'first record type')
    check.equal(verification['type'], 'verification', 'last record type')
    for field in ('passed', 'complete'):
        check.equal(verification[field], True, 'verification.' + field)
    check.equal(sum(row['type'] == 'manifest' for row in rows), 1, 'manifest count')
    check.equal(sum(row['type'] == 'verification' for row in rows), 1, 'verification count')
    check.data['report_sha256'] = digest_file(path)
    host = read_json(folder / (name + '.host.json'))
    check.equal(host['exit_code'], 0, 'process exit code')
    times = []
    for stage in ('before', 'after'):
        snap = host[stage]
        stamp = datetime.datetime.fromisoformat(snap['utc'])
        check.require(stamp.tzinfo is not None, stage + ' host timestamp lacks timezone')
        times.append(stamp.timestamp() * 1000)
        check.require(len(snap['load_average_1_5_15_minutes']) == 3 and
                      all(finite(v) for v in snap['load_average_1_5_15_minutes']),
                      stage + ' host load snapshot is invalid')
        check.require(type(snap['thermal_report_exit_code']) is int,
                      stage + ' thermal status missing')
    check.require(times[0] <= times[1], 'host snapshots reversed')
    check.require((folder / (name + '.time')).stat().st_size > 0,
                  'process timing report is empty')
    check.data['host_diagnostics'] = host
    check.data['host_interval_unix_ms'] = times
    return rows, manifest, verification, times


def common_manifest(check, manifest, protocol, frozen, benchmark):
    settings = protocol['settings']
    check.equal(manifest['profiling_enabled'], False, 'profiling enabled')
    check.equal(manifest['performance_gate_eligible'], True, 'performance eligibility')
    env = manifest['environment']
    build = env['build']
    compiled_build = frozen['_compiled_build']['build']
    for field in ('debug', 'git_dirty', 'git_head', 'opt_level', 'profile', 'rustc',
                  'source_sha256', 'target'):
        check.equal(build[field], compiled_build[field], 'frozen build ' + field)
    check.equal(build['source_sha256'], frozen['compiled_source_sha256'], 'compiled source')
    check.equal(build['profile'], 'release', 'build profile')
    check.equal(build['opt_level'], '3', 'optimization level')
    check.equal(build['debug'], 'false', 'debug build')
    check.require(build['rustflags'] in ('', []), 'unexpected compiler flags')
    check.equal(env['rayon_threads'], settings['threads'], 'worker count')
    check.equal(env['rayon_num_threads_env'], str(settings['threads']), 'worker environment')
    check.equal(env['matmul_tile_env'], str(settings['matmul_tile']), 'matmul tile')
    if benchmark:
        check.equal(env['attention_lanes_per_thread_env'],
                    str(settings['attention_lanes_per_thread']), 'attention lanes')
    check.equal(env['arch'], 'aarch64', 'host architecture')
    check.equal(env['os'], 'macos', 'host OS')
    check.equal(env['cpu'], 'Apple M2', 'host CPU')
    check.equal(env['simd'], 'neon', 'SIMD backend')
    check.equal(build['target'], 'aarch64-apple-darwin', 'build target')
    if benchmark:
        check.equal(manifest['checkpoint_sha256'], protocol['checkpoint_sha256'], 'checkpoint')
        check.equal(manifest['config_sha256'], frozen['config_sha256'], 'model config')
        check.equal(manifest['quantization'].lower(), settings['quantization'], 'quantization')
        check.equal(manifest['synthetic_weight_fixture'], False, 'synthetic weights')
        check.equal(manifest['synthetic_kv'], False, 'synthetic KV')
    else:
        identity = manifest['identity']
        check.equal(identity['weights_sha256'], protocol['checkpoint_sha256'], 'checkpoint')
        check.equal(identity['model_config_sha256'], frozen['config_sha256'], 'model config')
        check.equal(identity['quantization'].lower(), settings['quantization'], 'quantization')
        check.equal(identity['synthetic_fixture'], False, 'synthetic fixture')


def analyze_decode(folder, name, percentage, protocol, frozen):
    check = Check(name)
    def inspect():
        rows, manifest, verification, host_times = load_report(folder, name, check)
        common_manifest(check, manifest, protocol, frozen, True)
        settings = protocol['settings']
        batch, steps, repeats = (settings[key] for key in ('batch', 'steps', 'pairs_per_case'))
        for key, expected in [('batch', batch), ('steps', steps), ('repeats', repeats),
                              ('context', settings['context']), ('block_size', 16),
                              ('requested_shared_percentage', percentage)]:
            check.equal(manifest[key], expected, 'manifest.' + key)
        check.equal(manifest['benchmark'], 'shared_decode', 'benchmark identity')
        shape = manifest['shape']
        expected_shape = {'layers': 16, 'hidden': 2048, 'heads': 32, 'kv_heads': 8,
                          'head_dim': 64, 'vocabulary': 128256}
        check.equal(shape, expected_shape, 'model shape')
        shared_tokens = (settings['context'] // 16 * percentage // 100) * 16
        check.equal(manifest['mapped_common_prefix_tokens'], shared_tokens, 'mapped prefix')
        check.equal(manifest['inputs_sha256'], canonical_digest(manifest['inputs']), 'input digest')
        inputs = manifest['inputs']
        check.equal(len(inputs['common_content_prompt']), settings['context'], 'prompt length')
        for field in ('initial_held_tokens', 'private_prompt_tokens'):
            check.equal(len(inputs[field]), batch, field + ' count')
        check.equal(inputs['initial_held_position'], settings['context'] + 1, 'held position')
        check.equal(inputs['measured_steps'], steps, 'input steps')
        hash_spec = manifest['warmup_logit_hash']
        check.equal(hash_spec['algorithm'], 'SHA-256', 'warmup hash algorithm')
        check.equal(hash_spec['timed_loop_hashing'], False, 'timed hashing')
        for key, expected in [('steps_per_run', steps), ('entries_per_step', batch),
                              ('logits_per_entry', shape['vocabulary'])]:
            check.equal(hash_spec[key], expected, 'warmup hash ' + key)
        runs = [row for row in rows if row['type'] == 'run']
        check.equal(len(runs), repeats * 2, 'run count')
        check.equal([row['type'] for row in rows],
                    ['manifest'] + ['run'] * (repeats * 2) + ['summary', 'verification'],
                    'complete record sequence')
        by_key = {}
        signatures = []
        per_run = []
        previous_start = None
        for index, run in enumerate(runs):
            repeat = index // 2 + 1
            order = index % 2
            enabled = bool((repeat - 1) % 2 ^ order)
            variant = 'shared_prefix' if enabled else 'baseline'
            check.equal(run['repeat'], repeat, 'run repeat')
            check.equal(run['order_index'], order, 'run order')
            check.equal(run['variant'], variant, 'run variant')
            key = (run['repeat'], run['variant'])
            check.require(key not in by_key, 'duplicate run key ' + str(key))
            by_key[key] = run
            tokens = run['generated_tokens']
            check.require(isinstance(tokens, list) and len(tokens) == batch and
                          all(isinstance(row, list) and len(row) == steps and
                              all(integer(v) and v < shape['vocabulary'] for v in row)
                              for row in tokens), 'run does not contain complete valid tokens')
            for field in ('verified_finite_logits', 'verified_complete_greedy_output',
                          'verified_complete_warmup_logits', 'verified_warmup_timed_greedy_output'):
                check.equal(run[field], True, field)
            check.equal(run['profile'], [], 'run profiling counters')
            check.equal(run['finish_reason'], 'fixed_step_budget', 'finish reason')
            check.equal(run['warmup_logit_steps_hashed'], steps, 'warmup step count')
            for field in ('final_logits_sha256', 'warmup_logits_sha256'):
                check.require(is_digest(run[field]), 'invalid ' + field)
            signatures.append((tokens, run['final_logits_sha256'], run['warmup_logits_sha256']))
            expected_calls = steps * shape['layers'] if enabled and percentage else 0
            check.equal(run['shared_layer_calls'], expected_calls, 'shared layer count')
            check.equal(run['shared_query_tokens'], expected_calls * shared_tokens * batch,
                        'shared query-token count')
            check.equal(run['selected_path'], 'shared_prefix' if expected_calls else 'fallback',
                        'selected attention path')
            scratch = run['candidate_extra_scratch_bytes']
            check.require(integer(scratch) and (scratch > 0 if expected_calls else scratch == 0),
                          'unexpected extra scratch allocation')
            check.require(finite(run['elapsed_ms'], True), 'invalid elapsed time')
            check.require(finite(run['warmup_elapsed_ms'], True), 'invalid warmup time')
            check.close(run['tokens_per_second'], batch * steps * 1000 / run['elapsed_ms'],
                        'run throughput')
            step_times = run['step_ms']
            check.require(len(step_times) == steps and all(finite(v, True) for v in step_times),
                          'invalid/incomplete step timings')
            check.require(sum(step_times) <= run['elapsed_ms'] + 1e-6,
                          'step times exceed total elapsed time')
            dist = distribution(step_times)
            check_distribution(check, run['step_distribution_ms'], dist, 'step distribution')
            stamp = run['timed_loop_start_unix_ms']
            check.require(integer(stamp, 1) and host_times[0] - 1 <= stamp <= host_times[1] + 1,
                          'loop timestamp absent or outside host snapshots')
            if previous_start is not None:
                check.require(stamp >= previous_start, 'loop timestamps out of order')
            previous_start = stamp
            # Resource counters do not decide performance acceptance or explain
            # causes. Preserve the original optional values for diagnosis.
            per_run.append(dict(repeat=repeat, variant=variant, elapsed_ms=run['elapsed_ms'],
                                step_distribution_ms=dist,
                                timed_loop_start_unix_ms=stamp,
                                process_resource_usage=run['process_resource_usage']))
        check.require(bool(signatures) and all(s == signatures[0] for s in signatures),
                      'complete tokens/final logits/warmup logits differ between runs')
        check.equal(verification['runs'], repeats * 2, 'verification run count')
        check.equal(verification['warmup_logit_steps_per_run'], steps, 'verification warmup steps')
        check.equal(verification['warmup_logits_sha256'], signatures[0][2], 'verification digest')
        baseline = [by_key[(rep, 'baseline')]['elapsed_ms'] for rep in range(1, repeats + 1)]
        candidate = [by_key[(rep, 'shared_prefix')]['elapsed_ms'] for rep in range(1, repeats + 1)]
        ratios = [left / right for left, right in zip(baseline, candidate)]
        if not all(finite(value, True) for value in ratios + baseline + candidate):
            raise ValueError('non-finite/non-positive recomputed timing or paired ratio')
        estimate = median(ratios)
        interval = bootstrap_interval(ratios)
        aggregate = sum(baseline) / sum(candidate)
        if not finite(aggregate, True):
            raise ValueError('non-finite/non-positive aggregate timing ratio')
        pooled = {variant: distribution([value for run in runs if run['variant'] == variant
                                        for value in run['step_ms']])
                  for variant in ('baseline', 'shared_prefix')}
        order_medians = [median(ratios[parity::2]) for parity in (0, 1)]
        summary = next(row for row in rows if row['type'] == 'summary')
        check.equal(summary['timing_samples_retained'], repeats * 2, 'retained timing sample count')
        paired = summary['paired_elapsed_time_speedup']
        check.equal(paired['pair_count'], repeats, 'primary pair count')
        for field, expected in [('median', estimate), ('min', min(ratios)), ('max', max(ratios))]:
            check.close(paired[field], expected, 'primary ' + field)
        check.equal(len(paired['pairs']), repeats, 'primary pair records')
        for index, pair in enumerate(paired['pairs']):
            check.equal(pair['repeat'], index + 1, 'primary pair repeat')
            for field, expected in [('baseline_elapsed_ms', baseline[index]),
                                    ('candidate_elapsed_ms', candidate[index]),
                                    ('speedup', ratios[index])]:
                check.close(pair[field], expected, 'primary pair ' + field)
        bootstrap = paired['bootstrap_95']
        for key, expected in [('resamples', BOOTSTRAP_RESAMPLES),
                              ('seed_hex', f'0x{BOOTSTRAP_SEED:016x}'),
                              ('pairs_per_resample', repeats), ('status', 'descriptive_interval')]:
            check.equal(bootstrap[key], expected, 'bootstrap ' + key)
        check.close(bootstrap['confidence_level'], .95, 'bootstrap confidence')
        for key in ('lower', 'upper'):
            check.close(bootstrap['interval'][key], interval[key], 'bootstrap ' + key)
        check_distribution(check, summary['baseline_step_ms'], pooled['baseline'], 'baseline pooled')
        check_distribution(check, summary['candidate_step_ms'], pooled['shared_prefix'], 'candidate pooled')
        for key, values in [('baseline', baseline), ('candidate', candidate)]:
            check_distribution(check, summary[key + '_elapsed_ms'], distribution(values), key + ' elapsed')
            check.close(summary[key + '_median_tokens_per_second'],
                        batch * steps * 1000 / distribution(values)['p50'],
                        key + ' legacy median throughput')
        check.close(summary['median_speedup'], distribution(baseline)['p50'] /
                    distribution(candidate)['p50'], 'legacy ratio of marginal medians')
        diagnostics = summary['performance_diagnostics']
        check.equal(diagnostics['diagnostic_only'], True, 'diagnostic estimator designation')
        reported_aggregate = diagnostics['aggregate_throughput']
        for key, expected in [('speedup', aggregate), ('baseline_elapsed_ms', sum(baseline)),
                              ('candidate_elapsed_ms', sum(candidate)),
                              ('tokens_per_variant', batch * steps * repeats)]:
            check.close(reported_aggregate[key], expected, 'aggregate ' + key)
        strata = diagnostics['order_stratified_paired_speedup']
        check.equal(len(strata), 2, 'order strata count')
        for parity, stratum in enumerate(strata):
            numbers = list(range(parity + 1, repeats + 1, 2))
            check.equal(stratum['repeat_numbers'], numbers, 'order stratum repeats')
            check.equal(stratum['pair_count'], len(numbers), 'order stratum count')
            check.equal(stratum['first_variant'], ['baseline', 'shared_prefix'][parity],
                        'order stratum first variant')
            check.close(stratum['median'], order_medians[parity], 'order stratum median')
        check.data['evidence_valid'] = not check.reasons
        limits = protocol['performance_acceptance']
        kind = 'shared' if percentage else 'control'
        primary = limits[kind + '_each_session']
        extra = limits['additional_merge_safeguards']
        p95_ratio = pooled['shared_prefix']['p95'] / pooled['baseline']['p95']
        gates = {
            'paired_median': (estimate, '>=', primary['minimum_paired_median']),
            'descriptive_interval_lower': (interval['lower'], '>=', primary['minimum_descriptive_interval_lower']),
            'aggregate_throughput': (aggregate, '>=', extra['minimum_' + kind + '_aggregate_throughput_ratio']),
            'pooled_p95_candidate_over_baseline': (p95_ratio, '<=', extra['maximum_candidate_over_baseline_pooled_p95_ratio']),
            'baseline_first_median': (order_medians[0], '>=', extra['minimum_' + kind + '_order_stratum_median']),
            'candidate_first_median': (order_medians[1], '>=', extra['minimum_' + kind + '_order_stratum_median']),
        }
        gates = {key: dict(observed=value, comparison=op, threshold=threshold,
                           passed=value >= threshold if op == '>=' else value <= threshold)
                 for key, (value, op, threshold) in gates.items()}
        for key, gate in gates.items():
            check.require(gate['passed'], f'performance gate {key}: {gate["observed"]:.9g} '
                          f'does not satisfy {gate["comparison"]} {gate["threshold"]}')
        check.data.update(paired_ratios=ratios, paired_median=estimate,
                          descriptive_bootstrap_95=interval, aggregate_throughput_ratio=aggregate,
                          order_stratum_medians=dict(baseline_first=order_medians[0],
                                                    candidate_first=order_medians[1]),
                          pooled_step_ms=pooled, per_run=per_run, gates=gates,
                          output_signature_sha256=canonical_digest(signatures[0]),
                          input_sha256=manifest['inputs_sha256'],
                          compiled_source_sha256=manifest['environment']['build']['source_sha256'])
    check.attempt('decode evidence', inspect)
    check.data.setdefault('evidence_valid', False)
    return check


def analyze_lifecycle(folder, name, protocol, frozen):
    check = Check(name)
    def inspect():
        rows, manifest, verification, _ = load_report(folder, name, check)
        common_manifest(check, manifest, protocol, frozen, False)
        expected_files = ([name + '-baseline.json', name + '-shared.json'] if name == 'mixed'
                          else ['pressure-roomy.json', 'pressure-baseline.json', 'pressure-shared.json'])
        configs = [read_json(folder / filename) for filename in expected_files]
        expected_repeats = 2 if name == 'mixed' else 1
        expected_clock = 'milliseconds' if name == 'mixed' else 'steps'
        workload_file = folder / (name + '-workload.json')
        workload = read_json(workload_file)
        check.equal(manifest['workload'], workload, 'workload content')
        check.equal(manifest['identity']['workload_sha256'], digest_file(workload_file), 'workload digest')
        check.equal(workload['clock'], expected_clock, 'workload clock')
        check.equal(manifest['repeats'], expected_repeats, 'lifecycle repeats')
        check.equal(manifest['max_steps'], 1024, 'lifecycle step guard')
        check.close(manifest['timeout_ms'], 600000, 'lifecycle timeout')
        check.equal(verification['enabled'], True, 'lifecycle verification enabled')
        check.equal(verification['errors'], [], 'lifecycle verification errors')
        check.equal(len(manifest['configs']), len(configs), 'lifecycle config count')
        common_engines = []
        for filename, expected, actual in zip(expected_files, configs, manifest['configs']):
            check.equal(actual['name'], expected['name'], 'configuration name')
            check.equal(actual['source_sha256'], digest_file(folder / filename), 'configuration source hash')
            for key, value in expected['engine'].items():
                check.equal(actual['engine'][key], value, 'resolved configuration ' + key)
            engine = actual['engine']
            for key, value in [('block_size', 16), ('max_batch_size', 8), ('prefill_chunk_size', 32),
                               ('max_prefill_tokens_per_step', 32), ('enable_prefix_cache', True),
                               ('draft_tokens', 0), ('stream_tokens', True), ('eos_token', 0),
                               ('extra_eos_tokens', [])]:
                check.equal(engine[key], value, 'required configuration ' + key)
            common_engines.append({key: value for key, value in engine.items()
                                   if key not in ('shared_prefix_attention', 'total_blocks')})
        check.require(all(engine == common_engines[0] for engine in common_engines),
                      'lifecycle configurations differ outside toggle/block capacity')
        events = workload['events']
        check.require(all(event['type'] == 'submit' for event in events), 'unexpected lifecycle event')
        expected_requests = {event['id']: event for event in events}
        check.equal(len(expected_requests), 4 if name == 'mixed' else 1, 'declared request count')
        expected_sequences = sum(event['samples'] for event in events)
        expected_tokens = sum(event['samples'] * event['max_tokens'] for event in events)
        check.equal(expected_sequences, 8 if name == 'mixed' else 2, 'declared sequence count')
        check.equal(expected_tokens, 336 if name == 'mixed' else 128, 'declared output token count')
        runs = [row for row in rows if row['type'] == 'run']
        check.equal(len(runs), len(configs) * expected_repeats, 'lifecycle run count')
        check.equal([row['type'] for row in rows], ['manifest'] + ['run'] * len(runs) +
                    ['config_summary'] * len(configs) + ['verification'], 'lifecycle record sequence')
        summaries = [row for row in rows if row['type'] == 'config_summary']
        check.equal([row['config'] for row in summaries], [conf['name'] for conf in configs],
                    'configuration summary names')
        for row in summaries:
            check.equal(row['repeats'], expected_repeats, 'configuration summary repeats')
        signatures = []
        observations = []
        for index, run in enumerate(runs):
            repeat, slot = divmod(index, len(configs))
            expected_config = configs[(slot + repeat) % len(configs)]
            check.equal(run['repeat'], repeat + 1, 'lifecycle repeat')
            check.equal(run['order_in_repeat'], slot + 1, 'lifecycle order')
            check.equal(run['config'], expected_config['name'], 'lifecycle config order')
            check.equal(run['profile'], [], 'lifecycle profile')
            report = run['report']
            check.equal(report['clock'], expected_clock, 'report clock')
            check.require(finite(report['elapsed_ms'], True), 'invalid lifecycle elapsed time')
            requests = report['requests']
            check.equal(len(requests), len(expected_requests), 'observed request count')
            check.equal(sorted(req['id'] for req in requests), sorted(expected_requests), 'request identities')
            signature = []
            inter_delivery_gaps = []
            observed_token_gaps = []
            for request in sorted(requests, key=lambda req: req['id']):
                event = expected_requests[request['id']]
                check.equal(request['status'], 'finished', 'request status')
                check.equal(request['rejection'], None, 'request rejection')
                check.equal(request['scheduled_at'], event['at'], 'arrival deadline')
                for key in ('cancel_scheduled_at', 'cancel_dispatched_at_ms', 'cancel_dispatch_lag_ms'):
                    check.equal(request[key], None, 'unexpected cancellation')
                submitted, finished = request['submitted_at_ms'], request['finished_at_ms']
                check.require(finite(submitted) and finite(finished) and
                              submitted <= finished <= report['elapsed_ms'] + 1e-6,
                              'invalid request timestamps')
                origin = event['at'] if expected_clock == 'milliseconds' else submitted
                if expected_clock == 'milliseconds':
                    check.require(submitted >= origin, 'request dispatched before deadline')
                    check.close(request['dispatch_lag_ms'], submitted - origin, 'dispatch lag')
                else:
                    check.equal(request['dispatch_lag_ms'], None, 'step-clock dispatch lag')
                sequences = request['sequences']
                check.equal(len(sequences), event['samples'], 'request sample count')
                check.equal([seq['sample_index'] for seq in sequences], list(range(event['samples'])),
                            'sample identities')
                sequence_signatures = []
                for sequence in sequences:
                    check.equal(sequence['finish_reason'], 'length', 'sequence finish reason')
                    tokens = sequence['tokens']
                    check.require(len(tokens) == event['max_tokens'] and
                                  all(integer(token) and token < 128256 for token in tokens),
                                  'incomplete or invalid lifecycle output tokens')
                    deliveries = sequence['deliveries']
                    check.require(bool(deliveries), 'missing token deliveries')
                    times = [delivery['at_ms'] for delivery in deliveries]
                    check.require(all(finite(value) for value in times) and times == sorted(times),
                                  'invalid delivery timestamps')
                    check.require(all(integer(delivery['tokens'], 1) for delivery in deliveries),
                                  'invalid delivery token counts')
                    check.equal(sum(delivery['tokens'] for delivery in deliveries), len(tokens),
                                'delivery/token total')
                    # Match summarize() in src/replay.rs: a gap between
                    # nonempty deliveries contributes to both distributions;
                    # additional tokens within one delivery contribute zero
                    # observed token gaps, never invented delivery timestamps.
                    # Bound expansion by the complete token array, including
                    # for malformed evidence that already failed a check.
                    for delivery_index, delivery in enumerate(deliveries):
                        count = delivery['tokens']
                        if not integer(count, 1) or count > len(tokens):
                            raise ValueError('delivery token count exceeds valid sequence output')
                        if delivery_index:
                            gap = delivery['at_ms'] - deliveries[delivery_index - 1]['at_ms']
                            inter_delivery_gaps.append(gap)
                            observed_token_gaps.append(gap)
                        observed_token_gaps.extend([0.0] * (count - 1))
                    end = sequence['finished_at_ms']
                    check.require(finite(end) and submitted <= times[0] <= times[-1] <= end <= finished,
                                  'delivery/finish ordering')
                    check.close(sequence['ttft_ms'], times[0] - origin, 'deadline-derived TTFT')
                    check.close(sequence['end_to_end_ms'], end - origin, 'deadline-derived end-to-end')
                    sequence_signatures.append((sequence['sample_index'], tokens, sequence['finish_reason']))
                signature.append((request['id'], request['status'], request['rejection'], sequence_signatures))
            signatures.append(signature)
            summary, engine, memory = report['summary'], report['engine'], report['memory']
            for key, value in [('requests', len(events)), ('successful_requests', len(events)),
                               ('completed_sequences', expected_sequences), ('output_tokens', expected_tokens),
                               ('useful_output_tokens', expected_tokens), ('rejected_requests', 0),
                               ('failed_requests', 0), ('cancelled_requests', 0),
                               ('cancelled_sequences', 0), ('oom_sequences', 0)]:
                check.equal(summary[key], value, 'lifecycle summary ' + key)
            check.equal(summary['ttft_ms']['count'], expected_sequences, 'TTFT count')
            check.equal(summary['dispatch_lag_ms']['count'], len(events) if name == 'mixed' else 0,
                        'dispatch lag count')
            latency_values = {
                'ttft_ms': [seq['ttft_ms'] for req in requests for seq in req['sequences']],
                'end_to_end_ms': [seq['end_to_end_ms'] for req in requests for seq in req['sequences']],
                'dispatch_lag_ms': [req['dispatch_lag_ms'] for req in requests
                                    if req['dispatch_lag_ms'] is not None],
                'observed_inter_token_ms': observed_token_gaps,
                'inter_delivery_ms': inter_delivery_gaps,
            }
            check.equal(len(observed_token_gaps), expected_tokens - expected_sequences,
                        'delivery-derived observed token gap count')
            for label, values in latency_values.items():
                check.equal(summary[label]['count'], len(values), label + ' summary count')
                if values:
                    expected_distribution = distribution(values)
                    for field in ('p50', 'p95', 'p99', 'max'):
                        check.close(summary[label][field], expected_distribution[field],
                                    label + ' summary ' + field)
                else:
                    for field in ('p50', 'p95', 'p99', 'max'):
                        check.equal(summary[label][field], None, label + ' empty summary ' + field)
            check.close(summary['useful_tokens_per_second'], expected_tokens * 1000 / report['elapsed_ms'],
                        'lifecycle throughput')
            blocks = expected_config['engine']['total_blocks']
            enabled = expected_config['engine']['shared_prefix_attention']
            check.equal(memory['total_blocks'], blocks, 'pool capacity')
            check.require(integer(memory['peak_allocated_blocks']) and memory['peak_allocated_blocks'] <= blocks,
                          'invalid allocation peak')
            check.require(integer(memory['final_allocated_blocks']) and
                          memory['final_allocated_blocks'] <= memory['peak_allocated_blocks'],
                          'invalid final allocation')
            check.equal(engine['admitted_requests'], len(events), 'admitted request count')
            check.equal(engine['generated_tokens'], expected_tokens, 'engine generated tokens')
            for field, value in [('shared_attention_layer_calls', engine['shared_attention_layer_calls']),
                                 ('shared_attention_query_tokens', engine['shared_attention_query_tokens']),
                                 ('shared_attention_scratch_bytes', memory['shared_attention_scratch_bytes'])]:
                check.require(integer(value) and (value > 0 if enabled else value == 0),
                              'incorrect ' + field)
            if name == 'mixed' or blocks == 192:
                check.equal(engine['preemptions'], 0, 'unexpected preemption')
                check.equal(engine['recomputed_tokens'], 0, 'unexpected recomputation')
                check.equal(engine['prefill_preemptions'], 0, 'unexpected prefill preemption')
            else:
                check.equal(blocks, 132, 'tight pool size')
                check.equal(memory['peak_allocated_blocks'], 132, 'tight pool must reach capacity')
                for field in ('preemptions', 'recomputed_tokens', 'cow_copies'):
                    check.require(integer(engine[field], 1), 'pressure did not exercise ' + field)
            observations.append(dict(config=run['config'], repeat=run['repeat'],
                                     elapsed_ms=report['elapsed_ms'], summary=summary,
                                     engine=engine, memory=memory,
                                     output_signature_sha256=canonical_digest(signature)))
        check.require(bool(signatures) and all(sig == signatures[0] for sig in signatures),
                      'complete lifecycle outputs differ from first baseline/roomy oracle')
        check.data.update(observations=observations,
                          output_signature_sha256=canonical_digest(signatures[0]),
                          expected_requests=len(events), expected_sequences=expected_sequences,
                          expected_output_tokens=expected_tokens)
    check.attempt('lifecycle evidence', inspect)
    check.data['evidence_valid'] = not check.reasons
    return check


def provenance(folder, protocol, frozen, check_binaries=False):
    check = Check('provenance')
    def inspect():
        check.equal(frozen['protocol_sha256'], digest_file(folder / 'protocol.json'), 'frozen protocol hash')
        check.equal(frozen['checkpoint_sha256'], protocol['checkpoint_sha256'], 'frozen checkpoint hash')
        for field in ('compiled_source_sha256', 'config_sha256', 'checkpoint_sha256'):
            check.require(is_digest(frozen[field]), 'invalid frozen ' + field)
        compiled = frozen['_compiled_build']
        check.equal(compiled['profiling_enabled'], False, 'frozen profiling status')
        check.equal(compiled['compiled_source_sha256'], frozen['compiled_source_sha256'],
                    'frozen compiled source identity')
        check.equal(compiled['build']['source_sha256'], frozen['compiled_source_sha256'],
                    'frozen build source identity')
        binary_checks = {}
        for kind in ('benchmark', 'replay'):
            path = Path(frozen[kind + '_binary'])
            if not path.is_absolute():
                path = folder / path
            expected_digest = frozen[kind + '_binary_sha256']
            check.require(is_digest(expected_digest), 'invalid frozen ' + kind + ' binary hash')
            if not path.is_file() and (folder / path.name).is_file():
                path = folder / path.name
            if path.is_file():
                actual_digest = digest_file(path)
                check.equal(actual_digest, expected_digest, kind + ' executable hash')
                binary_checks[kind] = dict(status='rechecked', path=str(path), sha256=actual_digest)
            else:
                binary_checks[kind] = dict(status='unavailable_for_independent_recheck',
                                           recorded_path=str(path), recorded_sha256=expected_digest)
                check.require(not check_binaries, kind + ' executable required by --check-binaries is missing')
        check.data['binary_provenance'] = binary_checks
        for name in INPUT_FILES:
            check.equal(frozen['input_files_sha256'][name], digest_file(folder / name),
                        'frozen input ' + name)
        check.equal(protocol['settings']['features'], [], 'ordinary feature set')
        check.equal(protocol['sessions'], [{'name': 'A', 'sharing_percentages': [90, 0]},
                                          {'name': 'B', 'sharing_percentages': [0, 90]}],
                    'required independent session order')
        check.equal(protocol['settings'], dict(context=4096, batch=8, steps=16, pairs_per_case=12,
                                               threads=4, quantization='int8', matmul_tile=4,
                                               attention_lanes_per_thread=2, features=[]),
                    'fixed measurement settings')
        # A changed acceptance policy is not this predeclared experiment.
        limits = protocol['performance_acceptance']
        for kind, threshold in [('shared', 1.15), ('control', .95)]:
            check.equal(limits[kind + '_each_session'],
                        dict(minimum_paired_median=threshold, minimum_descriptive_interval_lower=threshold),
                        kind + ' fixed primary thresholds')
        check.equal(limits['additional_merge_safeguards'], {
            'minimum_shared_aggregate_throughput_ratio': 1.15,
            'minimum_control_aggregate_throughput_ratio': .95,
            'maximum_candidate_over_baseline_pooled_p95_ratio': 1.05,
            'minimum_shared_order_stratum_median': 1.0,
            'minimum_control_order_stratum_median': .95}, 'fixed additional safeguards')
        check.data.update(compiled_source_sha256=frozen['compiled_source_sha256'],
                          protocol_sha256=frozen['protocol_sha256'],
                          analyzer_sha256=digest_file(Path(__file__)))
    check.attempt('frozen provenance', inspect)
    return check


def analyze(folder, check_binaries=False):
    setup = Check('input_loading')
    protocol = setup.attempt('protocol.json', lambda: read_json(folder / 'protocol.json'))
    frozen = setup.attempt('build.json', lambda: read_json(folder / 'build.json'))
    compiled = setup.attempt('compiled-build.json', lambda: read_json(folder / 'compiled-build.json'))
    if protocol is None or frozen is None or compiled is None:
        return dict(measurement_gates_passed=False, ci_checked=False,
                    scope='measurement evidence only; no merge or CI determination',
                    cases=[], prerequisites=[setup.result()])
    frozen['_compiled_build'] = compiled
    source = provenance(folder, protocol, frozen, check_binaries)
    cases = [analyze_decode(folder, f'{session}-shared{percentage}', percentage, protocol, frozen)
             for session, percentage in [('A', 90), ('A', 0), ('B', 0), ('B', 90)]]
    cross = Check('cross_case_identity')
    for field in ('output_signature_sha256', 'input_sha256', 'compiled_source_sha256'):
        values = [case.data.get(field) for case in cases]
        cross.require(all(value is not None for value in values) and len(set(values)) == 1,
                      'all four decode cases must have the same ' + field)
    cases.extend(analyze_lifecycle(folder, name, protocol, frozen) for name in ('mixed', 'pressure'))
    intervals = [case.data.get('host_interval_unix_ms') for case in cases]
    cross.require(all(interval is not None for interval in intervals),
                  'every process needs complete before/after host timestamps')
    if all(interval is not None for interval in intervals):
        cross.require(all(previous[1] <= following[0]
                          for previous, following in zip(intervals, intervals[1:])),
                      'process timestamps overlap or violate fixed A90/A0/B0/B90/mixed/pressure order')
    prerequisites = [setup, source, cross]
    return dict(measurement_gates_passed=all(not check.reasons for check in prerequisites + cases),
                ci_checked=False, scope='measurement evidence only; no merge or CI determination',
                limitations=['Bootstrap intervals remain conditional descriptive estimates; serial drift, '
                             'desktop interference and order effects are not corrected.',
                             'Process resource counters are diagnostic only, never causal attribution.',
                             'Archived binary digests and source metadata are checked; absent executables '
                             'are explicitly unavailable for rehash unless --check-binaries is required.',
                             'All four decode cases and both lifecycle cases are required; no pooling '
                             'across sessions and no exclusion of completed samples.'],
                prerequisites=[check.result() for check in prerequisites],
                cases=[check.result() for check in cases])


def self_check():
    import tempfile
    assert median([9., 1., 5., 3.]) == 4.
    assert distribution([3., 1., 2.])['p95'] == 3.
    assert bootstrap_interval([1., 1., 1.]) == {'lower': 1., 'upper': 1.}
    for text in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":1e999}'):
        try:
            parse_json(text)
        except ValueError:
            pass
        else:
            raise AssertionError('invalid JSON accepted')
    with tempfile.TemporaryDirectory() as directory:
        result = analyze(Path(directory))
        assert result['measurement_gates_passed'] is False
        assert result['prerequisites'][0]['failure_reasons']
    print('Analyzer self-check passed; no performance evidence was generated.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, help='Create a new JSON report; never overwrite.')
    parser.add_argument('--self-check', action='store_true')
    parser.add_argument('--check-binaries', action='store_true',
                        help='Also fail when a frozen executable is unavailable for rehashing.')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return 0
    try:
        result = analyze(args.directory.resolve(), args.check_binaries)
    except INVALID_EVIDENCE as error:
        result = dict(measurement_gates_passed=False, ci_checked=False,
                      failure_reasons=[f'invalid protocol/evidence: {type(error).__name__}: {error}'])
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(text)
    else:
        sys.stdout.write(text)
    return 0 if result['measurement_gates_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
