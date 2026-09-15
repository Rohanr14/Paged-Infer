#!/usr/bin/env python3
"""Bounded negative checks; run only after the measurement pipeline finishes.

Copies small reports and metadata into temporary directories. Never copies or
hashes executables/checkpoints, edits original evidence, or runs inference.
Checks per-case evidence validity: a valid B0 performance failure is permitted.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile

CASES = ['A-shared90', 'A-shared0', 'B-shared0', 'B-shared90', 'mixed', 'pressure']


def load_module(path):
    spec = importlib.util.spec_from_file_location('readiness_analyzer_under_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def evidence_result(module, folder, case):
    protocol = module.read_json(folder / 'protocol.json')
    frozen = module.read_json(folder / 'build.json')
    frozen['_compiled_build'] = module.read_json(folder / 'compiled-build.json')
    if case in ('mixed', 'pressure'):
        return module.analyze_lifecycle(folder, case, protocol, frozen).result()
    percentage = int(case.split('shared')[1])
    return module.analyze_decode(folder, case, percentage, protocol, frozen).result()


def edit_runs(folder, case, edit):
    path = folder / (case + '.jsonl')
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    edit([row for row in rows if row['type'] == 'run'])
    # Preserve the generic success claim to ensure independent checks catch the
    # mutation rather than merely trusting the emitter's verification record.
    assert rows[-1]['type'] == 'verification' and rows[-1]['passed'] is True
    path.write_text(''.join(json.dumps(row, allow_nan=False) + '\n' for row in rows))


def truncate_report(folder):
    path = folder / 'A-shared90.jsonl'
    path.write_text('\n'.join(path.read_text().splitlines()[:-1]) + '\n')


def change_warmup(runs):
    original = runs[0]['warmup_logits_sha256']
    runs[0]['warmup_logits_sha256'] = ('0' if original != '0' * 64 else '1') * 64


def change_path_counter(runs):
    next(run for run in runs if run['variant'] == 'shared_prefix')['shared_layer_calls'] = 0


def shorten_lifecycle(runs, early_eos=False):
    # Mutate every variant/repeat identically, preserving cross-run token parity.
    for run in runs:
        report = run['report']
        request = min(report['requests'], key=lambda req: req['id'])
        sequence = request['sequences'][0]
        sequence['tokens'].pop()
        sequence['deliveries'][-1]['tokens'] -= 1
        if sequence['deliveries'][-1]['tokens'] == 0:
            sequence['deliveries'].pop()
        if early_eos:
            sequence['finish_reason'] = 'eos'
        for field in ('output_tokens', 'useful_output_tokens'):
            report['summary'][field] -= 1
        report['engine']['generated_tokens'] -= 1


def remove_pressure(runs):
    for run in runs:
        if run['report']['memory']['total_blocks'] == 132:
            for field in ('preemptions', 'recomputed_tokens', 'cow_copies'):
                run['report']['engine'][field] = 0


def change_deadline(runs):
    runs[0]['report']['requests'][0]['sequences'][0]['ttft_ms'] += 1.0


def change_summary(runs, field):
    runs[0]['report']['summary'][field]['p95'] += 1.0


def validate(folder, module):
    files = sorted(set(['protocol.json', 'build.json', *module.INPUT_FILES] +
                       [f'{case}.{suffix}' for case in CASES
                        for suffix in ('jsonl', 'host.json', 'time')]))
    # These are small evidence/metadata files only; no model or executable reads.
    before = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in files}
    baseline = {case: evidence_result(module, folder, case) for case in CASES}
    result = {'validation_passed': False, 'scope': 'Per-case evidence rejection; no inference or CI checks',
              'baseline': {case: {'evidence_valid': row.get('evidence_valid', False),
                                  'case_passed': row['passed'],
                                  'failure_reasons': row['failure_reasons']}
                           for case, row in baseline.items()},
              'negative_cases': []}
    if not all(row.get('evidence_valid') is True for row in baseline.values()):
        result['failure_reasons'] = ['Baseline evidence is not valid; mutation checks would be inconclusive.']
        return result

    def edit(case, function):
        return lambda temp: edit_runs(temp, case, function)

    checks = [
        ('missing_decode', 'A-shared90', lambda temp: (temp / 'A-shared90.jsonl').unlink(),
         ['FileNotFoundError']),
        ('truncated_decode', 'A-shared90', truncate_report, ['last record type']),
        ('warmup_digest_mismatch', 'A-shared90', edit('A-shared90', change_warmup),
         ['warmup logits differ']),
        ('wrong_shared_path_count', 'A-shared90', edit('A-shared90', change_path_counter),
         ['shared layer count']),
        ('equal_early_eos_outputs', 'mixed', edit('mixed', lambda runs: shorten_lifecycle(runs, True)),
         ['sequence finish reason']),
        ('equal_short_length_outputs', 'mixed', edit('mixed', shorten_lifecycle),
         ['incomplete or invalid lifecycle output tokens']),
        ('equal_outputs_without_pressure', 'pressure', edit('pressure', remove_pressure),
         ['pressure did not exercise preemptions', 'pressure did not exercise recomputed_tokens',
          'pressure did not exercise cow_copies']),
        ('incorrect_deadline_ttft', 'mixed', edit('mixed', change_deadline),
         ['deadline-derived TTFT']),
        ('incorrect_dispatch_summary', 'mixed', edit('mixed', lambda runs: change_summary(runs, 'dispatch_lag_ms')),
         ['dispatch_lag_ms summary p95']),
        ('incorrect_inter_token_summary', 'mixed', edit('mixed', lambda runs: change_summary(runs, 'observed_inter_token_ms')),
         ['observed_inter_token_ms summary p95']),
        ('incorrect_inter_delivery_summary', 'mixed', edit('mixed', lambda runs: change_summary(runs, 'inter_delivery_ms')),
         ['inter_delivery_ms summary p95']),
    ]
    for name, case, mutate, required_reasons in checks:
        with tempfile.TemporaryDirectory(prefix='pr16-analyzer-negative-') as directory:
            temporary = Path(directory)
            for filename in files:
                shutil.copyfile(folder / filename, temporary / filename)
            mutate(temporary)
            observed = evidence_result(module, temporary, case)
        reasons = observed['failure_reasons']
        matched = {expected: any(expected in reason for reason in reasons) for expected in required_reasons}
        result['negative_cases'].append({
            'name': name, 'case': case,
            'passed': observed.get('evidence_valid') is False and observed['passed'] is False
                      and all(matched.values()),
            'evidence_valid': observed.get('evidence_valid'),
            'expected_reasons_found': matched, 'failure_reasons': reasons,
        })
    after = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest() for name in files}
    result['original_files_unchanged'] = before == after
    result['original_files_checked'] = len(files)
    result['validation_passed'] = before == after and all(row['passed'] for row in result['negative_cases'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, help='Create a new JSON result; never overwrite.')
    args = parser.parse_args()
    try:
        folder = args.directory.resolve()
        module = load_module(folder / 'analyze_readiness.py')
        result = validate(folder, module)
    except Exception as error:
        result = {'validation_passed': False,
                  'failure_reasons': [f'{type(error).__name__}: {error}']}
    text = json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.output:
        with args.output.open('x') as stream:
            stream.write(text)
    else:
        sys.stdout.write(text)
    return 0 if result['validation_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
