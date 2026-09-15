"""Run the fixed M2 confirmation and lifecycle protocol, preserving every file."""
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys

folder = pathlib.Path(__file__).resolve().parent
benchmark, replay, checkpoint = [pathlib.Path(arg).resolve() for arg in sys.argv[1:]]
protocol = json.loads((folder / 'protocol.json').read_text())
cases = [(session['name'], fraction) for session in protocol['sessions']
         for fraction in session['sharing_percentages']]
names = [f'{session}-shared{fraction}' for session, fraction in cases] + ['mixed', 'pressure']
outputs = [folder / 'build.json']
for name in names:
    outputs.extend(folder / f'{name}.{suffix}' for suffix in ('jsonl', 'time', 'host.json'))
for path in outputs:
    if path.exists():
        raise SystemExit(f'Refusing to replace {path}')

def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()

compiled = json.loads((folder / 'compiled-build.json').read_text())
assert compiled['profiling_enabled'] is False
checkpoint_hash = digest(checkpoint)
assert checkpoint_hash == protocol['checkpoint_sha256']
benchmark_hash, replay_hash = digest(benchmark), digest(replay)
config_hash = digest(checkpoint.with_name('config.json'))
with (folder / 'build.json').open('x') as file:
    json.dump({'compiled_source_sha256': compiled['compiled_source_sha256'],
               'checkpoint_sha256': checkpoint_hash, 'config_sha256': config_hash,
               'benchmark_binary_sha256': benchmark_hash, 'replay_binary_sha256': replay_hash,
               'protocol_sha256': digest(folder / 'protocol.json'),
               'input_files_sha256': {path.name: digest(path) for path in folder.glob('*.json')
                                      if path.name != 'build.json'},
               'benchmark_binary': str(benchmark), 'replay_binary': str(replay)}, file, indent=2)
    file.write('\n')

def snapshot():
    thermal = subprocess.run(['pmset', '-g', 'therm'], capture_output=True, text=True)
    return {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'load_average_1_5_15_minutes': os.getloadavg(),
            'thermal_report_exit_code': thermal.returncode,
            'thermal_report': thermal.stdout.strip(), 'thermal_report_error': thermal.stderr.strip()}

env = dict(os.environ, RAYON_NUM_THREADS='4', PAGED_INFER_MATMUL_TILE='4',
           PAGED_INFER_ATTN_LANES_PER_THREAD='2')

def run(name, args, extra_env=None, report_on_stdout=False):
    before = snapshot()
    output = folder / f'{name}.jsonl'
    process_env = dict(env, **(extra_env or {}))
    with (folder / f'{name}.time').open('x') as stderr:
        if report_on_stdout:
            with output.open('x') as stdout:
                result = subprocess.run(['/usr/bin/time', '-l', *args], env=process_env,
                                        stdout=stdout, stderr=stderr)
        else:
            result = subprocess.run(['/usr/bin/time', '-l', *args, '--output', str(output)],
                                    env=process_env, stdout=subprocess.DEVNULL, stderr=stderr)
    after = snapshot()
    with (folder / f'{name}.host.json').open('x') as file:
        json.dump({'before': before, 'after': after, 'exit_code': result.returncode}, file, indent=2)
        file.write('\n')
    if result.returncode:
        raise SystemExit(f'{name} exited {result.returncode}; partial evidence retained')
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[0]['profiling_enabled'] is False, name
    assert rows[-1]['type'] == 'verification' and rows[-1]['passed'] is True, name
    print(f'Completed {name}; all reported output checks passed.', flush=True)

for session, fraction in cases:
    run(f'{session}-shared{fraction}', [str(benchmark)],
        {'MODEL_PATH': str(checkpoint), 'QUANT': 'int8', 'SHARED_DECODE_CONTEXT': '4096',
         'SHARED_DECODE_BATCH': '8', 'SHARED_DECODE_STEPS': '16', 'SHARED_DECODE_REPS': '12',
         'SHARED_DECODE_PERCENTAGE': str(fraction)}, True)

common = [str(replay), '--model', str(checkpoint), '--threads', '4', '--quant', 'int8',
          '--max-steps', '1024', '--timeout-secs', '600', '--verify']
run('mixed', [*common, '--workload', str(folder / 'mixed-workload.json'),
              '--config', str(folder / 'mixed-baseline.json'),
              '--config', str(folder / 'mixed-shared.json'), '--repeats', '2'])
run('pressure', [*common, '--workload', str(folder / 'pressure-workload.json'),
                 '--config', str(folder / 'pressure-roomy.json'),
                 '--config', str(folder / 'pressure-baseline.json'),
                 '--config', str(folder / 'pressure-shared.json'), '--repeats', '1'])
