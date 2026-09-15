"""Fixed three-pair cold-prefill experiment; run from the repository root."""
import hashlib
import json
import os
import pathlib
import subprocess
import sys

folder = pathlib.Path(__file__).resolve().parent
binary = pathlib.Path(sys.argv[1]).resolve()
model = pathlib.Path(sys.argv[2]).resolve()
protocol = json.loads((folder / 'protocol.json').read_text())
digest_path = folder / 'ordinary-binary.sha256'
outputs = [digest_path]
for pair, order in enumerate(protocol['ordinary_comparison']['order'], 1):
    for tile in order:
        outputs.extend(folder / f'pair{pair}-tile{tile}.{suffix}'
                       for suffix in ('jsonl', 'stdout', 'time'))
for output in outputs:
    if output.exists():
        raise SystemExit(f'Refusing to replace {output}')
digest = hashlib.sha256(binary.read_bytes()).hexdigest()
with digest_path.open('x') as file:
    file.write(digest + '\n')
prior = folder / 'pair1-tile4.jsonl'
for pair, order in enumerate(protocol['ordinary_comparison']['order'], 1):
    for tile in order:
        stem = folder / f'pair{pair}-tile{tile}'
        output = stem.with_suffix('.jsonl')
        if output.exists():
            raise SystemExit(f'Refusing to replace {output}')
        args = [str(binary), '--model', str(model), '--workload', str(folder / 'workload.json'),
                '--config', str(folder / 'config.json'), '--quant', 'int8', '--threads', '4',
                '--repeats', '1', '--max-steps', '128', '--timeout-secs', '180', '--verify',
                '--output', str(output)]
        if output != prior:
            args += ['--compare-to', str(prior)]
        env = dict(os.environ, PAGED_INFER_MATMUL_TILE=str(tile))
        with stem.with_suffix('.stdout').open('x') as stdout, stem.with_suffix('.time').open('x') as stderr:
            subprocess.run(['/usr/bin/time', '-l', *args], env=env, stdout=stdout, stderr=stderr, check=True)
        records = [json.loads(line) for line in output.read_text().splitlines()]
        assert records[0]['profiling_enabled'] is False
        assert records[0]['environment']['matmul_tile_env'] == str(tile)
        assert records[-1]['type'] == 'verification' and records[-1]['passed'] is True
        run = next(row for row in records if row['type'] == 'run')
        assert run['profile'] == []
        print(json.dumps({'pair': pair, 'tile': tile, 'prefill_ms': run['report']['engine']['prefill_ms'],
                          'summary': run['report']['summary']['ttft_ms']}), flush=True)
