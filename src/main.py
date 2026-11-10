"""Run SPET on one streaming benchmark scenario."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import sys
sys.dont_write_bytecode = True
import argparse
from datetime import datetime
from functools import partial
import hashlib
import json
from pathlib import Path
import config

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--function', type=int, choices=range(1,9), default=config.FUNCTION)
    parser.add_argument('--drift', type=int, choices=range(1,6), default=config.DRIFT)
    parser.add_argument('--data-seed', type=int, choices=[3101]+list(range(4101,4111)), default=config.DATA_SEED)
    parser.add_argument('--windows', type=int, choices=range(3,21), default=config.WINDOWS)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    from SPET import create_spet
    from dataset import load_benchmark
    from benchmark.evaluator import run, atomic_json
    from threadpoolctl import threadpool_limits
    import torch
    payload = load_benchmark(args.function, args.drift, args.data_seed)
    output = args.output or config.ROOT / 'results/runs' / (
        f'F{args.function}_D{args.drift}_data{args.data_seed}_' + datetime.now().strftime('%Y%m%d_%H%M%S'))
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed((config.SEARCH_SEED*1000003+args.function*1009+args.drift*101) % (2**63-1))
    job = {'algorithm': 'SPET', 'seed': config.SEARCH_SEED, 'function': args.function,
           'drift': args.drift, 'window_limit': 0 if args.windows == 20 else args.windows,
           'run_id': output.name, 'data_seed': args.data_seed}
    job['manifest_digest'] = hashlib.sha256(json.dumps(job, sort_keys=True).encode()).hexdigest()
    with threadpool_limits(limits=1):
        result = run(job, output, payload['snapshot'], payload['schedule'],
                     partial(create_spet, alpha=config.ALPHA))
    atomic_json(output / 'task.json', result)
    print('Completed:', output)
    print('Windows:', args.windows, 'Active evaluations:', result['active_fe'])

if __name__ == '__main__':
    main()
