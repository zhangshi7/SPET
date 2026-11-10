"""Read a benchmark dataset or extract an application dataset."""
import argparse
from pathlib import Path
import pickle
import zipfile
from config import ROOT

def load_benchmark(function, drift, data_seed):
    name = f'data_seed_{data_seed}/F{function}_D{drift}/dataset.pkl'
    with zipfile.ZipFile(ROOT / 'data/benchmark.zip') as archive:
        return pickle.loads(archive.read(name))

def prepare_case(case):
    target = (ROOT / 'data' / case).resolve()
    with zipfile.ZipFile(ROOT / 'data' / (case + '.zip')) as archive:
        for name in archive.namelist():
            if not (target / name).resolve().is_relative_to(target):
                raise ValueError('Invalid dataset member')
        archive.extractall(target)
    return target

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', choices=['task_allocation', 'database'])
    print(prepare_case(parser.parse_args().case))
