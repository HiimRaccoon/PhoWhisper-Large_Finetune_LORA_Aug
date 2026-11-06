"""Explicit network preparation. Runtime CLI commands always load local-only."""
import argparse
from pathlib import Path

from asr.assets import prepare_snapshot
from asr.config import load_config

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config',type=Path,default=Path('configs/official.yaml'))
parser.add_argument('--snapshot',type=Path)
args=parser.parse_args()
config=load_config(args.config)
if args.snapshot: config=config.with_paths(snapshot=args.snapshot)
result=prepare_snapshot(config)
print(f'Prepared {result.repository}@{result.revision}: {result.path}; identity={result.identity_hash}')
