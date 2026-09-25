"""Training hand-off scaffold (legacy pointer).

The real Phase 1 Mask R-CNN trainer now lives in train_maskrcnn.py, e.g.:

    python3 train_maskrcnn.py --dataset data/human_giemsa --config train_config.json

This file is kept only so `python3 train_model.py --config train_config.json`
still validates the config schema without requiring torch to be installed.
"""
import json
from pathlib import Path
import argparse
p=argparse.ArgumentParser(); p.add_argument('--config',default='train_config.json'); a=p.parse_args()
config=json.loads(Path(a.config).read_text(encoding='utf-8'))
assert config['species'] in {'human','mouse'}
print('Configuration valid. See train_maskrcnn.py for the actual Phase 1 trainer.')
