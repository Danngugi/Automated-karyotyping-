import argparse
from pathlib import Path
from karyo_ai.datasets import create_manifest

p=argparse.ArgumentParser(description='Create a local dataset manifest; it does not download datasets.')
p.add_argument('dataset_folder'); p.add_argument('--output',default='dataset_manifest.json'); a=p.parse_args()
m=create_manifest(a.dataset_folder,a.output); print(f"Wrote {a.output}: {m['image_count']} images")
