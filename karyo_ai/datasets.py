import json
from pathlib import Path

PUBLIC_REFERENCES={
 "AutoKary2022":"https://github.com/wangjuncongyu/chromosome-instance-segmentation-dataset",
 "UT Austin ADIR M-FISH":"https://live.ece.utexas.edu/research/mfish.html",
 "Kaggle chromosome karyotype images":"https://www.kaggle.com/datasets/aliabedimadiseh/chromosome-image-dataset-karyotype",
 "Kaggle overlapping chromosomes":"https://www.kaggle.com/datasets/jeanpat/overlapping-chromosomes"}

def create_manifest(root: str, output: str) -> dict:
    base=Path(root); images=[p for p in base.rglob('*') if p.suffix.lower() in {'.png','.jpg','.jpeg','.tif','.tiff'}]
    coco=[str(p.relative_to(base)) for p in base.rglob('*.json') if 'coco' in p.name.lower() or 'annotation' in p.name.lower()]
    manifest={"root":str(base.resolve()),"image_count":len(images),"images":[str(p.relative_to(base)) for p in images],"annotation_json":coco,"public_references":PUBLIC_REFERENCES}
    Path(output).write_text(json.dumps(manifest,indent=2),encoding='utf-8'); return manifest
