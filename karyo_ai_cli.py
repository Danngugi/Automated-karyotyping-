import argparse, csv
from pathlib import Path
from karyo_ai.pipeline import analyze_image
from karyo_ai.storage import log_run

def annotations(path):
    if not path: return {}
    with open(path,newline='',encoding='utf-8-sig') as f:
        return {int(r['object_id']):r for r in csv.DictReader(f) if r.get('object_id','').isdigit()}

parser=argparse.ArgumentParser(description='Research-only draft karyotype image workflow.')
parser.add_argument('image'); parser.add_argument('--out-dir',default='runs'); parser.add_argument('--species',choices=['human','mouse'],default='human')
parser.add_argument('--modality',choices=['auto','giemsa','fluorescent'],default='auto'); parser.add_argument('--annotations')
parser.add_argument('--reviewer',default=''); parser.add_argument('--approve',action='store_true'); parser.add_argument('--database')
args=parser.parse_args(); result=analyze_image(args.image,args.out_dir,args.species,args.modality,annotations(args.annotations),args.reviewer,args.approve)
if args.database: log_run(args.database,args.image,args.species,len(result.objects),args.approve)
print(f'Created draft analysis for {len(result.objects)} objects in {Path(args.out_dir).resolve()}')
