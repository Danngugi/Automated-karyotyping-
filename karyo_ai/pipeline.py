from pathlib import Path
import numpy as np
from PIL import ImageDraw
from .domain import AnalysisResult
from .preprocessing import detect_modality, image_qc, load_image, prepare
from .segmentation import segment
from .features import add_features
from .classification import apply_annotations, classify_heuristic
from .assembly import count_flags, overlap_quality_gate, render_karyogram
from .reporting import write_json, write_report

def analyze_image(source: str, out_dir: str, species="human", modality="auto", annotations=None, reviewer="", approved=False):
    out=Path(out_dir); out.mkdir(parents=True,exist_ok=True)
    image=load_image(source); selected=detect_modality(image) if modality == "auto" else modality
    gray,mask=prepare(image,selected); objects=segment(mask); add_features(objects,np.asarray(gray))
    apply_annotations(objects, annotations or {})
    classify_heuristic(objects,species)
    gate_verdict, gate_detail = overlap_quality_gate(objects, species)
    flags = [f"QUALITY GATE — {gate_verdict.upper()}: {gate_detail}"] + count_flags(objects,species)
    result=AnalysisResult(str(source), species, selected, objects, image_qc(image), flags, reviewer, approved)
    annotated=image.copy(); draw=ImageDraw.Draw(annotated)
    for o in objects:
        draw.rectangle(o.bbox,outline="red" if o.touching else "lime",width=2); draw.text((o.bbox[0],max(0,o.bbox[1]-12)),f"{o.object_id}:{o.label}",fill="red")
    stem=Path(source).stem
    annotated.save(out/f"{stem}_annotated.png"); render_karyogram(image,objects,species,str(out/f"{stem}_draft_karyogram.png"))
    write_json(result,str(out/f"{stem}_result.json")); write_report(result,str(out/f"{stem}_report.md"))
    return result
