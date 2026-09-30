"""Optional FastAPI wrapper. Install requirements-model.txt, then: uvicorn api_app:app"""
from pathlib import Path
import tempfile
try:
    from fastapi import FastAPI, UploadFile, File, Form
    from fastapi.responses import FileResponse
except ImportError as e:
    raise RuntimeError('Install requirements-model.txt to use the API.') from e
from karyo_ai.pipeline import analyze_image
app=FastAPI(title='KaryoAI research prototype')
@app.post('/analyze')
async def analyze(image: UploadFile=File(...), species: str=Form('human'), modality: str=Form('auto')):
    folder=Path(tempfile.mkdtemp()); source=folder/image.filename; source.write_bytes(await image.read())
    analyze_image(str(source),str(folder/'output'),species,modality)
    return FileResponse(folder/'output'/f'{source.stem}_result.json',media_type='application/json')
