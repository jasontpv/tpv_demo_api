# app.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from tinydb import TinyDB, Query
import os
import uuid
import shutil
import time
import subprocess
import whisper
import cv2
from ultralytics import YOLO
from glob import glob

app = FastAPI()
db = TinyDB("analysis_db.json")
AUDIT_LOG_PATH = "audit.log"
UPLOAD_DIR = "uploads"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---- Audio Functions ----

def extract_audio(input_path: str, output_path: str) -> None:
    cmd = ["ffmpeg", "-y", "-i", input_path, "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", output_path]
    subprocess.run(cmd, check=True)

def split_audio(input_wav: str, chunk_seconds: int, work_dir: str = "chunks_audio"):
    os.makedirs(work_dir, exist_ok=True)
    template = os.path.join(work_dir, "chunk%03d.wav")
    cmd = ["ffmpeg", "-y", "-i", input_wav, "-f", "segment", "-segment_time", str(chunk_seconds), "-c", "copy", template]
    subprocess.run(cmd, check=True)
    return sorted(glob(os.path.join(work_dir, "chunk*.wav")))

def transcribe_audio(chunks, model_size: str, language: str, chunk_length: int):
    model = whisper.load_model(model_size)
    speech_events = []
    for idx, wav in enumerate(chunks, start=1):
        result = model.transcribe(wav, language=language, word_timestamps=True)
        offset = (idx - 1) * chunk_length
        for seg in result.get("segments", []):
            for w in seg.get("words", []):
                start = round(w["start"] + offset, 3)
                end = round(w["end"] + offset, 3)
                speech_events.append({"id": f"speech_{start}", "type": "speech", "text": w["word"].strip(), "start": start, "end": end})
    return speech_events

# ---- Video Functions ----

def process_video(input_path: str, model_size: str, conf_thresh: float, skip: int):
    model = YOLO(f'yolov8{model_size}.pt')
    model.conf = conf_thresh
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    object_events = []
    skip_count = 0
    frame_idx = 0
    counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        timestamp = round(frame_idx / fps if fps else 0, 3)
        frame_idx += 1
        if skip > 1 and (frame_idx % skip) != 0:
            skip_count += 1
            continue
        results = model(frame)[0]
        for box in results.boxes:
            label = model.names[int(box.cls.cpu())]
            counter += 1
            object_events.append({"id": f"object_{timestamp}_{counter}", "type": "object", "label": label, "time": timestamp})
    cap.release()
    stats = {"skip_frames": skip_count, "total_frames": total_frames, "fps": fps, "object_count": len(object_events), "unique_objects": len({e['label'] for e in object_events})}
    return object_events, stats

# ---- Utility ----

def log_audit(message: str):
    with open(AUDIT_LOG_PATH, "a") as f:
        f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - {message}\n")

def debug(msg: str):
    print(f"[DEBUG] {msg}")
    log_audit(msg)

# ---- API ----

@app.post("/analyze")
async def analyze_file(
    file: UploadFile = File(...),
    mode: str = Form("both"),
    chunk_length: int = Form(30),
    whisper_model: str = Form("base"),
    language: str = Form(None),
    yolo_model: str = Form("n"),
    conf: float = Form(0.25),
    skip: int = Form(1)
):
    uid = str(uuid.uuid4())
    filename = f"{uid}_{file.filename}"
    save_path = os.path.join(UPLOAD_DIR, filename)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    debug(f"Uploaded: {filename}")

    events = []
    summary = {"case_id": uid, "source_file": filename, "mode": mode, "created_at": time.strftime('%Y-%m-%d %H:%M:%S')}
    start_time = time.time()

    if mode in ("audio", "both"):
        wav_path = f"{uid}.wav"
        extract_audio(save_path, wav_path)
        chunks = split_audio(wav_path, chunk_length)
        speech = transcribe_audio(chunks, whisper_model, language, chunk_length)
        events.extend(speech)
        summary.update({"audio_chunks": len(chunks), "speech_word_count": len(speech)})

    if mode in ("video", "both"):
        objects, stats = process_video(save_path, yolo_model, conf, skip)
        events.extend(objects)
        summary.update(stats)

    end_time = time.time()
    summary["processing_time_sec"] = round(end_time - start_time, 3)

    record = {"id": uid, "summary": summary, "events": events}
    db.insert(record)
    debug(f"Analysis complete for {filename} → {uid}")

    return JSONResponse({"message": "Analysis complete", "case_id": uid, "summary": summary})

@app.get("/cases")
def list_cases():
    return db.all()

@app.get("/case/{case_id}")
def get_case(case_id: str):
    Case = Query()
    return db.search(Case.id == case_id)

@app.get("/search")
def search_events(q: str):
    q = q.lower()
    result = []
    for item in db.all():
        matched = [e for e in item["events"] if q in str(e.get("text", "")).lower() or q in str(e.get("label", "")).lower()]
        if matched:
            result.append({"id": item["id"], "matches": matched})
    return result
