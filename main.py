# face_api_with_comments.py
# A FastAPI-based facial embedding and matching system using InsightFace and FAISS

from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse
import numpy as np
import shutil
import cv2
import json
from pathlib import Path
from insightface.app import FaceAnalysis
import faiss
import logging
import argparse
import os
from io import BytesIO
import base64

# === Logging Setup ===
debug_mode = os.getenv("DEBUG", "0") == "1"
log_level = logging.DEBUG if debug_mode else logging.INFO
logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("face-api")

# === Path Definitions ===
DB_PATH = Path("./face_db/face_db.json")
TMP_PATH = Path("./tmp/input.jpg")
UPLOAD_DIR = Path("./upload_dir")
ORIGINAL_DIR = Path("./face_db/original_faces")

# Ensure required directories exist
for path in [DB_PATH.parent, TMP_PATH.parent, UPLOAD_DIR, ORIGINAL_DIR]:
    path.mkdir(parents=True, exist_ok=True)

# === Load Face Detection & Embedding Model ===
face_app = FaceAnalysis(name="antelopev2")
face_app.prepare(ctx_id=0)  # Use GPU context if available

# === FastAPI Initialization ===
app = FastAPI()

# === Serve Static Files for Target Images ===
from fastapi.staticfiles import StaticFiles

app.mount(
    "/target_db/original_targets",  # This becomes the public URL path
    StaticFiles(directory="target_db/original_targets"),  # This is the actual local directory
    name="target_images"
)

# === Face DB Initialization ===
if DB_PATH.exists():
    with open(DB_PATH) as f:
        face_db = json.load(f)
else:
    face_db = []

# === FAISS Index Builder for Face DB ===
def build_faiss():
    if not face_db:
        return None
    xb = np.array([entry["embedding"] for entry in face_db], dtype=np.float32)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(512)
    index.add(xb)
    return index

faiss_index = build_faiss()

# Save DB to disk
def save_db():
    with open(DB_PATH, "w") as f:
        json.dump(face_db, f, indent=2)

# === Target Pattern Matching DB Setup ===
TARGET_DB_PATH = Path("./target_db/target_db.json")
TARGET_ORIG_DIR = Path("./target_db/original_targets")
TARGET_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
TARGET_ORIG_DIR.mkdir(parents=True, exist_ok=True)

# Load Target DB
if TARGET_DB_PATH.exists():
    with open(TARGET_DB_PATH) as f:
        target_db = json.load(f)
else:
    target_db = []

# Build FAISS index for Target DB
def build_target_faiss():
    if not target_db:
        return None
    xb = np.array([entry["embedding"] for entry in target_db], dtype=np.float32)
    xb /= np.linalg.norm(xb, axis=1, keepdims=True)
    index = faiss.IndexFlatIP(512)
    index.add(xb)
    return index

def save_target_db():
    with open(TARGET_DB_PATH, "w") as f:
        json.dump(target_db, f, indent=2)

target_faiss_index = build_target_faiss()

# === Route to Embed a Target Face ===
@app.post("/embed-target")
async def embed_target(file: UploadFile, label: str = Form(...), notes: str = Form("")):
    with open(TMP_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    img = cv2.imread(str(TMP_PATH))
    faces = face_app.get(img)

    if not faces:
        return {"error": "No face detected"}

    added = []
    for i, face in enumerate(faces):
        try:
            emb = face.embedding / np.linalg.norm(face.embedding)
            item_name = f"{label}_{i+1}" if len(faces) > 1 else label
            save_path = TARGET_ORIG_DIR / f"{item_name}.jpg"
            cv2.imwrite(str(save_path), img)
            target_db.append({
                "name": item_name,
                "info": {"notes": notes, "image_path": str(save_path)},
                "embedding": emb.tolist()
            })
            added.append(item_name)
        except Exception as e:
            logger.error(f"Failed to embed target {i+1}: {e}")

    save_target_db()
    global target_faiss_index
    target_faiss_index = build_target_faiss()

    return {"targets_added": added}

# === Route to Match a Face Against Target DB ===
@app.post("/match-target")
async def match_target(file: UploadFile):
    with open(TMP_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    img = cv2.imread(str(TMP_PATH))
    faces = face_app.get(img)

    if not faces:
        return {"error": "No face detected"}

    global target_faiss_index
    if not target_faiss_index:
        return {"error": "Target FAISS index is empty"}

    query = faces[0].embedding.reshape(1, -1).astype(np.float32)
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    D, I = target_faiss_index.search(query, 5)

    results = []
    for score, idx in zip(D[0], I[0]):
        if score >= 0.45:
            entry = target_db[idx]
            image_path = entry["info"].get("image_path")
            try:
                with open(image_path, "rb") as img_file:
                    b64_image = base64.b64encode(img_file.read()).decode("utf-8")
            except Exception as e:
                b64_image = None
            results.append({
                "name": entry["name"],
                "score": float(score),
                "notes": entry["info"].get("notes"),
                "image_path": image_path,
                "image_base64": b64_image
            })

    return {"matches": results, "threshold": 0.45}

# === Route to Embed a Face to Main DB ===
@app.post("/embed-face")
async def embed_face(file: UploadFile, name: str = Form(...), notes: str = Form("")):
    with open(TMP_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    img = cv2.imread(str(TMP_PATH))
    faces = face_app.get(img)

    added = []
    response_data = []
    for i, face in enumerate(faces):
        try:
            emb = face.embedding / np.linalg.norm(face.embedding)
            face_name = f"{name}_{i+1}" if len(faces) > 1 else name
            gender = face.sex if hasattr(face, 'sex') else "Unknown"
            age = face.age if hasattr(face, 'age') else "Unknown"
            description = notes if notes else f"Guessed Gender: {gender}, Age: {age}, Mood: neutral"
            save_path = ORIGINAL_DIR / f"{face_name}.jpg"
            cv2.imwrite(str(save_path), img)
            face_db.append({
                "name": face_name,
                "info": {"notes": description, "image_path": str(save_path)},
                "embedding": emb.tolist()
            })
            added.append(face_name)
            response_data.append({"name": face_name, "notes": description, "image_path": str(save_path)})
        except Exception as e:
            logger.error(f"Failed to embed face {i+1}: {e}")

    save_db()
    global faiss_index
    faiss_index = build_faiss()

    return {"faces_added": added, "data": response_data}

# === Route to Match Face Against Main DB ===
@app.post("/match-face")
async def match_face(file: UploadFile):
    with open(TMP_PATH, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    img = cv2.imread(str(TMP_PATH))
    faces = face_app.get(img)

    logger.debug(f"Received image: {file.filename}")
    logger.debug(f"Detected {len(faces)} face(s)")

    if not faces:
        return {"error": "No face detected"}

    global faiss_index
    if not faiss_index:
        return {"error": "FAISS index is empty"}

    query = faces[0].embedding.reshape(1, -1).astype(np.float32)
    query /= np.linalg.norm(query, axis=1, keepdims=True)
    D, I = faiss_index.search(query, 5)

    results = []
    for score, idx in zip(D[0], I[0]):
        if score >= 0.45:
            entry = face_db[idx]
            image_path = entry["info"].get("image_path")
            try:
                with open(image_path, "rb") as img_file:
                    b64_image = base64.b64encode(img_file.read()).decode("utf-8")
            except Exception as e:
                b64_image = None
            results.append({
                "name": entry["name"],
                "score": float(score),
                "notes": entry["info"].get("notes"),
                "image_path": image_path,
                "image_base64": b64_image
            })

    return {"matches": results, "threshold": 0.45}

# === Route to Bulk Embed Faces from Directory ===
@app.post("/embed-directory")
def embed_directory():
    files = list(UPLOAD_DIR.glob("*.jpg")) + list(UPLOAD_DIR.glob("*.jpeg")) + list(UPLOAD_DIR.glob("*.png"))
    count = 0
    for path in files:
        img = cv2.imread(str(path))
        faces = face_app.get(img)
        for i, face in enumerate(faces):
            try:
                emb = face.embedding / np.linalg.norm(face.embedding)
                gender = face.sex if hasattr(face, 'sex') else "Unknown"
                age = face.age if hasattr(face, 'age') else "Unknown"
                name = f"{path.stem}_{i+1}"
                save_path = ORIGINAL_DIR / f"{name}.jpg"
                cv2.imwrite(str(save_path), img)
                face_db.append({
                    "name": name,
                    "info": {"notes": f"Guessed Gender: {gender}, Age: {age}, Mood: neutral", "image_path": str(save_path)},
                    "embedding": emb.tolist()
                })
                count += 1
            except Exception as e:
                logger.error(f"Failed to embed from {path.name}: {e}")

    save_db()
    global faiss_index
    faiss_index = build_faiss()
    return {"faces_added": count}

# === Routes to List or Reset Face DB ===
@app.get("/list-faces")
def list_faces():
    return {"faces": [entry["name"] for entry in face_db]}

@app.get("/count-faces")
def count_faces():
    return {"total_faces": len(face_db)}

@app.get("/face-stats")
def face_stats():
    genders = {"male": 0, "female": 0, "unknown": 0}
    ages = []
    for entry in face_db:
        notes = entry.get("info", {}).get("notes", "")
        if "Gender: 1" in notes:
            genders["male"] += 1
        elif "Gender: 0" in notes:
            genders["female"] += 1
        else:
            genders["unknown"] += 1
        if "Age:" in notes:
            try:
                age = float(notes.split("Age:")[1].split(",")[0].strip())
                ages.append(age)
            except:
                continue
    return {
        "total_faces": len(face_db),
        "gender_distribution": genders,
        "age_stats": {
            "min": min(ages) if ages else None,
            "max": max(ages) if ages else None,
            "average": round(sum(ages)/len(ages), 2) if ages else None
        }
    }

@app.post("/reset-db")
def reset_db():
    global face_db, faiss_index
    face_db = []
    faiss_index = None
    save_db()
    return {"status": "reset", "total_faces": 0}
        