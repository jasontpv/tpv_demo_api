# 🧠 FastAPI Face Recognition Server

This project is a fully functional face recognition API built using **FastAPI**, **InsightFace**, and **FAISS**. It enables you to embed faces from images, organize them into databases (main and target), and perform similarity matching on-the-fly.

Perfect for real-time or batch face search systems, surveillance-style applications, or experimental facial analysis with pre-trained models.

---

## 🗂 Project Structure
ace_api/
├── face_db/
│ ├── face_db.json # Main DB of embedded faces
│ └── original_faces/ # Original images of added faces
├── target_db/
│ ├── target_db.json # Target DB for pattern-matching
│ └── original_targets/ # Original images of target faces
├── upload_dir/ # Bulk embedding input folder
├── tmp/
│ └── input.jpg # Temp file for uploads
├── main.py # 🚀 FastAPI app
├── requirements.txt # Python dependencies

---

## 🚀 Running the Server

### 1. Clone the Repo

```bash
git clone https://github.com/yourname/face-api.git
cd face-api



###2. Create Virtual Environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate



###3. Install Requirements
pip install -r requirements.txt

###4. Run the API
uvicorn main:app --reload

#📌 /embed-face

Description: Adds face(s) to the main DB.

Method: POST
Form Data:

    file: image with face(s)

    name: base name for the person

    notes: optional notes (e.g., context)

Response:

{
  "faces_added": ["John_1"],
  "data": [
    {
      "name": "John_1",
      "notes": "Guessed Gender: male, Age: 28, Mood: neutral",
      "image_path": "./face_db/original_faces/John_1.jpg"
    }
  ]
}

#📌 /match-face

Description: Matches uploaded face with those in the main DB.

Method: POST
Form Data:

    file: image containing a face

Response:

{
  "matches": [
    {
      "name": "John_1",
      "score": 0.93,
      "notes": "Guessed Gender: male, Age: 28, Mood: neutral",
      "image_path": "...",
      "image_base64": "..."
    }
  ],
  "threshold": 0.45
}

#🎯 /embed-target

Description: Adds target face(s) for pattern match scenarios.

Method: POST
Form Data:

    file: image with face(s)

    label: name/label

    notes: optional description

Response:

{
  "targets_added": ["Suspect_1"]
}

#🎯 /match-target

Description: Matches uploaded face against target DB.

Method: POST
Form Data:

    file: image with one face

Response:

{
  "matches": [
    {
      "name": "Suspect_1",
      "score": 0.88,
      "notes": "...",
      "image_path": "...",
      "image_base64": "..."
    }
  ],
  "threshold": 0.45
}

#📦 /embed-directory

Description: Batch embeds all .jpg, .png, .jpeg from upload_dir.

Method: POST
Response:

{
  "faces_added": 23
}

📃 /list-faces

Description: List all names in the main DB.

Method: GET
Response:

{
  "faces": ["John_1", "Alice_1"]
}

#📊 /count-faces

Description: Count of embedded faces.

Method: GET
Response:

{
  "total_faces": 42
}

#📈 /face-stats

Description: Returns basic demographic stats.

Method: GET
Response:

{
  "total_faces": 42,
  "gender_distribution": {
    "male": 30,
    "female": 10,
    "unknown": 2
  },
  "age_stats": {
    "min": 19,
    "max": 67,
    "average": 34.56
  }
}

#🔄 /reset-db

Description: Clears main DB and FAISS index.

Method: POST
Response:

{
  "status": "reset",
  "total_faces": 0
}

#⚙️ Dependencies

    fastapi

    uvicorn

    opencv-python

    insightface

    faiss-cpu (or faiss-gpu)

    numpy

#📌 Notes

    Default FAISS similarity threshold is 0.45.

    Only the first face in match operations is used.

    Uploads are temporarily saved to ./tmp/input.jpg and overwritten on each request.

🧠 License

MIT – use it, remix it, ship it.
