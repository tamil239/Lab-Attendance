# 🎓 Lab Attendance System

> YOLOv8 face detection + FaceNet recognition + ID card detection, with a real-time web dashboard.

---

## ✨ Features

| Feature | Technology |
|---|---|
| **Face Detection** | YOLOv8-nano |
| **Face Recognition** | FaceNet (InceptionResnetV1) |
| **ID Card Detection** | YOLOv8 (fine-tunable) |
| **Database** | SQLite |
| **Web Dashboard** | Flask + HTML/CSS/JS + Chart.js |
| **CSV Export** | Built-in |

---

## 📁 Project Structure

```
lab_attendance/
├── datasets/faces/{train,val,test}/    # Face images per person
├── datasets/id_cards/{images,labels}/  # YOLO-format ID card data
├── models/                             # Trained weights
├── src/
│   ├── database.py            # SQLite CRUD
│   ├── utils.py               # Helper functions
│   ├── data_preparation.py    # Dataset splitting
│   ├── train_id_card.py       # YOLOv8 ID-card training
│   ├── train_face.py          # FaceNet fine-tuning
│   ├── registration.py        # Webcam student registration
│   └── attendance_system.py   # Real-time attendance
├── web/
│   ├── app.py                 # Flask API
│   ├── templates/index.html   # Dashboard UI
│   └── static/{css,js}/       # Styles & scripts
├── config.yaml
├── requirements.txt
└── run.py                     # CLI entry point
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
cd lab_attendance
pip install -r requirements.txt
```

### 2. Initialise the database

```bash
python run.py initdb
```

### 3. Register students

```bash
python run.py register
```

- Enter Student ID, Name, Email
- Capture 10 face samples via webcam (press **SPACE**)

### 4. Start attendance system

```bash
python run.py attendance
```

- **q** — quit
- **s** — show statistics
- Green box = recognised, Red = unknown, Orange = ID card

### 5. Launch web dashboard

```bash
python run.py dashboard
```

Open **http://localhost:5000** in any browser (desktop / mobile / tablet).

---

## 🏋️ Training (Optional)

### Train ID Card Detector

Place your dataset in `datasets/id_cards/images/{train,val}` with YOLO-format labels, then:

```bash
python run.py train
# Choose option 1
```

### Train Face Recognizer

Organise face images as `datasets/faces/train/<person_name>/`, then:

```bash
python run.py train
# Choose option 2
```

---

## ⚙️ Configuration

Edit `config.yaml` to change:

- Model paths
- Recognition threshold (default `0.65`)
- Camera resolution & FPS
- Dashboard port (default `5000`)

---

## 📊 Web Dashboard Pages

| Page | Description |
|---|---|
| **Dashboard** | Live stat cards + today's attendance table |
| **Records** | Date-range search + CSV export |
| **Students** | Registered student list |
| **Analytics** | Daily trend chart + hourly distribution |

---

## 🧪 Testing

```bash
# Verify database
python run.py initdb

# Quick syntax check
python -m py_compile src/database.py
python -m py_compile web/app.py
python -m py_compile run.py

# Test with webcam
python run.py register      # register a face
python run.py attendance     # verify recognition
```

---

## 📝 License

MIT
