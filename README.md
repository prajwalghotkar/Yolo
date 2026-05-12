# 🔭 Vision AI — YOLOv3 Object Detection
### Built by Prajwal Ghotkar

---

## Files Inside This Project
```
vision_ai_final/
├── app.py              ← Flask backend
├── model.h5            ← YOLOv3 weights (238MB)
├── yolo_algorithm.py   ← YOLOv3 architecture
├── requirements.txt
├── Procfile            ← For Railway deployment
├── runtime.txt
├── static/
│   ├── zebra.jpg       ← Test image
│   ├── uploads/
│   └── results/
└── templates/
    └── index.html      ← World Map UI
```

---

## LOCAL SETUP

### Step 1
Open Command Prompt in this folder

### Step 2
```
pip install -r requirements.txt
```

### Step 3
```
python app.py
```

### Step 4
Open browser: http://localhost:5000

---

## DEPLOY ON RAILWAY

1. Create GitHub repo
2. Push this entire folder to GitHub
3. Go to railway.app → Sign up
4. New Project → Deploy from GitHub
5. Select your repo
6. Done! Get public URL in 2 minutes

---

## END TO END DESCRIPTION

### Frontend
- Animated world map showing 26 cities
- Live detection threads between countries
- Confidence threshold slider
- Real-time activity log
- Detection results with confidence bars
- Original vs Detected image toggle
- Animated stats counters

### Backend
- Flask REST API
- YOLOv3 inference on uploaded image
- 3-scale detection (13x13, 26x26, 52x52)
- NMS filtering
- Bounding box drawing with OpenCV
- Result served back to browser

### Model
- Architecture: YOLOv3 (Darknet-53 backbone)
- Dataset: COCO (80 classes)
- Input: 416x416 pixels
- Output: 3 detection scales
- Weights: 237MB pretrained

---

Built by Prajwal Ghotkar | 2024
