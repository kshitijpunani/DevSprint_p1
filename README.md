# ⏱️ QueueSense

**Intelligent Queue Monitoring & Waiting-Time Estimation**  
*AI-Powered Queue Monitoring & Waiting-Time Estimation · DEVSPRINT '26*

QueueSense is a computer vision and deep learning solution that monitors physical queues in real time. Using state-of-the-art YOLOv8 object detection, QueueSense tracks customer counts, assesses queue congestion, and estimates waiting times to optimize service throughput.

---

## 🚀 Features (Phase 1)

- **Real-Time Person Detection**: Powered by lightweight YOLOv8 (`yolov8n.pt`) focused on COCO person detection (`cls=0`).
- **Interactive Streamlit Web Dashboard**: Live video playback, annotated bounding boxes with confidence scores, ground foot tracking points, and performance diagnostics.
- **Flexible Video Ingestion**:
  - Upload custom queue footage (`.mp4`, `.avi`, `.mov`, `.mkv`).
  - Preloaded sample video support.
- **Dynamic Tuning**:
  - Real-time confidence threshold slider (0.15 - 0.85).
  - Configurable frame sampling / skip interval for high-FPS CPU inference.
- **Live Metrics**:
  - Detected People count.
  - Queue Status indicator (Active / Clear).
  - Inference & Processing FPS monitor.

---

## 📁 Project Structure

```text
DevSprint_p1/
├── app.py                     # Streamlit web application entry point
├── requirements.txt           # Python package dependencies
├── README.md                  # Project documentation
├── core/
│   ├── __init__.py            # Core package exports
│   └── detector.py            # YOLOv8 PersonDetector class & visualization
└── data/
    └── sample_videos/         # Sample queue footage
        └── sample_queue.mp4
```

---

## 🛠️ Installation & Setup

1. **Clone Repository & Navigate to Workspace**:
   ```bash
   git clone <repo-url>
   cd DevSprint_p1
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run Streamlit Application**:
   ```bash
   streamlit run app.py
   ```

4. **Access Dashboard**:
   Open your browser and navigate to `http://localhost:8501`.
