# ⏱️ QueueSense

**Intelligent Queue Monitoring, MOT Tracking & Waiting-Time Estimation Engine**  
*AI-Powered Queue Monitoring & Waiting-Time Estimation · DEVSPRINT '26*

QueueSense is a comprehensive computer vision and queueing theory solution engineered to monitor physical queues, track waiting individuals in real time, estimate waiting times using Little's Law, and trigger proactive bottleneck alerts.

---

## 🌟 Key Architecture & Capabilities

### 1. 🔍 High-Speed Person Detection (`core/detector.py`)
- Powered by Ultralytics **YOLOv8** (`yolov8n.pt`) fine-filtered strictly for COCO class 0 (`Person`).
- Computes bounding boxes, centroid coordinates, and ground contact points (`foot_point`) for accurate ground-plane zone association.
- Optimized for real-time CPU and GPU inference.

### 2. 📍 Region of Interest (ROI) & Queue Zones (`core/zone.py`)
- Configurable **Queue Monitoring Zone** (polygon or rectangular ROI).
- **Passerby Filtering**: Individuals walking outside the queue zone are categorized as passersby and excluded from queue length counts.
- **Service Counter Zone**: Front region where active customers are marked as `BEING_SERVED` and counted as `SERVED` upon completion.
- Layout presets: *Center Queue*, *Left Corridor*, *Right Corridor*, *Full Frame*, and *Custom Region Sliders*.

### 3. 🎯 Multi-Object Tracking (`core/tracker.py`)
- Persistent **Track IDs** assigned to each individual across frames.
- Robust Hungarian algorithm association combining Intersection-over-Union (IoU) and Euclidean centroid distance.
- Trajectory trail buffer rendering historical motion paths.
- Resilient against brief occlusions.

### 4. ⏳ Predictive Waiting-Time Estimation (`core/estimator.py`)
- **Individual Dwell Time**: Tracks exact seconds each person has been waiting in the queue.
- **Little's Law Estimation**: $W = \frac{L}{\mu}$ (estimated wait for next arrival based on line length $L$ and effective service rate $\mu$).
- **Statistical Aggregations**: Real-time Average Dwell Time, Max Waiting Time, and cumulative Served Count.
- **Congestion Classifier**: Real-time tiering into *Clear*, *Smooth*, *Moderate*, *Congested*, and *Critical Bottleneck*.
- **Bottleneck Alerts**: Visual alert banners triggered when queue size or wait time crosses management thresholds.

### 5. 🎨 Glassmorphic HUD & Overlays (`core/visualizer.py`)
- Dynamic zone illumination changing color with queue congestion (Green $\rightarrow$ Amber $\rightarrow$ Red).
- Floating pill badges above each queued person displaying `ID #X · Wait: 1m 24s`.
- Fading motion trajectory trails behind pedestrians.
- Top operational HUD banner.

### 6. 📊 Interactive Multi-Tab Streamlit Dashboard (`app.py`)
- **Tab 1: 📹 Live Queue Monitor**: Live video feed, real-time KPI metrics, and bottleneck notifications.
- **Tab 2: 📈 Real-time Analytics & Charts**: Dynamic interactive line charts for queue length, wait times, and throughput.
- **Tab 3: 📋 Customer Audit & Bottleneck Logs**: Searchable tabular log of all visitors and **CSV Export** button.

---

## 📁 Project Directory Structure

```text
DevSprint_p1/
├── app.py                     # Streamlit multi-tab dashboard entry point
├── requirements.txt           # Python dependencies
├── README.md                  # Comprehensive documentation
├── run.ps1                    # One-click launcher for PowerShell
├── run.bat                    # One-click launcher for Command Prompt
├── core/
│   ├── __init__.py            # Core package exports
│   ├── detector.py            # YOLOv8 PersonDetector
│   ├── tracker.py             # Multi-Object Tracker (MOT) & Track state
│   ├── zone.py                # QueueZone ROI & Service Counter geometry
│   ├── estimator.py           # Waiting-Time Estimator & Little's Law
│   └── visualizer.py          # HUD overlays, trails, & badges
└── data/
    └── sample_videos/
        └── sample_queue.mp4   # Included sample pedestrian video
```

---

## 🚀 How to Run

### Option A: Using the Launcher Script (Recommended)
In PowerShell:
```powershell
.\run.ps1
```
In Command Prompt:
```cmd
run.bat
```

### Option B: Direct Streamlit Run
```powershell
streamlit run app.py
```

The web dashboard will launch at `http://localhost:8501`.
