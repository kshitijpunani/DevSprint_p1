"""
QueueSense - Intelligent Queue Monitoring & Waiting-Time Estimation
Streamlit Web Application (Phase 1 Entry Point)
"""

import os
# Safeguard for Windows OpenBLAS thread allocation
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import tempfile
import time
import cv2
import streamlit as st

from core.detector import PersonDetector

# Page configuration
st.set_page_config(
    page_title="QueueSense | Queue Monitoring",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom header styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0px;
        background: linear-gradient(120deg, #ff8c00, #ff0080);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .sub-header {
        font-size: 1rem;
        color: #8892b0;
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading YOLOv8 person detector...")
def get_detector(conf_threshold: float = 0.35) -> PersonDetector:
    """Load model once and cache across app reruns."""
    return PersonDetector(model_name="yolov8n.pt", conf_threshold=conf_threshold)


def main():
    st.markdown('<div class="main-header">⏱️ QueueSense</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">AI-Powered Queue Monitoring & Waiting-Time Estimation · DEVSPRINT \'26</div>',
        unsafe_allow_html=True,
    )

    # Sidebar Controls
    st.sidebar.header("Configuration")

    # 1. Video Source
    video_source_type = st.sidebar.radio(
        "Select Video Input:",
        ("Upload Video", "Use Sample Video"),
        index=0,
    )

    video_path = None
    temp_file_path = None

    if video_source_type == "Upload Video":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Queue Video", type=["mp4", "avi", "mov", "mkv"]
        )
        if uploaded_file is not None:
            # Write to temporary file so OpenCV can read it via file path
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
            video_path = temp_file_path
    else:
        sample_path = os.path.join("data", "sample_videos", "sample_queue.mp4")
        if os.path.exists(sample_path):
            video_path = sample_path
            st.sidebar.success(f"Using: {sample_path}")
        else:
            st.sidebar.warning(f"Sample not found at `{sample_path}`. Please upload a video.")

    # 2. Model Settings
    conf_threshold = st.sidebar.slider(
        "Detection Confidence",
        min_value=0.15,
        max_value=0.85,
        value=0.35,
        step=0.05,
        help="Higher values reduce false positives; lower values detect partially occluded people.",
    )

    frame_skip = st.sidebar.slider(
        "Frame Sampling Interval",
        min_value=1,
        max_value=10,
        value=3,
        help="Process every Nth frame to ensure smooth real-time performance on CPU.",
    )

    start_processing = st.sidebar.button("▶ Start Processing Video", type="primary")

    # Main Dashboard Area
    col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
    metric_count = col_stat1.empty()
    metric_wait = col_stat2.empty()
    metric_congestion = col_stat3.empty()
    metric_fps = col_stat4.empty()

    # Initial metric placeholders
    metric_count.metric("Detected People", "--")
    metric_wait.metric("Estimated Wait", "Phase 2")
    metric_congestion.metric("Queue Status", "Ready")
    metric_fps.metric("Processing FPS", "--")

    video_placeholder = st.empty()
    status_box = st.empty()

    if not video_path:
        video_placeholder.info("👈 Please select or upload a video from the sidebar to begin.")
        return

    # Load detector
    detector = get_detector(conf_threshold=conf_threshold)
    detector.set_confidence(conf_threshold)

    if start_processing:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error: Could not open the specified video file.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        status_box.info(f"Processing video: {total_frames} frames @ {orig_fps:.1f} FPS")

        progress_bar = st.progress(0.0)
        stop_btn = st.sidebar.button("⏹ Stop")

        frame_idx = 0
        last_time = time.time()

        try:
            while cap.isOpened():
                if stop_btn:
                    status_box.warning("Processing stopped by user.")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # Skip frames for CPU inference throughput
                if frame_idx % frame_skip != 0:
                    continue

                # Resize frame to standard display width if large
                h, w = frame.shape[:2]
                if w > 800:
                    scale = 800 / w
                    frame = cv2.resize(frame, (800, int(h * scale)))

                # Run person detector
                t0 = time.time()
                detections = detector.detect(frame)
                infer_time = time.time() - t0

                # Render annotations
                annotated_frame = detector.draw_detections(frame, detections)
                annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Compute approximate processing FPS
                now = time.time()
                fps = 1.0 / max(now - last_time, 1e-4)
                last_time = now

                # Update live dashboard components
                video_placeholder.image(
                    annotated_rgb,
                    channels="RGB",
                    caption=f"Frame {frame_idx}/{total_frames} · Inferred in {infer_time*1000:.1f}ms",
                    use_container_width=True,
                )

                metric_count.metric("Detected People", len(detections))
                metric_congestion.metric("Queue Status", "Active" if len(detections) > 0 else "Clear")
                metric_fps.metric("Processing FPS", f"{fps:.1f}")

                if total_frames > 0:
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))

            status_box.success("Video processing completed.")

        finally:
            cap.release()
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception:
                    pass


if __name__ == "__main__":
    main()
