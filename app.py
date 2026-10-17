"""
QueueSense - Intelligent Queue Monitoring & Waiting-Time Estimation
Production Prototype · DEVSPRINT '26
"""

import os
# Safeguard for Windows OpenBLAS thread allocation
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import tempfile
import time
from typing import Optional, Dict, Any, List

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from core.detector import PersonDetector
from core.tracker import QueueTracker, Track
from core.zone import QueueZone
from core.estimator import WaitingTimeEstimator
from core.visualizer import QueueVisualizer


# -----------------------------------------------------------------------------
# Streamlit App Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="QueueSense | Queue Intelligence",
    page_icon="⏱️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Gradient Main Title */
    .brand-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        margin-bottom: 0px;
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E53 50%, #FFA84A 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .brand-subtitle {
        font-size: 0.95rem;
        color: #94a3b8;
        margin-top: 2px;
        margin-bottom: 18px;
    }
    /* Metric Card Styling */
    div[data-testid="stMetric"] {
        background: rgba(30, 41, 59, 0.45);
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 10px;
        padding: 12px 16px;
        backdrop-filter: blur(8px);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        transition: transform 0.15s ease-in-out;
    }
    div[data-testid="stMetric"]:hover {
        border-color: rgba(255, 142, 83, 0.4);
    }
    /* Alert Banner */
    .alert-banner {
        background: rgba(239, 68, 68, 0.18);
        border: 1px solid #ef4444;
        border-radius: 8px;
        color: #fca5a5;
        padding: 10px 16px;
        font-weight: 600;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Initializing YOLOv8 Neural Engine...")
def get_detector(conf_threshold: float = 0.35) -> PersonDetector:
    """Load and cache the YOLOv8 person detector across Streamlit reruns."""
    return PersonDetector(model_name="yolov8n.pt", conf_threshold=conf_threshold)


def main():
    # Header Banner
    st.markdown('<div class="brand-title">⏱️ QueueSense</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-subtitle">'
        'Intelligent Queue Monitoring, MOT Tracking & Waiting-Time Estimation Engine · DEVSPRINT \'26'
        '</div>',
        unsafe_allow_html=True,
    )

    # -------------------------------------------------------------------------
    # Sidebar Configuration Controls
    # -------------------------------------------------------------------------
    st.sidebar.header("🕹️ Control Panel")

    # 1. Video Input Source
    source_choice = st.sidebar.radio(
        "Video Input Source:",
        ("Use Sample Queue Video", "Upload Video File", "Live Webcam Stream"),
        index=0,
    )

    video_path = None
    temp_file_path = None
    is_webcam = False

    if source_choice == "Upload Video File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Queue Video", type=["mp4", "avi", "mov", "mkv"]
        )
        if uploaded_file is not None:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
            video_path = temp_file_path
    elif source_choice == "Use Sample Queue Video":
        sample_path = os.path.join("data", "sample_videos", "sample_queue.mp4")
        if os.path.exists(sample_path):
            video_path = sample_path
            st.sidebar.success(f"Loaded: `{os.path.basename(sample_path)}`")
        else:
            st.sidebar.error("Sample video not found. Please upload a video.")
    else:
        is_webcam = True
        webcam_idx = st.sidebar.number_input("Webcam Index", min_value=0, max_value=4, value=0)
        video_path = webcam_idx

    st.sidebar.markdown("---")

    # 2. Queue Zone (ROI) Configuration
    st.sidebar.subheader("📍 Queue Zone (ROI)")
    zone_preset = st.sidebar.selectbox(
        "Queue Zone Layout:",
        ("Center Queue", "Full Frame", "Left Corridor", "Right Corridor", "Custom Region"),
        index=0,
    )

    custom_coords = None
    if zone_preset == "Custom Region":
        col_c1, col_c2 = st.sidebar.columns(2)
        xmin = col_c1.slider("X Min", 0.0, 0.8, 0.2, 0.05)
        xmax = col_c2.slider("X Max", 0.2, 1.0, 0.8, 0.05)
        ymin = col_c1.slider("Y Min", 0.0, 0.8, 0.1, 0.05)
        ymax = col_c2.slider("Y Max", 0.2, 1.0, 0.95, 0.05)
        custom_coords = (xmin, ymin, xmax, ymax)

    counter_ratio = st.sidebar.slider(
        "Counter Service Depth",
        min_value=0.10,
        max_value=0.35,
        value=0.20,
        step=0.05,
        help="Fraction of the queue zone front considered the cashier/service counter.",
    )

    st.sidebar.markdown("---")

    # 3. Model & Queue Estimation Parameters
    st.sidebar.subheader("⚙️ Queue Estimation Model")
    service_time_sec = st.sidebar.slider(
        "Baseline Service Time per Customer",
        min_value=15,
        max_value=180,
        value=45,
        step=5,
        help="Average seconds required to serve one customer (used in Little's Law W = L / μ).",
    )

    alert_wait_min = st.sidebar.slider(
        "Bottleneck Wait Alert (mins)",
        min_value=1.0,
        max_value=15.0,
        value=4.0,
        step=0.5,
    )

    alert_queue_len = st.sidebar.slider(
        "Bottleneck Queue Alert (persons)",
        min_value=2,
        max_value=20,
        value=6,
        step=1,
    )

    conf_threshold = st.sidebar.slider(
        "Detection Confidence",
        min_value=0.15,
        max_value=0.85,
        value=0.30,
        step=0.05,
    )

    frame_skip = st.sidebar.slider(
        "Frame Sampling Interval",
        min_value=1,
        max_value=8,
        value=2,
        help="Process every Nth frame for smooth CPU throughput.",
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("🎨 Visual Overlays")
    show_trajectories = st.sidebar.checkbox("Motion Trajectory Trails", value=True)
    show_zone_overlay = st.sidebar.checkbox("Queue Zone Illumination", value=True)
    show_hud_banner = st.sidebar.checkbox("Top HUD Status Bar", value=True)

    start_btn = st.sidebar.button("▶ Start QueueSense Monitor", type="primary", use_container_width=True)

    # -------------------------------------------------------------------------
    # Top KPI Summary Metric Cards
    # -------------------------------------------------------------------------
    col_m1, col_m2, col_m3, col_m4, col_m5, col_m6 = st.columns(6)
    card_queue_len = col_m1.empty()
    card_est_wait = col_m2.empty()
    card_avg_wait = col_m3.empty()
    card_max_wait = col_m4.empty()
    card_served = col_m5.empty()
    card_status = col_m6.empty()

    # Initial metric placeholders
    card_queue_len.metric("Queue Length", "--", help="People currently inside the queue zone")
    card_est_wait.metric("Est. Wait Time", "--", help="Predicted wait for a new customer joining now")
    card_avg_wait.metric("Avg Dwell Time", "--", help="Average time current members have been waiting")
    card_max_wait.metric("Max Wait Time", "--", help="Longest waiting customer in queue")
    card_served.metric("Served Customers", "0", help="Total customers processed through service counter")
    card_status.metric("Queue Congestion", "Ready", help="Current congestion classification")

    # Main Dashboard Tabs
    tab_stream, tab_analytics, tab_audit = st.tabs([
        "📹 Live Queue Monitor",
        "📈 Real-time Analytics & Charts",
        "📋 Customer Audit & Bottleneck Logs",
    ])

    with tab_stream:
        alert_container = st.empty()
        video_placeholder = st.empty()
        status_box = st.empty()
        progress_bar = st.empty()

    with tab_analytics:
        st.subheader("Queue Dynamics & Throughput Over Time")
        col_c1, col_c2 = st.columns(2)
        chart_queue_holder = col_c1.empty()
        chart_wait_holder = col_c2.empty()
        chart_served_holder = st.empty()

    with tab_audit:
        st.subheader("Real-Time Customer Flow Log")
        audit_table_holder = st.empty()
        st.subheader("Bottleneck Incident Timeline")
        alert_log_holder = st.empty()
        export_placeholder = st.empty()

    if not video_path and not is_webcam:
        video_placeholder.info("👈 Please select or upload a video from the sidebar, then click **Start QueueSense Monitor**.")
        return

    # Initialize components
    detector = get_detector(conf_threshold=conf_threshold)
    detector.set_confidence(conf_threshold)

    queue_zone = QueueZone(
        preset_name=zone_preset,
        custom_norm_coords=custom_coords,
        counter_height_ratio=counter_ratio,
    )
    tracker = QueueTracker(max_missed=15, iou_threshold=0.25, max_distance=130.0)
    estimator = WaitingTimeEstimator(
        default_service_sec=service_time_sec,
        alert_wait_min=alert_wait_min,
        alert_queue_len=alert_queue_len,
    )

    if start_btn:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            st.error("Error: Could not open the specified video source.")
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        status_box.info(f"🟢 Processing video: {total_frames if total_frames > 0 else 'Live'} frames @ {orig_fps:.1f} FPS")

        stop_btn = st.sidebar.button("⏹ Stop Monitoring", use_container_width=True)

        frame_idx = 0
        last_time = time.time()
        last_chart_update = 0.0

        try:
            while cap.isOpened():
                if stop_btn:
                    status_box.warning("Processing stopped by user.")
                    break

                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # Frame skip for smooth CPU inference
                if frame_idx % frame_skip != 0:
                    continue

                # Standard display resize
                h, w = frame.shape[:2]
                if w > 800:
                    scale = 800.0 / w
                    frame = cv2.resize(frame, (800, int(h * scale)))
                    h, w = frame.shape[:2]

                timestamp = frame_idx / orig_fps

                # 1. Detection
                t0 = time.time()
                detections = detector.detect(frame)
                infer_time = time.time() - t0

                # 2. Tracking
                active_tracks = tracker.update(detections, timestamp)

                # 3. Estimation & Queue Analytics
                stats = estimator.update(active_tracks, tracker.tracks, queue_zone, w, h, timestamp)

                # 4. Visualization & Overlays
                annotated = QueueVisualizer.render(
                    frame=frame,
                    active_tracks=active_tracks,
                    queue_zone=queue_zone,
                    stats=stats,
                    show_trajectories=show_trajectories,
                    show_zone=show_zone_overlay,
                    show_hud=show_hud_banner,
                )
                annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

                # FPS Calculation
                now = time.time()
                fps = 1.0 / max(now - last_time, 1e-4)
                last_time = now

                # -------------------------------------------------------------
                # Update UI Dashboard
                # -------------------------------------------------------------
                # Metrics Cards
                card_queue_len.metric("Queue Length", f"{stats['queue_length']} people")
                card_est_wait.metric("Est. Wait Time", stats["est_wait_str"])
                card_avg_wait.metric("Avg Dwell Time", stats["avg_dwell_str"])
                card_max_wait.metric("Max Wait Time", stats["max_dwell_str"])
                card_served.metric("Served Customers", f"{stats['served_count']}")
                card_status.metric(
                    "Queue Congestion",
                    stats["congestion_label"],
                    delta="Normal" if stats["severity"] <= 1 else "Elevated",
                    delta_color="normal" if stats["severity"] <= 1 else "inverse",
                )

                # Bottleneck alert banner
                if stats["is_alert"]:
                    alert_container.markdown(
                        f"""
                        <div class="alert-banner">
                            ⚠️ <b>QUEUE BOTTLENECK ALERT:</b> Estimated wait is <b>{stats['est_wait_str']}</b>
                            with <b>{stats['queue_length']}</b> persons in line. Consider opening an additional counter!
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                else:
                    alert_container.empty()

                # Video Frame
                video_placeholder.image(
                    annotated_rgb,
                    channels="RGB",
                    caption=f"Frame {frame_idx}/{total_frames} · Inferred in {infer_time*1000:.1f}ms · {fps:.1f} FPS",
                    use_container_width=True,
                )

                if total_frames > 0:
                    progress_bar.progress(min(frame_idx / total_frames, 1.0))

                # Periodically update analytics charts (every 1.5 seconds)
                if (now - last_chart_update) > 1.5:
                    last_chart_update = now
                    df_history = pd.DataFrame(estimator.time_series_history)
                    if not df_history.empty:
                        # Chart 1: Queue Length
                        chart_queue_holder.line_chart(
                            df_history.set_index("time_sec")[["queue_length"]],
                            use_container_width=True,
                        )
                        # Chart 2: Wait Times
                        chart_wait_holder.line_chart(
                            df_history.set_index("time_sec")[["est_wait_sec", "avg_wait_sec"]],
                            use_container_width=True,
                        )

            status_box.success("🎉 Video processing and queue analytics completed.")

        finally:
            cap.release()
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except Exception:
                    pass

        # ---------------------------------------------------------------------
        # Final Post-Run Analytics & Export
        # ---------------------------------------------------------------------
        df_history = pd.DataFrame(estimator.time_series_history)
        if not df_history.empty:
            chart_queue_holder.line_chart(
                df_history.set_index("time_sec")[["queue_length"]],
                use_container_width=True,
            )
            chart_wait_holder.line_chart(
                df_history.set_index("time_sec")[["est_wait_sec", "avg_wait_sec"]],
                use_container_width=True,
            )
            chart_served_holder.area_chart(
                df_history.set_index("time_sec")[["served_cumulative"]],
                use_container_width=True,
            )

        # Audit Table
        all_tracks_list = list(tracker.tracks.values())
        if all_tracks_list:
            audit_rows = []
            for t in all_tracks_list:
                audit_rows.append({
                    "Customer ID": f"#{t.track_id}",
                    "Status": t.status,
                    "Total Queue Time": WaitingTimeEstimator.format_duration(t.total_queue_time),
                    "First Seen (s)": f"{t.first_seen:.1f}s",
                    "Last Seen (s)": f"{t.last_seen:.1f}s",
                    "Detections": t.hits,
                })
            df_audit = pd.DataFrame(audit_rows)
            audit_table_holder.dataframe(df_audit, use_container_width=True)

            # CSV Download
            csv_data = df_audit.to_csv(index=False).encode("utf-8")
            export_placeholder.download_button(
                label="📥 Download Customer Audit Report (CSV)",
                data=csv_data,
                file_name="queuesense_audit_report.csv",
                mime="text/csv",
            )

        # Alerts Table
        if estimator.alerts:
            df_alerts = pd.DataFrame(estimator.alerts)
            alert_log_holder.dataframe(df_alerts[["time_str", "queue_len", "est_wait", "reason"]], use_container_width=True)
        else:
            alert_log_holder.info("No critical bottleneck incidents recorded during this monitoring session.")


if __name__ == "__main__":
    main()
