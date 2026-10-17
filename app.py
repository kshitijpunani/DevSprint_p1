"""
QueueSense - Intelligent Queue Monitoring & Waiting-Time Estimation
Production Prototype · DEVSPRINT '26
"""

import os
# Safeguard for Windows OpenBLAS thread allocation
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

# Route model downloads and caches to F drive to preserve C drive space
os.environ.setdefault("TORCH_HOME", "F:/.cache/torch")
os.environ.setdefault("HF_HOME", "F:/.cache/huggingface")
os.environ.setdefault("YOLO_CONFIG_DIR", "F:/.cache/ultralytics")

import socket
import tempfile
import time
from typing import Optional, Dict, Any, List, Tuple

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as stc

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
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    /* Global Typography & Canvas */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        -webkit-font-smoothing: antialiased;
    }

    .stApp {
        background: radial-gradient(circle at 10% 10%, #0d1527 0%, #080c14 100%) !important;
        color: #f1f5f9;
    }

    /* Clean Streamlit Top Margin & Header */
    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1400px !important;
    }
    header[data-testid="stHeader"] {
        background: transparent !important;
        pointer-events: none !important;
        z-index: 99990 !important;
    }
    /* CRITICAL FIX: Re-enable pointer events on header's direct children
       so the sidebar collapsed-control button remains clickable.
       Without this, intermediate wrapper divs inherit pointer-events: none
       from the header and block clicks on the reopen button. */
    header[data-testid="stHeader"] > div {
        pointer-events: auto !important;
    }
    div[data-testid="stToolbar"] {
        visibility: hidden !important;
        height: 0% !important;
        position: fixed !important;
        pointer-events: none !important;
    }
    div[data-testid="stDecoration"] {
        visibility: hidden !important;
        height: 0% !important;
        position: fixed !important;
        pointer-events: none !important;
    }

    /* Fixed, High-Visibility Sidebar Reopen Button */
    div[data-testid="collapsedControl"],
    div[data-testid="stSidebarCollapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        z-index: 999999 !important;
        top: 12px !important;
        left: 12px !important;
        position: fixed !important;
        pointer-events: auto !important;
    }
    div[data-testid="collapsedControl"] button,
    div[data-testid="stSidebarCollapsedControl"] button {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.98) 100%) !important;
        border: 1.5px solid rgba(255, 107, 107, 0.6) !important;
        border-radius: 10px !important;
        color: #ff9980 !important;
        box-shadow: 0 4px 18px rgba(0, 0, 0, 0.6), 0 0 14px rgba(255, 107, 107, 0.35) !important;
        width: 38px !important;
        height: 38px !important;
        align-items: center !important;
        justify-content: center !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
        pointer-events: auto !important;
    }
    div[data-testid="collapsedControl"] button:hover,
    div[data-testid="stSidebarCollapsedControl"] button:hover {
        background: linear-gradient(135deg, #FF5E62 0%, #FF8E53 100%) !important;
        border-color: rgba(255, 255, 255, 0.5) !important;
        color: #ffffff !important;
        transform: scale(1.08) !important;
        box-shadow: 0 6px 22px rgba(255, 107, 107, 0.5) !important;
    }
    div[data-testid="collapsedControl"] svg,
    div[data-testid="stSidebarCollapsedControl"] svg {
        color: #ffffff !important;
        fill: #ffffff !important;
        stroke: #ffffff !important;
        width: 20px !important;
        height: 20px !important;
        display: block !important;
    }

    /* Sidebar Close Button */
    button[data-testid="stSidebarCollapseButton"],
    section[data-testid="stSidebar"] button[kind="header"] {
        color: #f1f5f9 !important;
        background: rgba(30, 41, 59, 0.8) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    button[data-testid="stSidebarCollapseButton"]:hover,
    section[data-testid="stSidebar"] button[kind="header"]:hover {
        background: rgba(239, 68, 68, 0.2) !important;
        border-color: rgba(239, 68, 68, 0.5) !important;
        color: #ff9999 !important;
    }

    /* Top Navigation / Brand Header */
    .header-wrapper {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 4px 0 18px 0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.07);
        margin-bottom: 22px;
    }
    .brand-group {
        display: flex;
        align-items: center;
        gap: 14px;
    }
    .brand-icon-box {
        width: 44px;
        height: 44px;
        border-radius: 12px;
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.22) 0%, rgba(255, 142, 83, 0.12) 100%);
        border: 1px solid rgba(255, 107, 107, 0.4);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.4rem;
        box-shadow: 0 0 20px rgba(255, 107, 107, 0.18);
    }
    .brand-title {
        font-size: 1.85rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0;
        background: linear-gradient(135deg, #ffffff 0%, #e2e8f0 60%, #ff8e53 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .brand-badge {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        padding: 3px 8px;
        border-radius: 20px;
        background: rgba(255, 107, 107, 0.12);
        border: 1px solid rgba(255, 107, 107, 0.3);
        color: #ff9980;
        text-transform: uppercase;
        display: inline-block;
        margin-left: 8px;
        vertical-align: middle;
    }
    .brand-subtitle {
        font-size: 0.86rem;
        color: #94a3b8;
        margin-top: 2px;
        font-weight: 400;
    }
    .header-chips {
        display: flex;
        gap: 10px;
        align-items: center;
    }
    .chip {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.74rem;
        font-weight: 500;
        padding: 5px 12px;
        border-radius: 20px;
        background: rgba(15, 23, 42, 0.65);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #cbd5e1;
        backdrop-filter: blur(8px);
    }
    .chip-dot {
        width: 7px;
        height: 7px;
        border-radius: 50%;
        background: #10b981;
        box-shadow: 0 0 8px #10b981;
        animation: pulse-dot 2s infinite;
    }
    @keyframes pulse-dot {
        0% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.4; transform: scale(0.9); }
        100% { opacity: 1; transform: scale(1); }
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.35) 100%) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 14px !important;
        padding: 14px 16px !important;
        backdrop-filter: blur(14px) !important;
        box-shadow: 0 4px 18px -2px rgba(0, 0, 0, 0.35) !important;
        transition: all 0.2s cubic-bezier(0.16, 1, 0.3, 1) !important;
        position: relative !important;
        overflow: hidden !important;
    }
    div[data-testid="stMetric"]::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(255, 107, 107, 0.5), transparent);
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 107, 107, 0.35) !important;
        box-shadow: 0 10px 24px -4px rgba(0, 0, 0, 0.5), 0 0 18px rgba(255, 107, 107, 0.12) !important;
    }
    div[data-testid="stMetricLabel"] p {
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.06em !important;
        text-transform: uppercase !important;
        color: #94a3b8 !important;
        margin-bottom: 4px !important;
    }
    div[data-testid="stMetricValue"] div {
        font-size: 1.55rem !important;
        font-weight: 700 !important;
        color: #f8fafc !important;
        letter-spacing: -0.02em !important;
        font-feature-settings: 'tnum' !important;
    }
    div[data-testid="stMetricDelta"] {
        font-size: 0.78rem !important;
        font-weight: 600 !important;
    }

    /* Tabs Styling */
    div[data-testid="stTabs"] {
        margin-top: 14px;
        margin-bottom: 22px;
    }
    div[data-testid="stTabs"] button[role="tab"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        border-radius: 10px !important;
        padding: 9px 18px !important;
        margin-right: 6px !important;
        transition: all 0.2s ease !important;
        border: 1px solid transparent !important;
        background: transparent !important;
    }
    div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
        color: #ffffff !important;
        background: rgba(255, 107, 107, 0.12) !important;
        border: 1px solid rgba(255, 107, 107, 0.35) !important;
        box-shadow: 0 0 16px rgba(255, 107, 107, 0.15) !important;
    }
    div[data-testid="stTabs"] button[role="tab"]:hover {
        color: #f1f5f9 !important;
        background: rgba(255, 255, 255, 0.04) !important;
    }
    div[data-testid="stTabs"] [data-baseweb="tab-highlight"],
    div[data-testid="stTabs"] [data-baseweb="tab-border"] {
        display: none !important;
    }

    /* Hero Showcase Card */
    .hero-card {
        background: linear-gradient(145deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.35) 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 32px 30px;
        margin: 8px 0 20px 0;
        backdrop-filter: blur(16px);
        box-shadow: 0 8px 32px -4px rgba(0, 0, 0, 0.4);
        position: relative;
        overflow: hidden;
    }
    .hero-card::after {
        content: '';
        position: absolute;
        top: -60px;
        right: -60px;
        width: 240px;
        height: 240px;
        background: radial-gradient(circle, rgba(255, 107, 107, 0.12) 0%, transparent 70%);
        pointer-events: none;
    }
    .hero-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #ff9980;
        background: rgba(255, 107, 107, 0.12);
        border: 1px solid rgba(255, 107, 107, 0.25);
        padding: 4px 12px;
        border-radius: 20px;
        margin-bottom: 12px;
    }
    .hero-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: #f8fafc;
        letter-spacing: -0.02em;
        margin-bottom: 6px;
    }
    .hero-desc {
        font-size: 0.9rem;
        color: #94a3b8;
        line-height: 1.6;
        max-width: 720px;
        margin-bottom: 22px;
    }
    .features-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-top: 18px;
    }
    .feature-box {
        background: rgba(15, 23, 42, 0.55);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 16px;
        transition: all 0.2s ease;
    }
    .feature-box:hover {
        border-color: rgba(255, 107, 107, 0.3);
        background: rgba(30, 41, 59, 0.4);
        transform: translateY(-2px);
    }
    .feature-icon {
        font-size: 1.25rem;
        margin-bottom: 8px;
    }
    .feature-name {
        font-size: 0.84rem;
        font-weight: 600;
        color: #f1f5f9;
        margin-bottom: 4px;
    }
    .feature-text {
        font-size: 0.75rem;
        color: #94a3b8;
        line-height: 1.45;
    }

    /* Sidebar Custom Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b1120 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.06) !important;
    }
    .sidebar-section-header {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #cbd5e1;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-top: 16px;
        margin-bottom: 10px;
        padding-bottom: 6px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }

    /* Primary Action Button */
    div.stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #FF5E62 0%, #FF8E53 100%) !important;
        border: none !important;
        border-radius: 12px !important;
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.02em !important;
        padding: 12px 20px !important;
        box-shadow: 0 4px 18px rgba(255, 94, 98, 0.35) !important;
        transition: all 0.2s ease !important;
        margin-top: 8px !important;
    }
    div.stButton > button[kind="primary"]:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 24px rgba(255, 94, 98, 0.5) !important;
    }

    /* Stop Button */
    div.stButton > button:not([kind="primary"]) {
        background: rgba(239, 68, 68, 0.12) !important;
        border: 1px solid rgba(239, 68, 68, 0.35) !important;
        border-radius: 12px !important;
        color: #fca5a5 !important;
        font-weight: 600 !important;
        font-size: 0.92rem !important;
        padding: 10px 18px !important;
        transition: all 0.2s ease !important;
    }
    div.stButton > button:not([kind="primary"]):hover {
        background: rgba(239, 68, 68, 0.22) !important;
        border-color: rgba(239, 68, 68, 0.6) !important;
        color: #ffffff !important;
    }

    /* Alert Banner */
    .alert-banner {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.18) 0%, rgba(185, 28, 28, 0.1) 100%);
        border: 1px solid rgba(239, 68, 68, 0.45);
        border-radius: 12px;
        color: #fecaca;
        padding: 12px 18px;
        font-weight: 500;
        margin-bottom: 14px;
        display: flex;
        align-items: center;
        gap: 12px;
        box-shadow: 0 4px 16px -2px rgba(239, 68, 68, 0.25);
        backdrop-filter: blur(8px);
        animation: alert-pulse 2s infinite;
    }
    @keyframes alert-pulse {
        0% { border-color: rgba(239, 68, 68, 0.45); }
        50% { border-color: rgba(239, 68, 68, 0.85); box-shadow: 0 4px 22px rgba(239, 68, 68, 0.35); }
        100% { border-color: rgba(239, 68, 68, 0.45); }
    }

    /* Video Frame Container */
    div[data-testid="stImage"] {
        border-radius: 14px;
        overflow: hidden;
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 30px -4px rgba(0, 0, 0, 0.5);
    }

    /* Download CSV Button */
    div[data-testid="stDownloadButton"] > button {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.2) 0%, rgba(5, 150, 105, 0.1) 100%) !important;
        border: 1px solid rgba(16, 185, 129, 0.4) !important;
        color: #6ee7b7 !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        padding: 10px 20px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.3) 0%, rgba(5, 150, 105, 0.2) 100%) !important;
        border-color: rgba(16, 185, 129, 0.7) !important;
        box-shadow: 0 0 16px rgba(16, 185, 129, 0.25) !important;
        color: #ffffff !important;
        transform: translateY(-1px) !important;
    }

    /* ——— Mobile Responsive Sidebar ——— */
    @media (max-width: 768px) {
        section[data-testid="stSidebar"] {
            z-index: 999999 !important;
        }
        .features-grid {
            grid-template-columns: repeat(2, 1fr) !important;
        }
        .header-chips {
            display: none !important;
        }
        .brand-title {
            font-size: 1.3rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ─── Persistent Sidebar Toggle Button ─────────────────────────────────────
# Injects a fixed-position hamburger menu button via client-side JavaScript.
# The button is appended to the parent document.body so it persists across
# Streamlit reruns and is always visible regardless of sidebar state.
# Falls back to Streamlit's native collapsedControl (CSS-fixed above) if the
# iframe cannot access the parent DOM due to browser security restrictions.
stc.html("""
<script>
(function() {
    try { var d = window.parent.document; } catch(e) { return; }
    if (d.getElementById('qs-sidebar-toggle')) return;

    /* ── Create hamburger toggle button ── */
    var b = d.createElement('button');
    b.id = 'qs-sidebar-toggle';
    b.setAttribute('aria-label', 'Toggle sidebar');
    b.innerHTML = '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round"><line x1="4" y1="6" x2="20" y2="6"></line><line x1="4" y1="12" x2="16" y2="12"></line><line x1="4" y1="18" x2="20" y2="18"></line></svg>';
    b.style.cssText = 'position:fixed;top:14px;left:14px;z-index:1000001;width:40px;height:40px;background:linear-gradient(135deg,rgba(30,41,59,0.95),rgba(15,23,42,0.98));border:1.5px solid rgba(255,107,107,0.6);border-radius:10px;display:flex;align-items:center;justify-content:center;cursor:pointer;box-shadow:0 4px 18px rgba(0,0,0,0.6),0 0 14px rgba(255,107,107,0.35);transition:all 0.25s cubic-bezier(0.16,1,0.3,1);padding:0;outline:none;';

    b.onmouseenter = function() {
        this.style.background = 'linear-gradient(135deg,#FF5E62,#FF8E53)';
        this.style.transform = 'scale(1.08)';
        this.style.boxShadow = '0 6px 22px rgba(255,107,107,0.5)';
        this.style.borderColor = 'rgba(255,255,255,0.5)';
    };
    b.onmouseleave = function() {
        this.style.background = 'linear-gradient(135deg,rgba(30,41,59,0.95),rgba(15,23,42,0.98))';
        this.style.transform = 'scale(1)';
        this.style.boxShadow = '0 4px 18px rgba(0,0,0,0.6),0 0 14px rgba(255,107,107,0.35)';
        this.style.borderColor = 'rgba(255,107,107,0.6)';
    };

    b.addEventListener('click', function(e) {
    e.stopPropagation(); e.preventDefault();
    var sb = d.querySelector('section[data-testid="stSidebar"]');
    if (!sb) return;
    var collapsed = sb.getAttribute('data-collapsed') === 'true'
                 || sb.getAttribute('aria-expanded') === 'false';
    if (collapsed) {
        var o = d.querySelector('[data-testid="stExpandSidebarButton"]')
             || d.querySelector('[data-testid="stSidebarCollapsedControl"] button')
             || d.querySelector('[data-testid="collapsedControl"] button');
        if (o) o.click();
    } else {
        var c = sb.querySelector('[data-testid="stSidebarCollapseButton"] button')
             || sb.querySelector('button[kind="header"]');
        if (c) c.click();
    }
});
    d.body.appendChild(b);

    /* ── Hide native collapsed control (our button replaces it) ── */
    var hs = d.createElement('style');
    hs.id = 'qs-hide-native-toggle';
    hs.textContent = 'div[data-testid="collapsedControl"],div[data-testid="stSidebarCollapsedControl"]{opacity:0!important;pointer-events:none!important;width:0!important;height:0!important;overflow:hidden!important;position:fixed!important;top:-9999px!important;}';
    d.head.appendChild(hs);

    /* ── Mobile backdrop overlay ── */
    var bd = d.createElement('div');
    bd.id = 'qs-sidebar-backdrop';
    bd.style.cssText = 'display:none;position:fixed;top:0;left:0;width:100vw;height:100vh;background:rgba(0,0,0,0.5);backdrop-filter:blur(4px);-webkit-backdrop-filter:blur(4px);z-index:999998;transition:opacity 0.3s ease;';
    bd.addEventListener('click', function() {
        var sb = d.querySelector('section[data-testid="stSidebar"]');
        if (sb) {
            var c = sb.querySelector('[data-testid="stSidebarCollapseButton"]')
                  || sb.querySelector('button[kind="header"]');
            if (c) c.click();
        }
        this.style.display = 'none';
    });
    d.body.appendChild(bd);

    /* ── Observe sidebar state changes for mobile backdrop ── */
    var sb = d.querySelector('section[data-testid="stSidebar"]');
    if (sb) {
        new MutationObserver(function() {
            if (window.parent.innerWidth > 768) { bd.style.display = 'none'; return; }
            bd.style.display = sb.getAttribute('aria-expanded') === 'true' ? 'block' : 'none';
        }).observe(sb, { attributes: true, attributeFilter: ['aria-expanded'] });
    }
})();
</script>
""", height=0)


def get_local_ip() -> str:
    """Get the local network IP address for mobile phone access."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "localhost"


def scan_cameras(max_to_check: int = 4) -> List[int]:
    """Detect available local and virtual webcam indices."""
    available = []
    for i in range(max_to_check):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available if available else [0]


def test_stream_connection(url: str, timeout_sec: float = 2.5) -> Tuple[bool, str]:
    """Test connection to an IP camera or RTSP phone stream."""
    try:
        cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(timeout_sec * 1000))
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(timeout_sec * 1000))
        if not cap.isOpened():
            cap.release()
            return False, "Could not connect to stream URL. Ensure phone & PC are on the same Wi-Fi network."
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            h, w = frame.shape[:2]
            return True, f"Connected! Active phone stream ({w}x{h} px)"
        return False, "Connection opened, but failed to retrieve initial video frame."
    except Exception as e:
        return False, f"Stream error: {e}"


@st.cache_resource(show_spinner="Initializing YOLOv8 Neural Engine...")
def get_detector(conf_threshold: float = 0.35) -> PersonDetector:
    """Load and cache the YOLOv8 person detector across Streamlit reruns."""
    return PersonDetector(model_name="yolov8n.pt", conf_threshold=conf_threshold)


def main():
    local_ip = get_local_ip()

    # Session State Initialization
    if "monitoring_active" not in st.session_state:
        st.session_state["monitoring_active"] = False
    if "scanned_cameras" not in st.session_state:
        st.session_state["scanned_cameras"] = [0]
    if "phone_stream_url" not in st.session_state:
        st.session_state["phone_stream_url"] = "http://192.168.1.15:8080/video"

    # Top Modern Header Banner
    header_html = f"""
    <div class="header-wrapper">
        <div class="brand-group">
            <div class="brand-icon-box">⏱️</div>
            <div>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span class="brand-title">QueueSense</span>
                    <span class="brand-badge">DEVSPRINT '26</span>
                </div>
                <div class="brand-subtitle">
                    Intelligent Queue Monitoring, MOT Tracking & Waiting-Time Estimation Engine
                </div>
            </div>
        </div>
        <div class="header-chips">
            <div class="chip">
                <span class="chip-dot"></span>
                <span>SYSTEM ONLINE</span>
            </div>
            <div class="chip">
                <span>⚡ YOLOv8n MOT</span>
            </div>
            <div class="chip">
                <span>📱 LAN: http://{local_ip}:8501</span>
            </div>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

    # -------------------------------------------------------------------------
    # Sidebar Configuration Controls
    # -------------------------------------------------------------------------
    st.sidebar.markdown('<div class="sidebar-section-header">🕹️ VIDEO INPUT SOURCE</div>', unsafe_allow_html=True)

    source_choice = st.sidebar.radio(
        "Select Camera / Video Source:",
        (
            "📁 Sample Queue Video",
            "📤 Upload Video File",
            "💻 Laptop / USB Webcam",
            "📱 Phone Camera (Wi-Fi / IP Stream)",
            "🌐 In-Browser Camera (Phone / Laptop Web)",
        ),
        index=0,
    )

    video_path = None
    temp_file_path = None
    is_webcam = False
    is_ip_stream = False
    is_browser_camera = False

    if source_choice == "📁 Sample Queue Video":
        sample_path = os.path.join("data", "sample_videos", "sample_queue.mp4")
        if os.path.exists(sample_path):
            video_path = sample_path
            st.sidebar.markdown(
                f'<div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 6px 12px; color: #6ee7b7; font-size: 0.8rem; margin-top: 6px;">'
                f'✓ Loaded: <b>{os.path.basename(sample_path)}</b></div>',
                unsafe_allow_html=True,
            )
        else:
            st.sidebar.error("Sample video not found. Please upload a video.")

    elif source_choice == "📤 Upload Video File":
        uploaded_file = st.sidebar.file_uploader(
            "Upload Queue Video", type=["mp4", "avi", "mov", "mkv"]
        )
        if uploaded_file is not None:
            f_temp_dir = os.path.join(os.path.dirname(__file__), "data", "temp")
            os.makedirs(f_temp_dir, exist_ok=True)
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4", dir=f_temp_dir)
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name
            video_path = temp_file_path
            st.sidebar.markdown(
                f'<div style="background: rgba(16, 185, 129, 0.12); border: 1px solid rgba(16, 185, 129, 0.3); border-radius: 8px; padding: 6px 12px; color: #6ee7b7; font-size: 0.8rem; margin-top: 6px;">'
                f'✓ Uploaded: <b>{uploaded_file.name}</b></div>',
                unsafe_allow_html=True,
            )

    elif source_choice == "💻 Laptop / USB Webcam":
        is_webcam = True
        col_c_sel, col_c_scan = st.sidebar.columns([3, 2])
        scanned_list = st.session_state.get("scanned_cameras", [0])
        webcam_idx = col_c_sel.selectbox(
            "Camera Device:",
            scanned_list,
            format_func=lambda x: f"Camera #{x}" + (" (Default/Laptop)" if x == 0 else " (External/Virtual)"),
        )
        if col_c_scan.button("🔍 Scan", use_container_width=True, help="Scan for USB and Virtual Webcams"):
            with st.spinner("Scanning cameras..."):
                found = scan_cameras()
                st.session_state["scanned_cameras"] = found
                st.rerun()

        # Camera test snapshot button
        if st.sidebar.button("📸 Test Camera Snapshot", use_container_width=True):
            cap_test = cv2.VideoCapture(int(webcam_idx), cv2.CAP_DSHOW)
            if not cap_test.isOpened():
                cap_test = cv2.VideoCapture(int(webcam_idx))
            if cap_test.isOpened():
                ret_t, frame_t = cap_test.read()
                cap_test.release()
                if ret_t and frame_t is not None:
                    rgb_t = cv2.cvtColor(frame_t, cv2.COLOR_BGR2RGB)
                    st.sidebar.image(rgb_t, caption=f"Camera #{webcam_idx} Active (640x480)", use_container_width=True)
                    st.sidebar.success("✓ Camera is working perfectly!")
                else:
                    st.sidebar.error("Failed to read frame from camera.")
            else:
                st.sidebar.error("Could not open camera. Ensure no other app is using it.")

        video_path = int(webcam_idx)

    elif source_choice == "📱 Phone Camera (Wi-Fi / IP Stream)":
        is_ip_stream = True
        with st.sidebar.expander("📖 How to stream your phone camera", expanded=False):
            st.markdown(
                """
                **Free & Easy Setup:**
                1. **Android**: Install free app **IP Webcam** (by Pavel Khlebovich) or **DroidCam**.
                2. **iPhone**: Install **DroidCam** or **IP Camera Lite**.
                3. Ensure your phone and laptop are connected to the **same Wi-Fi**.
                4. In the app, tap **Start Server**.
                5. Enter the stream URL shown on your phone screen below.
                """
            )
        
        # Preset buttons for quick URL filling
        st.sidebar.caption("Quick Presets:")
        col_pr1, col_pr2, col_pr3 = st.sidebar.columns(3)
        if col_pr1.button("IP Webcam", use_container_width=True):
            st.session_state["phone_stream_url"] = f"http://{local_ip.rsplit('.', 1)[0]}.15:8080/video"
            st.rerun()
        if col_pr2.button("DroidCam", use_container_width=True):
            st.session_state["phone_stream_url"] = f"http://{local_ip.rsplit('.', 1)[0]}.15:4747/video"
            st.rerun()
        if col_pr3.button("RTSP", use_container_width=True):
            st.session_state["phone_stream_url"] = f"rtsp://{local_ip.rsplit('.', 1)[0]}.15:8554/live"
            st.rerun()

        input_url = st.sidebar.text_input(
            "Phone Video Stream URL:",
            value=st.session_state.get("phone_stream_url", "http://192.168.1.15:8080/video"),
            help="Stream URL shown on your phone's camera app",
        )
        st.session_state["phone_stream_url"] = input_url
        video_path = input_url

        if st.sidebar.button("🔗 Test Stream Connection", use_container_width=True):
            with st.spinner("Connecting to phone stream..."):
                ok, msg = test_stream_connection(input_url)
                if ok:
                    st.sidebar.success(f"✓ {msg}")
                else:
                    st.sidebar.error(f"⚠️ {msg}")

    elif source_choice == "🌐 In-Browser Camera (Phone / Laptop Web)":
        is_browser_camera = True
        st.sidebar.markdown(
            f"""
            <div style="background: rgba(14, 165, 233, 0.12); border: 1px solid rgba(14, 165, 233, 0.35); border-radius: 10px; padding: 12px; color: #bae6fd; font-size: 0.8rem; margin-top: 6px;">
                <b>📱 Connect Mobile Phone:</b><br>
                Open your phone's browser and go to:<br>
                <code style="color: #38bdf8; font-weight: 700;">http://{local_ip}:8501</code><br>
                Your phone's camera will activate directly inside the web browser!
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Queue Zone (ROI) Configuration
    st.sidebar.markdown('<div class="sidebar-section-header">📍 QUEUE ZONE (ROI)</div>', unsafe_allow_html=True)
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

    # 3. Model & Queue Estimation Parameters
    st.sidebar.markdown('<div class="sidebar-section-header">⚙️ ESTIMATION ENGINE</div>', unsafe_allow_html=True)
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
        max_value=25.0,
        value=4.0,
        step=0.5,
    )

    alert_queue_len = st.sidebar.slider(
        "Bottleneck Queue Alert (persons)",
        min_value=2,
        max_value=25,
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

    st.sidebar.markdown('<div class="sidebar-section-header">🎨 VISUAL OVERLAYS</div>', unsafe_allow_html=True)
    show_trajectories = st.sidebar.checkbox("Motion Trajectory Trails", value=True)
    show_zone_overlay = st.sidebar.checkbox("Queue Zone Illumination", value=True)
    show_hud_banner = st.sidebar.checkbox("Top HUD Status Bar", value=True)

    # Sidebar Start / Stop button (for continuous stream sources)
    if not is_browser_camera:
        col_btn1, col_btn2 = st.sidebar.columns(2)
        if not st.session_state.get("monitoring_active", False):
            if col_btn1.button("▶ Start Monitor", type="primary", use_container_width=True):
                st.session_state["monitoring_active"] = True
                st.rerun()
        else:
            if col_btn2.button("⏹ Stop Monitor", use_container_width=True):
                st.session_state["monitoring_active"] = False
                st.rerun()

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
        st.markdown('<div class="sidebar-section-header">📈 QUEUE DYNAMICS & THROUGHPUT TIMELINE</div>', unsafe_allow_html=True)
        col_c1, col_c2 = st.columns(2)
        chart_queue_holder = col_c1.empty()
        chart_wait_holder = col_c2.empty()
        chart_served_holder = st.empty()

    with tab_audit:
        st.markdown('<div class="sidebar-section-header">📋 REAL-TIME CUSTOMER FLOW AUDIT</div>', unsafe_allow_html=True)
        audit_table_holder = st.empty()
        st.markdown('<div class="sidebar-section-header">⚠️ BOTTLENECK INCIDENT TIMELINE</div>', unsafe_allow_html=True)
        alert_log_holder = st.empty()
        export_placeholder = st.empty()

    # Initialize AI & Queue Components
    detector = get_detector(conf_threshold=conf_threshold)
    detector.set_confidence(conf_threshold)

    queue_zone = QueueZone(
        preset_name=zone_preset,
        custom_norm_coords=custom_coords,
        counter_height_ratio=counter_ratio,
    )
    tracker = QueueTracker(max_missed=25, iou_threshold=0.25, max_distance=130.0)
    estimator = WaitingTimeEstimator(
        default_service_sec=service_time_sec,
        alert_wait_min=alert_wait_min,
        alert_queue_len=alert_queue_len,
        alert_cooldown_sec=25.0,
    )

    # -------------------------------------------------------------------------
    # CASE A: In-Browser Direct Camera (Mobile Phone or Laptop Browser)
    # -------------------------------------------------------------------------
    if is_browser_camera:
        with tab_stream:
            st.markdown(
                f"""
                <div style="background: rgba(15, 23, 42, 0.7); border: 1px solid rgba(56, 189, 248, 0.3); border-radius: 12px; padding: 14px 18px; margin-bottom: 16px;">
                    <div style="display: flex; align-items: center; justify-content: space-between;">
                        <div>
                            <span style="font-size: 1.1rem; font-weight: 700; color: #f8fafc;">🌐 In-Browser Camera Active</span>
                            <div style="font-size: 0.84rem; color: #94a3b8; margin-top: 2px;">
                                Point your camera at the queue and tap <b>Take Photo</b> to run live detection, MOT tracking, and wait estimation.
                            </div>
                        </div>
                        <div style="background: rgba(56, 189, 248, 0.15); border: 1px solid rgba(56, 189, 248, 0.35); border-radius: 20px; padding: 4px 12px; color: #7dd3fc; font-size: 0.75rem; font-weight: 600;">
                            📱 Phone URL: http://{local_ip}:8501
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            camera_buffer = st.camera_input("📷 Live Camera View (Phone or Laptop Camera)", key="browser_cam_input")

        if camera_buffer is not None:
            bytes_data = camera_buffer.getvalue()
            frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)

            h, w = frame.shape[:2]
            if w > 800:
                scale = 800.0 / w
                frame = cv2.resize(frame, (800, int(h * scale)))
                h, w = frame.shape[:2]

            timestamp = time.time()
            t0 = time.time()
            detections = detector.detect(frame)
            infer_time = time.time() - t0

            active_tracks = tracker.update(detections, timestamp)
            stats = estimator.update(active_tracks, tracker.tracks, queue_zone, w, h, timestamp)
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

            # Update Metric Cards
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

            # Video Frame display
            video_placeholder.image(
                annotated_rgb,
                channels="RGB",
                caption=f"Frame Inferred in {infer_time*1000:.1f}ms · {len(active_tracks)} People Tracked · {stats['queue_length']} in Queue",
                use_container_width=True,
            )

            if stats["is_alert"]:
                alert_container.markdown(
                    f"""
                    <div class="alert-banner">
                        <span style="font-size: 1.3rem;">⚠️</span>
                        <div>
                            <b>QUEUE BOTTLENECK ALERT:</b> Estimated wait is <b>{stats['est_wait_str']}</b>
                            with <b>{stats['queue_length']}</b> persons in line. Consider opening an additional counter!
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                alert_container.empty()

            # Audit Table update
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
        return

    # -------------------------------------------------------------------------
    # CASE B: Continuous Video Streams (Laptop Webcam, Phone IP Stream, Videos)
    # -------------------------------------------------------------------------
    is_running = st.session_state.get("monitoring_active", False)

    # Idle state when not currently streaming
    if not is_running:
        source_display_name = (
            "Sample Pedestrian Video (sample_queue.mp4)"
            if source_choice == "📁 Sample Queue Video"
            else (
                "Custom Video Upload"
                if source_choice == "📤 Upload Video File"
                else (
                    f"Laptop / USB Webcam (Device #{video_path})"
                    if source_choice == "💻 Laptop / USB Webcam"
                    else f"Phone Wi-Fi Stream ({video_path})"
                )
            )
        )
        hero_html = f"""
        <div class="hero-card">
            <div class="hero-badge">⚡ Real-time Multi-Object Tracking & Little's Law Engine</div>
            <div class="hero-title">Queue Intelligence Dashboard</div>
            <div class="hero-desc">
                Real-time queue monitoring, MOT trajectory tracking, and Little's Law waiting-time estimation engine.
                <br><br>
                • Active Stream Source: <span style="color: #ff9980; font-weight: 600;">{source_display_name}</span><br>
                • Region of Interest: <span style="color: #38bdf8; font-weight: 600;">{zone_preset} Layout</span><br>
                • Service Rate Baseline: <span style="color: #34d399; font-weight: 600;">{service_time_sec}s / customer</span><br>
                • Local Network Access: <span style="color: #a78bfa; font-weight: 600;">http://{local_ip}:8501</span> (Access from Phone)
                <br><br>
                Click <b style="color: #ff8e53;">▶ Start Monitor</b> in the sidebar to begin processing frames with YOLOv8 person detection and Hungarian tracking.
            </div>
            <div class="features-grid">
                <div class="feature-box">
                    <div class="feature-icon">🎯</div>
                    <div class="feature-name">YOLOv8 Person MOT</div>
                    <div class="feature-text">High-precision detection & tracking with ground foot-point estimation.</div>
                </div>
                <div class="feature-box">
                    <div class="feature-icon">⏱️</div>
                    <div class="feature-name">Little's Law Estimator</div>
                    <div class="feature-text">Predictive waiting time ($W = L / \\mu$) calculated dynamically per frame.</div>
                </div>
                <div class="feature-box">
                    <div class="feature-icon">📍</div>
                    <div class="feature-name">Passerby Filtering</div>
                    <div class="feature-text">Ground-plane ROI ignores passersby outside the queue boundary.</div>
                </div>
                <div class="feature-box">
                    <div class="feature-icon">🚨</div>
                    <div class="feature-name">Bottleneck Alarms</div>
                    <div class="feature-text">Proactive warning alerts triggered when line or wait times peak.</div>
                </div>
            </div>
        </div>
        """
        video_placeholder.markdown(hero_html, unsafe_allow_html=True)
        chart_queue_holder.info("📊 Line chart for Queue Length over time will stream live here during monitoring.")
        chart_wait_holder.info("⏳ Line chart for Estimated vs. Average Wait Times will stream live here during monitoring.")
        chart_served_holder.info("👥 Area chart for Cumulative Served Customers will appear here upon session run.")
        audit_table_holder.info("📝 Individual customer IDs, dwell times, and entry timestamps will be cataloged here.")
        alert_log_holder.info("🔔 High-congestion bottleneck incidents will be logged here with timestamps and queue metrics.")
        return

    # Validate video_path
    if video_path is None:
        video_placeholder.warning("👈 Please select or configure a video source from the sidebar.")
        st.session_state["monitoring_active"] = False
        return

    # Open capture with optimized hardware backend
    if is_webcam:
        cap = cv2.VideoCapture(int(video_path), cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(int(video_path))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
    elif is_ip_stream:
        cap = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap = cv2.VideoCapture(str(video_path))
    else:
        cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error(
            f"❌ Could not open video source: `{video_path}`. "
            "If using a webcam or phone stream, ensure it is connected and not currently used by another program."
        )
        st.session_state["monitoring_active"] = False
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if (not is_webcam and not is_ip_stream) else 0
    orig_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    status_box.info(f"🟢 Active Stream: {total_frames if total_frames > 0 else 'Live Camera / Stream'} @ ~{orig_fps:.1f} FPS")

    frame_idx = 0
    last_time = time.time()
    last_chart_update = 0.0

    try:
        while cap.isOpened() and st.session_state.get("monitoring_active", False):
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

            timestamp = frame_idx / orig_fps if total_frames > 0 else time.time()

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

            # Update Metrics Cards
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
                        <span style="font-size: 1.3rem;">⚠️</span>
                        <div>
                            <b>QUEUE BOTTLENECK ALERT:</b> Estimated wait is <b>{stats['est_wait_str']}</b>
                            with <b>{stats['queue_length']}</b> persons in line. Consider opening an additional counter!
                        </div>
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
                caption=f"Frame {frame_idx} · Inferred in {infer_time*1000:.1f}ms · {fps:.1f} FPS",
                use_container_width=True,
            )

            if total_frames > 0:
                progress_bar.progress(min(frame_idx / total_frames, 1.0))

            # Periodically update analytics charts (every 1.5 seconds)
            if (now - last_chart_update) > 1.5:
                last_chart_update = now
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

            # Brief cooperative yield for responsive UI events
            time.sleep(0.005)

        status_box.success("🎉 Video processing and queue analytics session completed.")

    finally:
        cap.release()
        st.session_state["monitoring_active"] = False
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Final Post-Run Analytics & Export
    # -------------------------------------------------------------------------
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
