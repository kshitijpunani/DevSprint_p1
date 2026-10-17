@echo off
rem QueueSense Launcher Script (Isolated on F: Drive)
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set TORCH_HOME=F:\.cache\torch
set HF_HOME=F:\.cache\huggingface
set YOLO_CONFIG_DIR=F:\.cache\ultralytics

rem Run with dedicated virtual environment on F drive
call ".\.venv\Scripts\streamlit.exe" run app.py
