# QueueSense Launcher Script (Isolated on F: Drive)
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:TORCH_HOME = "F:\.cache\torch"
$env:HF_HOME = "F:\.cache\huggingface"
$env:YOLO_CONFIG_DIR = "F:\.cache\ultralytics"

# Run with dedicated virtual environment on F drive
& ".\.venv\Scripts\streamlit.exe" run app.py
