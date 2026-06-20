import os
import sys
import time
import numpy as np
import cv2
import streamlit as st
import pandas as pd
# ==========================================
# CORE DEPENDENCY IMPORTS & HARDWARE DETECTIONS
# ==========================================
# Check for PyTorch to see GPU status
try:
    import torch
    HAS_TORCH = True
    GPU_NAME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None"
except Exception:
    HAS_TORCH = False
    GPU_NAME = "None"
# Check for psutil for system stats
try:
    import psutil
    HAS_PSUTIL = True
except Exception:
    HAS_PSUTIL = False
# Check for qrcode library for generating real QR codes
try:
    import qrcode
    HAS_QRCODE = True
except Exception:
    HAS_QRCODE = False
# Import Decoders
try:
    import zxingcpp
    HAS_ZXING = True
except Exception:
    HAS_ZXING = False
try:
    from pyzbar.pyzbar import decode as zbar_decode
    HAS_PYZBAR = True
except Exception:
    HAS_PYZBAR = False
# Check OpenCV CUDA availability
HAS_CV_CUDA = False
try:
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        HAS_CV_CUDA = True
except AttributeError:
    pass
# ==========================================
# CORE UTILITY FUNCTIONS (MERGED FROM BENCHMARK)
# ==========================================
def get_cpu_temp():
    """Retrieves CPU temperature if available on Linux/Jetson."""
    if sys.platform.startswith('linux'):
        for i in range(5):
            path = f"/sys/class/thermal/thermal_zone{i}/temp"
            if os.path.exists(path):
                try:
                    with open(path, 'r') as f:
                        temp = float(f.read().strip()) / 1000.0
                        return f"{temp:.1f}°C"
                except Exception:
                    pass
    return "N/A"
def generate_synthetic_qr(text="https://github.com/saideveshadapa/med-price-scanner"):
    """Generates a high-quality QR code image or a fallback representation."""
    if HAS_QRCODE:
        qr = qrcode.QRCode(version=3, box_size=15, border=4)
        qr.add_data(text)
        qr.make(fit=True)
        img_pil = qr.make_image(fill_color="black", back_color="white")
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    
    # Generate 400x400 canvas with simulated QR patterns
    img = np.ones((400, 400, 3), dtype=np.uint8) * 255
    # Draw finder patterns
    cv2.rectangle(img, (20, 20), (100, 100), (0, 0, 0), -1)
    cv2.rectangle(img, (30, 30), (90, 90), (255, 255, 255), -1)
    cv2.rectangle(img, (40, 40), (80, 80), (0, 0, 0), -1)
    
    cv2.rectangle(img, (300, 20), (380, 100), (0, 0, 0), -1)
    cv2.rectangle(img, (310, 30), (370, 90), (255, 255, 255), -1)
    cv2.rectangle(img, (320, 40), (360, 80), (0, 0, 0), -1)
    
    cv2.rectangle(img, (20, 300), (100, 380), (0, 0, 0), -1)
    cv2.rectangle(img, (30, 310), (90, 370), (255, 255, 255), -1)
    cv2.rectangle(img, (40, 320), (80, 360), (0, 0, 0), -1)
    
    np.random.seed(42)
    for y in range(120, 280, 10):
        for x in range(20, 380, 10):
            if np.random.rand() > 0.5:
                cv2.rectangle(img, (x, y), (x+10, y+10), (0, 0, 0), -1)
    for y in range(20, 120, 10):
        for x in range(120, 280, 10):
            if np.random.rand() > 0.5:
                cv2.rectangle(img, (x, y), (x+10, y+10), (0, 0, 0), -1)
    for y in range(280, 380, 10):
        for x in range(120, 380, 10):
            if np.random.rand() > 0.5:
                cv2.rectangle(img, (x, y), (x+10, y+10), (0, 0, 0), -1)
    return img
def decode_single_pass(img_proc, engine):
    """Runs a single decoding sweep using the selected engine."""
    if engine == "zxing" and HAS_ZXING:
        binarizers = [zxingcpp.Binarizer.LocalAverage, zxingcpp.Binarizer.GlobalHistogram]
        for b_type in binarizers:
            results = zxingcpp.read_barcodes(img_proc, binarizer=b_type)
            for res in results:
                if res.text:
                    return res.text, f"zxing:{res.format}"
                        
    elif engine == "pyzbar" and HAS_PYZBAR:
        decoded_objects = zbar_decode(img_proc)
        for obj in decoded_objects:
            qr_data = obj.data.decode('utf-8')
            if qr_data:
                return qr_data, "pyzbar"
                    
    elif engine == "opencv":
        detector = cv2.QRCodeDetector()
        data, bbox, _ = detector.detectAndDecode(img_proc)
        if data and bbox is not None and len(bbox) > 0:
            return data, "opencv"
                
    return None, None
def decode_frame_lazy(gray, engine, passes_config, use_gpu, orig_w, orig_h):
    """
    Optimized lazy-evaluative multi-pass scanning.
    Preprocesses and decodes sequentially, stopping immediately on success
    to minimize CPU load and PCIe bus transfer overhead.
    """
    # Pass 1: Direct Grayscale (Fastest path, no prep cost)
    t_start_dec = time.perf_counter()
    data, scan_type = decode_single_pass(gray, engine)
    t_end_dec = time.perf_counter()
    
    if data:
        return data, scan_type, 0.0, (t_end_dec - t_start_dec) * 1000.0, "Direct Grayscale"
        
    if passes_config == "single":
        return None, None, 0.0, (t_end_dec - t_start_dec) * 1000.0, "Direct Grayscale (Failed)"
        
    # Pass 2: White quiet zone padding
    t_start_pre = time.perf_counter()
    if use_gpu and HAS_CV_CUDA:
        gpu_gray = cv2.cuda.GpuMat()
        gpu_gray.upload(gray)
        gpu_pad = cv2.cuda.copyMakeBorder(gpu_gray, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
        pad = gpu_pad.download()
    else:
        pad = cv2.copyMakeBorder(gray, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    t_end_pre = time.perf_counter()
    
    t_start_dec = time.perf_counter()
    data, scan_type = decode_single_pass(pad, engine)
    t_end_dec = time.perf_counter()
    
    pre_ms = (t_end_pre - t_start_pre) * 1000.0
    dec_ms = (t_end_dec - t_start_dec) * 1000.0
    if data:
        return data, scan_type, pre_ms, dec_ms, "Padded Quiet Zone"
        
    # Pass 3: 1200px Resize + Padding
    t_start_pre = time.perf_counter()
    target_w = 1200
    scale = target_w / orig_w
    target_h = int(orig_h * scale)
    if use_gpu and HAS_CV_CUDA:
        gpu_gray = cv2.cuda.GpuMat()
        gpu_gray.upload(gray)
        gpu_resized = cv2.cuda.resize(gpu_gray, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        gpu_pad = cv2.cuda.copyMakeBorder(gpu_resized, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
        resized_padded = gpu_pad.download()
    else:
        resized = cv2.resize(gray, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        resized_padded = cv2.copyMakeBorder(resized, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255)
    t_end_pre = time.perf_counter()
    
    t_start_dec = time.perf_counter()
    data, scan_type = decode_single_pass(resized_padded, engine)
    t_end_dec = time.perf_counter()
    
    pre_ms += (t_end_pre - t_start_pre) * 1000.0
    dec_ms += (t_end_dec - t_start_dec) * 1000.0
    if data:
        return data, scan_type, pre_ms, dec_ms, "Resized 1200px + Padded"
        
    # Pass 4: Blur + Padding
    t_start_pre = time.perf_counter()
    if use_gpu and HAS_CV_CUDA:
        gpu_gray = cv2.cuda.GpuMat()
        gpu_gray.upload(gray)
        try:
            gpu_blur = cv2.cuda.createGaussianFilter(cv2.CV_8UC1, cv2.CV_8UC1, (3, 3), 0.5)
            gpu_blurred = gpu_blur.apply(gpu_gray)
            gpu_pad = cv2.cuda.copyMakeBorder(gpu_blurred, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
            med_padded = gpu_pad.download()
        except Exception:
            median_3 = cv2.medianBlur(gray, 3)
            med_padded = cv2.copyMakeBorder(median_3, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    else:
        median_3 = cv2.medianBlur(gray, 3)
        med_padded = cv2.copyMakeBorder(median_3, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    t_end_pre = time.perf_counter()
    
    t_start_dec = time.perf_counter()
    data, scan_type = decode_single_pass(med_padded, engine)
    t_end_dec = time.perf_counter()
    
    pre_ms += (t_end_pre - t_start_pre) * 1000.0
    dec_ms += (t_end_dec - t_start_dec) * 1000.0
    if data:
        return data, scan_type, pre_ms, dec_ms, "Blur Filter + Padded"
        
    # Pass 5: Bilateral Filtering + Padding
    t_start_pre = time.perf_counter()
    if use_gpu and HAS_CV_CUDA:
        gpu_gray = cv2.cuda.GpuMat()
        gpu_gray.upload(gray)
        try:
            gpu_bilat = cv2.cuda.bilateralFilter(gpu_gray, 5, 50, 50)
            gpu_pad = cv2.cuda.copyMakeBorder(gpu_bilat, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
            bilat_padded = gpu_pad.download()
        except Exception:
            bilat_5 = cv2.bilateralFilter(gray, 5, 50, 50)
            bilat_padded = cv2.copyMakeBorder(bilat_5, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    else:
        bilat_5 = cv2.bilateralFilter(gray, 5, 50, 50)
        bilat_padded = cv2.copyMakeBorder(bilat_5, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    t_end_pre = time.perf_counter()
    
    t_start_dec = time.perf_counter()
    data, scan_type = decode_single_pass(bilat_padded, engine)
    t_end_dec = time.perf_counter()
    
    pre_ms += (t_end_pre - t_start_pre) * 1000.0
    dec_ms += (t_end_dec - t_start_dec) * 1000.0
    if data:
        return data, scan_type, pre_ms, dec_ms, "Bilateral Filter + Padded"
        
    # Pass 6: CLAHE + Padding
    t_start_pre = time.perf_counter()
    if use_gpu and HAS_CV_CUDA:
        gpu_gray = cv2.cuda.GpuMat()
        gpu_gray.upload(gray)
        try:
            gpu_clahe = cv2.cuda.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gpu_clahe_img = gpu_clahe.apply(gpu_gray, cv2.cuda.Stream.Null())
            gpu_pad = cv2.cuda.copyMakeBorder(gpu_clahe_img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
            clahe_padded = gpu_pad.download()
        except Exception:
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            clahe_img = clahe.apply(gray)
            clahe_padded = cv2.copyMakeBorder(clahe_img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    else:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        clahe_img = clahe.apply(gray)
        clahe_padded = cv2.copyMakeBorder(clahe_img, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    t_end_pre = time.perf_counter()
    
    t_start_dec = time.perf_counter()
    data, scan_type = decode_single_pass(clahe_padded, engine)
    t_end_dec = time.perf_counter()
    
    pre_ms += (t_end_pre - t_start_pre) * 1000.0
    dec_ms += (t_end_dec - t_start_dec) * 1000.0
    if data:
        return data, scan_type, pre_ms, dec_ms, "CLAHE Contrast + Padded"
        
    # Pass 7: Adaptive Threshold + Padding
    t_start_pre = time.perf_counter()
    adaptive_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 5
    )
    thresh_padded = cv2.copyMakeBorder(adaptive_thresh, 30, 30, 30, 30, cv2.BORDER_CONSTANT, value=255)
    t_end_pre = time.perf_counter()
    
    t_start_dec = time.perf_counter()
    data, scan_type = decode_single_pass(thresh_padded, engine)
    t_end_dec = time.perf_counter()
    
    pre_ms += (t_end_pre - t_start_pre) * 1000.0
    dec_ms += (t_end_dec - t_start_dec) * 1000.0
    if data:
        return data, scan_type, pre_ms, dec_ms, "Adaptive Threshold + Padded"
        
    return None, None, pre_ms, dec_ms, "All 7 Passes Failed"
# ==========================================
# STREAMLIT USER INTERFACE
# ==========================================
# Premium Custom CSS Styling (Glassmorphism, dark mode, custom card margins)
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet">
    <style>
        * {
            font-family: 'Outfit', sans-serif;
        }
        .main {
            background-color: #0f0f11;
            color: #ffffff;
        }
        .metric-container {
            display: flex;
            flex-direction: column;
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 16px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(10px);
            margin-bottom: 12px;
            transition: all 0.2s ease-in-out;
        }
        .metric-container:hover {
            transform: translateY(-2px);
            border-color: rgba(9, 132, 227, 0.5);
            box-shadow: 0 4px 20px 0 rgba(9, 132, 227, 0.15);
        }
        .metric-label {
            font-size: 13px;
            color: #a0a0a5;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            font-weight: 600;
        }
        .metric-value {
            font-size: 26px;
            font-weight: 700;
            color: #00cec9;
            margin-top: 4px;
        }
        .metric-sub {
            font-size: 11px;
            color: #d1d1d6;
            margin-top: 2px;
        }
        .payload-card {
            background: rgba(85, 239, 196, 0.1);
            border-radius: 10px;
            padding: 14px;
            border: 1px solid rgba(85, 239, 196, 0.3);
            margin-top: 15px;
            color: #55efc4;
        }
    </style>
""", unsafe_allow_html=True)
# App Hero Header
st.markdown("""
    <div style='text-align: center; padding: 15px 0 25px 0;'>
        <h1 style='font-size: 38px; font-weight: 700; background: linear-gradient(135deg, #00cec9, #0984e3); -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
            Edge AI QR Scanner Benchmark ⚡
        </h1>
        <p style='color: #a0a0a5; font-size: 15px; font-weight: 300;'>
            Compare real-time frame rates and latency characteristics between the <strong>Raspberry Pi 5</strong> (CPU) and <strong>NVIDIA Jetson Orin Nano</strong> (GPU/CUDA).
        </p>
    </div>
""", unsafe_allow_html=True)
# Setup Sidebar Configuration Control Center
st.sidebar.markdown("<h3 style='color: #00cec9; margin-top: 0;'>Scan Settings</h3>", unsafe_allow_html=True)
camera_idx = st.sidebar.number_input("Camera Index", min_value=0, max_value=5, value=0, step=1)
engine_options = []
if HAS_ZXING: engine_options.append("zxing")
if HAS_PYZBAR: engine_options.append("pyzbar")
engine_options.append("opencv")
selected_engine = st.sidebar.selectbox("Decoding Engine", engine_options)
passes_config = st.sidebar.selectbox("Preprocessing Configuration", ["single", "multi"], index=1)
gpu_checkbox_disabled = not HAS_CV_CUDA
gpu_pre = st.sidebar.checkbox(
    "GPU Preprocessing (CUDA)", 
    value=HAS_CV_CUDA, 
    disabled=gpu_checkbox_disabled,
    help="Leverage OpenCV CUDA filters for the multi-pass preprocessing pipeline (Jetson Orin Nano only)."
)
if gpu_checkbox_disabled:
    st.sidebar.caption("⚠️ OpenCV CUDA is unavailable on this device (running CPU mode).")
st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='color: #0984e3;'>Platform Information</h3>", unsafe_allow_html=True)
# Hardware Diagnostic details in Sidebar
st.sidebar.text(f"OS Platform:   {sys.platform.upper()}")
st.sidebar.text(f"CPU Cores:     {os.cpu_count()}")
st.sidebar.text(f"Temp (Thermal): {get_cpu_temp()}")
st.sidebar.text(f"OpenCV CUDA:   {'Available' if HAS_CV_CUDA else 'Unavailable'}")
st.sidebar.text(f"Torch CUDA:    {'Available' if HAS_TORCH and torch.cuda.is_available() else 'Unavailable'}")
if HAS_TORCH and torch.cuda.is_available():
    st.sidebar.caption(f"GPU: {GPU_NAME}")
# Create Tabs
tab_live, tab_stress = st.tabs(["📹 Live Camera Scan", "⚡ Peak Performance Stress Test"])
# ==========================================
# TAB 1: LIVE VIDEO BENCHMARK STREAM
# ==========================================
with tab_live:
    run_stream = st.checkbox("▶️ Start Camera Stream & Benchmarking", value=False)
    if run_stream:
        # Set up layout structure
        col_video, col_metrics = st.columns([5, 3])
        
        with col_video:
            st.markdown("### 📹 Live Scan Feed")
            video_placeholder = st.empty()
            
        with col_metrics:
            st.markdown("### 📊 Performance Analytics")
            fps_metric = st.empty()
            lat_metric = st.empty()
            cpu_metric = st.empty()
            payload_metric = st.empty()
            
            st.markdown("#### Latency History (ms)")
            chart_placeholder = st.empty()
        # Open Camera Device
        cap = cv2.VideoCapture(camera_idx)
        if not cap.isOpened():
            st.error(f"Failed to open camera index {camera_idx}. Please verify camera connections.")
            st.stop()
            
        fps_history = []
        latency_history = []
        
        last_time = time.perf_counter()
        
        # Stream Loop
        while run_stream:
            ret, frame = cap.read()
            if not ret:
                st.warning("Failed to retrieve camera frame. Stream ended.")
                break
                
            orig_h, orig_w = frame.shape[:2]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Optimized Lazy Multi-pass pre-processing and decoding
            data, scan_type, pre_ms, dec_ms, active_pass = decode_frame_lazy(
                gray, selected_engine, passes_config, gpu_pre, orig_w, orig_h
            )
            e2e_ms = pre_ms + dec_ms
            
            # FPS Calculation
            now = time.perf_counter()
            fps = 1.0 / (now - last_time) if (now - last_time) > 0 else 0
            last_time = now
            
            # Store metrics history
            fps_history.append(fps)
            latency_history.append(e2e_ms)
            if len(fps_history) > 40:
                fps_history.pop(0)
                latency_history.pop(0)
                
            avg_fps = np.mean(fps_history)
            avg_lat = np.mean(latency_history)
            
            # Draw bounding boxes and text overlays on the frame
            frame_disp = frame.copy()
            if data:
                cv2.putText(
                    frame_disp, 
                    "QR DETECTED", 
                    (30, 45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.9, 
                    (0, 255, 0), 
                    3, 
                    cv2.LINE_AA
                )
                cv2.rectangle(frame_disp, (10, 10), (orig_w - 10, orig_h - 10), (0, 255, 0), 4)
                
            cv2.putText(
                frame_disp, 
                f"FPS: {fps:.1f} ({selected_engine.upper()})", 
                (30, orig_h - 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.8, 
                (255, 255, 255), 
                2, 
                cv2.LINE_AA
            )
            # Update Streamlit Screen Elements
            video_placeholder.image(frame_disp, channels="BGR", use_container_width=True)
            
            # Update Metric Panels using custom HTML/CSS
            fps_metric.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Throughput</div>
                    <div class="metric-value" style="color: #00cec9;">{avg_fps:.1f} FPS</div>
                    <div class="metric-sub">Pipelined execution frame-rate</div>
                </div>
            """, unsafe_allow_html=True)
            
            lat_metric.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">End-to-End Latency</div>
                    <div class="metric-value" style="color: #0984e3;">{avg_lat:.1f} ms</div>
                    <div class="metric-sub">Active Pass: {active_pass} (Preproc: {pre_ms:.1f}ms | Decode: {dec_ms:.1f}ms)</div>
                </div>
            """, unsafe_allow_html=True)
            
            if HAS_PSUTIL:
                cpu_load = psutil.cpu_percent()
                mem_load = psutil.virtual_memory().percent
                cpu_metric.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">System Load</div>
                        <div class="metric-value" style="color: #fdcb6e;">CPU: {cpu_load}% | MEM: {mem_load}%</div>
                        <div class="metric-sub">Thermal: {get_cpu_temp()}</div>
                    </div>
                """, unsafe_allow_html=True)
            else:
                cpu_metric.markdown(f"""
                    <div class="metric-container">
                        <div class="metric-label">System Load</div>
                        <div class="metric-value" style="color: #fdcb6e;">N/A</div>
                        <div class="metric-sub">Thermal: {get_cpu_temp()}</div>
                    </div>
                """, unsafe_allow_html=True)
                
            if data:
                payload_metric.markdown(f"""
                    <div class="payload-card">
                        <strong>🟢 Scanned Payload:</strong><br/>
                        <code style="word-break: break-all;">{data}</code>
                    </div>
                """, unsafe_allow_html=True)
            else:
                payload_metric.markdown(f"""
                    <div class="metric-container" style="background: rgba(255, 255, 255, 0.02); border-color: rgba(255, 255, 255, 0.05);">
                        <span style="color: #b2bec3; font-style: italic;">Align a QR or DataMatrix code in front of the lens...</span>
                    </div>
                """, unsafe_allow_html=True)
                
            # Draw moving chart
            chart_placeholder.line_chart(latency_history, height=180)
            
            time.sleep(0.01)
            
        cap.release()
    else:
        st.info("💡 Check the box above to start your camera stream and run live benchmarking.")
# ==========================================
# TAB 2: PEAK PERFORMANCE STRESS TEST BENCHMARK
# ==========================================
with tab_stress:
    st.markdown("### ⚡ Peak Performance Benchmarking Mode")
    st.write("Run a high-frequency stress benchmark loop (100 iterations) using a synthetic QR target to calculate peak hardware performance on this board.")
    
    col_bench_act, col_bench_img = st.columns([2, 1])
    
    with col_bench_img:
        # Show synthetic QR preview
        synthetic_img = generate_synthetic_qr()
        st.image(synthetic_img, caption="Synthetic Benchmark Target", width=250)
        
    with col_bench_act:
        bench_runs = st.slider("Benchmark Runs", min_value=10, max_value=200, value=100, step=10)
        bench_engine = st.selectbox("Benchmark Decoder", engine_options, key="bench_eng")
        
        run_cpu_bench = st.button("🚀 Run CPU Preprocessing Benchmark")
        run_gpu_bench = st.button("🔥 Run GPU CUDA Preprocessing Benchmark", disabled=not HAS_CV_CUDA)
        
    if run_cpu_bench or run_gpu_bench:
        use_gpu_bench = run_gpu_bench
        
        st.write(f"[*] Initializing stress test loop of **{bench_runs}** runs on **{'GPU (CUDA)' if use_gpu_bench else 'CPU'}** preprocessing...")
        
        gray = cv2.cvtColor(synthetic_img, cv2.COLOR_BGR2GRAY)
        orig_h, orig_w = gray.shape[:2]
        
        # Warmup
        st.write("[*] Warming up pipeline caches...")
        for _ in range(5):
            _, _, _, _, _ = decode_frame_lazy(gray, bench_engine, "multi", use_gpu_bench, orig_w, orig_h)
            
        # Run Stress Loop
        pre_times = []
        dec_times = []
        e2e_times = []
        success_count = 0
        
        progress_bar = st.progress(0)
        
        start_time = time.perf_counter()
        
        for i in range(bench_runs):
            t_start = time.perf_counter()
            data, _, pre_ms, dec_ms, _ = decode_frame_lazy(
                gray, bench_engine, "multi", use_gpu_bench, orig_w, orig_h
            )
            t_end = time.perf_counter()
            
            pre_times.append(pre_ms)
            dec_times.append(dec_ms)
            e2e_times.append((t_end - t_start) * 1000.0)
            
            if data:
                success_count += 1
                
            progress_bar.progress((i + 1) / bench_runs)
            
        end_time = time.perf_counter()
        total_time_s = end_time - start_time
        
        peak_fps = bench_runs / total_time_s
        avg_pre = np.mean(pre_times)
        avg_dec = np.mean(dec_times)
        avg_e2e = np.mean(e2e_times)
        
        st.success("Benchmark completed successfully!")
        
        # Display Results Cards
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1:
            st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Peak Throughput</div>
                    <div class="metric-value" style="color: #00cec9;">{peak_fps:.1f} FPS</div>
                    <div class="metric-sub">Processed {bench_runs} frames in {total_time_s:.2f}s</div>
                </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Mean E2E Latency</div>
                    <div class="metric-value" style="color: #0984e3;">{avg_e2e:.2f} ms</div>
                    <div class="metric-sub">Min: {np.min(e2e_times):.1f}ms | Max: {np.max(e2e_times):.1f}ms</div>
                </div>
            """, unsafe_allow_html=True)
            
        with m_col3:
            st.markdown(f"""
                <div class="metric-container">
                    <div class="metric-label">Preproc / Decode</div>
                    <div class="metric-value" style="color: #fdcb6e;">{avg_pre:.2f}ms / {avg_dec:.2f}ms</div>
                    <div class="metric-sub">Foil multi-pass check success: {(success_count/bench_runs)*100:.1f}%</div>
                </div>
            """, unsafe_allow_html=True)
            
        # Draw Bar Chart of the timing breakdown
        chart_data = pd.DataFrame({
            "Frame": list(range(1, bench_runs + 1)),
            "Preprocessing Latency (ms)": pre_times,
            "Decoding Latency (ms)": dec_times
        }).set_index("Frame")
        
        st.markdown("#### Latency Breakdown Profile (Preprocess vs. Decode)")
        st.bar_chart(chart_data)
