import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import threading

# Image Processing Logic
#ELA
#__________________________

def detect_tampering_ela(input_path, quality=90):
    original = cv2.imread(input_path)
    if original is None:
        raise ValueError("Image not found")

    temp_file = "temp_resaved.jpg"
    cv2.imwrite(temp_file, original, [cv2.IMWRITE_JPEG_QUALITY, quality])
    resaved = cv2.imread(temp_file)

    ela = cv2.absdiff(original, resaved)
    max_diff = np.max(ela)
    if max_diff == 0: max_diff = 1
    ela = cv2.multiply(ela, 255.0 / max_diff)

    if os.path.exists(temp_file):
        os.remove(temp_file)
    return original, ela

# COPY_MOVE
#___________________

def detect_copy_move(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) < 10:
        return image, 0

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors, descriptors, k=3)
    goodmatches = []
    for m_list in matches:
        if len(m_list) < 3: continue
        m, n = m_list[1], m_list[2]
        if m.distance < 0.6 * n.distance:
            p1, p2 = keypoints[m.queryIdx].pt, keypoints[m.trainIdx].pt
            dist = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
            if dist > 30:
                goodmatches.append(m)

    result = cv2.drawMatches(image, keypoints, image, keypoints, goodmatches, None, flags=2)
    return result, len(goodmatches)

# DETECT NOISE
#__________________________________________
def detect_noise_inconsistency(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)

    #NOISE DIV
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    noise_map = cv2.absdiff(gray, blurred)

    block_size = 32
    h, w = noise_map.shape
    block_stds = []
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            block = noise_map[y:y+block_size, x:x+block_size]
            block_stds.append(np.std(block))

    if not block_stds:
        return image, image, 0.0

    mean_std = np.mean(block_stds)
    threshold = mean_std * 2.0  

    # FAKE AEREA
    result = image.copy()
    suspicious_blocks = 0
    idx = 0
    for y in range(0, h - block_size, block_size):
        for x in range(0, w - block_size, block_size):
            if block_stds[idx] > threshold:
                cv2.rectangle(result, (x, y), (x+block_size, y+block_size), (0, 255, 255), 1)
                suspicious_blocks += 1
            idx += 1

    # COLORED NOISE
    noise_visual = cv2.normalize(noise_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    noise_colored = cv2.applyColorMap(noise_visual, cv2.COLORMAP_HOT)

    noise_ratio = suspicious_blocks / len(block_stds) if block_stds else 0
    return noise_colored, result, noise_ratio

def detect_jpeg_ghost(image_path):
    original = cv2.imread(image_path).astype(np.float32)
    h, w = original.shape[:2]
    ghost_map = np.zeros((h, w), dtype=np.float32)

    for quality in [60, 70, 80]:
        temp = "temp_ghost.jpg"
        cv2.imwrite(temp, original, [cv2.IMWRITE_JPEG_QUALITY, quality])
        compressed = cv2.imread(temp).astype(np.float32)
        diff = np.mean(np.abs(original - compressed), axis=2)
        ghost_map += (diff < np.mean(diff) * 0.5).astype(np.float32)
        if os.path.exists(temp):
            os.remove(temp)

    # normalize 
    ghost_norm = cv2.normalize(ghost_map, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    ghost_colored = cv2.applyColorMap(ghost_norm, cv2.COLORMAP_COOL)

    _, ghost_thresh = cv2.threshold(ghost_norm, 180, 255, cv2.THRESH_BINARY)
    ghost_ratio = np.sum(ghost_thresh > 0) / ghost_thresh.size

    result = cv2.imread(image_path)
    contours, _ = cv2.findContours(ghost_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) > 500:
            x, y, ww, hh = cv2.boundingRect(c)
            cv2.rectangle(result, (x, y), (x+ww, y+hh), (255, 0, 255), 2)

    return ghost_colored, result, ghost_ratio

# RUN
def run_full_analysis(path):
    original, ela = detect_tampering_ela(path)
    gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    ela_heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    _, thresh = cv2.threshold(gray, 25, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detection_img = original.copy()
    suspicious_area = 0
    for c in contours:
        area = cv2.contourArea(c)
        if area > 200:
            suspicious_area += area
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(detection_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    ratio = suspicious_area / thresh.size
    decision = "Possibly Tampered" if ratio > 0.08 else "Likely Authentic"

    return original, ela_heatmap, thresh, detection_img, ratio, decision

# GUI
#____________________________________
class ForensicsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔍 Digital Image Forensics Expert")
        self.root.geometry("1150x850")
        self.root.configure(bg="#0d0d0d")

        self.image_path = None
        self.tk_images = []

        self._setup_styles()
        self._build_header()
        self._build_controls()
        self._build_tabs()
        self._build_statusbar()

    def _setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TNotebook", background="#0d0d0d", borderwidth=0)
        style.configure("TNotebook.Tab", font=("Segoe UI", 10, "bold"), padding=[15, 5], background="#1a1a1a", foreground="#888")
        style.map("TNotebook.Tab", background=[("selected", "#00ff99")], foreground=[("selected", "#000")])

    def _build_header(self):
        header = tk.Frame(self.root, bg="#0d0d0d", pady=20)
        header.pack(fill="x")
        tk.Label(header, text="IMAGE FORENSICS SYSTEM", font=("Impact", 28), fg="#00ff99", bg="#0d0d0d").pack()
        tk.Label(header, text="ELA | Copy-Move | Noise | JPEG Ghost Detection", font=("Consolas", 10), fg="#555", bg="#0d0d0d").pack()

    def _build_controls(self):
        ctrl_frame = tk.Frame(self.root, bg="#151515", padx=20, pady=10)
        ctrl_frame.pack(fill="x", padx=20, pady=5)

        self.path_var = tk.StringVar(value="Please select an image file to start...")
        tk.Label(ctrl_frame, textvariable=self.path_var, font=("Arial", 9), fg="#aaa", bg="#151515", width=70, anchor="w").pack(side="left")

        tk.Button(ctrl_frame, text="📂 OPEN IMAGE", command=self._load_image, bg="#333", fg="white", font=("Arial", 9, "bold"), borderwidth=0, padx=15, pady=5, cursor="hand2").pack(side="left", padx=10)
        tk.Button(ctrl_frame, text="🚀 ANALYZE", command=self._start_analysis, bg="#0066ff", fg="white", font=("Arial", 9, "bold"), borderwidth=0, padx=20, pady=5, cursor="hand2").pack(side="left")

    def _build_tabs(self):
        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=20, pady=10)

        # ── Tab 1: ELA ──
        self.tab_ela = tk.Frame(self.nb, bg="#0d0d0d")
        self.nb.add(self.tab_ela, text=" ERROR LEVEL ANALYSIS ")
        self.ela_grid = tk.Frame(self.tab_ela, bg="#0d0d0d")
        self.ela_grid.pack(expand=True)
        self.ela_panels = []
        for i, title in enumerate(["Original Image", "ELA Heatmap", "Binary Mask", "Detected Tampering"]):
            frame = tk.Frame(self.ela_grid, bg="#111", padx=5, pady=5, highlightbackground="#222", highlightthickness=1)
            frame.grid(row=i//2, column=i%2, padx=10, pady=10)
            tk.Label(frame, text=title, font=("Arial", 8, "bold"), fg="#00ff99", bg="#111").pack()
            lbl = tk.Label(frame, bg="#000", width=400, height=300)
            lbl.pack()
            self.ela_panels.append(lbl)

        # ── Tab 2: Copy-Move ──
        self.tab_cm = tk.Frame(self.nb, bg="#0d0d0d")
        self.nb.add(self.tab_cm, text=" COPY-MOVE DETECTION ")
        self.cm_display = tk.Label(self.tab_cm, bg="#000", relief="flat")
        self.cm_display.pack(fill="both", expand=True, padx=20, pady=20)
        self.cm_info = tk.Label(self.tab_cm, text="Matches: -", font=("Consolas", 12), fg="#00ff99", bg="#0d0d0d")
        self.cm_info.pack(pady=5)

        # ── Tab 3: Noise & Ghost ──
        self.tab_ng = tk.Frame(self.nb, bg="#0d0d0d")
        self.nb.add(self.tab_ng, text=" NOISE & JPEG GHOST ")
        self.ng_grid = tk.Frame(self.tab_ng, bg="#0d0d0d")
        self.ng_grid.pack(expand=True)
        self.ng_panels = []
        for i, title in enumerate(["Noise Map", "Noise Anomalies", "JPEG Ghost Map", "Ghost Regions"]):
            frame = tk.Frame(self.ng_grid, bg="#111", padx=5, pady=5, highlightbackground="#222", highlightthickness=1)
            frame.grid(row=i//2, column=i%2, padx=10, pady=10)
            tk.Label(frame, text=title, font=("Arial", 8, "bold"), fg="#00ff99", bg="#111").pack()
            lbl = tk.Label(frame, bg="#000", width=400, height=300)
            lbl.pack()
            self.ng_panels.append(lbl)
        self.ng_info = tk.Label(self.tab_ng, text="Noise Ratio: - | Ghost Ratio: -", font=("Consolas", 11), fg="#00ff99", bg="#0d0d0d")
        self.ng_info.pack(pady=5)

        # ── Tab 4: Final Verdict ──
        self.tab_sum = tk.Frame(self.nb, bg="#0d0d0d")
        self.nb.add(self.tab_sum, text=" FINAL VERDICT ")
        self.res_vars = {
            "ela_dec":  tk.StringVar(value="-"),
            "ela_rat":  tk.StringVar(value="-"),
            "cm_match": tk.StringVar(value="-"),
            "noise_r":  tk.StringVar(value="-"),
            "ghost_r":  tk.StringVar(value="-"),
            "final":    tk.StringVar(value="WAITING")
        }
        summary_box = tk.Frame(self.tab_sum, bg="#111", padx=50, pady=40)
        summary_box.place(relx=0.5, rely=0.5, anchor="center")
        lbl_style = {"bg": "#111", "font": ("Arial", 12)}

        tk.Label(summary_box, text="ELA DECISION:",         fg="#777", **lbl_style).grid(row=0, column=0, pady=8, sticky="e")
        tk.Label(summary_box, textvariable=self.res_vars["ela_dec"],  fg="#fff", **lbl_style).grid(row=0, column=1, padx=20, sticky="w")

        tk.Label(summary_box, text="SUSPICIOUS RATIO:",     fg="#777", **lbl_style).grid(row=1, column=0, pady=8, sticky="e")
        tk.Label(summary_box, textvariable=self.res_vars["ela_rat"],  fg="#fff", **lbl_style).grid(row=1, column=1, padx=20, sticky="w")

        tk.Label(summary_box, text="COPY-MOVE KEYPOINTS:",  fg="#777", **lbl_style).grid(row=2, column=0, pady=8, sticky="e")
        tk.Label(summary_box, textvariable=self.res_vars["cm_match"], fg="#fff", **lbl_style).grid(row=2, column=1, padx=20, sticky="w")

        tk.Label(summary_box, text="NOISE ANOMALY RATIO:",  fg="#777", **lbl_style).grid(row=3, column=0, pady=8, sticky="e")
        tk.Label(summary_box, textvariable=self.res_vars["noise_r"],  fg="#fff", **lbl_style).grid(row=3, column=1, padx=20, sticky="w")

        tk.Label(summary_box, text="JPEG GHOST RATIO:",     fg="#777", **lbl_style).grid(row=4, column=0, pady=8, sticky="e")
        tk.Label(summary_box, textvariable=self.res_vars["ghost_r"],  fg="#fff", **lbl_style).grid(row=4, column=1, padx=20, sticky="w")

        self.final_lbl = tk.Label(summary_box, textvariable=self.res_vars["final"], font=("Arial", 24, "bold"), bg="#111", pady=20)
        self.final_lbl.grid(row=5, column=0, columnspan=2)

    def _build_statusbar(self):
        self.status_var = tk.StringVar(value="System Ready")
        tk.Label(self.root, textvariable=self.status_var, bd=1, relief="flat", anchor="w",
                 bg="#1a1a1a", fg="#666", font=("Arial", 8), padx=20).pack(fill="x", side="bottom")

    #  Helper & Events

    def _load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if file_path:
            self.image_path = file_path
            self.path_var.set(os.path.basename(file_path))
            self.status_var.set(f"Selected: {file_path}")

    def _start_analysis(self):
        if not self.image_path:
            self.status_var.set("❌ Please load an image first!")
            return
        self.status_var.set("⏳ Processing... please wait")
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            orig, heat, mask, det, ratio, ela_dec = run_full_analysis(self.image_path)
            cm_img, cm_cnt                         = detect_copy_move(self.image_path)
            noise_map, noise_det, noise_ratio      = detect_noise_inconsistency(self.image_path)
            ghost_map, ghost_det, ghost_ratio      = detect_jpeg_ghost(self.image_path)

            self.root.after(0, lambda: self._update_results(
                orig, heat, mask, det, ratio, ela_dec,
                cm_img, cm_cnt,
                noise_map, noise_det, noise_ratio,
                ghost_map, ghost_det, ghost_ratio
            ))
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"❌ Error: {str(e)}"))

    def _update_results(self, orig, heat, mask, det, ratio, ela_dec,
                        cm_img, cm_cnt,
                        noise_map, noise_det, noise_ratio,
                        ghost_map, ghost_det, ghost_ratio):
        self.tk_images = []

        # ELA panels
        for panel, img in zip(self.ela_panels, [orig, heat, mask, det]):
            tk_img = self._cv2_to_tk(img, (400, 300))
            self.tk_images.append(tk_img)
            panel.config(image=tk_img)

        # Copy-Move
        cm_tk = self._cv2_to_tk(cm_img, (900, 450))
        self.tk_images.append(cm_tk)
        self.cm_display.config(image=cm_tk)
        self.cm_info.config(text=f"Copy-Move Matches Found: {cm_cnt}")

        # Noise & Ghost panels
        for panel, img in zip(self.ng_panels, [noise_map, noise_det, ghost_map, ghost_det]):
            tk_img = self._cv2_to_tk(img, (400, 300))
            self.tk_images.append(tk_img)
            panel.config(image=tk_img)
        self.ng_info.config(text=f"Noise Anomaly Ratio: {noise_ratio*100:.1f}%  |  Ghost Ratio: {ghost_ratio*100:.1f}%")

        # Summary
        self.res_vars["ela_dec"].set(ela_dec)
        self.res_vars["ela_rat"].set(f"{ratio*100:.2f}%")
        self.res_vars["cm_match"].set(str(cm_cnt))
        self.res_vars["noise_r"].set(f"{noise_ratio*100:.1f}%")
        self.res_vars["ghost_r"].set(f"{ghost_ratio*100:.1f}%")

        # tampered
        signals = [
            ela_dec == "Possibly Tampered" and ratio > 0.08,
            cm_cnt > 40,
            noise_ratio > 0.30,
            ghost_ratio > 0.25,
        ]
        is_tampered = sum(signals) >= 2

        self.res_vars["final"].set("🚨 TAMPERED" if is_tampered else "✅ AUTHENTIC")
        self.final_lbl.config(fg="#ff4444" if is_tampered else "#00ff99")
        self.status_var.set("✅ Analysis Complete")
        self.nb.select(3)  

    def _cv2_to_tk(self, img, size):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size)
        return ImageTk.PhotoImage(Image.fromarray(img))

#MAIN
#__________________________
if __name__ == "__main__":
    root = tk.Tk()
    # root.iconbitmap("icon.ico")
    app = ForensicsApp(root)
    root.mainloop()