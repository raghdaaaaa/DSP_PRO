import cv2
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import threading


# ══════════════════════════════════════════════
#  ELA Analysis
# ══════════════════════════════════════════════
def detect_tampering_ela(input_path, quality=90):
    original = cv2.imread(input_path)
    if original is None:
        raise ValueError("Image not found")

    temp_file = "temp_resaved.jpg"
    cv2.imwrite(temp_file, original, [cv2.IMWRITE_JPEG_QUALITY, quality])
    resaved = cv2.imread(temp_file)

    ela = cv2.absdiff(original, resaved)
    max_diff = np.max(ela)
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela = cv2.multiply(ela, scale)

    if os.path.exists(temp_file):
        os.remove(temp_file)

    return ela


# ══════════════════════════════════════════════
#  Copy-Move Detection
# ══════════════════════════════════════════════
def detect_copy_move(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) < 3:
        return image, 0

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors, descriptors, k=3)

    goodmatches = []
    for match in matches:
        m = match[1]
        n = match[2]
        if m.distance < 0.5 * n.distance and m.queryIdx != m.trainIdx:
            p1 = keypoints[m.queryIdx].pt
            p2 = keypoints[m.trainIdx].pt
            distance = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
            if distance > 5:
                goodmatches.append(m)

    result = cv2.drawMatches(
        image, keypoints, image, keypoints,
        goodmatches, None, flags=2
    )

    return result, len(goodmatches)


# ══════════════════════════════════════════════
#  ELA Full Analysis
# ══════════════════════════════════════════════
def run_ela_analysis(path):
    original = cv2.imread(path)
    ela = detect_tampering_ela(path, quality=90)
    gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
    ela_heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_JET)

    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    kernel = np.ones((3, 3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output = original.copy()
    for c in contours:
        if cv2.contourArea(c) > 100:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)

    suspicious_pixels = np.sum(thresh == 255)
    total_pixels = thresh.size
    ratio = suspicious_pixels / total_pixels
    decision = "Possibly Tampered" if ratio > 0.05 else "Likely Authentic"

    return original, ela_heatmap, thresh, output, ratio, decision


# ══════════════════════════════════════════════
#  Helper: cv2 image → Tk PhotoImage
# ══════════════════════════════════════════════
def cv2_to_tk(cv_img, size=(380, 280)):
    if len(cv_img.shape) == 2:
        cv_img = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2BGR)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    cv_img = cv2.resize(cv_img, size)
    pil_img = Image.fromarray(cv_img)
    return ImageTk.PhotoImage(pil_img)


# ══════════════════════════════════════════════
#  GUI
# ══════════════════════════════════════════════
class ForensicsApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔍 Image Forensics System")
        self.root.configure(bg="#0d0d0d")
        self.root.geometry("1100x780")
        self.root.resizable(False, False)

        self.image_path = None
        self.tk_images = []  # keep references

        self._build_ui()

    # ── UI Layout ──────────────────────────────
    def _build_ui(self):
        # ── Header
        header = tk.Frame(self.root, bg="#0d0d0d")
        header.pack(fill="x", padx=20, pady=(18, 0))

        tk.Label(header, text="IMAGE FORENSICS",
                 font=("Courier New", 22, "bold"),
                 fg="#00ff99", bg="#0d0d0d").pack(side="left")

        tk.Label(header, text="ELA + COPY-MOVE DETECTION",
                 font=("Courier New", 10),
                 fg="#444", bg="#0d0d0d").pack(side="left", padx=14, pady=6)

        # ── Control Bar
        ctrl = tk.Frame(self.root, bg="#111", pady=10)
        ctrl.pack(fill="x", padx=20, pady=10)

        self.path_var = tk.StringVar(value="No image selected...")
        tk.Label(ctrl, textvariable=self.path_var,
                 font=("Courier New", 9), fg="#666", bg="#111",
                 width=60, anchor="w").pack(side="left", padx=10)

        tk.Button(ctrl, text="📂  LOAD IMAGE",
                  font=("Courier New", 10, "bold"),
                  bg="#00ff99", fg="#0d0d0d",
                  activebackground="#00cc77",
                  relief="flat", cursor="hand2",
                  padx=14, pady=6,
                  command=self._load_image).pack(side="left", padx=6)

        tk.Button(ctrl, text="▶  ANALYZE",
                  font=("Courier New", 10, "bold"),
                  bg="#0066ff", fg="white",
                  activebackground="#0044cc",
                  relief="flat", cursor="hand2",
                  padx=14, pady=6,
                  command=self._run_analysis).pack(side="left", padx=6)

        # ── Notebook (tabs)
        style = ttk.Style()
        style.theme_use("default")
        style.configure("TNotebook", background="#0d0d0d", borderwidth=0)
        style.configure("TNotebook.Tab",
                        background="#1a1a1a", foreground="#666",
                        font=("Courier New", 10, "bold"),
                        padding=[16, 8])
        style.map("TNotebook.Tab",
                  background=[("selected", "#00ff99")],
                  foreground=[("selected", "#0d0d0d")])

        self.nb = ttk.Notebook(self.root)
        self.nb.pack(fill="both", expand=True, padx=20, pady=(0, 10))

        # Tab 1: ELA
        self.tab_ela = tk.Frame(self.nb, bg="#0d0d0d")
        self.nb.add(self.tab_ela, text="  ELA Analysis  ")
        self._build_ela_tab()

        # Tab 2: Copy-Move
        self.tab_cm = tk.Frame(self.nb, bg="#0d0d0d")
        self.nb.add(self.tab_cm, text="  Copy-Move Detection  ")
        self._build_cm_tab()

        # Tab 3: Summary
        self.tab_sum = tk.Frame(self.nb, bg="#0d0d0d")
        self.nb.add(self.tab_sum, text="  Summary  ")
        self._build_summary_tab()

        # ── Status bar
        self.status_var = tk.StringVar(value="Ready — load an image to begin")
        tk.Label(self.root, textvariable=self.status_var,
                 font=("Courier New", 9), fg="#444", bg="#0d0d0d",
                 anchor="w").pack(fill="x", padx=22, pady=(0, 8))

    # ── ELA Tab ───────────────────────────────
    def _build_ela_tab(self):
        grid = tk.Frame(self.tab_ela, bg="#0d0d0d")
        grid.pack(padx=10, pady=10)

        labels = ["Original", "ELA Heatmap", "Binary Mask", "Detected Regions"]
        self.ela_panels = []

        for i, label in enumerate(labels):
            col = i % 2
            row = i // 2
            frame = tk.Frame(grid, bg="#1a1a1a", padx=4, pady=4)
            frame.grid(row=row, column=col, padx=8, pady=8)

            tk.Label(frame, text=label,
                     font=("Courier New", 9, "bold"),
                     fg="#00ff99", bg="#1a1a1a").pack(anchor="w", padx=4)

            lbl = tk.Label(frame, bg="#111", width=380, height=280)
            lbl.pack()
            self.ela_panels.append(lbl)

    # ── Copy-Move Tab ─────────────────────────
    def _build_cm_tab(self):
        frame = tk.Frame(self.tab_cm, bg="#0d0d0d")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        tk.Label(frame, text="Match Visualization",
                 font=("Courier New", 9, "bold"),
                 fg="#00ff99", bg="#0d0d0d").pack(anchor="w", padx=4)

        self.cm_panel = tk.Label(frame, bg="#111")
        self.cm_panel.pack(fill="both", expand=True, padx=4, pady=4)

        self.cm_count_var = tk.StringVar(value="Good Matches: —")
        tk.Label(frame, textvariable=self.cm_count_var,
                 font=("Courier New", 11, "bold"),
                 fg="#0066ff", bg="#0d0d0d").pack(pady=6)

    # ── Summary Tab ───────────────────────────
    def _build_summary_tab(self):
        frame = tk.Frame(self.tab_sum, bg="#0d0d0d")
        frame.pack(expand=True)

        self.summary_labels = {}
        fields = [
            ("ELA Decision", "ela_decision"),
            ("Suspicious Ratio", "ela_ratio"),
            ("Copy-Move Matches", "cm_matches"),
            ("Overall Verdict", "verdict"),
        ]

        for i, (title, key) in enumerate(fields):
            tk.Label(frame, text=title + ":",
                     font=("Courier New", 12),
                     fg="#555", bg="#0d0d0d").grid(row=i, column=0,
                                                    sticky="e", pady=10, padx=20)
            var = tk.StringVar(value="—")
            color = "#00ff99" if key != "verdict" else "#ff4444"
            lbl = tk.Label(frame, textvariable=var,
                           font=("Courier New", 14, "bold"),
                           fg=color, bg="#0d0d0d")
            lbl.grid(row=i, column=1, sticky="w", pady=10, padx=10)
            self.summary_labels[key] = (var, lbl)

    # ── Actions ───────────────────────────────
    def _load_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
        )
        if path:
            self.image_path = path
            self.path_var.set(os.path.basename(path))
            self.status_var.set(f"Loaded: {path}")

    def _run_analysis(self):
        if not self.image_path:
            self.status_var.set("⚠  Please load an image first!")
            return
        self.status_var.set("⏳  Analyzing... please wait")
        self.root.update()
        threading.Thread(target=self._analyze_thread, daemon=True).start()

    def _analyze_thread(self):
        try:
            # ELA
            orig, heatmap, mask, detected, ratio, decision = run_ela_analysis(self.image_path)

            # Copy-Move
            cm_result, cm_count = detect_copy_move(self.image_path)

            self.root.after(0, lambda: self._update_ui(
                orig, heatmap, mask, detected,
                ratio, decision,
                cm_result, cm_count
            ))
        except Exception as e:
            self.root.after(0, lambda: self.status_var.set(f"❌ Error: {e}"))

    def _update_ui(self, orig, heatmap, mask, detected,
                   ratio, decision, cm_result, cm_count):
        self.tk_images.clear()

        # ELA panels
        imgs = [orig, heatmap, mask, detected]
        for panel, img in zip(self.ela_panels, imgs):
            tk_img = cv2_to_tk(img)
            self.tk_images.append(tk_img)
            panel.configure(image=tk_img)

        # Copy-Move panel (wide)
        cm_tk = cv2_to_tk(cm_result, size=(980, 400))
        self.tk_images.append(cm_tk)
        self.cm_panel.configure(image=cm_tk)
        self.cm_count_var.set(f"Good Matches Found: {cm_count}")

        # Summary
        self.summary_labels["ela_decision"][0].set(decision)
        self.summary_labels["ela_ratio"][0].set(f"{ratio:.4f}")
        self.summary_labels["cm_matches"][0].set(str(cm_count))

        tampered = decision == "Possibly Tampered" or cm_count > 10
        verdict = "⚠  TAMPERED" if tampered else "✅  LIKELY AUTHENTIC"
        verdict_color = "#ff4444" if tampered else "#00ff99"
        self.summary_labels["verdict"][0].set(verdict)
        self.summary_labels["verdict"][1].configure(fg=verdict_color)

        self.status_var.set(f"✅  Analysis complete — {os.path.basename(self.image_path)}")
        self.nb.select(2)  # jump to summary


# ══════════════════════════════════════════════
#  Entry Point
# ══════════════════════════════════════════════
if __name__ == "__main__":
    root = tk.Tk()
    app = ForensicsApp(root)
    root.mainloop()