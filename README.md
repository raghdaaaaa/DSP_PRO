# 🔍 Image Forensics Detector

A desktop application for detecting image tampering using two complementary techniques:
**Error Level Analysis (ELA)** and **SIFT-based Copy-Move Detection**.

---

## 🖥️ Features

- 📂 Load any JPG/PNG image
- 🔥 ELA Heatmap — reveals re-saved or edited regions
- 🔗 Copy-Move Detection — finds duplicated regions using SIFT keypoints
- 🧠 Automatic verdict: Likely Authentic / Possibly Tampered
- 🗂️ Clean tabbed GUI built with Tkinter

---

## 🚀 Getting Started

### Requirements
- Python 3.8+
- OpenCV
- Pillow

### Install dependencies
pip install opencv-python pillow

### Run
python image_forensics.py

---

## 🧪 How It Works

### ELA (Error Level Analysis)
Re-saves the image at a known JPEG quality, then computes the pixel difference.
Tampered regions retain higher error levels than the rest of the image.

### Copy-Move Detection
Uses SIFT to extract keypoints and descriptors, then matches them against
themselves using BFMatcher (k=3). Suspicious matches with spatial distance > 5px
are flagged as potential copy-move forgeries.

---

## 📸 Usage

1. Click **LOAD IMAGE** to select an image
2. Click **ANALYZE** to run both detections
3. Review results across the three tabs:
   - **ELA Analysis** — heatmap and detected regions
   - **Copy-Move Detection** — keypoint match visualization
   - **Summary** — final verdict

---

## 🛠️ Built With

- [OpenCV](https://opencv.org/) — image processing and SIFT
- [Tkinter](https://docs.python.org/3/library/tkinter.html) — GUI
- [Pillow](https://python-pillow.org/) — image rendering in GUI
- [NumPy](https://numpy.org/) — numerical operations

---

## 📁 Project Structure

image-forensics-detector/
│
├── image_forensics.py   # Main application
├── README.md            
└── test.jpg             # Sample image (optional)

---

## ⚠️ Limitations

- ELA works best on JPEG images
- Copy-Move detection may miss heavily post-processed forgeries
- Results are indicative, not conclusive

---

## 📜 License
MIT License
