
# Patching Estimation in Highways 🛣️

> An intelligent, AI-driven solution for real-time pothole detection and cost estimation using YOLOv11 and semantic segmentation.

---

## 🧠 Project Overview

**Patching Estimation in Highways** is a deep learning-powered system that detects potholes in road surfaces from images or videos and calculates the estimated repair cost. This project leverages **YOLOv11** for object detection, **semantic segmentation** for precise area measurement, and **a cost estimation algorithm** to predict material and labor expenses.

---

## 🚀 Features

- 🔍 **YOLOv11-based Real-time Pothole Detection**
- 📐 **Semantic Segmentation** for dimension analysis
- 💸 **Automated Cost Estimation** (materials + labor)
- 📊 **Performance Metrics** (Precision, Recall, F1-Score)
- 🖼️ Annotated image outputs & CSV cost reports
- 📈 Cost graphs generated using `matplotlib`

---

## 📂 Project Structure

```
📦 PatchingEstimation
├── final_code.p             # Final Python script
├── data.yaml                # Dataset configuration file
├── test/                    # Test data folder
├── train/                   # Training data folder
├── valid/                   # Validation data folder
├── results/                 # Output results and annotations
├── runs/                    # YOLO run outputs
│   └── segment/             # Segmentation results and weights
└── README.md                # Project documentation
```

---

## 🛠️ Technologies Used

- **Python** (OpenCV, NumPy, Pandas, Matplotlib, Scikit-learn)
- **Deep Learning:** YOLOv11 (Ultralytics)
- **Computer Vision:** Semantic Segmentation
- **Cost Estimation Module:** Custom algorithm
- **IDE:** VS Code / Jupyter Notebook

---

## 📸 Sample Output

- Bounding boxes on detected potholes
- Real-world area estimation
- Cost breakdown per pothole
- Graphical report of costs

---

## 🧪 Installation & Setup

1. **Clone the repo**
```bash
git clone https://github.com/<your-username>/PatchingEstimation.git
cd PatchingEstimation
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Place input files**
- Put your test images/videos in the `data/` folder.

4. **Run the script**
```bash
final_code.py
```

---

## 📊 Output

- Annotated images: `results/annotated_<image>.jpg`
- CSV report: `results/report_<image>.csv`
- Graphs: `results/cost_graph_<image>.png`
- Accuracy Metrics: Precision, Recall, F1-score

---

## ⚙️ Cost Estimation Formula

```text
Material Cost = Area × Rate_per_sq_meter
Labor Cost = Time × Workers × Rate_per_hour
Total Cost = Material + Labor + Equipment + Overhead
```

---

## 🧑‍💻 Team Members

- Sandeep Yadav – [10800122197]
- SK Irfan – [10800122207]
- Alok Pandey – [10800121145]
- Abinash Acherjee – [10800122203]

**Supervisor:** Dr. Kailash Pati Mandal  
**Institution:** Asansol Engineering College

---

## 📚 References

- YOLOv11 - Ultralytics Docs
- Semantic Segmentation with CNNs
- Deep Learning for Infrastructure Monitoring

---

## 📌 Future Enhancements

- Integrate GPS for geo-tagging potholes
- Deploy as mobile/web application
- Expand to detect cracks and other road defects
- Add drone-based detection support

---

## 📄 License

This project is academic and research-based. Feel free to fork, improve, and build upon with proper citation.

---

## 💬 Acknowledgments

Special thanks to all faculty and lab assistants who supported the project. Gratitude to our parents for their constant motivation and support. 🙏

---

### 🚀 Making Roads Safer with AI!
