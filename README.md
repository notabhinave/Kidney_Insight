<div align="center">
  <h1>🩺 KidneyInsight AI v2</h1>
  <p><strong>Next-Level Renal Tumour Analysis powered by AI</strong></p>
  <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B.svg" alt="Streamlit">
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/OpenCV-Image%20Processing-5C3EE8.svg" alt="OpenCV">
</div>

---

## 📖 Overview

**KidneyInsight AI** is an advanced, AI-powered diagnostic application designed to assist in the analysis and detection of renal (kidney) tumours from CT scans. Built with a focus on clinical workflow, it features a multi-model ensemble system, DICOM file support, role-based access control, and comprehensive PDF report generation.

> **⚠️ Disclaimer:** This application is for educational and research purposes only. It is **not** a certified medical diagnostic device. Always consult a licensed medical professional.

---

## ✨ Key Features

### 🚀 Tier 1: Core Capabilities
- **Multi-Model Ensemble:** Loads multiple `.h5` models, averages predictions, and displays per-model scores alongside a disagreement indicator.
- **DICOM Support:** Accepts `.dcm` files natively (via `pydicom`), auto-windowed and converted for seamless analysis.
- **Severity Scoring:** Translates AI confidence into a 4-stage clinical severity scale.
- **Annotation & Visualization:** Highlights potential tumour regions using bounding boxes and Grad-CAM overlays.
- **Patient Records:** Attach patient name, age, ID, and clinical notes to every generated PDF report.

### 🌟 Tier 2: Advanced Capabilities
- **Batch Processing:** Upload multiple scans, analyze them simultaneously, and export a merged PDF.
- **Tumour Size Estimation:** Converts pixel measurements to millimeters using configurable CT Field-of-View (FOV).
- **Longitudinal Tracking:** Compare a single patient's scans over time.
- **Radiologist Feedback:** In-app thumbs up/down feedback system for AI predictions, stored in a local SQLite database.
- **Role-Based Access Control (RBAC):** Distinct roles (Admin, Radiologist, Viewer) with tailored capabilities.
- **Database Persistence & Audit Logging:** Immutable audit logs and scan history stored reliably in SQLite, ensuring continuity between sessions.

---

## 🛠️ Technology Stack

- **Frontend/UI:** [Streamlit](https://streamlit.io/)
- **Deep Learning:** [TensorFlow](https://www.tensorflow.org/) / Keras
- **Image Processing:** [OpenCV](https://opencv.org/), [Pillow](https://python-pillow.org/)
- **Medical Imaging:** [pydicom](https://pydicom.github.io/)
- **Data Manipulation:** [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/)
- **Report Generation:** [ReportLab](https://www.reportlab.com/)
- **Database:** SQLite

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Surya-K-Ratheesh/KidneyInsight.git
   cd KidneyInsight
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Add Models:**
   - Place your primary model as `best_model.h5` in the root directory.
   - For ensemble predictions, place additional `.h5` models in the `models/` folder.

5. **Run the application:**
   ```bash
   streamlit run app.py
   ```

---

## 🔐 Demo Accounts

Use the following credentials to explore different roles within the application:

| Role          | Username    | Password    |
|---------------|-------------|-------------|
| **Admin**     | admin       | `admin123`  |
| **Radiologist**| radiologist | `radio123`  |
| **Viewer**    | viewer      | `view123`   |

---

## 📊 Application Preview

1. **Dashboard:** Upload CT scans (JPG, PNG, DCM) and instantly view predictions, severity, and Grad-CAM bounding box overlays.
2. **Batch Analysis:** Analyze a folder of scans in one click.
3. **History & Logging:** Keep track of all past predictions and user actions in the built-in SQLite dashboard.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! 
Feel free to check out the [issues page](https://github.com/Surya-K-Ratheesh/KidneyInsight/issues).

---

<div align="center">
  <p>Built with ❤️ for the Medical AI Community</p>
</div>
