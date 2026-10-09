---
short_description: Elephant trajectory prediction and GIS risk assessment
---

# 🐘 Elephant Trajectory Prediction AI

This deployment uses the final working Gradio pipeline from the project notebook.

## Pipeline

1. YOLOv10 elephant detection
2. ByteTrack tracking
3. Movement feature extraction
4. TFT short-term trajectory prediction
5. GIS spatial risk assessment
6. Movement analytics
7. PDF report generation

## Folder structure

```text
Elephant_Prediction/
├── app.py
├── README.md
├── requirements.txt
├── packages.txt
└── models/
    ├── yolov/
    │   └── yolov10_elephants/
    │       └── weights/
    │           └── best.pt
    ├── gis/
    │   ├── combined_risk_zones.gpkg
    │   └── koundinya_study_area_75km.gpkg
    ├── multi_video_tft_trajectory_model_best.pt
    ├── multi_video_tft_feature_scaler.pkl
    └── multi_video_tft_target_scaler.pkl
```

The application uses paths relative to `app.py`; it does not depend on Google Drive or Colab paths.

## Hardware

A GPU such as a T4 is recommended for practical video inference.

## Run locally

```bash
pip install -r requirements.txt
python app.py
```
