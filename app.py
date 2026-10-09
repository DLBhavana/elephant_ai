# ============================================================
# ELEPHANT TRAJECTORY PREDICTION AI
# Deployment version of the final working Gradio pipeline
# ============================================================

import os
import cv2
import base64
import uuid
import subprocess
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import gradio as gr
import torch
import torch.nn as nn
import joblib

from ultralytics import YOLO

# ------------------------------------------------------------
# Deployment paths
# ------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HERO_IMAGE_PATH = os.path.join(
    BASE_DIR,
    "assets",
    "elephant_hero.jpg"
)
print("Hero image path:", HERO_IMAGE_PATH)
print("Hero image exists:", os.path.exists(HERO_IMAGE_PATH))
with open(HERO_IMAGE_PATH, "rb") as f:
    HERO_IMAGE_BASE64 = base64.b64encode(f.read()).decode("utf-8")

HERO_IMAGE_URL = f"data:image/jpeg;base64,{HERO_IMAGE_BASE64}"

MODEL_DIR = os.path.join(BASE_DIR, "models")
YOLO_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "yolov",
    "yolov10s_elephant",
    "weights",
    "best.pt"
)

TFT_MODEL_PATH = os.path.join(
    MODEL_DIR,
    "multi_video_tft_trajectory_model_best.pt"
)

FEATURE_SCALER_PATH = os.path.join(
    MODEL_DIR,
    "multi_video_tft_feature_scaler.pkl"
)

TARGET_SCALER_PATH = os.path.join(
    MODEL_DIR,
    "multi_video_tft_target_scaler.pkl"
)

GIS_DIR = os.path.join(MODEL_DIR, "gis")

RISK_GPKG = os.path.join(
    GIS_DIR,
    "combined_risk_zones.gpkg"
)

STUDY_GPKG = os.path.join(
    GIS_DIR,
    "koundinya_study_area_75km.gpkg"
)

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------------------------------------------------------
# Validate required deployment files
# ------------------------------------------------------------
required_files = [
    YOLO_MODEL_PATH,
    TFT_MODEL_PATH,
    FEATURE_SCALER_PATH,
    TARGET_SCALER_PATH,
    RISK_GPKG,
    STUDY_GPKG,
]

missing_files = [p for p in required_files if not os.path.exists(p)]

if missing_files:
    raise FileNotFoundError(
        "Required deployment files are missing:\n" +
        "\n".join(missing_files)
    )

# ------------------------------------------------------------
# Device
# ------------------------------------------------------------
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("========================================")
print("🐘 Elephant Trajectory Prediction AI")
print("========================================")
print("Device:", device)

# ------------------------------------------------------------
# YOLOv10
# ------------------------------------------------------------
yolo_model = YOLO(YOLO_MODEL_PATH)

if torch.cuda.is_available():
    yolo_model.to("cuda")

print("✅ YOLOv10 loaded")

# ------------------------------------------------------------
# TFT scalers
# ------------------------------------------------------------
feature_scaler = joblib.load(FEATURE_SCALER_PATH)
target_scaler = joblib.load(TARGET_SCALER_PATH)

print("✅ TFT scalers loaded")

# ------------------------------------------------------------
# Exact trained TFT architecture
# ------------------------------------------------------------
checkpoint = torch.load(
    TFT_MODEL_PATH,
    map_location="cpu",
    weights_only=False
)

input_size = checkpoint["input_size"]
hidden_size = checkpoint["hidden_size"]
num_heads = checkpoint["num_heads"]
dropout = checkpoint["dropout"]
sequence_length = checkpoint["sequence_length"]


class TrajectoryTFT(nn.Module):

    def __init__(
        self,
        input_size,
        hidden_size,
        num_heads,
        dropout
    ):
        super().__init__()

        self.input_projection = nn.Linear(
            input_size,
            hidden_size
        )

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm = nn.LayerNorm(
            hidden_size
        )

        self.output_layer = nn.Linear(
            hidden_size,
            2
        )

    def forward(self, x):

        x = self.input_projection(x)

        lstm_out, _ = self.lstm(x)

        attention_out, _ = self.attention(
            lstm_out,
            lstm_out,
            lstm_out
        )

        x = lstm_out + attention_out

        x = self.norm(x)

        x = x[:, -1, :]

        return self.output_layer(x)


tft_model = TrajectoryTFT(
    input_size=input_size,
    hidden_size=hidden_size,
    num_heads=num_heads,
    dropout=dropout
)

tft_model.load_state_dict(
    checkpoint["model_state_dict"]
)

tft_model = tft_model.to(device)
tft_model.eval()

print("✅ TFT model loaded")
print("Sequence length:", sequence_length)


# ============================================================
# PDF REPORT GENERATION - CORRECTED
# ============================================================

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    PageBreak
)
import os


def generate_pdf_report(
    report_path,
    run_id,
    fps,
    total_frames,
    video_width,
    video_height,
    total_detections,
    unique_ids,
    predictions_df,
    risk_df,
    analytics_df,
    trajectory_path,
    gis_map_path,
    analytics_path
):

    print("========================================")
    print("Generating PDF report...")
    print("========================================")

    doc = SimpleDocTemplate(
        report_path,
        pagesize=A4,
        rightMargin=40,
        leftMargin=40,
        topMargin=40,
        bottomMargin=40
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ReportTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontSize=18,
        spaceAfter=12
    )

    subtitle_style = ParagraphStyle(
        "ReportSubtitle",
        parent=styles["BodyText"],
        alignment=TA_CENTER,
        fontSize=11,
        spaceAfter=15
    )

    heading_style = ParagraphStyle(
        "ReportHeading",
        parent=styles["Heading2"],
        fontSize=13,
        spaceBefore=12,
        spaceAfter=8
    )

    body_style = ParagraphStyle(
        "ReportBody",
        parent=styles["BodyText"],
        fontSize=9,
        leading=13
    )

    story = []

    # ========================================================
    # TITLE
    # ========================================================

    story.append(
        Paragraph(
            "Elephant Trajectory Prediction and GIS Risk Assessment",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Koundinya Wildlife Sanctuary Study Area",
            subtitle_style
        )
    )

    story.append(
        Paragraph(
            f"<b>Analysis ID:</b> {run_id}",
            body_style
        )
    )

    story.append(Spacer(1, 12))

    # ========================================================
    # 1. VIDEO INFORMATION
    # ========================================================

    story.append(
        Paragraph(
            "1. Video Information",
            heading_style
        )
    )

    video_data = [
        ["Parameter", "Value"],
        ["FPS", f"{fps:.2f}"],
        ["Total Frames", str(total_frames)],
        ["Resolution", f"{video_width} x {video_height}"]
    ]

    table = Table(
        video_data,
        colWidths=[180, 280]
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ])
    )

    story.append(table)
    story.append(Spacer(1, 12))

    # ========================================================
    # 2. DETECTION AND TRACKING
    # ========================================================

    story.append(
        Paragraph(
            "2. Detection and Tracking",
            heading_style
        )
    )

    detection_data = [
        ["Parameter", "Value"],
        ["Detection Model", "YOLOv10"],
        ["Tracking Algorithm", "ByteTrack"],
        ["Total Detections", str(total_detections)],
        ["Tracked Elephants", str(len(unique_ids))]
    ]

    table = Table(
        detection_data,
        colWidths=[180, 280]
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ])
    )

    story.append(table)
    story.append(Spacer(1, 12))

    # ========================================================
    # 3. TFT
    # ========================================================

    story.append(
        Paragraph(
            "3. TFT Trajectory Prediction",
            heading_style
        )
    )

    tft_data = [
        ["Parameter", "Value"],
        ["Model", "Temporal Fusion Transformer (TFT)"],
        ["Sequence Length", "20"],
        ["Prediction Windows", str(len(predictions_df))]
    ]

    table = Table(
        tft_data,
        colWidths=[180, 280]
    )

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 9),
        ])
    )

    story.append(table)
    story.append(Spacer(1, 12))

    # ========================================================
    # 4. LATEST PREDICTIONS
    # ========================================================

    story.append(
        Paragraph(
            "4. Latest TFT Predictions",
            heading_style
        )
    )

    if not predictions_df.empty:

        latest = (
            predictions_df
            .sort_values("end_frame")
            .groupby("track_id")
            .tail(1)
            .copy()
        )

        prediction_data = [
            [
                "Track ID",
                "Current X",
                "Current Y",
                "Predicted X",
                "Predicted Y"
            ]
        ]

        for _, row in latest.iterrows():

            prediction_data.append([
                str(int(row["track_id"])),
                f'{row["current_x"]:.2f}',
                f'{row["current_y"]:.2f}',
                f'{row["predicted_x"]:.2f}',
                f'{row["predicted_y"]:.2f}'
            ])

        table = Table(
            prediction_data,
            colWidths=[70, 90, 90, 90, 90]
        )

        table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
            ])
        )

        story.append(table)

    story.append(Spacer(1, 12))

    # ========================================================
    # 5. GIS RISK
    # ========================================================

    story.append(
        Paragraph(
            "5. GIS Risk Assessment",
            heading_style
        )
    )

    if not risk_df.empty:

        counts = risk_df["risk_level"].value_counts()

        risk_data = [
            ["Risk Level", "Count"],
            ["LOW", str(int(counts.get("LOW", 0)))],
            ["MEDIUM", str(int(counts.get("MEDIUM", 0)))],
            ["HIGH", str(int(counts.get("HIGH", 0)))],
            ["CRITICAL", str(int(counts.get("CRITICAL", 0)))],
            [
                "OUTSIDE DEFINED RISK ZONE",
                str(
                    int(
                        counts.get(
                            "OUTSIDE_DEFINED_RISK_ZONE",
                            0
                        )
                    )
                )
            ]
        ]

        table = Table(
            risk_data,
            colWidths=[280, 180]
        )

        table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
            ])
        )

        story.append(table)

    story.append(PageBreak())

    # ========================================================
    # 6. TRAJECTORY
    # ========================================================

    story.append(
        Paragraph(
            "6. Elephant Trajectory",
            heading_style
        )
    )

    if os.path.exists(trajectory_path):

        img = RLImage(
            trajectory_path,
            width=6.8 * inch,
            height=4.5 * inch
        )

        story.append(img)

    story.append(PageBreak())

    # ========================================================
    # 7. GIS MAP
    # ========================================================

    story.append(
        Paragraph(
            "7. GIS Risk Map",
            heading_style
        )
    )

    if os.path.exists(gis_map_path):

        img = RLImage(
            gis_map_path,
            width=6.8 * inch,
            height=5.1 * inch
        )

        story.append(img)

    story.append(PageBreak())

    # ========================================================
    # 8. MOVEMENT ANALYTICS
    # ========================================================

    story.append(
        Paragraph(
            "8. Movement Analytics",
            heading_style
        )
    )

    if os.path.exists(analytics_path):

        img = RLImage(
            analytics_path,
            width=6.8 * inch,
            height=4.2 * inch
        )

        story.append(img)

    story.append(Spacer(1, 15))

    # ========================================================
    # 9. TRACK ANALYTICS
    # ========================================================

    story.append(
        Paragraph(
            "9. Track Analytics",
            heading_style
        )
    )

    if analytics_df is not None and not analytics_df.empty:

        columns = list(analytics_df.columns)

        analytics_data = [
            [str(c) for c in columns]
        ]

        for _, row in analytics_df.iterrows():

            analytics_data.append([
                str(row[c])
                for c in columns
            ])

        table = Table(
            analytics_data,
            repeatRows=1
        )

        table.setStyle(
            TableStyle([
                ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
            ])
        )

        story.append(table)

    story.append(Spacer(1, 20))

    # ========================================================
    # FINAL STATUS
    # ========================================================

    story.append(
        Paragraph(
            "10. Pipeline Status",
            heading_style
        )
    )

    story.append(
        Paragraph(
            "YOLOv10 - Completed<br/>"
            "ByteTrack - Completed<br/>"
            "TFT Trajectory Prediction - Completed<br/>"
            "GIS Risk Assessment - Completed<br/>"
            "Movement Analytics - Completed",
            body_style
        )
    )

    # ========================================================
    # BUILD PDF
    # ========================================================

    doc.build(story)

    print("========================================")
    print("PDF REPORT GENERATED SUCCESSFULLY")
    print("========================================")
    print("Path:", report_path)
    print(
        "Size:",
        round(
            os.path.getsize(report_path) / (1024 * 1024),
            2
        ),
        "MB"
    )

    return report_path

# ============================================================
# FINAL OPTIMIZED GRADIO PIPELINE
# ONE YOLO RUN + REAL GRADIO PROGRESS
# ============================================================

import os
import cv2
import uuid
import subprocess
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import gradio as gr


# ============================================================
# PATHS
# ============================================================

# Deployment paths are defined at the top of this file.


# ============================================================
# GIS SETTINGS
# ============================================================

UTM_CRS = "EPSG:32644"

# ============================================================
# TFT SETTINGS
# ============================================================

sequence_length = 20

print("TFT sequence length:", sequence_length)

STUDY_MIN_X = 120504.4986
STUDY_MIN_Y = 1317007.2849
STUDY_MAX_X = 368657.7497
STUDY_MAX_Y = 1565160.5360

REFERENCE_VIDEO_WIDTH = 1280
REFERENCE_VIDEO_HEIGHT = 720


# ============================================================
# GIS CACHE
# ============================================================

RISK_GDF_CACHE = None
STUDY_GDF_CACHE = None


# ============================================================
# LOAD GIS
# ============================================================

def load_gis():

    global RISK_GDF_CACHE
    global STUDY_GDF_CACHE

    if RISK_GDF_CACHE is None:

        print("Loading risk layer...")

        RISK_GDF_CACHE = gpd.read_file(
            RISK_GPKG,
            columns=[
                "village_risk",
                "railway_risk",
                "road_risk",
                "risk_score",
                "risk_level",
                "geometry"
            ]
        )

        if RISK_GDF_CACHE.crs != UTM_CRS:

            RISK_GDF_CACHE = (
                RISK_GDF_CACHE
                .to_crs(UTM_CRS)
            )

        print(
            "Risk features:",
            len(RISK_GDF_CACHE)
        )

    if STUDY_GDF_CACHE is None:

        print("Loading study boundary...")

        STUDY_GDF_CACHE = gpd.read_file(
            STUDY_GPKG
        )

        if STUDY_GDF_CACHE.crs != UTM_CRS:

            STUDY_GDF_CACHE = (
                STUDY_GDF_CACHE
                .to_crs(UTM_CRS)
            )

    return (
        RISK_GDF_CACHE,
        STUDY_GDF_CACHE
    )


# ============================================================
# PIXEL → UTM
# ============================================================

def pixel_to_utm(
    x,
    y,
    video_width,
    video_height
):

    x_ref = (
        x *
        REFERENCE_VIDEO_WIDTH /
        video_width
    )

    y_ref = (
        y *
        REFERENCE_VIDEO_HEIGHT /
        video_height
    )

    utm_x = (
        STUDY_MIN_X +
        (
            x_ref /
            REFERENCE_VIDEO_WIDTH
        ) *
        (
            STUDY_MAX_X -
            STUDY_MIN_X
        )
    )

    utm_y = (
        STUDY_MAX_Y -
        (
            y_ref /
            REFERENCE_VIDEO_HEIGHT
        ) *
        (
            STUDY_MAX_Y -
            STUDY_MIN_Y
        )
    )

    return utm_x, utm_y


# ============================================================
# ONE YOLO + BYTETRACK PASS
# CLEAN TRACKING - NO TRACK -1 IN ANALYTICS
# ============================================================

def run_yolo_tracking_once(
    video_path,
    detected_output,
    progress
):

    progress(
        0.03,
        desc="📹 Reading video..."
    )

    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    if fps <= 0:
        fps = 25.0

    print("FPS:", fps)
    print("Frames:", total_frames)
    print("Resolution:", width, "x", height)

    # ========================================================
    # YOLO + BYTETRACK
    # ========================================================

    progress(
        0.05,
        desc="🐘 Starting YOLOv10 + ByteTrack..."
    )

    results = yolo_model.track(
        source=video_path,
        tracker="bytetrack.yaml",
        stream=True,
        persist=True,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False
    )

    writer = None
    tracking_rows = []
    total_detections = 0
    unique_ids = set()
    frame_number = 0

    # ========================================================
    # PROCESS VIDEO ONCE
    # ========================================================

    for result in results:

        frame_number += 1

        # ----------------------------------------------------
        # ANNOTATED VIDEO FRAME
        # ----------------------------------------------------

        annotated_frame = result.plot(
            conf=True,
            labels=True,
            boxes=True
        )

        # ----------------------------------------------------
        # VIDEO WRITER
        # ----------------------------------------------------

        if writer is None:

            h, w = annotated_frame.shape[:2]

            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            writer = cv2.VideoWriter(
                detected_output,
                fourcc,
                fps,
                (w, h)
            )

        # ----------------------------------------------------
        # FRAME INFORMATION
        # ----------------------------------------------------

        cv2.putText(
            annotated_frame,
            f"Frame: {frame_number}/{total_frames}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            annotated_frame,
            "YOLOv10 + ByteTrack",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        writer.write(annotated_frame)

        # ----------------------------------------------------
        # TRACKING DATA
        # ----------------------------------------------------

        if result.boxes is not None:

            boxes = result.boxes

            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()

            if boxes.id is not None:
                track_ids = (
                    boxes.id
                    .cpu()
                    .numpy()
                    .astype(int)
                )
            else:
                track_ids = [-1] * len(xyxy)

            for box, conf, track_id in zip(
                xyxy,
                confs,
                track_ids
            ):

                total_detections += 1

                # IMPORTANT:
                # Do not send unassigned ByteTrack ID -1
                # into trajectory, TFT, GIS, or analytics.
                # The video annotation above is still preserved.
                if int(track_id) < 0:
                    continue

                x1, y1, x2, y2 = box

                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2

                tracking_rows.append({
                    "frame": frame_number - 1,
                    "track_id": int(track_id),
                    "x": float(center_x),
                    "y": float(center_y),
                    "confidence": float(conf)
                })

                unique_ids.add(int(track_id))

        # ----------------------------------------------------
        # LIVE PROGRESS
        # ----------------------------------------------------

        if total_frames > 0:

            fraction = frame_number / total_frames
            percentage = 5 + fraction * 50

            progress(
                percentage / 100,
                desc=(
                    f"🐘 YOLOv10 + ByteTrack | "
                    f"{frame_number}/{total_frames} frames | "
                    f"{total_detections} detections"
                )
            )

    # ========================================================
    # RELEASE VIDEO
    # ========================================================

    if writer is not None:
        writer.release()

    # ========================================================
    # CREATE CLEAN TRACKING DATAFRAME
    # ========================================================

    tracking_df = pd.DataFrame(tracking_rows)

    print()
    print("========================================")
    print("YOLO + tracking complete")
    print("========================================")
    print("Total detections:", total_detections)
    print("Valid tracking rows:", len(tracking_df))
    print("Valid elephant tracks:", len(unique_ids))
    print("Track IDs:", sorted(unique_ids))
    print("========================================")

    return (
        tracking_df,
        fps,
        total_frames,
        width,
        height,
        total_detections,
        unique_ids
    )


# ============================================================
# CONVERT VIDEO TO BROWSER-FRIENDLY H264
# ============================================================

def convert_video_h264(
    input_video,
    progress
):

    progress(
        0.54,
        desc="🎥 Preparing detected video..."
    )

    output_video = (
        input_video.replace(
            ".mp4",
            "_h264.mp4"
        )
    )

    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_video,
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_video
    ]

    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    if result.returncode != 0:

        print(
            "FFmpeg warning:"
        )

        print(
            result.stderr.decode(
                errors="ignore"
            )[-1000:]
        )

        return input_video

    return output_video


# ============================================================
# FEATURES
# ============================================================

def create_features(
    tracking_df,
    fps,
    progress
):

    progress(
        0.56,
        desc="📊 Creating movement features..."
    )

    if tracking_df.empty:

        return pd.DataFrame()

    tracks = []

    track_ids = (
        tracking_df[
            "track_id"
        ]
        .unique()
    )

    for i, track_id in enumerate(
        track_ids
    ):

        track = tracking_df[
            tracking_df[
                "track_id"
            ] == track_id
        ].copy()

        track = track.sort_values(
            "frame"
        )

        track["dx"] = (
            track["x"]
            .diff()
            .fillna(0)
        )

        track["dy"] = (
            track["y"]
            .diff()
            .fillna(0)
        )

        track["distance"] = np.sqrt(
            track["dx"] ** 2 +
            track["dy"] ** 2
        )

        track["speed"] = (
            track["distance"] *
            fps
        )

        window = 5

        track["x_smooth"] = (
            track["x"]
            .rolling(window)
            .mean()
            .bfill()
        )

        track["y_smooth"] = (
            track["y"]
            .rolling(window)
            .mean()
            .bfill()
        )

        track["dx_smooth"] = (
            track["dx"]
            .rolling(window)
            .mean()
            .bfill()
        )

        track["dy_smooth"] = (
            track["dy"]
            .rolling(window)
            .mean()
            .bfill()
        )

        track["distance_smooth"] = (
            track["distance"]
            .rolling(window)
            .mean()
            .bfill()
        )

        track["speed_smooth"] = (
            track["speed"]
            .rolling(window)
            .mean()
            .bfill()
        )

        track[
            "direction_smooth"
        ] = np.degrees(
            np.arctan2(
                track["dy_smooth"],
                track["dx_smooth"]
            )
        )

        tracks.append(
            track
        )

        fraction = (
            (i + 1) /
            len(track_ids)
        )

        progress(
            0.56 +
            fraction * 0.06,
            desc=(
                f"📊 Movement features | "
                f"Track {i+1}/{len(track_ids)}"
            )
        )

    return pd.concat(
        tracks,
        ignore_index=True
    )


# ============================================================
# TFT
# ============================================================

def run_tft(
    features_df,
    progress
):

    progress(
        0.63,
        desc="🧠 Starting TFT trajectory prediction..."
    )

    if features_df.empty:

        return pd.DataFrame()

    feature_names = [
        "x_smooth",
        "y_smooth",
        "dx_smooth",
        "dy_smooth",
        "distance_smooth",
        "speed_smooth",
        "direction_smooth"
    ]

    predictions = []

    track_ids = (
        features_df[
            "track_id"
        ].unique()
    )

    for track_index, track_id in enumerate(
        track_ids
    ):

        track = features_df[
            features_df[
                "track_id"
            ] == track_id
        ].sort_values(
            "frame"
        ).reset_index(
            drop=True
        )

        if len(track) < sequence_length:

            continue

        total_windows = (
            len(track) -
            sequence_length +
            1
        )

        for i in range(
            total_windows
        ):

            window = track.iloc[
                i:
                i + sequence_length
            ]

            X = window[
                feature_names
            ].values

            X_scaled = (
                feature_scaler
                .transform(X)
            )

            X_tensor = torch.tensor(
                X_scaled,
                dtype=torch.float32
            ).unsqueeze(0)

            X_tensor = (
                X_tensor.to(device)
            )

            with torch.no_grad():

                prediction_scaled = (
                    tft_model(
                        X_tensor
                    )
                    .cpu()
                    .numpy()
                )

            prediction = (
                target_scaler
                .inverse_transform(
                    prediction_scaled
                )[0]
            )

            current = (
                window.iloc[-1]
            )

            predictions.append({

                "track_id":
                    int(track_id),

                "start_frame":
                    int(
                        window.iloc[0]["frame"]
                    ),

                "end_frame":
                    int(
                        current["frame"]
                    ),

                "current_x":
                    float(
                        current["x_smooth"]
                    ),

                "current_y":
                    float(
                        current["y_smooth"]
                    ),

                "predicted_x":
                    float(
                        prediction[0]
                    ),

                "predicted_y":
                    float(
                        prediction[1]
                    )
            })

            fraction = (
                (
                    track_index *
                    total_windows +
                    i + 1
                )
                /
                (
                    len(track_ids) *
                    total_windows
                )
            )

            progress(
                0.63 +
                min(fraction, 1) * 0.15,
                desc=(
                    f"🧠 TFT Prediction | "
                    f"Track {track_index+1}/"
                    f"{len(track_ids)} | "
                    f"Window {i+1}/"
                    f"{total_windows}"
                )
            )

    return pd.DataFrame(
        predictions
    )


# ============================================================
# CLEAN TRAJECTORY GRAPH
# ============================================================

def create_trajectory(
    features_df,
    predictions_df,
    path,
    progress
):

    progress(
        0.79,
        desc="📈 Creating clean trajectory graph..."
    )

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    # --------------------------------------------------------
    # Valid tracks only
    # --------------------------------------------------------

    valid_tracks = sorted(
        features_df["track_id"]
        .dropna()
        .unique()
    )

    # --------------------------------------------------------
    # Plot each elephant
    # --------------------------------------------------------

    for track_id in valid_tracks:

        # ================================================
        # ACTUAL MOVEMENT
        # ================================================

        track = features_df[
            features_df["track_id"] == track_id
        ].sort_values(
            "frame"
        )

        if track.empty:
            continue

        ax.plot(
            track["x_smooth"],
            track["y_smooth"],
            linewidth=2.5,
            label=f"Actual Track {int(track_id)}"
        )

        # ================================================
        # CURRENT POSITION
        # ================================================

        current_x = track.iloc[-1]["x_smooth"]
        current_y = track.iloc[-1]["y_smooth"]

        ax.scatter(
            current_x,
            current_y,
            s=90,
            marker="o",
            edgecolors="black",
            linewidths=1.2,
            zorder=5
        )

        ax.annotate(
            f"Current {int(track_id)}",
            (
                current_x,
                current_y
            ),
            xytext=(8, 8),
            textcoords="offset points",
            fontsize=9
        )

        # ================================================
        # TFT PREDICTIONS
        # ================================================

        if not predictions_df.empty:

            pred = predictions_df[
                predictions_df["track_id"] == track_id
            ].sort_values(
                "end_frame"
            )

            if not pred.empty:

                # ------------------------------------------------
                # Sample predictions to avoid overcrowding
                # ------------------------------------------------

                if len(pred) > 25:

                    sample_indices = np.linspace(
                        0,
                        len(pred) - 1,
                        25
                    ).astype(int)

                    pred_plot = pred.iloc[
                        sample_indices
                    ]

                else:

                    pred_plot = pred

                # ------------------------------------------------
                # Predicted trajectory
                # ------------------------------------------------

                ax.plot(
                    pred_plot["predicted_x"],
                    pred_plot["predicted_y"],
                    linestyle="--",
                    linewidth=2,
                    alpha=0.8,
                    label=f"TFT Predicted {int(track_id)}"
                )

                # ------------------------------------------------
                # Final predicted position
                # ------------------------------------------------

                final_prediction = pred.iloc[-1]

                predicted_x = (
                    final_prediction["predicted_x"]
                )

                predicted_y = (
                    final_prediction["predicted_y"]
                )

                ax.scatter(
                    predicted_x,
                    predicted_y,
                    s=120,
                    marker="X",
                    edgecolors="black",
                    linewidths=1.2,
                    zorder=6
                )

                ax.annotate(
                    f"Predicted {int(track_id)}",
                    (
                        predicted_x,
                        predicted_y
                    ),
                    xytext=(8, -18),
                    textcoords="offset points",
                    fontsize=9
                )

                # ------------------------------------------------
                # Current → predicted direction
                # ------------------------------------------------

                dx = (
                    predicted_x -
                    current_x
                )

                dy = (
                    predicted_y -
                    current_y
                )

                ax.annotate(
                    "",
                    xy=(
                        predicted_x,
                        predicted_y
                    ),
                    xytext=(
                        current_x,
                        current_y
                    ),
                    arrowprops=dict(
                        arrowstyle="->",
                        linewidth=1.8,
                        alpha=0.8
                    )
                )

    # --------------------------------------------------------
    # Reverse Y-axis
    #
    # Video coordinates normally start at the top-left.
    # This keeps the graph visually consistent with the video.
    # --------------------------------------------------------

    ax.invert_yaxis()

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel(
        "X Position (pixels)",
        fontsize=11
    )

    ax.set_ylabel(
        "Y Position (pixels)",
        fontsize=11
    )

    ax.set_title(
        "Elephant Trajectory: Actual vs TFT Predicted",
        fontsize=15,
        fontweight="bold"
    )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.grid(
        alpha=0.25
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    ax.legend(
        fontsize=9,
        loc="best"
    )

    # --------------------------------------------------------
    # Layout
    # --------------------------------------------------------

    plt.tight_layout()

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "✅ Clean trajectory graph saved:"
    )

    print(
        path
    )

    return path


# ============================================================
# GIS — ONE SPATIAL JOIN
# ============================================================

def calculate_gis(
    predictions_df,
    video_width,
    video_height,
    progress
):

    progress(
        0.81,
        desc="🗺️ Loading GIS risk data..."
    )

    if predictions_df.empty:

        return pd.DataFrame()

    risk_gdf, _ = load_gis()

    progress(
        0.83,
        desc="🗺️ Converting TFT predictions to UTM..."
    )

    utm_points = []

    for _, row in predictions_df.iterrows():

        x, y = pixel_to_utm(
            row["predicted_x"],
            row["predicted_y"],
            video_width,
            video_height
        )

        utm_points.append(
            (x, y)
        )

    gis_points = gpd.GeoDataFrame(
        predictions_df.copy(),
        geometry=[
            gpd.points_from_xy(
                [p[0]],
                [p[1]]
            )[0]
            for p in utm_points
        ],
        crs=UTM_CRS
    )

    gis_points["utm_x"] = [
        p[0]
        for p in utm_points
    ]

    gis_points["utm_y"] = [
        p[1]
        for p in utm_points
    ]

    progress(
        0.85,
        desc="🗺️ Performing spatial risk assessment..."
    )

    # ONE spatial join instead of one join per prediction
    joined = gpd.sjoin(
        gis_points,
        risk_gdf[
            [
                "risk_score",
                "risk_level",
                "geometry"
            ]
        ],
        how="left",
        predicate="within"
    )

    # Remove duplicate matches
    if "index_right" in joined.columns:

        joined = (
            joined
            .sort_values(
                "risk_score",
                ascending=False,
                na_position="last"
            )
            .drop_duplicates(
                subset=[
                    "track_id",
                    "end_frame"
                ]
            )
        )

    joined["risk_level"] = (
        joined["risk_level"]
        .fillna(
            "OUTSIDE_DEFINED_RISK_ZONE"
        )
    )

    return joined


# ============================================================
# ANALYTICS
# ============================================================

def create_analytics(
    features_df,
    graph_path,
    progress
):

    progress(
        0.89,
        desc="📊 Calculating movement analytics..."
    )

    rows = []

    for track_id in sorted(
        features_df[
            "track_id"
        ].unique()
    ):

        track = features_df[
            features_df[
                "track_id"
            ] == track_id
        ]

        rows.append({

            "Track ID":
                int(track_id),

            "Average Pixel Speed":
                round(
                    track[
                        "speed_smooth"
                    ].mean(),
                    2
                ),

            "Maximum Pixel Speed":
                round(
                    track[
                        "speed_smooth"
                    ].max(),
                    2
                ),

            "Minimum Pixel Speed":
                round(
                    track[
                        "speed_smooth"
                    ].min(),
                    2
                ),

            "Total Movement Distance":
                round(
                    track[
                        "distance"
                    ].sum(),
                    2
                ),

            "Average Direction":
                round(
                    track[
                        "direction_smooth"
                    ].mean(),
                    2
                )
        })

    analytics_df = pd.DataFrame(
        rows
    )

    plt.figure(
        figsize=(12, 6)
    )

    for track_id in sorted(
        features_df[
            "track_id"
        ].unique()
    ):

        track = features_df[
            features_df[
                "track_id"
            ] == track_id
        ]

        plt.plot(
            track["frame"],
            track["speed_smooth"],
            linewidth=2,
            label=f"Track {track_id}"
        )

    plt.xlabel(
        "Frame"
    )

    plt.ylabel(
        "Pixel Speed"
    )

    plt.title(
        "Elephant Movement Speed Analytics"
    )

    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        graph_path,
        dpi=150
    )

    plt.close()

    return analytics_df


# ============================================================
# GIS MAP
# ============================================================

# ============================================================
# CLEAN GIS RISK MAP
# ============================================================

def create_gis_map(
    risk_df,
    path,
    progress
):

    progress(
        0.94,
        desc="🗺️ Creating presentation GIS risk map..."
    )

    # --------------------------------------------------------
    # Load study boundary
    # --------------------------------------------------------

    _, study_gdf = load_gis()

    # --------------------------------------------------------
    # Create figure
    # --------------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(12, 9)
    )

    # --------------------------------------------------------
    # Study area boundary
    # --------------------------------------------------------

    if study_gdf is not None:

        study_gdf.boundary.plot(
            ax=ax,
            linewidth=2.5,
            label="Koundinya Study Area"
        )

    # --------------------------------------------------------
    # Risk levels
    # --------------------------------------------------------

    risk_levels = [
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL",
        "OUTSIDE_DEFINED_RISK_ZONE"
    ]

    # --------------------------------------------------------
    # Plot prediction points
    # --------------------------------------------------------

    if (
        risk_df is not None
        and not risk_df.empty
    ):

        for level in risk_levels:

            subset = risk_df[
                risk_df["risk_level"] == level
            ].copy()

            if subset.empty:
                continue

            # --------------------------------------------
            # Limit displayed points
            #
            # This keeps the map clean when hundreds
            # of prediction windows exist.
            # --------------------------------------------

            if len(subset) > 150:

                sample_indices = np.linspace(
                    0,
                    len(subset) - 1,
                    150
                ).astype(int)

                subset = subset.iloc[
                    sample_indices
                ]

            # --------------------------------------------
            # Point sizes
            # --------------------------------------------

            if level == "LOW":

                point_size = 30

            elif level == "MEDIUM":

                point_size = 45

            elif level == "HIGH":

                point_size = 65

            elif level == "CRITICAL":

                point_size = 90

            else:

                point_size = 18

            # --------------------------------------------
            # Plot
            # --------------------------------------------

            ax.scatter(
                subset["utm_x"],
                subset["utm_y"],
                s=point_size,
                alpha=0.75,
                label=level,
                edgecolors="black",
                linewidths=0.4
            )

    # --------------------------------------------------------
    # Mark latest prediction for each elephant
    # --------------------------------------------------------

    if (
        risk_df is not None
        and not risk_df.empty
    ):

        latest = (
            risk_df
            .sort_values("end_frame")
            .groupby("track_id")
            .tail(1)
        )

        for _, row in latest.iterrows():

            ax.scatter(
                row["utm_x"],
                row["utm_y"],
                s=180,
                marker="X",
                edgecolors="black",
                linewidths=1.5,
                zorder=10
            )

            ax.annotate(
                f"Elephant {int(row['track_id'])}",
                (
                    row["utm_x"],
                    row["utm_y"]
                ),
                xytext=(8, 8),
                textcoords="offset points",
                fontsize=9,
                fontweight="bold"
            )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    ax.set_xlabel(
        "UTM Easting (EPSG:32644)",
        fontsize=11
    )

    ax.set_ylabel(
        "UTM Northing (EPSG:32644)",
        fontsize=11
    )

    ax.set_title(
        "Koundinya Wildlife Sanctuary\n"
        "Elephant Trajectory & GIS Risk Assessment",
        fontsize=15,
        fontweight="bold"
    )

    # --------------------------------------------------------
    # Grid
    # --------------------------------------------------------

    ax.grid(
        alpha=0.25
    )

    # --------------------------------------------------------
    # Legend
    # --------------------------------------------------------

    ax.legend(
        fontsize=9,
        loc="best",
        frameon=True
    )

    # --------------------------------------------------------
    # Equal geographic scale
    # --------------------------------------------------------

    ax.set_aspect(
        "equal",
        adjustable="box"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=180,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(
        "✅ Presentation GIS map saved:"
    )

    print(
        path
    )

    return path


# ============================================================
# COMPLETE ANALYSIS
# ============================================================

def complete_analysis(
    video_path,
    progress=gr.Progress()
):

    if video_path is None:

        raise gr.Error(
            "Please upload an elephant video."
        )

    run_id = uuid.uuid4().hex[:8]

    run_dir = os.path.join(
        OUTPUT_DIR,
        run_id
    )

    os.makedirs(
        run_dir,
        exist_ok=True
    )

    print(
        "\n========================================"
    )

    print(
        "NEW ANALYSIS:",
        run_id
    )

    print(
        "========================================"
    )

    # ========================================================
    # 1. YOLO + BYTE TRACK
    # ========================================================

    raw_detected = os.path.join(
        run_dir,
        "detected_raw.mp4"
    )

    (
        tracking_df,
        fps,
        total_frames,
        video_width,
        video_height,
        total_detections,
        unique_ids
    ) = run_yolo_tracking_once(
        video_path,
        raw_detected,
        progress
    )

    # ========================================================
    # 2. VIDEO CONVERSION
    # ========================================================

    detected_video = (
        convert_video_h264(
            raw_detected,
            progress
        )
    )
    print("========================================")
    print("🎥 FINAL DETECTED VIDEO")
    print("Path:", detected_video)
    print("Exists:", os.path.exists(detected_video))

    if os.path.exists(detected_video):
        print(
            "Size:",
            round(os.path.getsize(detected_video) / (1024 * 1024), 2),
            "MB"
        )

    print("========================================")

    # ========================================================
    # 3. FEATURES
    # ========================================================

    features_df = create_features(
        tracking_df,
        fps,
        progress
    )

    features_df.to_csv(
        os.path.join(
            run_dir,
            "features.csv"
        ),
        index=False
    )

    # ========================================================
    # 4. TFT
    # ========================================================

    predictions_df = run_tft(
        features_df,
        progress
    )

    predictions_df.to_csv(
        os.path.join(
            run_dir,
            "predictions.csv"
        ),
        index=False
    )

    # ========================================================
    # 5. TRAJECTORY
    # ========================================================

    trajectory_path = os.path.join(
        run_dir,
        "trajectory.png"
    )

    create_trajectory(
        features_df,
        predictions_df,
        trajectory_path,
        progress
    )

    # ========================================================
    # 6. GIS
    # ========================================================

    risk_df = calculate_gis(
        predictions_df,
        video_width,
        video_height,
        progress
    )

    risk_df.to_csv(
        os.path.join(
            run_dir,
            "gis_risk.csv"
        ),
        index=False
    )

    # ========================================================
    # 7. ANALYTICS
    # ========================================================

    analytics_path = os.path.join(
        run_dir,
        "analytics.png"
    )

    analytics_df = create_analytics(
        features_df,
        analytics_path,
        progress
    )

    analytics_df.to_csv(
        os.path.join(
            run_dir,
            "analytics.csv"
        ),
        index=False
    )

    # ========================================================
    # 8. GIS MAP
    # ========================================================

    gis_map_path = os.path.join(
        run_dir,
        "gis_map.png"
    )

    create_gis_map(
        risk_df,
        gis_map_path,
        progress
    )

    # ========================================================
    # 9. LATEST PREDICTIONS
    # ========================================================

    if not predictions_df.empty:

        latest = (
            predictions_df
            .sort_values(
                "end_frame"
            )
            .groupby(
                "track_id"
            )
            .tail(1)
            .copy()
        )

        latest = latest[
            [
                "track_id",
                "current_x",
                "current_y",
                "predicted_x",
                "predicted_y"
            ]
        ]

        latest.columns = [
            "Track ID",
            "Current X",
            "Current Y",
            "Predicted X",
            "Predicted Y"
        ]

        latest = latest.round(2)

    else:

        latest = pd.DataFrame()

    # ========================================================
    # 10. RISK COUNTS
    # ========================================================

    if not risk_df.empty:

        counts = (
            risk_df[
                "risk_level"
            ]
            .value_counts()
        )

    else:

        counts = pd.Series(
            dtype=int
        )

    low = int(
        counts.get("LOW", 0)
    )

    medium = int(
        counts.get("MEDIUM", 0)
    )

    high = int(
        counts.get("HIGH", 0)
    )

    critical = int(
        counts.get("CRITICAL", 0)
    )

    outside = int(
        counts.get(
            "OUTSIDE_DEFINED_RISK_ZONE",
            0
        )
    )

    # ========================================================
    # 10.5 PDF REPORT
    # ========================================================

    progress(
        0.97,
        desc="📄 Generating PDF report..."
    )

    report_path = os.path.join(
        run_dir,
        "elephant_analysis_report.pdf"
    )

    generate_pdf_report(
        report_path=report_path,
        run_id=run_id,
        fps=fps,
        total_frames=total_frames,
        video_width=video_width,
        video_height=video_height,
        total_detections=total_detections,
        unique_ids=unique_ids,
        predictions_df=predictions_df,
        risk_df=risk_df,
        analytics_df=analytics_df,
        trajectory_path=trajectory_path,
        gis_map_path=gis_map_path,
        analytics_path=analytics_path
    )

    print("📄 Report:", report_path)
    # ========================================================
    # 11. FINAL
    # ========================================================

    progress(
        0.99,
        desc="📋 Preparing final results..."
    )

    summary = f"""
# Elephant Trajectory Prediction

### 🎥 Video Analysis
- **FPS:** {fps:.2f}
- **Frames:** {total_frames}
- **Resolution:** {video_width} × {video_height}

### 🐘 Detection & Tracking
- **Total detections:** {total_detections}
- **Elephant tracks:** {len(unique_ids)}

### 🗺️ GIS Risk Summary

| Risk Level | Count |
|---|---:|
| LOW | {low} |
| MEDIUM | {medium} |
| HIGH | {high} |
| CRITICAL | {critical} |
| OUTSIDE | {outside} |
"""

    progress(
        1.0,
        desc="🎉 COMPLETE — 100%"
    )

    progress_html = """
    <div class="analysis-progress-wrap completed">
        <div class="analysis-progress-head">
            <span>Analysis Progress</span>
            <strong>100%</strong>
        </div>
        <div class="analysis-progress-track">
            <div class="analysis-progress-fill" style="width:100%"></div>
        </div>
        <div class="analysis-progress-status">🎉 Analysis completed successfully</div>
    </div>
    """

    return (
        detected_video,
        trajectory_path,
        gis_map_path,
        analytics_path,
        latest,
        analytics_df,
        summary,
        report_path,
        progress_html
    )


# ============================================================
# ============================================================
# PROFESSIONAL WEBSITE-STYLE GRADIO UI — V8
# ============================================================

CUSTOM_CSS = """
/* ============================================================
   GLOBAL WEBSITE LAYOUT
   ============================================================ */
html, body {
    margin: 0 !important;
    padding: 0 !important;
    background: #f7faf8 !important;
    color: #17352a !important;
    font-family: Inter, Arial, sans-serif !important;
}

.gradio-container {
    width: 100% !important;
    max-width: 100% !important;
    min-width: 0 !important;
    margin: 0 !important;
    padding: 0 0 50px !important;
    background: #f7faf8 !important;
}

/* FULL-VIEWPORT OUTER SHELL */
html, body, gradio-app {
    width: 100vw !important;
    max-width: 100vw !important;
    min-width: 100vw !important;
    overflow-x: hidden !important;
}
.gradio-container .contain,
.gradio-container > .contain,
.gradio-container .main,
.gradio-container .wrap {
    width: 100% !important;
    max-width: none !important;
    min-width: 0 !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
    box-sizing: border-box !important;
}


/* Remove Gradio's default page chrome */
footer { display: none !important; }

/* ============================================================
   TOP NAVIGATION — ABOVE HERO
   ============================================================ */
.tabs {
    width: 100% !important;
    margin: 0 !important;
    padding: 0 !important;
}

.tab-nav {
    width: 100% !important;
    max-width: 100% !important;
    min-height: 74px !important;
    margin: 0 !important;
    padding: 0 24px !important;
    box-sizing: border-box !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-end !important;
    gap: 12px !important;
    flex-wrap: nowrap !important;
    background: #ffffff !important;
    border: 0 !important;
    border-bottom: 1px solid #e1e9e4 !important;
    box-shadow: 0 2px 10px rgba(24,60,42,.06) !important;
}

/* Brand on the left, just like a real website navbar */
.tab-nav::before {
    content: "Elephant Prediction AI";
    margin-right: auto !important;
    color: #075b35 !important;
    font-size: 20px !important;
    font-weight: 850 !important;
    white-space: nowrap !important;
}

.tab-nav button {
    flex: 0 0 auto !important;
    min-width: 138px !important;
    height: 44px !important;
    padding: 10px 20px !important;
    border-radius: 8px !important;
    background: #ffffff !important;
    color: #17603e !important;
    border: 1px solid #17603e !important;
    font-weight: 650 !important;
    font-size: 13px !important;
    transition: all .2s ease !important;
}

.tab-nav button:hover {
    background: #edf8f1 !important;
}

.tab-nav button.selected {
    background: #075b35 !important;
    color: #ffffff !important;
    border-color: #075b35 !important;
}

/* ============================================================
   HERO — FULL WIDTH, DIRECTLY UNDER NAV
   ============================================================ */
#hero {
    position: relative !important;
    width: 100% !important;
    max-width: none !important;
    min-height: 560px !important;
    margin-left: 0 !important;
    margin-right: 0 !important;
    margin-top: 0 !important;
    margin-bottom: 0 !important;
    box-sizing: border-box !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
    padding: 80px 7vw !important;
    overflow: hidden !important;
    background-image:
        linear-gradient(
            90deg,
            rgba(0,25,16,.45),
            rgba(0,25,16,.20),
            rgba(0,25,16,.40)
        ),
        url("/gradio_api/file/C:/Users/HP/Downloads/Elephant_Prediction/assets/elephant_hero.jpg") !important;
    background-size: cover !important;
    background-position: center 55% !important;
}

#hero::after {
    content: "" !important;
    position: absolute !important;
    inset: 0 !important;
    background: linear-gradient(to bottom, transparent 68%, rgba(4,20,13,.68)) !important;
    pointer-events: none !important;
}

.hero-content {
    position: relative !important;
    z-index: 2 !important;
    width: min(1100px, 92vw) !important;
    max-width: 1100px !important;
}

.hero-title {
    color: #ffffff !important;
    font-size: clamp(44px, 6.5vw, 82px) !important;
    line-height: 1.04 !important;
    font-weight: 850 !important;
    margin: 0 0 22px !important;
    text-shadow: 0 5px 25px rgba(0,0,0,.50) !important;
}

.hero-title,
.hero-title span {
    color: #ffffff !important;
}

.hero-subtitle {
    color: #ffffff !important;
    font-size: clamp(17px, 1.7vw, 23px) !important;
    line-height: 1.55 !important;
    max-width: 1050px !important;
    margin: 0 auto !important;
    text-shadow: 0 3px 15px rgba(0,0,0,.60) !important;
}

/* ============================================================
   HERO START ANALYSIS BUTTON
   ============================================================ */
.hero-start-button {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin-top: 28px !important;
    min-width: 210px !important;
    padding: 15px 30px !important;
    border-radius: 10px !important;
    background: #0b9658 !important;
    color: #ffffff !important;
    border: 2px solid rgba(255,255,255,.75) !important;
    text-decoration: none !important;
    font-size: 17px !important;
    font-weight: 850 !important;
    box-shadow: 0 10px 28px rgba(0,0,0,.25) !important;
    cursor: pointer !important;
    transition: transform .2s ease, background .2s ease !important;
}
.hero-start-button:hover {
    background: #087342 !important;
    transform: translateY(-2px) !important;
}

/* ============================================================
   HOME CONTENT — BELOW HERO
   ============================================================ */
.home-content {
    width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    box-sizing: border-box !important;
    width: 100% !important;
    max-width: none !important;
    margin: 34px 0 0 !important;
}

.upload-card {
    width: 100% !important;
    box-sizing: border-box !important;
    background: #ffffff !important;
    border: 1px solid #dce7df !important;
    border-radius: 18px !important;
    padding: 24px 5vw !important;
    box-shadow: 0 8px 28px rgba(25,65,45,.08) !important;
}

.run-analysis-button {
    width: 100% !important;
    margin-top: 18px !important;
    min-height: 62px !important;
    font-size: 18px !important;
}

button.primary {
    background: linear-gradient(135deg, #087342, #0b9658) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 11px !important;
    font-weight: 850 !important;
    box-shadow: 0 8px 22px rgba(8,115,66,.22) !important;
}

button.primary:hover {
    filter: brightness(1.06);
    transform: translateY(-1px);
}

/* ============================================================
   ANALYSIS PROGRESS — BELOW RUN BUTTON
   ============================================================ */
.analysis-progress-wrap {
    display: block !important;
    margin-top: 18px !important;
    padding: 17px 20px !important;
    border: 1px solid #dce7df !important;
    border-radius: 14px !important;
    background: #f8fcf9 !important;
}

.analysis-progress-head {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    color: #17603e !important;
    font-size: 14px !important;
    font-weight: 750 !important;
    margin-bottom: 10px !important;
}

.analysis-progress-track {
    width: 100% !important;
    height: 11px !important;
    background: #dfeae3 !important;
    border-radius: 999px !important;
    overflow: hidden !important;
}

.analysis-progress-fill {
    height: 100% !important;
    border-radius: 999px !important;
    background: linear-gradient(90deg, #087342, #16a765) !important;
    transition: width .4s ease !important;
}

.analysis-progress-status {
    margin-top: 9px !important;
    color: #087342 !important;
    font-size: 13px !important;
    font-weight: 700 !important;
}

.next-page-button {
    width: 150px !important;
    margin: 18px 0 0 auto !important;
    background: #07864f !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 750 !important;
}
/* ============================================================
   FORCE ALL TAB PAGES TO USE THE FULL SCREEN WIDTH
   ============================================================ */
.tabs,
.tabitem,
.tabitem > div,
.tabitem > .block,
.tabitem .form,
.tabitem .block,
.tabitem .gradio-container,
.tabitem .contain,
.tabitem .wrap {
    width: 100vw !important;
    max-width: none !important;
    min-width: 0 !important;
    box-sizing: border-box !important;
}

/* ============================================================
   TAB CONTENT — FULL WEBSITE WIDTH
   ============================================================ */
.tabitem {
    width: 100% !important;
    max-width: 100% !important;
    margin: 0 !important;
    padding: 0 0 35px !important;
    box-sizing: border-box !important;
}

.tabitem > div {
    width: 100% !important;
    box-sizing: border-box !important;
}

/* ============================================================
   SECTION HEADERS — DETECTION / GIS / TRAJECTORY / ANALYTICS
   ============================================================ */
.page-section-hero {
    width: 100% !important;
    max-width: none !important;
    margin: 34px 0 24px !important;
    box-sizing: border-box !important;
    display: flex !important;
    align-items: center !important;
    gap: 18px !important;
    padding: 22px 5vw !important;
    border-radius: 20px !important;
    background: linear-gradient(135deg, #075b35, #0b9658) !important;
    color: #ffffff !important;
    box-shadow: 0 12px 30px rgba(8,115,66,.16) !important;
}

.page-section-icon {
    width: 58px !important;
    height: 58px !important;
    flex: 0 0 58px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    border-radius: 16px !important;
    background: rgba(255,255,255,.16) !important;
    font-size: 30px !important;
}

.page-section-title {
    font-size: 28px !important;
    line-height: 1.15 !important;
    font-weight: 850 !important;
    color: #ffffff !important;
}

.page-section-subtitle {
    margin-top: 6px !important;
    color: rgba(255,255,255,.90) !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
}

.page-content-card {
    width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    box-sizing: border-box !important;
    background: #ffffff !important;
    border: 1px solid #dce7df !important;
    border-radius: 18px !important;
    padding: 24px 5vw !important;
    box-shadow: 0 8px 28px rgba(25,65,45,.07) !important;
}

/* ============================================================
   IMAGES / VIDEO / TABLES
   ============================================================ */
video, img {
    border-radius: 12px !important;
    max-width: 100% !important;
}

input, textarea, .wrap, .block {
    background: #ffffff !important;
    color: #17352a !important;
    border-color: #d7e4db !important;
}

.dataframe {
    background: #ffffff !important;
}

/* ============================================================
   REPORTS
   ============================================================ */
.report-section-heading {
    width: 100% !important;
    max-width: none !important;
    box-sizing: border-box !important;
    padding-left: 5vw !important;
    padding-right: 5vw !important;
    margin: 28px 0 7px !important;
    font-size: 22px !important;
    font-weight: 850 !important;
    color: #075b35 !important;
}

.report-section-note {
    width: 100% !important;
    max-width: none !important;
    box-sizing: border-box !important;
    padding-left: 5vw !important;
    padding-right: 5vw !important;
    margin: 0 0 12px !important;
    color: #718177 !important;
}

.report-summary {
    width: 100% !important;
    max-width: none !important;
    margin: 0 !important;
    box-sizing: border-box !important;
    padding: 24px 5vw !important;
    border: 1px solid #dce7df !important;
    border-radius: 18px !important;
    background: #ffffff !important;
    box-shadow: 0 8px 28px rgba(25,65,45,.07) !important;
}

.report-summary h1 {
    color: #075b35 !important;
    font-size: 28px !important;
}

.report-summary h2, .report-summary h3 {
    color: #17603e !important;
}

.download-card {
    width: 100% !important;
    max-width: none !important;
    margin: 28px 0 10px !important;
    box-sizing: border-box !important;
    display: flex !important;
    align-items: center !important;
    gap: 15px !important;
    padding: 20px 5vw !important;
    border-radius: 16px !important;
    border: 1px solid #cfe1d5 !important;
    background: linear-gradient(135deg, #f4fbf6, #ffffff) !important;
}

.download-icon { font-size: 28px; }
.download-title { color: #075b35; font-size: 17px; font-weight: 850; }
.download-subtitle { color: #718177; margin-top: 4px; font-size: 13px; }

/* ============================================================
   FOOTER
   ============================================================ */
.site-footer {
    width: 100% !important;
    margin: 50px 0 0 !important;
    padding: 25px !important;
    text-align: center !important;
    color: #718177 !important;
    border-top: 1px solid #dce7df !important;
}

.footer-title {
    color: #075b35 !important;
    font-size: 20px !important;
    font-weight: 850 !important;
    margin-bottom: 7px !important;
}

/* ============================================================
   MOBILE
   ============================================================ */
@media (max-width: 900px) {
    .tab-nav {
        min-height: auto !important;
        padding: 12px !important;
        justify-content: center !important;
        flex-wrap: wrap !important;
    }

    .tab-nav::before {
        width: 100% !important;
        text-align: center !important;
        margin: 0 0 5px !important;
    }

    .tab-nav button {
        min-width: 135px !important;
    }

    #hero {
        min-height: 500px !important;
        padding: 55px 24px !important;
    }

    .page-section-hero,
    .page-content-card,
    .report-summary,
    .download-card,
    .home-content {
        width: 100% !important;
        max-width: none !important;
    }
}

/* V10: One consistent page gutter and a single balanced navbar */
:root { --site-gutter: clamp(16px, 2.5vw, 48px); }
html, body, gradio-app { width: 100% !important; min-width: 0 !important; max-width: 100% !important; }
.gradio-container, .gradio-container .main, .gradio-container .contain {
  width: 100% !important; max-width: 100% !important; min-width: 0 !important;
  box-sizing: border-box !important;
}
#site-tabs { width: 100% !important; max-width: 100% !important; padding: 0 var(--site-gutter) !important; box-sizing: border-box !important; }
#site-tabs .tabitem, #site-tabs .tabitem > div, #site-tabs .tabitem .block,
#site-tabs .tabitem .form, #site-tabs .tabitem .wrap {
  width: 100% !important; max-width: 100% !important; min-width: 0 !important;
  box-sizing: border-box !important;
}
#site-tabs .tab-nav, #site-tabs [role="tablist"] {
  width: 100% !important; max-width: 100% !important;
  min-height: 78px !important; box-sizing: border-box !important;
  display: flex !important; align-items: center !important;
  justify-content: flex-end !important; gap: clamp(6px, .8vw, 14px) !important;
  flex-wrap: nowrap !important; padding: 8px 0 !important;
}
#site-tabs .tab-nav::before, #site-tabs [role="tablist"]::before {
  content: "Elephant Prediction AI" !important;
  display: flex !important; align-items: center !important;
  margin-right: auto !important; flex: 1 1 auto !important;
  min-width: 180px !important; white-space: nowrap !important;
  color: #075b35 !important; font-size: clamp(16px, 1.25vw, 21px) !important;
  font-weight: 800 !important;
}
#site-tabs .tab-nav button, #site-tabs [role="tablist"] > button {
  flex: 0 1 auto !important; min-width: 0 !important;
  padding: 10px clamp(7px, 1vw, 18px) !important;
  white-space: nowrap !important; font-size: clamp(11px, .85vw, 14px) !important;
}
#hero { width: 100% !important; max-width: 100% !important; margin: 0 !important; }
#site-tabs .page-section-hero, #site-tabs .content-card,
#site-tabs .gradio-video, #site-tabs .gradio-image, #site-tabs .gradio-dataframe {
  max-width: 100% !important; box-sizing: border-box !important;
}
@media (max-width: 1050px) {
 #site-tabs .tab-nav, #site-tabs [role="tablist"] { flex-wrap: wrap !important; justify-content: center !important; }
 #site-tabs .tab-nav::before, #site-tabs [role="tablist"]::before { flex-basis: 100% !important; justify-content: center !important; }
}
/* Make Gradio progress bar and percentage visible */
.progress-bar,
.progress-bar-wrap,
.progress-text,
.wrap .progress-bar {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
}

.progress-bar {
    min-height: 8px !important;
    background: #22c55e !important;
    border-radius: 10px !important;
}

.progress-bar-wrap {
    height: 8px !important;
    background: #e5e7eb !important;
    border-radius: 10px !important;
    overflow: visible !important;
}

.progress-text {
    color: #166534 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}
/* Fix invisible navigation text on deployment */
#site-tabs button,
#site-tabs button span,
#site-tabs [role="tab"],
#site-tabs [role="tab"] span {
    color: #1f2937 !important;
    opacity: 1 !important;
}

/* Keep the active tab green */
#site-tabs button.selected,
#site-tabs [role="tab"][aria-selected="true"] {
    color: #059669 !important;
    opacity: 1 !important;
}

"""

CUSTOM_CSS = CUSTOM_CSS.replace(
    "{HERO_IMAGE_URL}",
    HERO_IMAGE_URL
)

with gr.Blocks(
    title="Elephant Trajectory Prediction",
    css=CUSTOM_CSS,
    theme=gr.themes.Base(
        primary_hue="green",
        secondary_hue="green",
        neutral_hue="slate"
    )
) as app:

    # ========================================================
    # TOP NAV + HOME HERO
    # ========================================================
    # Brand is rendered inside the navigation bar by .tab-nav::before
    # so the brand and tabs stay on one horizontal row.
    with gr.Tabs(elem_id="site-tabs"):

        # ====================================================
        # HOME
        # ====================================================
        with gr.Tab("🏠 HOME"):

            gr.HTML("""
            <section id="hero">
                <div class="hero-content">
                    <div class="hero-title">
                        Elephant <span>Trajectory Prediction</span>
                    </div>
                    <div class="hero-subtitle">
                        AI-Based Elephant Trajectory Prediction and GIS-Enabled
                        Risk Assessment System
                    </div>
                    <a class="hero-start-button" href="#home-upload"
                       onclick="setTimeout(function(){document.getElementById('home-upload')?.scrollIntoView({behavior:'smooth', block:'start'});}, 50); return true;">
                        ▶ START ANALYSIS
                    </a>
                </div>
            </section>
            """)

            with gr.Column(elem_classes="home-content"):
                with gr.Column(elem_classes="upload-card", elem_id="home-upload"):
                    video_input = gr.Video(
                        label="📤 Upload Elephant Video",
                        sources=["upload"]
                    )

                    run_button = gr.Button(
                        "▶ RUN COMPLETE ANALYSIS",
                        variant="primary",
                        size="lg",
                        elem_classes="run-analysis-button"
                    )

                    analysis_progress = gr.HTML(
                        value="""
                        <div class="analysis-progress-wrap">
                            <div class="analysis-progress-head">
                                <span>Analysis Progress</span>
                                <strong>0%</strong>
                            </div>
                            <div class="analysis-progress-track">
                                <div class="analysis-progress-fill" style="width:0%"></div>
                            </div>
                        </div>
                        """,
                        label=""
                    )
                    next_detection_btn = gr.Button(
                        "NEXT →",
                        variant="primary",
                        elem_classes=["next-page-button"],
                        visible=False
                    )

        # ====================================================
        # DETECTION
        # ====================================================
        with gr.Tab("🎥 DETECTION", id="detection"):
            gr.HTML("""
            <div class="page-section-hero">
                <div class="page-section-icon">🎥</div>
                <div>
                    <div class="page-section-title">Elephant Detection &amp; Tracking</div>
                    <div class="page-section-subtitle">Detected elephants and movement tracks generated from the uploaded video.</div>
                </div>
            </div>
            """)

            with gr.Column(elem_classes="page-content-card"):
                detected_output = gr.Video(
                    label="Detected Elephant Video",
                    format="mp4"
                )

                next_gis_btn = gr.Button(
                    "NEXT →",
                    variant="primary",
                    elem_classes=["next-page-button"]
                )


        # ====================================================
        # GIS
        # ====================================================
        with gr.Tab("🗺️ GIS RISK"):
            gr.HTML("""
            <div class="page-section-hero">
                <div class="page-section-icon">🗺️</div>
                <div>
                    <div class="page-section-title">GIS Risk Assessment</div>
                    <div class="page-section-subtitle">Predicted elephant movement mapped against the defined spatial risk zones.</div>
                </div>
            </div>
            """)

            with gr.Column(elem_classes="page-content-card"):
                gis_output = gr.Image(label="Koundinya GIS Risk Map")
                gr.Markdown("""
                **Risk levels:** 🟢 LOW &nbsp;&nbsp; 🟠 MEDIUM &nbsp;&nbsp; 🔴 HIGH &nbsp;&nbsp; 🟥 CRITICAL
                """)

                next_trajectory_btn = gr.Button(
                    "NEXT →",
                    variant="primary",
                    elem_classes=["next-page-button"]
                )


        # ====================================================
        # TRAJECTORY
        # ====================================================
        with gr.Tab("📈 TRAJECTORY"):
            gr.HTML("""
            <div class="page-section-hero">
                <div class="page-section-icon">📈</div>
                <div>
                    <div class="page-section-title">Elephant Trajectory Prediction</div>
                    <div class="page-section-subtitle">Actual elephant movement compared with the predicted trajectory.</div>
                </div>
            </div>
            """)

            with gr.Column(elem_classes="page-content-card"):
                trajectory_output = gr.Image(
                    label="Actual vs Predicted Trajectory"
                )

                next_analytics_btn = gr.Button(
                    "NEXT →",
                    variant="primary",
                    elem_classes=["next-page-button"]
                )

        # ====================================================
        # ANALYTICS
        # ====================================================
        with gr.Tab("📊 ANALYTICS"):
            gr.HTML("""
            <div class="page-section-hero">
                <div class="page-section-icon">📊</div>
                <div>
                    <div class="page-section-title">Movement Analytics</div>
                    <div class="page-section-subtitle">Track-level speed, direction and movement statistics.</div>
                </div>
            </div>
            """)

            with gr.Column(elem_classes="page-content-card"):
                analytics_output = gr.Image(label="Movement Analytics")
                analytics_table = gr.Dataframe(
                    interactive=False,
                    label="Track Analytics"
                )

                next_reports_btn = gr.Button(
                    "NEXT →",
                    variant="primary",
                    elem_classes=["next-page-button"]
                )


        # ====================================================
        # REPORTS
        # ====================================================
        with gr.Tab("📄 REPORTS"):
            gr.HTML("""
            <div class="page-section-hero">
                <div class="page-section-icon">📄</div>
                <div>
                    <div class="page-section-title">Analysis Report</div>
                    <div class="page-section-subtitle">Latest trajectory predictions, GIS risk summary and downloadable analysis report.</div>
                </div>
            </div>
            """)

            gr.HTML("""
            <div class="report-section-heading">📍 Latest Trajectory Predictions</div>
            <div class="report-section-note">Current position compared with the predicted next position.</div>
            """)

            latest_output = gr.Dataframe(
                interactive=False,
                label="Current vs Predicted Position"
            )

            gr.HTML("""
            <div class="report-section-heading">🚨 GIS Risk Summary</div>
            <div class="report-section-note">Risk distribution from the final spatial assessment.</div>
            """)

            summary_output = gr.Markdown(elem_classes="report-summary")

            gr.HTML("""
            <div class="download-card">
                <div class="download-icon">📥</div>
                <div>
                    <div class="download-title">Download Full Analysis Report</div>
                    <div class="download-subtitle">Save the complete PDF report for documentation and presentation.</div>
                </div>
            </div>
            """)

            report_output = gr.File(label="PDF Report")


    # ========================================================
    # RUN EVENT
    # ========================================================
    run_event = run_button.click(
        fn=complete_analysis,
        inputs=video_input,
        outputs=[
            detected_output,
            trajectory_output,
            gis_output,
            analytics_output,
            latest_output,
            analytics_table,
            summary_output,
            report_output,
            analysis_progress
        ],
        show_progress="full"
    ).then(
        fn=lambda: gr.update(visible=True),
        inputs=None,
        outputs=next_detection_btn
    )
    next_detection_btn.click(
        fn=None,
        inputs=None,
        outputs=None,
        js="""
        () => {
            const tabs = Array.from(
                document.querySelectorAll('#site-tabs button')
            ).filter(tab =>
                ['HOME', 'DETECTION', 'GIS RISK', 'TRAJECTORY', 'ANALYTICS', 'REPORTS']
                    .some(name => tab.innerText.includes(name))
            );

            const currentIndex = tabs.findIndex(
                tab => tab.getAttribute('aria-selected') === 'true'
            );

            if (currentIndex >= 0 && currentIndex < tabs.length - 1) {
                tabs[currentIndex + 1].click();
            }
        }
        """
    )
    next_gis_btn.click(
        fn=None,
        inputs=None,
        outputs=None,
        js="""() => {
            const tabs = document.querySelectorAll('#site-tabs [role="tab"]');
            for (const tab of tabs) {
                if (tab.innerText.includes('GIS RISK')) {
                    tab.click();
                    break;
                }
            }
        }"""
    )
    next_trajectory_btn.click(
        fn=None,
        inputs=None,
        outputs=None,
        js="""() => {
            const tabs = document.querySelectorAll('#site-tabs [role="tab"]');
            for (const tab of tabs) {
                if (tab.innerText.includes('TRAJECTORY')) {
                    tab.click();
                    break;
                }
            }
        }"""
    )
    next_analytics_btn.click(
        fn=None,
        inputs=None,
        outputs=None,
        js="""() => {
            const tabs = document.querySelectorAll('#site-tabs [role="tab"]');
            for (const tab of tabs) {
                if (tab.innerText.includes('ANALYTICS')) {
                    tab.click();
                    break;
                }
            }
        }"""
    )
    next_reports_btn.click(
        fn=None,
        inputs=None,
        outputs=None,
        js="""() => {
            const tabs = document.querySelectorAll('#site-tabs [role="tab"]');
            for (const tab of tabs) {
                if (tab.innerText.includes('REPORTS')) {
                    tab.click();
                    break;
                }
            }
        }"""
    )
    gr.HTML("""
    <div class="site-footer">
        <div class="footer-title">🐘 Elephant Trajectory Prediction</div>
        <div>Developed by <b>Trio Team</b></div>
    </div>
    """)

print("🚀 Starting Elephant Trajectory Prediction website...")

app.queue(max_size=2).launch(
    allowed_paths=[
        os.path.join(BASE_DIR, "assets")
    ]
)
