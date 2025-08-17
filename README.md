# HAR-Data-Preparation

---

## 📝 Project Description

### **Human Activity Recognition: Data Preparation using IMU Sensors and Video Annotations**

This project focuses on the preparation and annotation of synchronized **wearable sensor data** and **video recordings** for **Human Activity Recognition (HAR)**. The primary objective is to create high-quality, time-aligned datasets that can be used to train machine learning models for detecting and classifying human activities such as walking, sitting, standing, and other gestures.

---

### 📦 **Data Sources**

1. **IMU Sensor Data**

   * Collected from **wearable devices** placed on participants (e.g., arms, legs).
   * Sensors capture:
     * Accelerometer (A_x, A_y, A_z)
     * Gyroscope (G_x, G_y, G_z)
     * Magnetometer (M_x, M_y, M_z)
     * Capacitance values
   * Stored in raw `.csv` files for each limb (e.g., `Star_P11_Left.csv`, `Star_P11_Right.csv`).

2. **Video Recordings**

   * Parallel video recordings were made during the sensor-based data collection.
   * These were used to manually annotate activities for each session.

---

### 🧠 **Manual Annotations**

* Annotations were done by **watching the video recordings** and marking the start/end times of each activity.
* These were saved in `.xlsx` format (e.g., `VideoLabelling_Person11_Annotated_Final.xlsx`).
* Converted to `.csv` format (e.g., `TVL11.csv`) to align with the sensor data pipeline.

---

### 🔁 **Data Processing Pipeline**

1. **Manual Annotation**  
   Annotators label activities in video and store them in `.xlsx` or `.csv` files.

2. **Session Labeling**  
   Using `GenerateSessions.py`, label files are optionally split into per-session CSVs (`S1` to `S5`), depending on activity duration or number of repetitions.

3. **Sensor Alignment**  
   Left and right IMU sensor files are aligned with label timings to generate `.npy` files per session using a custom merging script. These files contain synchronized features + labels.

4. **Subtitle Generation (Optional)**  
   The label CSVs can be converted into `.srt` subtitle files (`subtitle_generator.py`) to overlay activity labels on the video for visualization.

5. **Final Video Output (Optional)**  
   The subtitles are merged with the original video (e.g., using `ffmpeg`) to generate an annotated presentation video (e.g., `3xSpeed_Annotation_Video_With_Subtitles_For_Presentation.mp4`).

---

### 🗂️ **Folder Structure Overview**

```
HAR-Data-Preparation/
├── ManualAnnotations/       # Excel files with manual activity labels
├── LabelCSVs/               # Processed activity labels (AllSessions, Sessions, TVL)
├── SensorData/              # Raw IMU sensor files (.csv)
├── AlignedSensorData/       # Final processed .npy files
├── Outputs/
│   ├── subtitles/           # Generated .srt subtitle files
│   └── videos/              # Final annotated videos with overlays
├── Scripts/                 # All scripts for data prep, alignment, visualization
└── README.md
```

---

### 🧩 **Key Scripts**

| Script                     | Function                                                |
| -------------------------- | ------------------------------------------------------- |
| `GenerateSessions.py`      | Splits AllSessions label file into session-wise CSVs    |
| `imu_data_reader.py`       | Converts raw IMU sensor CSVs to structured numpy arrays |
| `subtitle_generator.py`    | Converts CSV labels to `.srt` subtitles                 |
| `sync_gesture_selector.py` | Visualizes sensor data for manual selection             |
| `read1.py`, `time.py`      | Utility scripts for debugging and time conversion       |

---

## 📝 Overview

This project includes:
- Manual video annotations using Excel
- Session-wise label splitting
- IMU sensor alignment with labels
- Subtitle file generation (`.srt`)
- Optional video overlay for presentations

---

## 🔁 Data Preparation Pipeline

1. **Manual Annotations**
   - Annotated in `VideoLabelling_PersonXX_Annotated_Final.xlsx`
   - Converted to `TVLXX.csv` for automation

2. **Session Label Splitting**
   - Done using `GenerateSessions.py`
   - Output: `Star_Label_P11_S1.csv` ... `S5.csv`

3. **Sensor Alignment**
   - `imu_data_reader.py` + your merging logic
   - Produces `Star_P11_S1.npy`, etc.

4. **Subtitle Generation**
   - `subtitle_generator.py` converts CSV → `.srt`

5. **Final Video**
   - `ffmpeg` or other tool to burn `.srt` into video

---

## 💻 Scripts

- `GenerateSessions.py`: Splits AllSessions label CSV
- `imu_data_reader.py`: Loads and converts sensor data
- `subtitle_generator.py`: Converts labels to `.srt`
- `sync_gesture_selector.py`: Visual IMU explorer
- `read1.py`, `time.py`: Utilities

---

## 🧩 Requirements

- Python 3.x
- Packages:
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `scipy`

---

## 🎥 Output

- Annotated video: `Outputs/videos/3xSpeed_Annotation_Video_With_Subtitles_For_Presentation.mp4`
- Subtitle file (CSV): `Outputs/subtitles/DFKI_Activity_Annotations.csv`
- Subtitle file (SRT): `Outputs/subtitles/DFKI_Activity_Annotations.srt`

---

## 📬 Contact

Prepared by **Keerti Prem Gadde**  
Date: **May 2022**
