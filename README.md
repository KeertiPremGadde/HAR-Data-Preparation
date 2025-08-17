**HAR-Data-Preparation**
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

4. **Subtitle Generation (Optional)**
   - `subtitle_generator.py` converts CSV → `.srt`

5. **Final Video (Optional)**
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
- Subtitle file(csv): `Outputs/subtitles/DFKI_Activity_Annotations.csv`
- Subtitle file(srt): `Outputs/subtitles/DFKI_Activity_Annotations.srt`

---

## 📬 Contact

Prepared by Keerti Prem Gadde 
Date: May 2022
