# GPU Climate Analytics

GPU-accelerated climate data analysis combining **NVIDIA CuPy** with **ArcGIS Maps SDK for JavaScript 5.0**. Processes 500 NOAA-like weather stations across the contiguous United States (CONUS).

## Live Demo

**[View Live App](https://garridolecca.github.io/Nvidia_esri-gpu-climate-analytics/)**

## GPU Analytics Pipeline

| Analysis | Method | Description |
|---|---|---|
| **Temperature Surface** | GPU IDW | Inverse Distance Weighted interpolation (120x60 grid) |
| **Precipitation Surface** | GPU IDW | Rainfall interpolation across CONUS |
| **Temperature Anomalies** | GPU Stats | Stations >5C from latitude-expected values |
| **Climate Zones** | GPU K-Means | 6-zone clustering on temp/precip/humidity/latitude |

## Tech Stack

- **GPU Compute**: NVIDIA RTX A4000 + CuPy (CUDA-accelerated NumPy)
- **Visualization**: ArcGIS Maps SDK for JavaScript 5.0 (Web Components)
- **Data**: 500 synthetic NOAA-like weather stations
- **Region**: Contiguous United States (CONUS)

## Setup

```bash
pip install -r requirements.txt
python scripts/download_data.py
python scripts/run_analytics.py
```

Then open `webapp/index.html` or deploy to GitHub Pages.

## Results

- **500 Stations** with temperature, precipitation, wind, humidity
- **7,200 Grid Points** interpolated via GPU IDW
- **6 Climate Zones** identified by GPU K-Means
- Full distance matrix computation on GPU via CuPy
