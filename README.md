# CardioMonitor Pro

**Cardiovascular signal processing system** — a C++ DSP engine wrapped as a Python C Extension, served by a Flask REST API, and visualised in a real-time web dashboard.

---

## Overview

CardioMonitor Pro analyses ECG recordings from the PhysioNet MIT-BIH Arrhythmia Database and runs a full clinical-grade pipeline: signal filtering → R-peak detection → HRV analysis → arrhythmia classification → composite risk scoring. All heavy computation runs in a hand-written C++ engine (`cardio_engine`) for maximum performance. If the C++ extension is unavailable, the server falls back to a pure-Python implementation automatically.

---

## Architecture

```
index.html          ← Browser UI (Chart.js, Three.js, dark theme)
    │  REST / SSE
server.py           ← Flask backend + CardioAnalyzer pipeline
    │  Python C API
cardio_engine.cpp   ← C++ DSP engine (compiled via setup.py)
```

---

## Features

### C++ DSP Engine (`cardio_engine.cpp`)
- **Cooley-Tukey FFT** — iterative in-place, power-of-2, with optional inverse
- **Welch PSD** — overlapping Hann-windowed segments for power spectral density
- **Butterworth bandpass filter** — bilinear-transform IIR, cascaded biquad sections
- **Pan-Tompkins R-peak detector** — adaptive threshold + 200 ms refractory period
- **HRV time-domain**: mean RR, SDNN, RMSSD, pNN50, SDSD
- **HRV frequency-domain**: VLF / LF / HF band powers, LF/HF ratio
- **Poincaré analysis**: SD1, SD2, CSI, CVI, ellipse area
- **Detrended Fluctuation Analysis (DFA)**: α1 and α2 scaling exponents
- **ECG interval measurements**: PR, QRS, QT, QTc (Bazett), ST elevation
- **Nonlinear entropy**: Approximate Entropy (ApEn) and Sample Entropy (SampEn)

### Python Server (`server.py`)
- Fetches real ECG data from **PhysioNet MIT-BIH** (48 records, format 212 and 16) over HTTP — no `wfdb` package required
- Falls back to a **realistic synthetic ECG generator** when PhysioNet is unreachable
- **Composite risk scoring** (0–100) combining heart rate deviation, SDNN, LF/HF ratio, QTc, Poincaré complexity, DFA α1, and ApEn
- **Arrhythmia classification**: Normal Sinus, Sinus Bradycardia/Tachycardia, Atrial Fibrillation, Ventricular patterns, AV Block, Bigeminy, Long QT
- In-memory record caching with thread-safe locking
- **Server-Sent Events (SSE)** endpoint for real-time ECG streaming

### REST API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/records` | List all available MIT-BIH record IDs |
| GET | `/api/load/<id>` | Load a record (returns 10-second preview) |
| GET | `/api/analyze/<id>` | Run full analysis pipeline |
| GET | `/api/stream/<id>` | Stream ECG samples via SSE |
| GET | `/api/synthetic` | Generate synthetic ECG (seed, hr, noise, duration) |
| GET | `/api/health` | Health check + C++ engine status |

Query parameters for `/api/analyze`: `seconds` (default 60, max 300), `offset` (seconds into record).

---

## Setup

### Requirements

- Python 3.10+
- GCC / Clang with C++17 support
- Python development headers (`python3-dev` / `python3-devel`)

```bash
pip install flask numpy scipy
```

### Build the C++ Extension

```bash
python setup.py build_ext --inplace
```

The extension compiles with `-O3 -march=native -std=c++17 -ffast-math`. No third-party C++ dependencies (no pybind11, no Boost).

### Run the Server

```bash
python server.py
```

Starts on port `8765` by default. Override with the `PORT` environment variable:

```bash
PORT=9000 python server.py
```

The server logs whether the C++ engine loaded successfully:

```
[C++] cardio_engine loaded ✓
Starting Cardiovascular Monitor on port 8765
```

### Open the Dashboard

Open `index.html` in a browser (or serve it statically). The UI connects to `http://localhost:8765`.

---

## Project Structure

```
.
├── cardio_engine.cpp   # C++ DSP engine (Python C Extension)
├── server.py           # Flask backend + analysis pipeline
├── index.html          # Frontend dashboard
└── setup.py            # Build script
```

---

## Fallback Behaviour

If the C++ extension cannot be imported (e.g. not yet compiled), `server.py` automatically uses its pure-Python fallback implementations for every analysis function. All API endpoints remain fully functional. The `/api/health` response includes `"cpp_engine": false` to indicate fallback mode.

---

## Data Source

ECG data is sourced from the **PhysioNet MIT-BIH Arrhythmia Database** (mitdb 1.0.0) and the **MIT-BIH Normal Sinus Rhythm Database** (nsrdb 1.0.0), fetched on demand via the PhysioNet WFDB REST API. Records are cached in memory for the lifetime of the server process.

> **Note:** This project is intended for research and educational purposes. It is not a certified medical device and should not be used for clinical diagnosis.
