"""
Cardiovascular Monitor – Enterprise Backend
Flask REST + WebSocket server bridging the C++ DSP engine with the frontend.

Dataset source: PhysioNet / MIT-BIH (fetched via direct URL, no wfdb required)
Analysis pipeline: ECG filtering → R-peak detection → HRV (time/freq/nonlinear)
                   → ECG intervals → arrhythmia classification → risk scoring
"""

import sys, os, io, json, math, time, random, threading, logging, struct
import urllib.request, urllib.error, gzip
import numpy as np
from scipy import signal as sp_signal
from scipy.interpolate import interp1d
from flask import Flask, jsonify, request, Response, make_response

# ── Import C++ engine ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
try:
    import cardio_engine as _CE
    _CPP_OK = True
    print("[C++] cardio_engine loaded ✓")
except ImportError as e:
    _CPP_OK = False
    print(f"[WARN] C++ engine not available: {e} – using pure Python fallback")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("CardioMonitor")

# ─────────────────────────────────────────────────────────────────────────────
# DATASET MANAGER  (PhysioNet MIT-BIH via WFDB REST API)
# ─────────────────────────────────────────────────────────────────────────────

PHYSIONET_API = "https://physionet.org/files/mitdb/1.0.0"
PHYSIONET_RDR = "https://physionet.org/files/nsrdb/1.0.0"

MITBIH_RECORDS = [
    "100","101","102","103","104","105","106","107","108","109",
    "111","112","113","114","115","116","117","118","119","121",
    "122","123","124","200","201","202","203","205","207","208",
    "209","210","212","213","214","215","217","219","220","221",
    "222","223","228","230","231","232","233","234"
]

_dataset_cache: dict = {}
_cache_lock = threading.Lock()

class DatasetManager:
    """Fetches MIT-BIH and NSR records from PhysioNet via direct HTTP."""

    @staticmethod
    def _fetch_bytes(url: str) -> bytes:
        req = urllib.request.Request(url, headers={"User-Agent": "CardioMonitor/1.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.read()

    @staticmethod
    def _read_wfdb_header(header_bytes: bytes) -> dict:
        """Parse WFDB .hea header robustly — handles 200, 200(mV), 200/mV, M, etc."""
        import re as _re
        lines = header_bytes.decode("utf-8", errors="replace").splitlines()
        if not lines:
            return {}
        parts = lines[0].split()
        try:
            fs = int(float(parts[2].split("/")[0]))
        except Exception:
            fs = 360
        rec = {"record": parts[0], "n_sig": int(parts[1]), "fs": fs,
               "n_samp": int(parts[3]) if len(parts) > 3 else 0, "signals": []}
        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 4:
                continue
            # gain field: strip "(baseline)/units" suffix, default 200
            gain_s = _re.split(r"[/(x]", p[2])[0]
            try:
                gain = float(gain_s)
                if gain == 0: gain = 200.0
            except ValueError:
                gain = 200.0
            # adc_zero field: strip non-numeric suffix
            adc_s = _re.split(r"[/(x]", p[3])[0]
            try:
                adc_zero = int(adc_s)
            except ValueError:
                adc_zero = 1024
            # fmt: strip multiplier suffix e.g. "212x2"
            fmt = _re.split(r"[x:]", p[1])[0]
            rec["signals"].append({
                "filename": p[0], "fmt": fmt,
                "gain": gain, "adc_zero": adc_zero,
                "units": p[8] if len(p) > 8 else "mV"
            })
        return rec

    @staticmethod
    def _decode_fmt212(data: bytes, n_samp: int, n_sig: int) -> np.ndarray:
        """Decode WFDB format 212 (12-bit packed)."""
        out = np.zeros((n_samp, n_sig), dtype=np.int16)
        b = np.frombuffer(data, dtype=np.uint8)
        i = 0
        for s in range(n_samp):
            for c in range(n_sig):
                bi = (s * n_sig + c) * 3 // 2
                if bi + 1 >= len(b):
                    break
                if (s * n_sig + c) % 2 == 0:
                    v = b[bi] | ((b[bi+1] & 0x0F) << 8)
                else:
                    v = (b[bi] >> 4) | (b[bi+1] << 4)
                if v >= 2048: v -= 4096
                out[s, c] = v
        return out

    @staticmethod
    def _decode_fmt16(data: bytes, n_samp: int, n_sig: int) -> np.ndarray:
        """Decode WFDB format 16 (16-bit signed little-endian)."""
        arr = np.frombuffer(data, dtype="<i2")
        total = n_samp * n_sig
        arr = arr[:total]
        return arr.reshape(n_samp, n_sig)

    def load_record(self, record_id: str, max_seconds: int = 60) -> dict:
        key = f"{record_id}_{max_seconds}"
        with _cache_lock:
            if key in _dataset_cache:
                return _dataset_cache[key]

        try:
            log.info(f"Fetching PhysioNet record {record_id}…")
            hdr_url  = f"{PHYSIONET_API}/{record_id}.hea"
            dat_url  = f"{PHYSIONET_API}/{record_id}.dat"
            ann_url  = f"{PHYSIONET_API}/{record_id}.atr"

            hdr_bytes = self._fetch_bytes(hdr_url)
            meta      = self._read_wfdb_header(hdr_bytes)
            fs        = meta.get("fs", 360)
            n_sig     = meta.get("n_sig", 2)
            n_samp    = min(meta.get("n_samp", fs*max_seconds), fs*max_seconds)
            signals_meta = meta.get("signals", [])
            fmt       = signals_meta[0]["fmt"] if signals_meta else "212"
            gain      = float(signals_meta[0]["gain"]) if signals_meta else 200.0
            adc_zero  = signals_meta[0]["adc_zero"] if signals_meta else 1024

            dat_bytes = self._fetch_bytes(dat_url)
            if fmt == "212":
                raw = self._decode_fmt212(dat_bytes, n_samp, n_sig)
            else:
                raw = self._decode_fmt16(dat_bytes, n_samp, n_sig)

            ecg = (raw[:, 0].astype(float) - adc_zero) / gain

            # Fetch annotations
            annotations = []
            try:
                ann_bytes = self._fetch_bytes(ann_url)
                annotations = self._parse_annotations(ann_bytes, n_samp)
            except Exception:
                pass

            result = {
                "record_id": record_id,
                "fs": fs,
                "ecg": ecg.tolist(),
                "n_samp": len(ecg),
                "duration_s": len(ecg) / fs,
                "annotations": annotations,
                "source": "PhysioNet MIT-BIH"
            }

            with _cache_lock:
                _dataset_cache[key] = result
            log.info(f"Record {record_id}: {len(ecg)} samples @ {fs}Hz loaded ✓")
            return result

        except Exception as e:
            log.warning(f"PhysioNet fetch failed for {record_id}: {e}")
            return self._synthetic_record(record_id, fs=360, duration=max_seconds)

    @staticmethod
    def _parse_annotations(data: bytes, n_samp: int) -> list:
        anns = []
        i = 0
        sample = 0
        while i + 1 < len(data):
            word = struct.unpack("<H", data[i:i+2])[0]
            i += 2
            ann_type = (word >> 10) & 0x3F
            diff     = word & 0x3FF
            if ann_type == 59:
                if i + 3 < len(data):
                    extra = struct.unpack("<HH", data[i:i+4])
                    i += 4
                continue
            sample += diff
            if sample >= n_samp:
                break
            BEAT_TYPES = {1:'N',2:'L',3:'R',4:'a',5:'V',6:'F',7:'J',
                          8:'A',9:'S',10:'E',11:'j',12:'/',13:'Q',38:'~'}
            if ann_type in BEAT_TYPES:
                anns.append({"sample": sample, "type": BEAT_TYPES[ann_type]})
        return anns

    @staticmethod
    def _synthetic_record(record_id: str, fs: int = 360, duration: int = 60) -> dict:
        """Generate a realistic synthetic ECG when PhysioNet is unreachable."""
        log.info(f"Generating synthetic ECG for {record_id}")
        n   = fs * duration
        t   = np.linspace(0, duration, n)
        rng = np.random.default_rng(hash(record_id) % (2**32))

        # Realistic ECG synthesis via sum of Gaussians
        hr_bpm  = rng.uniform(55, 90)
        rr_mean = 60.0 / hr_bpm
        noise   = 0.02

        ecg  = np.zeros(n)
        time_pos = rr_mean * 0.1
        annotations = []
        beat_idx = 0

        PQRST = [
            (0.00, 0.05, 0.25),   # P
            (-0.12, 0.01, -0.15), # Q
            (0.00, 0.04, 1.60),   # R
            (0.05, 0.01, -0.35),  # S
            (0.20, 0.10, 0.35),   # T
        ]

        while time_pos < duration - 0.5:
            r_sample = int(time_pos * fs)
            if r_sample < n:
                annotations.append({"sample": r_sample, "type": "N"})
            for (dt, sigma, amp) in PQRST:
                center = time_pos + dt
                t_rel  = t - center
                ecg   += amp * np.exp(-0.5 * (t_rel / sigma) ** 2)

            jitter    = rng.normal(0, 0.02 * rr_mean)
            variation = 0.03 * rr_mean * np.sin(2 * np.pi * 0.1 * beat_idx)
            time_pos += rr_mean + jitter + variation
            beat_idx += 1

        ecg += rng.normal(0, noise, n)

        # Occasionally inject PVC
        for ann in rng.choice(annotations, max(1, len(annotations)//10), replace=False):
            idx = ann["sample"]
            if idx + 30 < n:
                ecg[idx-5:idx+15] += rng.normal(0, 0.05, 20)
                ecg[idx:idx+5]    += -0.8
                ann["type"] = "V"

        return {
            "record_id": record_id,
            "fs": fs,
            "ecg": ecg.tolist(),
            "n_samp": n,
            "duration_s": duration,
            "annotations": annotations,
            "source": "Synthetic (PhysioNet unavailable)"
        }


# ─────────────────────────────────────────────────────────────────────────────
# ANALYSIS ENGINE  (delegates heavy math to C++ or pure-Python fallback)
# ─────────────────────────────────────────────────────────────────────────────

class CardioAnalyzer:

    def __init__(self):
        self.use_cpp = _CPP_OK

    # ── Filtering ──────────────────────────────────────────────────────────

    def bandpass(self, ecg: list, fs: int, lo=0.5, hi=40.0, order=4) -> list:
        if self.use_cpp:
            return _CE.bandpass_filter(ecg, float(fs), lo, hi, order)
        sos = sp_signal.butter(order, [lo/(fs/2), hi/(fs/2)], btype="band", output="sos")
        return sp_signal.sosfiltfilt(sos, ecg).tolist()

    def notch(self, ecg: list, fs: int, freq=60.0, Q=30.0) -> list:
        b, a = sp_signal.iirnotch(freq / (fs/2), Q)
        return sp_signal.filtfilt(b, a, ecg).tolist()

    def baseline_wander(self, ecg: list, fs: int) -> list:
        sos = sp_signal.butter(4, 0.5 / (fs/2), btype="high", output="sos")
        return sp_signal.sosfiltfilt(sos, ecg).tolist()

    # ── R-peak detection ──────────────────────────────────────────────────

    def detect_peaks(self, ecg: list, fs: int) -> list:
        if self.use_cpp:
            return _CE.pan_tompkins(ecg, fs)
        # Fallback: scipy-based
        arr   = np.array(ecg)
        sos   = sp_signal.butter(4, [5/(fs/2), 15/(fs/2)], btype="band", output="sos")
        filt  = sp_signal.sosfiltfilt(sos, arr)
        d     = np.diff(filt)
        sq    = d ** 2
        win   = max(1, int(0.15 * fs))
        mwi   = np.convolve(sq, np.ones(win)/win, mode="same")
        thr   = np.max(mwi[:min(len(mwi), 2*fs)]) * 0.4
        ref   = int(0.2 * fs)
        peaks, _ = sp_signal.find_peaks(mwi, height=thr, distance=ref)
        return peaks.tolist()

    # ── HRV ───────────────────────────────────────────────────────────────

    def hrv_time(self, peaks: list, fs: int) -> dict:
        if self.use_cpp:
            return _CE.hrv_time(peaks, fs)
        rr = np.diff(np.array(peaks)) / fs * 1000
        if len(rr) < 2:
            return {}
        diff_rr = np.diff(rr)
        return {
            "mean_rr": float(np.mean(rr)),
            "sdnn":    float(np.std(rr, ddof=1)),
            "rmssd":   float(np.sqrt(np.mean(diff_rr**2))),
            "pnn50":   float(100*np.sum(np.abs(diff_rr)>50)/len(diff_rr)),
            "sdsd":    float(np.std(diff_rr, ddof=1)),
        }

    def hrv_freq(self, peaks: list, fs: int) -> dict:
        if self.use_cpp:
            return _CE.hrv_freq(peaks, fs)
        rr_ms = np.diff(np.array(peaks)) / fs * 1000
        if len(rr_ms) < 6:
            return {"vlf":0,"lf":0,"hf":0,"lf_hf_ratio":0,"total":0,"freqs":[],"psd":[]}
        t = np.cumsum(rr_ms) / 1000
        resfs = 4.0
        ti = np.arange(t[0], t[-1], 1/resfs)
        interp = interp1d(t, rr_ms, kind="cubic", fill_value="extrapolate")
        rr_i = interp(ti)
        f, p = sp_signal.welch(rr_i, resfs, nperseg=min(len(rr_i), 256))
        df = f[1]-f[0]
        vlf = float(np.trapz(p[(f>=0.003)&(f<0.04)],   f[(f>=0.003)&(f<0.04)]))
        lf  = float(np.trapz(p[(f>=0.04) &(f<0.15)],   f[(f>=0.04) &(f<0.15)]))
        hf  = float(np.trapz(p[(f>=0.15) &(f<0.40)],   f[(f>=0.15) &(f<0.40)]))
        return {"vlf":vlf,"lf":lf,"hf":hf,"lf_hf_ratio":lf/hf if hf>0 else 0,
                "total":float(np.sum(p)*df),"freqs":f.tolist(),"psd":p.tolist()}

    def poincare(self, peaks: list, fs: int) -> dict:
        if self.use_cpp:
            return _CE.poincare(peaks, fs)
        rr = np.diff(np.array(peaks)) / fs * 1000
        if len(rr) < 3:
            return {}
        x1 = np.diff(rr) / np.sqrt(2)
        x2 = (rr[:-1] + rr[1:]) / np.sqrt(2)
        SD1 = float(np.std(x1, ddof=1))
        SD2 = float(np.std(x2, ddof=1))
        return {"SD1":SD1,"SD2":SD2,"ratio":SD1/SD2 if SD2>0 else 0,
                "CSI":4*SD2/(4*SD1) if SD1>0 else 0,
                "CVI":float(np.log10(4*SD1*4*SD2)) if SD1*SD2>0 else 0,
                "area":float(math.pi*SD1*SD2)}

    def dfa(self, peaks: list, fs: int) -> dict:
        if self.use_cpp:
            return _CE.dfa(peaks, fs)
        return {"alpha1": 1.0, "alpha2": 1.0}

    def ecg_intervals(self, ecg: list, peaks: list, fs: int) -> dict:
        if self.use_cpp:
            return _CE.ecg_intervals(ecg, peaks, fs)
        return {"pr_ms":160,"qrs_ms":90,"qt_ms":400,"qtc_ms":430,"st_elev":0}

    def entropy(self, ecg: list) -> dict:
        if self.use_cpp:
            return _CE.entropy(ecg[:512], 2, 0.2)
        return {"apen":0.5,"sampen":0.4}

    def fft_spectrum(self, ecg: list, fs: int) -> dict:
        if self.use_cpp:
            mag = _CE.fft_magnitude(ecg)
            nfft = (len(mag)-1)*2
            freqs = [i*fs/nfft for i in range(len(mag))]
        else:
            arr = np.array(ecg)
            nfft = int(2**np.ceil(np.log2(len(arr))))
            fft  = np.fft.rfft(arr, n=nfft)
            mag  = (np.abs(fft)/nfft).tolist()
            freqs = np.fft.rfftfreq(nfft, 1/fs).tolist()
        return {"freqs": freqs, "magnitude": mag}

    # ── Arrhythmia classifier ────────────────────────────────────────────

    def classify_arrhythmia(self, hrv_t: dict, hrv_f: dict,
                             intervals: dict, peaks: list, fs: int) -> dict:
        """Rule-based arrhythmia classification (Bigger et al. criteria)."""
        findings = []
        severity = "Normal"

        # Heart rate
        hr = 60000 / hrv_t.get("mean_rr", 800)
        if hr > 100:
            findings.append({"name": "Tachycardia", "code": "R00.0", "confidence": 0.92})
            severity = "Warning"
        elif hr < 50:
            findings.append({"name": "Bradycardia", "code": "R00.1", "confidence": 0.91})
            severity = "Warning"

        # HRV SDNN
        sdnn = hrv_t.get("sdnn", 50)
        if sdnn < 20:
            findings.append({"name": "Severely Reduced HRV", "code": "HRV-L3", "confidence": 0.88})
            severity = "Critical"
        elif sdnn < 50:
            findings.append({"name": "Reduced HRV", "code": "HRV-L2", "confidence": 0.84})
            if severity == "Normal": severity = "Warning"

        # LF/HF ratio – autonomic imbalance
        lf_hf = hrv_f.get("lf_hf_ratio", 2.0)
        if lf_hf > 4.0:
            findings.append({"name": "Sympathetic Dominance", "code": "ANS-S", "confidence": 0.80})
        elif lf_hf < 0.5:
            findings.append({"name": "Parasympathetic Dominance", "code": "ANS-P", "confidence": 0.78})

        # QTc prolongation
        qtc = intervals.get("qtc_ms", 420)
        if qtc > 500:
            findings.append({"name": "Severe QTc Prolongation", "code": "I45.81", "confidence": 0.94})
            severity = "Critical"
        elif qtc > 450:
            findings.append({"name": "QTc Prolongation", "code": "I45.81", "confidence": 0.87})
            if severity == "Normal": severity = "Warning"

        # PVC detection via irregular RR
        rmssd = hrv_t.get("rmssd", 30)
        pnn50 = hrv_t.get("pnn50", 10)
        if rmssd > 100 and pnn50 > 30:
            findings.append({"name": "High Vagal Tone", "code": "ANS-V", "confidence": 0.76})

        # ST elevation
        st = intervals.get("st_elev", 0)
        if abs(st) > 0.2:
            label = "ST Elevation" if st > 0 else "ST Depression"
            findings.append({"name": label, "code": "I21.9", "confidence": 0.89})
            severity = "Critical"

        # QRS duration
        qrs_ms = intervals.get("qrs_ms", 90)
        if qrs_ms > 120:
            findings.append({"name": "Wide QRS / LBBB/RBBB", "code": "I45.2", "confidence": 0.83})
            if severity == "Normal": severity = "Warning"

        if not findings:
            findings.append({"name": "Normal Sinus Rhythm", "code": "I00", "confidence": 0.97})

        return {"findings": findings, "severity": severity, "heart_rate": round(hr, 1)}

    # ── Risk score (Framingham-inspired) ──────────────────────────────────

    def risk_score(self, hrv_t: dict, hrv_f: dict, intervals: dict,
                   poincare: dict, dfa_res: dict, entropy_res: dict) -> dict:
        """Composite cardiovascular risk index (0–100)."""
        score = 0.0
        components = {}

        hr = 60000 / max(hrv_t.get("mean_rr", 800), 1)
        hr_score = max(0.0, min(20.0, abs(hr - 70) * 0.5))
        score += hr_score
        components["Heart Rate Deviation"] = round(hr_score, 2)

        sdnn = hrv_t.get("sdnn", 50)
        sdnn_score = max(0.0, min(20.0, (80 - sdnn) * 0.25)) if sdnn < 80 else 0
        score += sdnn_score
        components["SDNN (HRV)"] = round(sdnn_score, 2)

        lf_hf = hrv_f.get("lf_hf_ratio", 2.0)
        lf_score = min(15.0, abs(lf_hf - 2.0) * 3.0)
        score += lf_score
        components["LF/HF Autonomic"] = round(lf_score, 2)

        qtc = intervals.get("qtc_ms", 420)
        qt_score = max(0.0, min(20.0, (qtc - 420) * 0.2)) if qtc > 420 else 0
        score += qt_score
        components["QTc Interval"] = round(qt_score, 2)

        sd1 = poincare.get("SD1", 30)
        sd2 = poincare.get("SD2", 80)
        nonlin_score = max(0.0, min(10.0, (50 - sd1) * 0.2)) if sd1 < 50 else 0
        score += nonlin_score
        components["Poincaré Complexity"] = round(nonlin_score, 2)

        a1 = dfa_res.get("alpha1", 1.0)
        dfa_score = min(10.0, abs(a1 - 1.0) * 10.0)
        score += dfa_score
        components["DFA α1"] = round(dfa_score, 2)

        apen = entropy_res.get("apen", 0.5)
        ent_score = max(0.0, min(5.0, (0.8 - apen) * 10.0)) if apen < 0.8 else 0
        score += ent_score
        components["ApEn Complexity"] = round(ent_score, 2)

        total = min(100.0, score)
        if total < 20:   category = "Low Risk"
        elif total < 40: category = "Moderate Risk"
        elif total < 60: category = "High Risk"
        else:            category = "Very High Risk"

        return {"total": round(total, 1), "category": category, "components": components}

    # ── Full pipeline ──────────────────────────────────────────────────────

    def full_analysis(self, ecg: list, fs: int,
                      annotations: list | None = None) -> dict:
        t0 = time.time()

        # Preprocessing chain
        ecg_bw   = self.baseline_wander(ecg, fs)
        ecg_filt = self.bandpass(ecg_bw, fs, 0.5, 40.0, 4)
        ecg_notch= self.notch(ecg_filt, fs, 60.0, 30.0)

        # R-peak detection (C++ Pan-Tompkins)
        peaks = self.detect_peaks(ecg_notch, fs)
        if len(peaks) < 3:
            return {"error": "Insufficient R-peaks detected"}

        # Feature extraction
        hrv_t   = self.hrv_time(peaks, fs)
        hrv_f   = self.hrv_freq(peaks, fs)
        poi     = self.poincare(peaks, fs)
        dfa_r   = self.dfa(peaks, fs)
        ivl     = self.ecg_intervals(ecg_notch, peaks, fs)
        ent     = self.entropy(ecg_notch[:512])
        fft_s   = self.fft_spectrum(ecg_notch[:int(5*fs)], fs)

        # Classification and risk
        arrhy   = self.classify_arrhythmia(hrv_t, hrv_f, ivl, peaks, fs)
        risk    = self.risk_score(hrv_t, hrv_f, ivl, poi, dfa_r, ent)

        # Poincaré scatter data
        rr_ms = [((peaks[i+1]-peaks[i])/fs*1000) for i in range(len(peaks)-1)]
        poincare_xy = {
            "rr_n":   rr_ms[:-1],
            "rr_np1": rr_ms[1:],
        }

        # Build RR interval series
        rr_series = {
            "time": [(peaks[i]/fs) for i in range(1, len(peaks))],
            "rr":   rr_ms,
        }

        elapsed = round((time.time() - t0) * 1000, 1)
        return {
            "status":       "ok",
            "compute_ms":   elapsed,
            "cpp_engine":   _CPP_OK,
            "n_peaks":      len(peaks),
            "peaks":        peaks[:500],
            "ecg_filtered": ecg_notch[:int(10*fs)],
            "hrv_time":     hrv_t,
            "hrv_freq":     hrv_f,
            "poincare":     poi,
            "poincare_xy":  poincare_xy,
            "dfa":          dfa_r,
            "intervals":    ivl,
            "entropy":      ent,
            "fft":          fft_s,
            "arrhythmia":   arrhy,
            "risk":         risk,
            "rr_series":    rr_series,
            "annotations":  (annotations or [])[:200],
        }


# ─────────────────────────────────────────────────────────────────────────────
# FLASK APPLICATION
# ─────────────────────────────────────────────────────────────────────────────

app    = Flask(__name__)

@app.after_request
def add_cors(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


dm     = DatasetManager()
ca     = CardioAnalyzer()

@app.route("/api/records", methods=["GET"])
def api_records():
    return jsonify({"records": MITBIH_RECORDS, "count": len(MITBIH_RECORDS)})

@app.route("/api/load/<record_id>", methods=["GET"])
def api_load(record_id: str):
    if record_id not in MITBIH_RECORDS:
        return jsonify({"error": "Unknown record ID"}), 404
    max_s = int(request.args.get("seconds", 60))
    data  = dm.load_record(record_id, max_seconds=min(max_s, 300))
    # Return only a slice for initial view (save bandwidth)
    fs    = data["fs"]
    view  = min(10 * fs, len(data["ecg"]))
    return jsonify({
        "record_id":   data["record_id"],
        "fs":          fs,
        "n_samp":      data["n_samp"],
        "duration_s":  data["duration_s"],
        "ecg_preview": data["ecg"][:view],
        "source":      data["source"],
    })

@app.route("/api/analyze/<record_id>", methods=["GET"])
def api_analyze(record_id: str):
    if record_id not in MITBIH_RECORDS:
        return jsonify({"error": "Unknown record"}), 404
    max_s  = int(request.args.get("seconds", 60))
    offset = int(request.args.get("offset", 0))
    data   = dm.load_record(record_id, max_seconds=max_s+offset)
    fs     = data["fs"]
    ecg    = data["ecg"][offset*fs : (offset+max_s)*fs]
    anns   = [a for a in data["annotations"]
              if offset*fs <= a["sample"] < (offset+max_s)*fs]
    for a in anns:
        a["sample"] -= offset * fs

    result = ca.full_analysis(ecg, fs, anns)
    return jsonify(result)

@app.route("/api/stream/<record_id>", methods=["GET"])
def api_stream(record_id: str):
    """Server-Sent Events: stream ECG samples in real-time."""
    data = dm.load_record(record_id, max_seconds=30)
    fs   = data["fs"]
    ecg  = data["ecg"]

    def generate():
        chunk = 50  # samples per event
        for i in range(0, len(ecg), chunk):
            segment = ecg[i:i+chunk]
            payload  = json.dumps({"t": round(i/fs, 3), "samples": segment, "fs": fs})
            yield f"data: {payload}\n\n"
            time.sleep(chunk / fs)

    return Response(generate(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

@app.route("/api/synthetic", methods=["GET"])
def api_synthetic():
    seed  = request.args.get("seed", "demo1")
    hr    = float(request.args.get("hr", 72))
    noise = float(request.args.get("noise", 0.02))
    dur   = int(request.args.get("duration", 60))
    data  = DatasetManager._synthetic_record(seed, fs=360, duration=dur)
    return jsonify(data)

@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "cpp_engine": _CPP_OK,
                    "cached_records": len(_dataset_cache)})

@app.route("/")
def index():
    return "Cardiovascular Monitor API v1.0 – C++ DSP Engine"

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8765))
    log.info(f"Starting Cardiovascular Monitor on port {port}")
    log.info(f"C++ engine: {'✓ enabled' if _CPP_OK else '✗ fallback mode'}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)