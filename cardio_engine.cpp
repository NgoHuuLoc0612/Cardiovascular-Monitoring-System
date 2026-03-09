/**
 * Cardiovascular Monitor - C++ DSP Engine
 * Enterprise-grade signal processing core
 * Exposes Python API via Python C Extension (no pybind11 dependency)
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <complex>
#include <stdexcept>
#include <string>
#include <map>
#include <array>

// ─────────────────────────────────────────────────────────────────────────────
// DSP PRIMITIVES
// ─────────────────────────────────────────────────────────────────────────────

namespace dsp {

static const double PI = 3.14159265358979323846;
static const double TWO_PI = 2.0 * PI;

// Cooley-Tukey FFT (iterative, in-place, power-of-2)
void fft(std::vector<std::complex<double>>& x, bool inverse = false) {
    size_t n = x.size();
    if (n <= 1) return;

    // Bit-reversal permutation
    for (size_t i = 1, j = 0; i < n; ++i) {
        size_t bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(x[i], x[j]);
    }

    for (size_t len = 2; len <= n; len <<= 1) {
        double ang = TWO_PI / (double)len * (inverse ? -1.0 : 1.0);
        std::complex<double> wlen(std::cos(ang), std::sin(ang));
        for (size_t i = 0; i < n; i += len) {
            std::complex<double> w(1.0, 0.0);
            for (size_t j = 0; j < len / 2; ++j) {
                std::complex<double> u = x[i + j];
                std::complex<double> v = x[i + j + len / 2] * w;
                x[i + j]           = u + v;
                x[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
    if (inverse) {
        for (auto& c : x) c /= (double)n;
    }
}

// Next power of 2
size_t nextPow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

// Power Spectral Density via Welch's method
std::vector<double> welchPSD(const std::vector<double>& signal, int fs,
                              int nperseg = 256, double overlap = 0.5) {
    int step    = (int)(nperseg * (1.0 - overlap));
    int nfft    = (int)nextPow2(nperseg);
    int nsegs   = (int)((signal.size() - nperseg) / step) + 1;

    std::vector<double> psd(nfft / 2 + 1, 0.0);

    // Hann window
    std::vector<double> window(nperseg);
    double wsum2 = 0.0;
    for (int i = 0; i < nperseg; ++i) {
        window[i] = 0.5 * (1.0 - std::cos(TWO_PI * i / (nperseg - 1)));
        wsum2 += window[i] * window[i];
    }

    for (int s = 0; s < nsegs; ++s) {
        int start = s * step;
        std::vector<std::complex<double>> seg(nfft, 0.0);
        for (int i = 0; i < nperseg && (start + i) < (int)signal.size(); ++i)
            seg[i] = signal[start + i] * window[i];
        fft(seg);
        for (int i = 0; i <= nfft / 2; ++i)
            psd[i] += std::norm(seg[i]);
    }

    double scale = 1.0 / (fs * wsum2 * nsegs);
    for (int i = 1; i < (int)psd.size() - 1; ++i) psd[i] *= 2.0 * scale;
    psd[0] *= scale;
    psd.back() *= scale;

    return psd;
}

// Butterworth IIR filter coefficients (2nd-order sections)
struct Biquad { double b0,b1,b2,a1,a2; };

std::vector<Biquad> butterworthBandpass(double fs, double low_hz, double high_hz, int order = 4) {
    // Bilinear transform Butterworth bandpass
    std::vector<Biquad> sections;
    double w0_low  = 2.0 * std::tan(PI * low_hz  / fs);
    double w0_high = 2.0 * std::tan(PI * high_hz / fs);
    double bw = w0_high - w0_low;
    double w0 = std::sqrt(w0_low * w0_high);

    int half = order / 2;
    for (int k = 1; k <= half; ++k) {
        double theta = PI * (2.0 * k - 1.0) / (2.0 * half);
        double Qk = -2.0 * std::cos(theta + PI / 2.0);
        // LP prototype pole: s = Qk*j combined with bandpass transform
        double bw2 = bw * Qk;
        double b0  =  bw / (1.0 + bw2 + bw * bw / 4.0 + w0 * w0);
        double a1b = -2.0 * (1.0 - w0 * w0) / (1.0 + bw2 + bw * bw / 4.0 + w0 * w0);
        double a2b = (1.0 - bw2 + bw * bw / 4.0 + w0 * w0) /
                     (1.0 + bw2 + bw * bw / 4.0 + w0 * w0);
        sections.push_back({b0, 0.0, -b0, a1b, a2b});
    }
    if (sections.empty()) {
        double b0 = bw / (1.0 + bw + w0 * w0);
        double a1 = -2.0 * (1.0 - w0 * w0) / (1.0 + bw + w0 * w0);
        double a2 = (1.0 - bw + w0 * w0) / (1.0 + bw + w0 * w0);
        sections.push_back({b0, 0.0, -b0, a1, a2});
    }
    return sections;
}

std::vector<double> applyBiquad(const std::vector<double>& x, const std::vector<Biquad>& sos) {
    std::vector<double> y = x;
    for (const auto& s : sos) {
        double z1 = 0.0, z2 = 0.0;
        for (auto& v : y) {
            double w = v - s.a1 * z1 - s.a2 * z2;
            v = s.b0 * w + s.b1 * z1 + s.b2 * z2;
            z2 = z1; z1 = w;
        }
    }
    return y;
}

// Pan-Tompkins R-peak detector
std::vector<int> panTompkins(const std::vector<double>& ecg, int fs) {
    int n = (int)ecg.size();

    // 1. Bandpass 5-15 Hz
    auto bpSOS = butterworthBandpass(fs, 5.0, 15.0, 4);
    auto filtered = applyBiquad(ecg, bpSOS);

    // 2. Derivative
    std::vector<double> deriv(n, 0.0);
    for (int i = 2; i < n - 2; ++i)
        deriv[i] = (2*filtered[i+2] + filtered[i+1] - filtered[i-1] - 2*filtered[i-2]) / (8.0/fs);

    // 3. Squaring
    std::vector<double> sq(n);
    for (int i = 0; i < n; ++i) sq[i] = deriv[i] * deriv[i];

    // 4. Moving window integration (~150ms)
    int winSize = std::max(1, (int)(0.15 * fs));
    std::vector<double> mwi(n, 0.0);
    double s = 0.0;
    for (int i = 0; i < n; ++i) {
        s += sq[i];
        if (i >= winSize) s -= sq[i - winSize];
        mwi[i] = s / winSize;
    }

    // 5. Adaptive threshold + refractory period
    double threshold = *std::max_element(mwi.begin(), mwi.begin() + std::min(n, 2*fs)) * 0.5;
    int refractory   = (int)(0.2 * fs);
    std::vector<int> peaks;
    int last = -refractory;

    for (int i = 1; i < n - 1; ++i) {
        if (mwi[i] > threshold && mwi[i] > mwi[i-1] && mwi[i] >= mwi[i+1]
            && (i - last) > refractory) {
            peaks.push_back(i);
            last = i;
            threshold = 0.875 * threshold + 0.125 * mwi[i];
        }
    }
    return peaks;
}

// HRV Time-domain features
struct HRVTimeDomain {
    double mean_rr, sdnn, rmssd, pnn50, sdsd;
};

HRVTimeDomain computeHRVTime(const std::vector<int>& peaks, int fs) {
    if (peaks.size() < 3) return {0,0,0,0,0};
    std::vector<double> rr;
    for (size_t i = 1; i < peaks.size(); ++i)
        rr.push_back((peaks[i] - peaks[i-1]) * 1000.0 / fs);

    double mean = std::accumulate(rr.begin(), rr.end(), 0.0) / rr.size();
    double sdnn = 0.0, rmssd = 0.0, sdsd = 0.0;
    int nn50 = 0;

    std::vector<double> diffs;
    for (size_t i = 1; i < rr.size(); ++i) {
        double d = rr[i] - rr[i-1];
        diffs.push_back(d);
        rmssd += d * d;
        if (std::abs(d) > 50.0) ++nn50;
    }

    for (auto& v : rr)  sdnn += (v - mean) * (v - mean);
    sdnn  = std::sqrt(sdnn / rr.size());
    rmssd = std::sqrt(rmssd / diffs.size());

    double dmean = std::accumulate(diffs.begin(), diffs.end(), 0.0) / diffs.size();
    for (auto& d : diffs) sdsd += (d - dmean) * (d - dmean);
    sdsd = std::sqrt(sdsd / diffs.size());

    double pnn50 = 100.0 * nn50 / rr.size();

    return {mean, sdnn, rmssd, pnn50, sdsd};
}

// HRV Frequency domain (VLF/LF/HF power from PSD of RR intervals)
struct HRVFreqDomain {
    double vlf_power, lf_power, hf_power, lf_hf_ratio, total_power;
    std::vector<double> freqs, psd;
};

HRVFreqDomain computeHRVFreq(const std::vector<int>& peaks, int fs) {
    if (peaks.size() < 6) return {0,0,0,0,0,{},{}};

    // Build RR tachogram at 4Hz
    double resample_fs = 4.0;
    std::vector<double> rr_ms;
    std::vector<double> rr_times;
    for (size_t i = 1; i < peaks.size(); ++i) {
        rr_times.push_back((double)peaks[i] / fs);
        rr_ms.push_back((peaks[i] - peaks[i-1]) * 1000.0 / fs);
    }

    // Interpolate to uniform grid
    double t_start = rr_times.front();
    double t_end   = rr_times.back();
    int N = (int)((t_end - t_start) * resample_fs);
    if (N < 8) return {0,0,0,0,0,{},{}};

    std::vector<double> rr_interp(N);
    for (int i = 0; i < N; ++i) {
        double t = t_start + i / resample_fs;
        // Linear interpolation
        auto it = std::lower_bound(rr_times.begin(), rr_times.end(), t);
        if (it == rr_times.begin()) { rr_interp[i] = rr_ms[0]; continue; }
        if (it == rr_times.end())   { rr_interp[i] = rr_ms.back(); continue; }
        int idx = (int)(it - rr_times.begin());
        double alpha = (t - rr_times[idx-1]) / (rr_times[idx] - rr_times[idx-1]);
        rr_interp[i] = rr_ms[idx-1] * (1.0-alpha) + rr_ms[idx] * alpha;
    }

    // PSD via Welch
    int nperseg = std::min(N, 256);
    auto psd = welchPSD(rr_interp, (int)resample_fs, nperseg);
    int nfft  = (int)nextPow2(nperseg);

    // Frequency bins
    std::vector<double> freqs(psd.size());
    for (int i = 0; i < (int)psd.size(); ++i)
        freqs[i] = i * resample_fs / nfft;

    double df = resample_fs / nfft;
    double vlf = 0.0, lf = 0.0, hf = 0.0, total = 0.0;
    for (int i = 0; i < (int)psd.size(); ++i) {
        double f = freqs[i];
        total += psd[i] * df;
        if (f >= 0.003 && f < 0.04)  vlf += psd[i] * df;
        else if (f >= 0.04 && f < 0.15) lf  += psd[i] * df;
        else if (f >= 0.15 && f < 0.40) hf  += psd[i] * df;
    }

    return {vlf, lf, hf, (hf > 0) ? lf/hf : 0.0, total, freqs, psd};
}

// Poincaré plot features (SD1, SD2, CSI, CVI)
struct PoincareFeatures {
    double SD1, SD2, SD1_SD2_ratio, CSI, CVI, ellipse_area;
};

PoincareFeatures computePoincare(const std::vector<int>& peaks, int fs) {
    if (peaks.size() < 4) return {0,0,0,0,0,0};
    std::vector<double> rr;
    for (size_t i = 1; i < peaks.size(); ++i)
        rr.push_back((peaks[i] - peaks[i-1]) * 1000.0 / fs);

    double sd1sq = 0.0, sd2sq = 0.0;
    for (size_t i = 1; i < rr.size(); ++i) {
        double x1 = (rr[i] - rr[i-1]) / std::sqrt(2.0);
        double x2 = (rr[i] + rr[i-1]) / std::sqrt(2.0);
        sd1sq += x1 * x1;
        sd2sq += x2 * x2;
    }
    double N = rr.size() - 1.0;
    double SD1 = std::sqrt(sd1sq / N);
    double SD2 = std::sqrt(std::max(0.0, sd2sq / N - sd1sq / N));

    double mean_rr = std::accumulate(rr.begin(), rr.end(), 0.0) / rr.size();
    double L = 4.0 * SD2;
    double T = 4.0 * SD1;
    double CSI = L / T;
    double CVI = std::log10(L * T);
    double area = PI * SD1 * SD2;

    return {SD1, SD2, (SD2 > 0) ? SD1/SD2 : 0, CSI, CVI, area};
}

// DFA (Detrended Fluctuation Analysis) – α1 & α2 scaling exponents
struct DFAResult { double alpha1, alpha2; };

DFAResult computeDFA(const std::vector<int>& peaks, int fs) {
    if (peaks.size() < 16) return {0,0};
    std::vector<double> rr;
    for (size_t i = 1; i < peaks.size(); ++i)
        rr.push_back((peaks[i] - peaks[i-1]) * 1000.0 / fs);

    int n = (int)rr.size();
    double mean = std::accumulate(rr.begin(), rr.end(), 0.0) / n;

    // Cumulative sum (integrate)
    std::vector<double> y(n);
    double cum = 0.0;
    for (int i = 0; i < n; ++i) { cum += rr[i] - mean; y[i] = cum; }

    auto fluctuation = [&](int box) -> double {
        int numBoxes = n / box;
        if (numBoxes < 1) return 0.0;
        double F2 = 0.0;
        for (int b = 0; b < numBoxes; ++b) {
            int s = b * box, e = s + box;
            // Linear detrend within box
            double sx=0,sy=0,sxy=0,sx2=0;
            for (int i=s;i<e;++i){sx+=i;sy+=y[i];sxy+=i*y[i];sx2+=i*i;}
            double slope = (box*sxy - sx*sy) / (box*sx2 - sx*sx + 1e-12);
            double inter = (sy - slope*sx) / box;
            for (int i=s;i<e;++i){ double r=y[i]-slope*i-inter; F2+=r*r; }
        }
        return std::sqrt(F2 / (numBoxes * box));
    };

    std::vector<int> scales;
    for (int s = 4; s <= n/4; s = (int)(s * 1.2) + 1) scales.push_back(s);
    if (scales.size() < 4) return {0,0};

    std::vector<double> logS, logF;
    for (int s : scales) {
        double f = fluctuation(s);
        if (f > 0) { logS.push_back(std::log(s)); logF.push_back(std::log(f)); }
    }

    auto linreg = [&](int from, int to) -> double {
        if (to <= from) return 0.0;
        double sx=0,sy=0,sxy=0,sx2=0;
        int N2 = to-from;
        for (int i=from;i<to;++i){sx+=logS[i];sy+=logF[i];sxy+=logS[i]*logF[i];sx2+=logS[i]*logS[i];}
        return (N2*sxy - sx*sy) / (N2*sx2 - sx*sx + 1e-12);
    };

    int mid = (int)logS.size() / 2;
    double alpha1 = linreg(0, mid);
    double alpha2 = linreg(mid, (int)logS.size());

    return {alpha1, alpha2};
}

// QRS morphology: compute QRS width, QT interval, PR interval, ST elevation
struct ECGIntervals {
    double pr_ms, qrs_ms, qt_ms, qtc_ms, st_elev_mv;
};

ECGIntervals computeIntervals(const std::vector<double>& ecg,
                               const std::vector<int>& peaks, int fs) {
    if (peaks.size() < 2) return {0,0,0,0,0};

    double pr_sum=0, qrs_sum=0, qt_sum=0, st_sum=0;
    int cnt = 0;
    double mean_rr = 0;
    for (size_t i = 1; i < peaks.size(); ++i)
        mean_rr += (peaks[i] - peaks[i-1]) * 1000.0 / fs;
    mean_rr /= (peaks.size()-1);

    for (int ri = 0; ri < (int)peaks.size() && ri < 10; ++ri) {
        int r = peaks[ri];
        int search = (int)(0.1 * fs);

        // Q: minimum before R
        int q_start = std::max(0, r - search);
        int q_idx = q_start;
        for (int i = q_start; i < r; ++i)
            if (ecg[i] < ecg[q_idx]) q_idx = i;

        // S: minimum after R
        int s_end = std::min((int)ecg.size()-1, r + search);
        int s_idx = r;
        for (int i = r; i <= s_end; ++i)
            if (ecg[i] < ecg[s_idx]) s_idx = i;

        // P: max before Q in 0.2s window
        int p_start = std::max(0, q_idx - (int)(0.2*fs));
        int p_idx = p_start;
        for (int i = p_start; i < q_idx; ++i)
            if (ecg[i] > ecg[p_idx]) p_idx = i;

        // T end: 0.4s after S
        int t_end = std::min((int)ecg.size()-1, s_idx + (int)(0.4*fs));
        int t_idx = s_idx;
        for (int i = s_idx; i <= t_end; ++i)
            if (ecg[i] > ecg[t_idx]) t_idx = i;

        // ST elevation: 0.06s after S
        int st_point = std::min((int)ecg.size()-1, s_idx + (int)(0.06*fs));
        double baseline = (p_idx > 0) ? ecg[p_idx] : 0.0;
        double st_elev  = ecg[st_point] - baseline;

        pr_sum  += (q_idx - p_idx) * 1000.0 / fs;
        qrs_sum += (s_idx - q_idx) * 1000.0 / fs;
        qt_sum  += (t_idx - q_idx) * 1000.0 / fs;
        st_sum  += st_elev;
        ++cnt;
    }

    if (cnt == 0) return {0,0,0,0,0};
    double pr   = pr_sum  / cnt;
    double qrs  = qrs_sum / cnt;
    double qt   = qt_sum  / cnt;
    double qtc  = qt / std::sqrt(mean_rr / 1000.0); // Bazett
    double st   = st_sum  / cnt;

    return {pr, qrs, qt, qtc, st};
}

// Approximate Entropy (ApEn) – complexity measure
double approxEntropy(const std::vector<double>& data, int m = 2, double r_frac = 0.2) {
    int N = std::min((int)data.size(), 512);
    if (N < m + 2) return 0.0;

    double sd = 0.0, mean = std::accumulate(data.begin(), data.begin()+N, 0.0) / N;
    for (int i=0;i<N;++i) sd += (data[i]-mean)*(data[i]-mean);
    sd = std::sqrt(sd/N);
    double r = r_frac * sd;

    auto phi = [&](int m_) -> double {
        int cnt = 0; double C = 0.0;
        for (int i=0; i<N-m_; ++i) {
            for (int j=0; j<N-m_; ++j) {
                double mx=0.0;
                for (int k=0;k<m_;++k) mx = std::max(mx, std::abs(data[i+k]-data[j+k]));
                if (mx <= r) ++cnt;
            }
            C += std::log((double)cnt / (N-m_));
            cnt = 0;
        }
        return C / (N-m_);
    };

    return phi(m) - phi(m+1);
}

// Sample Entropy (SampEn) – more robust complexity
double sampleEntropy(const std::vector<double>& data, int m=2, double r_frac=0.2) {
    int N = std::min((int)data.size(), 512);
    if (N < m + 2) return 0.0;

    double sd=0.0, mean=std::accumulate(data.begin(), data.begin()+N, 0.0)/N;
    for (int i=0;i<N;++i) sd+=(data[i]-mean)*(data[i]-mean);
    sd=std::sqrt(sd/N);
    double r=r_frac*sd;

    long long A=0, B=0;
    for (int i=0;i<N-m;++i) {
        for (int j=i+1;j<N-m;++j) {
            bool match_m=true;
            for (int k=0;k<m&&match_m;++k)
                if (std::abs(data[i+k]-data[j+k])>r) match_m=false;
            if (match_m) {
                ++B;
                if (std::abs(data[i+m]-data[j+m])<=r) ++A;
            }
        }
    }
    if (B==0 || A==0) return 0.0;
    return -std::log((double)A / B);
}

} // namespace dsp

// ─────────────────────────────────────────────────────────────────────────────
// PYTHON EXTENSION METHODS
// ─────────────────────────────────────────────────────────────────────────────

static PyObject* py_fft_magnitude(PyObject*, PyObject* args) {
    PyObject* list;
    if (!PyArg_ParseTuple(args, "O!", &PyList_Type, &list)) return nullptr;

    int n = (int)PyList_Size(list);
    size_t nfft = dsp::nextPow2(n);
    std::vector<std::complex<double>> x(nfft, 0.0);
    for (int i = 0; i < n; ++i)
        x[i] = PyFloat_AsDouble(PyList_GetItem(list, i));

    dsp::fft(x);

    PyObject* result = PyList_New(nfft/2 + 1);
    for (int i = 0; i <= (int)nfft/2; ++i)
        PyList_SetItem(result, i, PyFloat_FromDouble(std::abs(x[i]) / nfft));
    return result;
}

static PyObject* py_welch_psd(PyObject*, PyObject* args) {
    PyObject* list; int fs; int nperseg;
    if (!PyArg_ParseTuple(args, "O!ii", &PyList_Type, &list, &fs, &nperseg))
        return nullptr;

    std::vector<double> signal(PyList_Size(list));
    for (int i = 0; i < (int)signal.size(); ++i)
        signal[i] = PyFloat_AsDouble(PyList_GetItem(list, i));

    auto psd = dsp::welchPSD(signal, fs, nperseg);
    int nfft  = (int)dsp::nextPow2(nperseg);

    PyObject* freqs = PyList_New(psd.size());
    PyObject* power = PyList_New(psd.size());
    for (int i = 0; i < (int)psd.size(); ++i) {
        PyList_SetItem(freqs, i, PyFloat_FromDouble(i * (double)fs / nfft));
        PyList_SetItem(power, i, PyFloat_FromDouble(psd[i]));
    }

    PyObject* result = PyTuple_New(2);
    PyTuple_SetItem(result, 0, freqs);
    PyTuple_SetItem(result, 1, power);
    return result;
}

static PyObject* py_pan_tompkins(PyObject*, PyObject* args) {
    PyObject* list; int fs;
    if (!PyArg_ParseTuple(args, "O!i", &PyList_Type, &list, &fs)) return nullptr;

    std::vector<double> ecg(PyList_Size(list));
    for (int i = 0; i < (int)ecg.size(); ++i)
        ecg[i] = PyFloat_AsDouble(PyList_GetItem(list, i));

    auto peaks = dsp::panTompkins(ecg, fs);

    PyObject* result = PyList_New(peaks.size());
    for (int i = 0; i < (int)peaks.size(); ++i)
        PyList_SetItem(result, i, PyLong_FromLong(peaks[i]));
    return result;
}

static PyObject* py_hrv_time(PyObject*, PyObject* args) {
    PyObject* list; int fs;
    if (!PyArg_ParseTuple(args, "O!i", &PyList_Type, &list, &fs)) return nullptr;

    std::vector<int> peaks(PyList_Size(list));
    for (int i = 0; i < (int)peaks.size(); ++i)
        peaks[i] = (int)PyLong_AsLong(PyList_GetItem(list, i));

    auto h = dsp::computeHRVTime(peaks, fs);
    return Py_BuildValue("{s:d,s:d,s:d,s:d,s:d}",
        "mean_rr", h.mean_rr, "sdnn", h.sdnn,
        "rmssd",   h.rmssd,   "pnn50", h.pnn50,
        "sdsd",    h.sdsd);
}

static PyObject* py_hrv_freq(PyObject*, PyObject* args) {
    PyObject* list; int fs;
    if (!PyArg_ParseTuple(args, "O!i", &PyList_Type, &list, &fs)) return nullptr;

    std::vector<int> peaks(PyList_Size(list));
    for (int i = 0; i < (int)peaks.size(); ++i)
        peaks[i] = (int)PyLong_AsLong(PyList_GetItem(list, i));

    auto h = dsp::computeHRVFreq(peaks, fs);

    PyObject* f_list = PyList_New(h.freqs.size());
    PyObject* p_list = PyList_New(h.psd.size());
    for (int i = 0; i < (int)h.freqs.size(); ++i) {
        PyList_SetItem(f_list, i, PyFloat_FromDouble(h.freqs[i]));
        PyList_SetItem(p_list, i, PyFloat_FromDouble(h.psd[i]));
    }

    return Py_BuildValue("{s:d,s:d,s:d,s:d,s:d,s:O,s:O}",
        "vlf", h.vlf_power, "lf", h.lf_power, "hf", h.hf_power,
        "lf_hf_ratio", h.lf_hf_ratio, "total", h.total_power,
        "freqs", f_list, "psd", p_list);
}

static PyObject* py_poincare(PyObject*, PyObject* args) {
    PyObject* list; int fs;
    if (!PyArg_ParseTuple(args, "O!i", &PyList_Type, &list, &fs)) return nullptr;

    std::vector<int> peaks(PyList_Size(list));
    for (int i = 0; i < (int)peaks.size(); ++i)
        peaks[i] = (int)PyLong_AsLong(PyList_GetItem(list, i));

    auto p = dsp::computePoincare(peaks, fs);
    return Py_BuildValue("{s:d,s:d,s:d,s:d,s:d,s:d}",
        "SD1", p.SD1, "SD2", p.SD2, "ratio", p.SD1_SD2_ratio,
        "CSI", p.CSI, "CVI", p.CVI, "area", p.ellipse_area);
}

static PyObject* py_dfa(PyObject*, PyObject* args) {
    PyObject* list; int fs;
    if (!PyArg_ParseTuple(args, "O!i", &PyList_Type, &list, &fs)) return nullptr;

    std::vector<int> peaks(PyList_Size(list));
    for (int i = 0; i < (int)peaks.size(); ++i)
        peaks[i] = (int)PyLong_AsLong(PyList_GetItem(list, i));

    auto d = dsp::computeDFA(peaks, fs);
    return Py_BuildValue("{s:d,s:d}", "alpha1", d.alpha1, "alpha2", d.alpha2);
}

static PyObject* py_ecg_intervals(PyObject*, PyObject* args) {
    PyObject* ecg_list, *peaks_list; int fs;
    if (!PyArg_ParseTuple(args, "O!O!i",
        &PyList_Type, &ecg_list, &PyList_Type, &peaks_list, &fs)) return nullptr;

    std::vector<double> ecg(PyList_Size(ecg_list));
    for (int i = 0; i < (int)ecg.size(); ++i)
        ecg[i] = PyFloat_AsDouble(PyList_GetItem(ecg_list, i));

    std::vector<int> peaks(PyList_Size(peaks_list));
    for (int i = 0; i < (int)peaks.size(); ++i)
        peaks[i] = (int)PyLong_AsLong(PyList_GetItem(peaks_list, i));

    auto iv = dsp::computeIntervals(ecg, peaks, fs);
    return Py_BuildValue("{s:d,s:d,s:d,s:d,s:d}",
        "pr_ms", iv.pr_ms, "qrs_ms", iv.qrs_ms,
        "qt_ms", iv.qt_ms, "qtc_ms", iv.qtc_ms,
        "st_elev", iv.st_elev_mv);
}

static PyObject* py_entropy(PyObject*, PyObject* args) {
    PyObject* list; int m; double r;
    if (!PyArg_ParseTuple(args, "O!id", &PyList_Type, &list, &m, &r)) return nullptr;

    std::vector<double> data(PyList_Size(list));
    for (int i = 0; i < (int)data.size(); ++i)
        data[i] = PyFloat_AsDouble(PyList_GetItem(list, i));

    double apen = dsp::approxEntropy(data, m, r);
    double sen  = dsp::sampleEntropy(data, m, r);
    return Py_BuildValue("{s:d,s:d}", "apen", apen, "sampen", sen);
}

static PyObject* py_bandpass_filter(PyObject*, PyObject* args) {
    PyObject* list; double fs, lo, hi; int order;
    if (!PyArg_ParseTuple(args, "O!dddi", &PyList_Type, &list, &fs, &lo, &hi, &order))
        return nullptr;

    std::vector<double> signal(PyList_Size(list));
    for (int i = 0; i < (int)signal.size(); ++i)
        signal[i] = PyFloat_AsDouble(PyList_GetItem(list, i));

    auto sos      = dsp::butterworthBandpass(fs, lo, hi, order);
    auto filtered = dsp::applyBiquad(signal, sos);

    PyObject* result = PyList_New(filtered.size());
    for (int i = 0; i < (int)filtered.size(); ++i)
        PyList_SetItem(result, i, PyFloat_FromDouble(filtered[i]));
    return result;
}

// ─────────────────────────────────────────────────────────────────────────────
// MODULE DEFINITION
// ─────────────────────────────────────────────────────────────────────────────

static PyMethodDef CardioMethods[] = {
    {"fft_magnitude",   py_fft_magnitude,  METH_VARARGS, "Compute FFT magnitude spectrum"},
    {"welch_psd",       py_welch_psd,      METH_VARARGS, "Welch power spectral density"},
    {"pan_tompkins",    py_pan_tompkins,   METH_VARARGS, "Pan-Tompkins R-peak detection"},
    {"hrv_time",        py_hrv_time,       METH_VARARGS, "HRV time-domain features"},
    {"hrv_freq",        py_hrv_freq,       METH_VARARGS, "HRV frequency-domain features"},
    {"poincare",        py_poincare,       METH_VARARGS, "Poincaré plot features"},
    {"dfa",             py_dfa,            METH_VARARGS, "Detrended Fluctuation Analysis"},
    {"ecg_intervals",   py_ecg_intervals,  METH_VARARGS, "ECG interval measurements"},
    {"entropy",         py_entropy,        METH_VARARGS, "ApEn and SampEn complexity"},
    {"bandpass_filter", py_bandpass_filter,METH_VARARGS, "Butterworth bandpass filter"},
    {nullptr, nullptr, 0, nullptr}
};

static PyModuleDef CardioModule = {
    PyModuleDef_HEAD_INIT, "cardio_engine", "C++ Cardiovascular DSP Engine", -1, CardioMethods
};

PyMODINIT_FUNC PyInit_cardio_engine() {
    return PyModule_Create(&CardioModule);
}
