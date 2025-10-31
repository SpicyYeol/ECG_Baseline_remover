# -*- coding: utf-8 -*-
# ECG Viewer — 1000→250 Hz | Hybrid BL++ (adaptive λ, variance-aware, hard-cut) + Residual Refit(옵션 해제)
# (AGC & Glitch 제거 버전)
# Masks(Sag/Step/Corner/Burst/Wave/HV)는 PROCESSED 신호(y_corr_eq=y_corr) 기준. 보간 없음.

from typing import List, Tuple, Set
import json, sys
from collections import defaultdict
from pathlib import Path
from time import perf_counter

import numpy as np
import pyqtgraph as pg
import pywt
from PyQt5 import QtWidgets, QtCore

from scipy import signal
from scipy.signal import (
    butter, filtfilt, find_peaks, lfilter, lfilter_zi,
    decimate, savgol_filter
)
from scipy.ndimage import (
    binary_dilation, median_filter as _mf, uniform_filter1d, percentile_filter
)
from scipy.linalg import solveh_banded
from scipy.interpolate import interp1d, PchipInterpolator

try:
    import neurokit2 as nk
except Exception:
    nk = None

# =========================
# Defaults (수치 파라미터)
# =========================
DEFAULTS = dict(
    # Baseline Hybrid BL++
    PER_WIN_S=3.2, PER_Q=8, ASLS_LAM=8e7, ASLS_P=0.01, ASLS_DECIM=8,
    LPF_FC=0.55, VOL_WIN=0.8, VOL_GAIN=2.0, LAM_FLOOR_PERCENT=0.5, BREAK_PAD_S=0.30,
    # Residual refit (현재 옵션 off)
    RES_K=2.8, RES_WIN_S=0.5, RES_PAD_S=0.20,
    # RR cap
    RR_EPS_UP=6.0, RR_EPS_DN=8.0, RR_T0_MS=80, RR_T1_MS=320,
    # Masks
    SAG_WIN_S=1.0, SAG_Q=20, SAG_K=3.5, SAG_MINDUR_S=0.25, SAG_PAD_S=0.25,
    STEP_SIGMA=5.0, STEP_ABS=0.0, STEP_HOLD_S=0.45,
    CORNER_L_MS=140, CORNER_K=5.5,
    BURST_WIN_MS=140, BURST_KD=5.0, BURST_KS=2.5, BURST_PAD_MS=140,
    WAVE_SIGMA=2.8, WAVE_BLEND_MS=80,
    HV_WIN=2000, HV_KSIGMA=4.0, HV_PAD=200
)

# =========================
# Lightweight Profiler
# =========================
_PROF = defaultdict(lambda: {"calls": 0, "total": 0.0})

def _prof_add(name: str, dt: float):
    d = _PROF[name]; d["calls"] += 1; d["total"] += float(dt)

class time_block:
    def __init__(self, name: str): self.name = name; self.t0 = None
    def __enter__(self): self.t0 = perf_counter(); return self
    def __exit__(self, exc_type, exc, tb): _prof_add(self.name, perf_counter() - self.t0)

def profiled(name: str = None):
    def deco(fn):
        label = name or fn.__name__
        def wrapped(*args, **kwargs):
            t0 = perf_counter()
            try: return fn(*args, **kwargs)
            finally: _prof_add(label, perf_counter() - t0)
        wrapped.__name__ = fn.__name__; wrapped.__doc__ = fn.__doc__
        return wrapped
    return deco

def profiler_report(topn: int = 30):
    rows = []
    for k, v in _PROF.items():
        calls = v["calls"] or 1; total = v["total"]; avg = total / calls
        rows.append((k, calls, total, avg))
    rows.sort(key=lambda r: r[2], reverse=True)
    print("\n[Profiler]  function | calls | total_ms | avg_ms")
    for k, c, tot, avg in rows[:topn]:
        print(f"[Profiler] {k:>20} | {c:5d} | {tot*1000:8.2f} | {avg*1000:7.2f}")
    return rows

# =========================
# Config
# =========================
FILE_PATH = Path('11646C1011258_test5_20250825T112545inPlainText.json')
FS_RAW = 250.0
FS = 250.0
DECIM = max(1, int(round(FS_RAW / FS)))

# =========================
# IO & Utils (필요한 것만)
# =========================
@profiled()
def extract_ecg(obj):
    if isinstance(obj, dict):
        if 'ECG' in obj and isinstance(obj['ECG'], list):
            return np.array(obj['ECG'], dtype=float)
        for v in obj.values():
            hit = extract_ecg(v)
            if hit is not None: return hit
    elif isinstance(obj, list):
        for it in obj:
            hit = extract_ecg(it)
            if hit is not None: return hit
    return None

@profiled()
def decimate_fir_zero_phase(x, q=4):
    return decimate(x, q, ftype='fir', zero_phase=True)

@profiled()
def decimate_if_needed(x, decim: int):
    if decim <= 1: return x
    try: return decimate_fir_zero_phase(x, decim)
    except Exception:
        n = (len(x)//decim)*decim
        return x[:n].reshape(-1, decim).mean(axis=1)

# =========================
# 신호 처리 유틸 (필요한 것만)
# =========================
def _smooth_binary(mask: np.ndarray, fs: float, blend_ms: int = 80) -> np.ndarray:
    L = max(3, int(round(blend_ms/1000.0 * fs)))
    if L % 2 == 0: L += 1
    win = np.hanning(L); win = win / win.sum()
    return np.convolve(mask.astype(float), win, mode='same')

def replace_with_bandlimited(y, fs, mask, fc=12.0):
    """마스크 구간만 저역통과 재구성한 신호로 치환 후 페이드."""
    b,a = butter(3, fc/(fs/2.0), btype='low')
    y_lp = filtfilt(b, a, y)
    win = int(0.10*fs)
    w = np.ones_like(y, float)
    d = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.flatnonzero(d==1); ends = np.flatnonzero(d==-1)
    for s,e in zip(starts, ends):
        a0 = max(0, s-win); b0 = min(len(y), s+win)
        a1 = max(0, e-win); b1 = min(len(y), e+win)
        if b0-a0 > 1: w[a0:b0] *= np.linspace(1, 0, b0-a0)
        if b1-a1 > 1: w[a1:b1] *= np.linspace(0, 1, b1-a1)
        w[s:e] = 0.0
    return y*w + y_lp*(1.0-w)

def highpass_zero_drift(x, fs, fc=0.3, order=2):
    if fc <= 0:
        return x - np.median(x)
    b, a = butter(order, fc/(fs/2.0), btype='high')
    y = filtfilt(b, a, np.asarray(x, float))
    return y - np.median(y)

def smooth_preserve_r(ecg, fs=250, target_fs=100):
    # 1) bandpass
    b, a = butter(2, [0.5/(fs/2), 35/(fs/2)], btype='band'); ecg_f = filtfilt(b, a, ecg)
    # 2) downsample
    decim = int(round(fs / target_fs)); ecg_d = decimate(ecg_f, decim, ftype='fir')
    # 3) upsample back
    t_d = np.linspace(0, len(ecg)/fs, len(ecg_d)); t = np.linspace(0, len(ecg)/fs, len(ecg))
    interp = interp1d(t_d, ecg_d, kind='cubic', fill_value='extrapolate')
    return interp(t)

# =========================
# R/Valley 검출기
# =========================
class rPeakDetector:
    def __init__(self, fs: int = 250, min_rr_ms: int = 250,
                 factor_10s: float = 0.80, factor_2s: float = 0.65,
                 use_prominence: bool = True, min_prom_frac: float = 0.05,
                 debug: bool = True,
                 use_bandpass: bool = True, bp_lo: float = 5.0, bp_hi: float = 20.0,
                 k_mad: float = 3.0, pctl_ref: float = 95.0,
                 width_ms: tuple = (30, 150), slope_quantile: float = 0.90):
        self.fs = int(fs); self.min_rr_ms = int(min_rr_ms)
        self.factor_10s = float(factor_10s); self.factor_2s = float(factor_2s)
        self.use_prominence = bool(use_prominence); self.min_prom_frac = float(min_prom_frac)
        self.debug = bool(debug); self.use_bandpass = bool(use_bandpass)
        self.bp_lo, self.bp_hi = float(bp_lo), float(bp_hi)
        self.k_mad = float(k_mad); self.pctl_ref = float(pctl_ref)
        self.width_ms = width_ms; self.slope_quantile = float(slope_quantile)
        self._refractory = max(1, int((self.min_rr_ms / 1000.0) * self.fs))
        self._win10 = 10 * self.fs; self._win2 = 2 * self.fs; self._step2 = int(0.5 * self.fs)

    def _bandpass(self, x):
        ny = 0.5 * self.fs
        lo = max(0.5, self.bp_lo) / ny; hi = min(self.bp_hi, 0.45 * self.fs) / ny
        b, a = signal.butter(3, [lo, hi], btype="band")
        padlen = min(3 * max(len(a), len(b)), max(0, x.size - 1))
        return signal.filtfilt(b, a, x, padlen=padlen)

    def _auto_polarity(self, x):
        q_hi, q_lo = np.percentile(x, [99, 1])
        return x if abs(q_hi) >= abs(q_lo) else -x

    def _fixed_windows(self, n: int, win: int) -> List[Tuple[int, int]]:
        starts = list(range(0, n, win)); last_start = max(0, n - win)
        if last_start not in starts: starts.append(last_start)
        return [(s, min(s + win, n)) for s in sorted(set(starts))]

    def _sliding_windows(self, n: int, win: int, step: int) -> List[Tuple[int, int]]:
        out = []; s = 0
        while s < n:
            e = min(s + win, n); out.append((s, e))
            if e == n: break
            s += step
        last_start = max(0, n - win)
        if out[-1][0] != last_start: out.append((last_start, n))
        return out

    def _dedup_by_refractory(self, ecg: np.ndarray, idxs: List[int], refr: int) -> List[int]:
        if not idxs: return []
        kept = [idxs[0]]
        for cur in idxs[1:]:
            last = kept[-1]
            if cur - last < refr:
                if ecg[cur] > ecg[last]: kept[-1] = cur
            else: kept.append(cur)
        return kept

    def _assert_full_coverage(self, n: int, fixed_wins, slide_wins) -> None:
        cover = np.zeros(n, dtype=bool)
        for s, e in fixed_wins + slide_wins: cover[s:e] = True
        if not np.all(cover):
            uncovered = np.where(~cover)[0]
            seconds = [(a / self.fs, b / self.fs) for a, b in zip(uncovered[::2], uncovered[1::2])]
            print(f"[RPEAK][WARN] uncovered: {seconds}")
        else:
            total_len_sec = n / self.fs
            print(f"[RPEAK] Full coverage OK — {total_len_sec:.2f}s")

    def _collect_over_windows(self, ecg: np.ndarray, wins: List[Tuple[int, int]],
                              factor: float, sink: Set[int], mode: str = "peak") -> None:
        fs = self.fs
        w_lo = max(1, int(round(self.width_ms[0] / 1000.0 * fs)))
        w_hi = max(w_lo + 1, int(round(self.width_ms[1] / 1000.0 * fs)))
        for s, e in wins:
            seg = ecg[s:e]
            if seg.size == 0: continue
            med = np.median(seg); mad = np.median(np.abs(seg - med)) + 1e-12
            seg_z = (seg - med) / (1.4826 * mad)
            use = seg_z if mode == "peak" else -seg_z
            ref = np.percentile(use, self.pctl_ref); thr_z = max(factor * ref, self.k_mad)
            kwargs = dict(distance=self._refractory, height=thr_z)
            if self.use_prominence: kwargs["prominence"] = 0.25 * thr_z
            peaks, _ = signal.find_peaks(use, **kwargs)
            if peaks.size == 0: continue
            w_res = signal.peak_widths(use, peaks, rel_height=0.5)
            widths = w_res[0]; keep_w = (widths >= w_lo) & (widths <= w_hi)
            d = np.gradient(use); rad = max(1, int(round(0.02 * fs)))
            slopes = []
            for p in peaks:
                a = max(0, p - rad); b = p
                slopes.append(np.mean(d[a:b]) if b > a else 0.0)
            slopes = np.asarray(slopes)
            s_thr = np.quantile(slopes, self.slope_quantile)
            keep = keep_w & (slopes >= s_thr * 0.3)
            for p in peaks[keep]: sink.add(s + int(p))

    def _refine_on_raw(self, ecg_raw: np.ndarray, idxs: List[int],
                       win_ms: int = 40, prefer_max_slope: bool = True,
                       backward_ms: int = 100, extrema: str = "max") -> List[int]:
        fs = self.fs
        rad = max(1, int(round((win_ms / 1000.0) * fs / 2)))
        bwd = max(1, int(round((backward_ms / 1000.0) * fs)))
        n = ecg_raw.size; out = []; g = np.gradient(ecg_raw)
        for i in idxs:
            a = max(0, i - rad); b = min(n, i + rad + 1); seg = ecg_raw[a:b]
            if seg.size < 3: local = i
            else:
                if extrema == "max":
                    local = a + int(np.argmax(seg))
                    if prefer_max_slope:
                        cand = [local] + [local + off for off in (-1,1,-2,2) if 0 <= local + off < n]
                        local = max(cand, key=lambda j:(ecg_raw[j], np.mean(g[max(0,j-rad//2):j]) if j>0 else 0))
                else:
                    local = a + int(np.argmin(seg))
                    if prefer_max_slope:
                        cand = [local] + [local + off for off in (-1,1,-2,2) if 0 <= local + off < n]
                        local = min(cand, key=lambda j: (ecg_raw[j],))
            b_start = max(0, local - bwd); b_end = local + 1
            if b_end - b_start >= 2:
                seg_bwd = ecg_raw[b_start:b_end]
                local = b_start + (int(np.argmax(seg_bwd)) if extrema == "max" else int(np.argmin(seg_bwd)))
            out.append(int(local))
        return self._dedup_by_refractory(ecg_raw, sorted(out), self._refractory)

    def detect_extrema(self, ecg: np.ndarray, mode: str = "peak") -> np.ndarray:
        ecg_raw = np.asarray(ecg, dtype=float); n = ecg_raw.size
        if n == 0: return np.asarray([], dtype=int)
        x = ecg_raw.copy()
        if self.use_bandpass:
            x = self._auto_polarity(self._bandpass(x)) if mode == "peak" else self._bandpass(x)
        cand: Set[int] = set()
        fixed_wins = self._fixed_windows(n, self._win10)
        self._collect_over_windows(x, fixed_wins, self.factor_10s, cand, mode=mode)
        slide_wins = self._sliding_windows(n, self._win2, self._step2)
        self._collect_over_windows(x, slide_wins, self.factor_2s, cand, mode=mode)
        if self.debug: self._assert_full_coverage(n, fixed_wins, slide_wins)
        if not cand:
            if mode == "peak":
                if nk is not None:
                    try:
                        _, info = nk.ecg_peaks(x, sampling_rate=self.fs)
                        p = info.get("ECG_R_Peaks", [])
                        if p: return np.asarray([int(i) for i in p], int)
                    except Exception: pass
                gmax = np.nanmax(x); thr = 0.55 * gmax
                p, _ = signal.find_peaks(x, distance=self._refractory, height=thr)
                return np.asarray(p, int)
            else:
                gmin = np.nanmin(x); thr = 0.55 * (-gmin)
                p, _ = signal.find_peaks(-x, distance=self._refractory, height=thr)
                return np.asarray(p, int)
        idx = self._dedup_by_refractory(x, sorted(cand), self._refractory)
        idx = self._refine_on_raw(ecg_raw, idx, win_ms=40, prefer_max_slope=True,
                                  backward_ms=100, extrema=("max" if mode=="peak" else "min"))
        return np.asarray(idx, int)

    def rPeakDetection(self, cur_ecg: np.ndarray) -> np.ndarray:
        return self.detect_extrema(cur_ecg, mode="peak")

    def vValleyDetection(self, cur_ecg: np.ndarray) -> np.ndarray:
        return self.detect_extrema(cur_ecg, mode="valley")

# =========================
# Peak–Valley 구간 억제
# =========================
def _pair_peak_valley(peaks: np.ndarray, valleys: np.ndarray) -> List[tuple]:
    p = np.asarray(peaks, int); v = np.asarray(valleys, int)
    tags = [(int(i),  1) for i in p] + [(int(j), -1) for j in v]
    tags.sort(key=lambda x: x[0])
    pairs = []
    for (idx0, t0), (idx1, t1) in zip(tags, tags[1:]):
        if t0 != t1:
            a, b = (idx0, idx1) if idx0 < idx1 else (idx1, idx0)
            pairs.append((a, b))
    return sorted(set(pairs))

def suppress_between_peak_valley(
    y: np.ndarray, fs: float,
    peaks: np.ndarray, valleys: np.ndarray,
    pad_ms: int = 30, mode: str = "band", fc: float = 18.0, blend_ms: int = 100
) -> tuple:
    x = np.asarray(y, float).copy(); N = x.size
    if N == 0: return x, np.zeros(0, bool)
    pairs = _pair_peak_valley(peaks, valleys)
    if not pairs: return x, np.zeros(N, bool)
    pad = int(round((pad_ms/1000.0) * fs))
    mask = np.zeros(N, bool); spans = []
    for a, b in pairs:
        s = max(0, min(a, b) + pad); e = min(N, max(a, b) - pad)
        if e - s >= max(3, int(0.04*fs)):
            spans.append((s, e)); mask[s:e] = True
    if not np.any(mask): return x, mask
    if mode == "band":
        y_out = replace_with_bandlimited(x, fs, mask, fc=fc)
        return y_out, mask
    y_out = x.copy()
    for s, e in spans:
        xs = np.array([max(0, s-1), e]); ys = np.array([y_out[xs[0]], y_out[xs[1]]], float)
        if xs[1] - xs[0] < 2: continue
        try:
            f = PchipInterpolator(xs, ys, extrapolate=True)
            y_seg = f(np.arange(s, e))
            L = e - s; bw = min(max(3, int(0.06*fs)), L//3)
            if bw > 1:
                left = np.linspace(0, 1, bw, endpoint=False)
                right = np.linspace(1, 0, bw, endpoint=False)
                y_seg[:bw]  = left  * y_seg[:bw]  + (1-left)  * y_out[s:s+bw]
                y_seg[-bw:] = right * y_seg[-bw:] + (1-right) * y_out[e-bw:e]
            y_out[s:e] = y_seg
        except Exception:
            y_out = replace_with_bandlimited(y_out, fs, (np.arange(N)>=s)&(np.arange(N)<e), fc=fc)
    return y_out, mask

# =========================
# Baseline core
# =========================

class BeatTemplateMemory:
    """
    P/ST 구간의 '모양'과 '레벨'을 지수이동평균(EMA) 템플릿으로 기억.
    - length: 각 구간을 고정 샘플수로 정규화(리샘플)하여 템플릿 저장
    - alpha: 템플릿 업데이트 계수(0.1~0.3 권장)
    """
    def __init__(self, p_len=80, st_len=120, alpha=0.18):
        self.p_len  = int(p_len)
        self.st_len = int(st_len)
        self.alpha  = float(alpha)
        self.p_tpl  = None  # shape [p_len]
        self.st_tpl = None  # shape [st_len]

    def _resample_fixed(self, seg: np.ndarray, L: int) -> np.ndarray:
        if seg.size <= 1:
            return np.zeros(L, float)
        xs = np.linspace(0.0, 1.0, num=seg.size, endpoint=True)
        xt = np.linspace(0.0, 1.0, num=L,       endpoint=True)
        f  = interp1d(xs, seg.astype(float), kind='linear', fill_value='extrapolate', assume_sorted=True)
        return f(xt)

    def update(self, p_seg: np.ndarray, st_seg: np.ndarray):
        if p_seg is not None and p_seg.size > 3:
            rp = self._resample_fixed(p_seg, self.p_len) - np.mean(p_seg)
            self.p_tpl = rp if self.p_tpl is None else (1.0 - self.alpha) * self.p_tpl + self.alpha * rp
        if st_seg is not None and st_seg.size > 3:
            rs = self._resample_fixed(st_seg, self.st_len) - np.mean(st_seg)
            self.st_tpl = rs if self.st_tpl is None else (1.0 - self.alpha) * self.st_tpl + self.alpha * rs


def _roi_bounds(idx, fs, pre_ms, post_ms, N):
    a = max(0, int(idx + pre_ms * 1e-3 * fs))
    b = min(N, int(idx + post_ms * 1e-3 * fs))
    return (a, b) if (b - a) >= max(5, int(0.04 * fs)) else (None, None)

def build_roi_windows(r_idx, fs, N, p_win=(-220, -100), st_win=(+120, +260)):
    wins = []
    for r in r_idx:
        a,b = _roi_bounds(r, fs, p_win[0], p_win[1], N)
        if a is not None: wins.append((a,b))
        a,b = _roi_bounds(r, fs, st_win[0], st_win[1], N)
        if a is not None: wins.append((a,b))
    return wins

def roi_pchip_fill_baseline(b, wins):
    """각 ROI에서 baseline을 양 끝 경계값으로 PCHIP 보간해 '평탄/경계보존'으로 채움."""
    b = np.asarray(b, float).copy(); N = b.size
    for (a, bnd) in wins:
        a0 = max(0, a-1); b0 = min(N-1, bnd)
        if b0 - a0 < 3: continue
        xs = np.array([a0, b0], dtype=float)
        ys = np.array([b[a0], b[b0]], dtype=float)
        f  = PchipInterpolator(xs, ys, extrapolate=True)
        b[a:bnd] = f(np.arange(a, bnd))
    return b

def roi_adaptive_mix(y_qvri_out, y_med_out, wins, fs, gamma=0.5, corr_min=0.15):
    """
    ROI에서 QVRi 출력과 메디안 출력의 가중 혼합.
    - 기본: y = (1-γ)*QVRi + γ*Median
    - 상관이 낮거나(QVRi가 P/ST를 누른 흔적) polarity 불안정하면 γ를 자동 상향.
    """
    yq = np.asarray(y_qvri_out, float).copy()
    ym = np.asarray(y_med_out,  float)
    N = yq.size
    for (a, b) in wins:
        a = max(0, int(a)); b = min(N, int(b))
        if b - a < 5: continue
        s_q = yq[a:b]; s_m = ym[a:b]
        # 상관/극성 평가
        m_q, m_m = s_q.mean(), s_m.mean()
        v_q = np.var(s_q) + 1e-9
        cov = float(np.dot(s_q - m_q, s_m - m_m)) / max(1.0, (b - a))
        rho = float(cov / (np.sqrt(v_q) * (np.std(s_m) + 1e-9)))
        g = float(gamma)
        if (rho < corr_min) or (cov < 0):
            g = min(0.85, max(gamma, 0.65))  # 혼합 가중 자동 상향(메디안 쪽 비중↑)
        # 경계 블렌딩(해닝)
        L = b - a
        Lw = max(3, int(round(0.06 * fs))); Lw += (Lw % 2 == 0)
        if 2*Lw < L:
            wL = np.linspace(0, 1, Lw, endpoint=False)
            wM = np.ones(L - 2*Lw)
            wR = np.linspace(1, 0, Lw, endpoint=False)
            edge = np.concatenate([wL, wM, wR])
        else:
            edge = np.linspace(0, 1, L, endpoint=False)
        mix = (1 - g) * s_q + g * s_m
        yq[a:b] = edge * mix + (1 - edge) * s_q  # 경계에서 QVRi 연속성 유지
    return yq

def build_protect_windows(r_idx, fs, N,
                          p_win=(-220, -100), st_win=(+120, +260)):
    wins = []
    for r in r_idx:
        a,b = _roi_bounds(r, fs, p_win[0], p_win[1], N)
        if a is not None: wins.append((a,b))
        a,b = _roi_bounds(r, fs, st_win[0], st_win[1], N)
        if a is not None: wins.append((a,b))
    return wins
def affine_restore_roi(y_before, y_after, windows,
                       gmin=0.92, gmax=1.15, off_cap=0.30,
                       blend_ms=60, fs=250.0,
                       corr_min=0.15,   # 상관 guard 임계
                       skip_if_negative=False):
    """
    y_after ≈ a*y_before + b 를 ROI별로 추정하되,
    - 상관이 낮거나 음수면 a=1(게인 X), b(오프셋)만 적용(또는 스킵)
    - a,b는 안전 캡
    """
    x0 = np.asarray(y_before, float)
    x1 = np.asarray(y_after,  float).copy()
    N = x0.size
    for (a, b) in windows:
        a = max(0, int(a)); b = min(N, int(b))
        if b - a < 5:
            continue
        s0 = x0[a:b]; s1 = x1[a:b]
        m0 = s0.mean(); m1 = s1.mean()
        v0 = np.var(s0) + 1e-9
        cov = float(np.dot(s0 - m0, s1 - m1)) / max(1.0, (b - a))
        # 피어슨 상관 (수치 안정)
        rho = float(cov / (np.sqrt(v0) * (np.std(s1) + 1e-9)))

        # 최소제곱 게인/오프셋
        a_hat = cov / v0
        b_hat = m1 - a_hat * m0

        # === 극성 가드 ===
        if (rho < corr_min) or (cov < 0):
            if skip_if_negative:
                # 완전 스킵
                continue
            # 게인은 금지(a=1), 오프셋만(레벨 정렬; 과보정 캡)
            a_hat = 1.0
            b_hat = float(np.clip(m0 - m1, -off_cap, off_cap))  # 원신호 평균에 맞춤
        else:
            # 정상 케이스: 안전 캡
            a_hat = float(np.clip(a_hat, gmin, gmax))
            b_hat = float(np.clip(b_hat, -off_cap, off_cap))

        # 블렌딩 적용
        L = b - a
        Lb = max(3, int(round(blend_ms * 1e-3 * fs))); Lb += (Lb % 2 == 0)
        if 2*Lb < L:
            left  = np.linspace(0.0, 1.0, Lb, endpoint=False)
            mid   = np.ones(L - 2*Lb)
            right = np.linspace(1.0, 0.0, Lb, endpoint=False)
            w = np.concatenate([left, mid, right])
        else:
            w = np.linspace(0.0, 1.0, L, endpoint=False)

        target = a_hat * s0 + b_hat
        x1[a:b] = (1 - w) * s1 + w * target
    return x1



def apply_template_guided_restore(
    y: np.ndarray, fs: float, r_idx: np.ndarray, mem: BeatTemplateMemory,
    p_win=(-220, -100), st_win=(+120, +260),
    add_cap=0.35,                   # 박자별 additive 복원 한도
    gmin=0.92, gmax=1.15,           # bounded gain 범위
    blend_ms=60                     # 가장자리 블렌딩
) -> np.ndarray:
    """
    이전 박자 템플릿을 현재 박자에 '약하게' 씌워 P/ST를 복원.
    - 1단계: additive offset(레벨) 보정(±add_cap)
    - 2단계: bounded gain(형태 유지하며 크기만 살짝)
    """
    x = np.asarray(y, float).copy()
    N = x.size
    if r_idx is None or len(r_idx) < 3:
        return x

    def _apply_add_gain(lo, hi, target_mean, add_cap, gain, blend_ms):
        L = hi - lo
        if L < 3: return
        seg = x[lo:hi]
        # 1) additive (레벨 복원)
        delta = np.clip(target_mean - float(np.mean(seg)), -float(add_cap), float(add_cap))
        seg2  = seg + delta
        # 2) bounded gain (형태 유지하며 살짝 키움/줄임)
        g = float(np.clip(gain, gmin, gmax))
        if g != 1.0:
            Lb = max(3, int(round(blend_ms * 1e-3 * fs))); Lb += (Lb % 2 == 0)
            if 2*Lb < L:
                left  = np.linspace(1.0, g, Lb, endpoint=False)
                mid   = np.full(L - 2*Lb, g)
                right = np.linspace(g, 1.0, Lb, endpoint=False)
                w = np.concatenate([left, mid, right])
            else:
                w = np.linspace(1.0, g, L, endpoint=False)
            seg2 = seg2 * w
        x[lo:hi] = seg2

    # 박자 순회: 이전 박자의 템플릿으로 현재 박자를 복원 → 그 다음 템플릿 업데이트
    for k in range(1, len(r_idx)-1):
        r_prev, r_cur = r_idx[k-1], r_idx[k]

        # --- P 구간 ---
        aP, bP = _roi_bounds(r_cur, fs, p_win[0], p_win[1], N)
        # --- ST 구간 ---
        aS, bS = _roi_bounds(r_cur, fs, st_win[0], st_win[1], N)

        # 템플릿이 있으면 '복원'
        if mem.p_tpl is not None and (aP is not None):
            curP = x[aP:bP]
            # 템플릿과 RMS 매칭으로 gain 추정(안정적)
            tplP = mem._resample_fixed(mem.p_tpl, curP.size)
            tplP = tplP - np.mean(tplP)
            rms_cur = np.sqrt(np.mean(curP**2)) + 1e-9
            rms_tpl = np.sqrt(np.mean(tplP**2)) + 1e-9
            gainP   = (rms_tpl / rms_cur) if rms_cur > 0 else 1.0
            _apply_add_gain(aP, bP, target_mean=0.0, add_cap=add_cap, gain=gainP, blend_ms=blend_ms)

        if mem.st_tpl is not None and (aS is not None):
            curS = x[aS:bS]
            tplS = mem._resample_fixed(mem.st_tpl, curS.size)
            tplS = tplS - np.mean(tplS)
            rms_cur = np.sqrt(np.mean(curS**2)) + 1e-9
            rms_tpl = np.sqrt(np.mean(tplS**2)) + 1e-9
            gainS   = (rms_tpl / rms_cur) if rms_cur > 0 else 1.0
            _apply_add_gain(aS, bS, target_mean=0.0, add_cap=add_cap, gain=gainS, blend_ms=blend_ms)

        # 현재 박자의(복원 후) P/ST로 템플릿 업데이트
        p_seg = (x[aP:bP] if aP is not None else None)
        s_seg = (x[aS:bS] if aS is not None else None)
        mem.update(p_seg, s_seg)

    return x

class ReferenceBeatMemory:
    """
    이전 박자의 PQ(등전위)와 ST 통계를 기억(EWMA)했다가
    QVRi 이후 신호에 (1) beat-wise 오프셋, (2) 제한 게인 보정을 적용.
    RR-비율 창을 써서 리듬 변화에 강함.
    """
    def __init__(self,
                 fs: float,
                 alpha: float = 0.25,     # 템플릿 EWMA
                 corr_cap: float = 0.30,  # 오프셋 보정 한도
                 gmin: float = 0.92, gmax: float = 1.14,  # 게인 한도
                 blend_ms: int = 60):
        self.fs = float(fs)
        self.alpha = float(alpha)
        self.corr_cap = float(corr_cap)
        self.gmin, self.gmax = float(gmin), float(gmax)
        self.blend_ms = int(blend_ms)
        # 템플릿(이전 beat들의 통계) — EWMA
        self.pq_level = None   # 등전위 메디안
        self.st_level = None   # ST 메디안
        self.p_rms   = None    # P RMS
        self.st_rms  = None    # ST RMS

    def _roi_idx(self, r_idx_prev: int, r_idx_cur: int, N: int,
                 pq_lo=0.28, pq_hi=0.12, st_lo=0.08, st_hi=0.22):
        """
        RR 비율 기반 창:
        - PQ: R_prev - [0.28..0.12]*RR
        - ST: R_cur  + [0.08..0.22]*RR
        """
        RR = max(1, r_idx_cur - r_idx_prev)
        a_pq = int(round(r_idx_prev - pq_lo * RR))
        b_pq = int(round(r_idx_prev - pq_hi * RR))
        a_st = int(round(r_idx_cur  + st_lo * RR))
        b_st = int(round(r_idx_cur  + st_hi * RR))
        a_pq = max(0, a_pq); b_pq = min(N, b_pq)
        a_st = max(0, a_st); b_st = min(N, b_st)
        if b_pq - a_pq < max(5, int(0.04*self.fs)): a_pq = b_pq = -1
        if b_st - a_st < max(5, int(0.04*self.fs)): a_st = b_st = -1
        return (a_pq, b_pq, a_st, b_st)

    def _ewma(self, old, new):
        if old is None: return new
        return (1.0 - self.alpha) * old + self.alpha * new

    def _apply_offset(self, x, lo, hi, delta):
        if lo < 0 or hi <= lo: return
        Lb = max(3, int(round(self.blend_ms*1e-3*self.fs)));
        if 2*Lb >= (hi-lo):
            w = np.linspace(0.0, 1.0, max(2, hi-lo), endpoint=False)
            w = np.concatenate([w, w[::-1]])[:hi-lo]
        else:
            left  = np.linspace(0.0, 1.0, Lb, endpoint=False)
            mid   = np.ones((hi-lo - 2*Lb,), float)
            right = np.linspace(1.0, 0.0, Lb, endpoint=False)
            w = np.concatenate([left, mid, right])
        x[lo:hi] -= delta * w

    def _apply_gain(self, x, lo, hi, g):
        if lo < 0 or hi <= lo or g <= 0: return
        Lb = max(3, int(round(self.blend_ms*1e-3*self.fs)))
        if 2*Lb >= (hi-lo):
            w = np.linspace(1.0, g, max(2, hi-lo), endpoint=False)
            w = np.concatenate([w, w[::-1]])[:hi-lo]
        else:
            left  = np.linspace(1.0, g, Lb, endpoint=False)
            mid   = np.full((hi-lo - 2*Lb,), g, float)
            right = np.linspace(g, 1.0, Lb, endpoint=False)
            w = np.concatenate([left, mid, right])
        x[lo:hi] *= w

    def update_and_apply(self, y_corr_eq: np.ndarray, r_idx: np.ndarray):
        """
        QVRi 이후 신호(y_corr_eq)에 대해 beat-wise로 보정 적용 + 템플릿 업데이트.
        반환: y_out, debug(dict)
        """
        x = np.asarray(y_corr_eq, float).copy()
        N = x.size
        if r_idx is None or len(r_idx) < 3:
            return x, {"applied": 0}

        applied = 0
        for k in range(1, len(r_idx)-1):
            r_prev, r_cur = int(r_idx[k-1]), int(r_idx[k])
            a_pq, b_pq, a_st, b_st = self._roi_idx(r_prev, r_cur, N)
            if a_pq < 0 or a_st < 0:
                continue

            seg_pq = x[a_pq:b_pq]; seg_st = x[a_st:b_st]
            # 통계 추출
            level_pq = float(np.median(seg_pq))
            level_st = float(np.median(seg_st))
            rms_p    = float(np.sqrt(np.mean(np.square(seg_pq - np.median(seg_pq))))) + 1e-12
            rms_st   = float(np.sqrt(np.mean(np.square(seg_st - np.median(seg_st))))) + 1e-12

            # 템플릿 업데이트(EWMA)
            self.pq_level = self._ewma(self.pq_level, level_pq)
            self.st_level = self._ewma(self.st_level, level_st)
            self.p_rms    = self._ewma(self.p_rms,    rms_p)
            self.st_rms   = self._ewma(self.st_rms,   rms_st)

            # (1) 오프셋 보정: ST를 PQ 템플릿에 맞춤(형태불변)
            if self.pq_level is not None:
                delta = np.clip(level_st - self.pq_level, -self.corr_cap, self.corr_cap)
                if abs(delta) > 1e-9:
                    self._apply_offset(x, a_st, b_st, delta)

            # (2) 게인 보정: P/ST RMS를 템플릿에 제한적으로 매칭
            if self.p_rms and rms_p > 0:
                gp = np.clip((self.p_rms / rms_p), self.gmin, self.gmax)
                if abs(gp - 1.0) > 1e-3:
                    self._apply_gain(x, a_pq, b_pq, gp)  # P에 가까운 구간 포함

            if self.st_rms and rms_st > 0:
                gs = np.clip((self.st_rms / rms_st), self.gmin, self.gmax)
                if abs(gs - 1.0) > 1e-3:
                    self._apply_gain(x, a_st, b_st, gs)

            applied += 1

        return x, {
            "applied": applied,
            "pq_level": self.pq_level, "st_level": self.st_level,
            "p_rms": self.p_rms, "st_rms": self.st_rms
        }


def _roi_bounds(idx, fs, pre_ms, post_ms, N):
    a = max(0, int(idx + pre_ms * 1e-3 * fs))
    b = min(N, int(idx + post_ms * 1e-3 * fs))
    return (a, b) if b - a >= max(5, int(0.04 * fs)) else (None, None)

def beatwise_isoelectric_st_anchor(y, fs, r_idx,
                                   pq_win=(-220, -100),   # R 이전: P–PQ(등전위) 근처
                                   st_win=(+120, +260),   # R 이후: ST 구간
                                   smooth_ms=120,
                                   corr_cap=0.35):
    """PQ(이전 박자)와 ST(현재 박자)의 평균 레벨 차이를 beat-wise로 조금만(±corr_cap) 보정."""
    x = np.asarray(y, float).copy()
    N = x.size
    if r_idx is None or len(r_idx) < 3:
        return x, np.zeros_like(x)

    corr = np.zeros(N, float)
    for k in range(1, len(r_idx)-1):
        r_prev = r_idx[k-1]; r_cur = r_idx[k]
        a_pq, b_pq = _roi_bounds(r_prev, fs, pq_win[0], pq_win[1], N)
        a_st, b_st = _roi_bounds(r_cur,  fs, st_win[0], st_win[1],  N)
        if any(v is None for v in (a_pq, b_pq, a_st, b_st)):
            continue
        m_pq_prev = float(np.median(x[a_pq:b_pq]))
        m_st_cur  = float(np.median(x[a_st:b_st]))
        delta = np.clip((m_st_cur - m_pq_prev), -float(corr_cap), float(corr_cap))
        if abs(delta) < 1e-12:
            continue
        mid = (a_st + b_st)//2
        wL = max(3, int(0.08*fs)); wR = wL
        lo = max(0, a_st - wL); hi = min(N, b_st + wR)
        if hi - lo < 3:
            continue
        ww = np.hanning(hi - lo)
        corr[lo:hi] += delta * ww

    if smooth_ms and smooth_ms > 0:
        w = max(3, int(round((smooth_ms/1000.0) * fs))); w += (w % 2 == 0)
        corr = uniform_filter1d(corr, size=w, mode='nearest')

    y_out = x - corr
    return y_out, corr

def beatwise_ps_bounded_gain(y, fs, r_idx,
                             p_win=(-220, -100), st_win=(+120, +260),
                             gmin=0.90, gmax=1.18, blend_ms=60):
    """P/ST RMS를 이전 박자와 맞추되 gain을 [gmin, gmax]로 제한(형태 보존)."""
    x = np.asarray(y, float).copy(); N = x.size
    if r_idx is None or len(r_idx) < 3:
        return x

    def _rms(seg):
        return float(np.sqrt(np.mean(np.square(seg)))) if seg.size else 0.0

    def _apply_gain(lo, hi, g):
        if hi - lo < 3 or g <= 0:
            return
        Lb = max(3, int(round(blend_ms*1e-3*fs))); Lb += (Lb % 2 == 0)
        if 2*Lb < (hi-lo):
            left = np.linspace(1.0, g, Lb, endpoint=False)
            mid  = np.full((hi-lo - 2*Lb,), g)
            right= np.linspace(g, 1.0, Lb, endpoint=False)
            w = np.concatenate([left, mid, right])
        else:
            w = np.linspace(1.0, g, hi-lo, endpoint=False)
        x[lo:hi] *= w

    for k in range(1, len(r_idx)-1):
        r_prev = r_idx[k-1]; r_cur = r_idx[k]
        # P
        a0,b0 = _roi_bounds(r_prev, fs, p_win[0], p_win[1], N)
        a1,b1 = _roi_bounds(r_cur,  fs, p_win[0], p_win[1], N)
        if not any(v is None for v in (a0,b0,a1,b1)):
            r_ref = _rms(x[a0:b0]); r_curv = _rms(x[a1:b1])
            if r_ref>0 and r_curv>0:
                g = np.clip(r_ref / r_curv, gmin, gmax)
                _apply_gain(a1, b1, g)
        # ST
        a0,b0 = _roi_bounds(r_prev, fs, st_win[0], st_win[1], N)
        a1,b1 = _roi_bounds(r_cur,  fs, st_win[0], st_win[1], N)
        if not any(v is None for v in (a0,b0,a1,b1)):
            r_ref = _rms(x[a0:b0]); r_curv = _rms(x[a1:b1])
            if r_ref>0 and r_curv>0:
                g = np.clip(r_ref / r_curv, gmin, gmax)
                _apply_gain(a1, b1, g)

    return x

def enhance_p_matched(y, fs, r_idx, p_win=(-220,-100), tpl_beats=7, alpha=0.20):
    x = np.asarray(y, float).copy(); N = x.size
    if len(r_idx) < tpl_beats+2:
        return x
    # 템플릿 생성(직전 tpl_beats개의 P 구간 평균)
    segs = []
    for k in range(1, min(len(r_idx)-1, 1+tpl_beats)):
        r = r_idx[k-1]
        a,b = _roi_bounds(r, fs, p_win[0], p_win[1], N)
        if any(v is None for v in (a,b)): continue
        segs.append(x[a:b])
    if not segs: return x
    L = min(map(len, segs))
    tpl = np.mean([s[:L] for s in segs if len(s)>=L], axis=0)
    tpl = tpl - np.mean(tpl)
    en = np.dot(tpl, tpl) + 1e-9
    # 현재 박자들에 투영 보강
    for k in range(1, len(r_idx)-1):
        r = r_idx[k]
        a,b = _roi_bounds(r, fs, p_win[0], p_win[1], N)
        if any(v is None for v in (a,b)) or (b-a)<L:
            continue
        seg = x[a:a+L]
        coeff = np.dot(seg - np.mean(seg), tpl) / en
        x[a:a+L] = seg + float(alpha) * coeff * tpl
    return x


def _qvri_residual_isoelectric(
    y: np.ndarray,
    fs: float,
    r_idx: np.ndarray,
    t0_ms: int = -240,    # ← PQ를 앵커로 권장
    t1_ms: int = -100,
    stride: int = 1,
    lam: float = 2e3,
    pin_strength: float = 1e9,
    protect_windows: list = None,   # [(a,b), ...] 보호 구간
    w_protect: float = 1e-6         # 보호 구간 데이터 가중치 (거의 0)
) -> tuple:
    """
    QVRi에 '보호 구간'을 주어 그 구간의 신호를 깎지 않도록 함.
    반환: (y_qvri, baseline_qvri)
    """
    x = np.asarray(y, float)
    N = x.size
    if N < 10 or r_idx is None or len(r_idx) < 2:
        return x, np.zeros_like(x)

    # 1) PQ 기반 결정점(앵커)
    t0 = int(round(t0_ms * 1e-3 * fs))
    t1 = int(round(t1_ms * 1e-3 * fs))
    idx_knot, val_knot = [], []
    for k, r in enumerate(r_idx[:-1]):
        if (k % max(1, int(stride))) != 0:
            continue
        a = r + t0; b = r + t1
        if a < 0 or b > N or (b - a) < max(5, int(0.04 * fs)):
            continue
        m = float(np.median(x[a:b]))
        idx_knot.append(int((a + b) // 2))
        val_knot.append(m)
    if len(idx_knot) == 0:
        return x, np.zeros_like(x)
    idx_knot = np.asarray(idx_knot, dtype=int)
    val_knot = np.asarray(val_knot, dtype=float)

    # 2) 데이터항 가중치 W: 기본 1, 보호구간은 w_protect, 결정점은 pin_strength
    w = np.ones(N, dtype=float)
    if protect_windows:
        for (a, b) in protect_windows:
            a = max(0, int(a)); b = min(N, int(b))
            if b > a:
                w[a:b] = float(w_protect)
    w[idx_knot] = float(pin_strength)

    # 3) y0: 결정점 위치는 등전위 값, 그 외는 원신호
    y0 = x.copy()
    y0[idx_knot] = val_knot

    # 4) (W + lam * D^T D) z = W * y0  를 대역행렬로 풉니다
    ab_u = np.zeros((3, N), dtype=float)
    ab_u[0, 2:] = lam * 1.0
    ab_u[1, 1:] = lam * (-4.0)
    ab_u[2, :]  = lam * 6.0
    ab_u[2, :] += w
    bvec = w * y0
    z = solveh_banded(ab_u, bvec, lower=False, overwrite_ab=False,
                      overwrite_b=True, check_finite=False)

    baseline_qvri = z - np.median(z)
    y_qvri        = (x - baseline_qvri) - np.median(x - baseline_qvri)
    return y_qvri, baseline_qvri


@profiled()
def baseline_asls_masked(y, lam=1e6, p=0.008, niter=10, mask=None,
                         cg_tol=1e-3, cg_maxiter=200, decim_for_baseline=1,
                         use_float32=True):
    y = np.asarray(y, np.float32 if use_float32 else np.float64); N = y.size
    if N < 3: return np.zeros_like(y)
    if decim_for_baseline > 1:
        q = int(decim_for_baseline); n = (N // q) * q
        if n < q: return np.zeros_like(y)
        y_head = y[:n]; y_ds = y_head.reshape(-1, q).mean(axis=1)
        z_ds = baseline_asls_masked(y_ds, lam=lam, p=p, niter=niter, mask=None,
                                    decim_for_baseline=1, use_float32=use_float32)
        idx = np.repeat(np.arange(z_ds.size), q); z_coarse = z_ds[idx]
        if z_coarse.size < N:
            z = np.empty(N, y.dtype); z[:z_coarse.size] = z_coarse; z[z_coarse.size:] = z_coarse[-1]
        else:
            z = z_coarse[:N]
        return z
    g = np.ones(N, dtype=y.dtype) if mask is None else np.where(mask, 1.0, 1e-3).astype(y.dtype)
    lam = y.dtype.type(lam)
    ab_u = np.zeros((3, N), dtype=y.dtype)
    ab_u[0, 2:] = lam * 1.0
    ab_u[1, 1:] = lam * (-4.0)
    ab_u[2, :]  = lam * 6.0
    base_niter = int(niter)
    if N < 0.5 * 250: base_niter = min(base_niter, 5)
    if N < 0.25 * 250: base_niter = min(base_niter, 4)
    w = np.ones(N, dtype=y.dtype); z = np.zeros(N, dtype=y.dtype)
    last_obj = None
    for it in range(base_niter):
        wg = (w * g).astype(y.dtype, copy=False)
        ab_u[2, :] = lam * 6.0 + wg
        b = wg * y
        z = solveh_banded(ab_u, b, lower=False, overwrite_ab=False,
                          overwrite_b=True, check_finite=False)
        w = p * (y > z) + (1.0 - p) * (y < z)
        if it >= 1:
            r = (y - z)
            data_term = float(np.dot((wg * r).astype(np.float64), r.astype(np.float64)))
            d2 = np.diff(z.astype(np.float64), n=2, prepend=float(z[0]), append=float(z[-1]))
            reg_term = float(lam) * float(np.dot(d2, d2))
            obj = data_term + reg_term
            if last_obj is not None and abs(last_obj - obj) <= 1e-5 * max(1.0, obj): break
            last_obj = obj
    return z.astype(np.float64, copy=False)

@profiled()
def make_qrs_mask(y, fs=250, r_pad_ms=180, t_pad_start_ms=80, t_pad_end_ms=300):
    info = nk.ecg_peaks(y, sampling_rate=fs)[1]
    r_idx = np.array(info.get("ECG_R_Peaks", []), dtype=int)
    mask = np.ones_like(y, dtype=bool)
    if r_idx.size == 0: return mask
    def clamp(a): return np.clip(a, 0, len(y)-1)
    r_pad = int(round(r_pad_ms * 1e-3 * fs))
    t_s   = int(round(t_pad_start_ms * 1e-3 * fs))
    t_e   = int(round(t_pad_end_ms   * 1e-3 * fs))
    for r in r_idx:
        mask[clamp(r-r_pad):clamp(r+r_pad)+1] = False
        mask[clamp(r+t_s):clamp(r+t_e)+1]     = False
    return mask

@profiled()
def _find_breaks(y, fs, k=7.0, min_gap_s=0.30):
    dy = np.diff(y, prepend=y[0]); med = np.median(dy); mad = np.median(np.abs(dy - med)) + 1e-12
    z = np.abs(dy - med) / (1.4826 * mad); idx = np.flatnonzero(z > float(k))
    if idx.size == 0: return []
    gap = int(round(min_gap_s * fs)); breaks = [int(idx[0])]
    for i in idx[1:]:
        if i - breaks[-1] > gap: breaks.append(int(i))
    return breaks

@profiled()
def _dilate_mask(mask, fs, pad_s=0.45):
    pad = int(round(pad_s * fs))
    if pad <= 0: return mask
    k = np.ones(pad*2+1, dtype=int)
    return (np.convolve(mask.astype(int), k, mode='same') > 0)

def rr_isoelectric_clamp(y, fs, r_idx=None, t0_ms=80, t1_ms=300):
    x = np.asarray(y, float)
    if r_idx is None or len(r_idx) < 2:
        try:
            info = nk.ecg_peaks(x, sampling_rate=fs)[1]
            r_idx = np.array(info.get("ECG_R_Peaks", []), int)
        except Exception:
            r_idx = np.array([], int)
    if r_idx.size < 2: return np.zeros_like(x)
    t0 = int(round(t0_ms * 1e-3 * fs)); t1 = int(round(t1_ms * 1e-3 * fs))
    pts_x, pts_y = [], []; N = x.size
    for r in r_idx[:-1]:
        a = max(0, r + t0); b = min(N, r + t1)
        if b - a < max(5, int(0.04 * fs)): continue
        m = float(np.median(x[a:b])); pts_x.append((a + b) // 2); pts_y.append(m)
    if len(pts_x) < 2: return np.zeros_like(x)
    xs = np.arange(N, dtype=float)
    baseline_rr = np.interp(xs, np.array(pts_x, float), np.array(pts_y, float))
    baseline_rr -= np.median(baseline_rr)
    return baseline_rr

@profiled()
def baseline_hybrid_plus_adaptive(
    y, fs,
    per_win_s=2.8, per_q=15,
    asls_lam=1e8, asls_p=0.01, asls_decim=12,
    qrs_aware=True, verylow_fc=0.55, clamp_win_s=6.0,
    vol_win_s=0.6, vol_gain=6.0, lam_floor_ratio=0.03,
    hard_cut=True, break_pad_s=0.30,
    rr_cap_enable=True, rr_eps_up=5.0, rr_eps_dn=8.0, rr_t0_ms=80, rr_t1_ms=320,
    r_idx=None, qrs_mask=None, lam_bins=6, min_seg_s=0.50, max_seg_s=6.0
):
    x = np.asarray(y, float); N = x.size
    if N < 8: return np.zeros_like(x), np.zeros_like(x)

    def _odd(n): n = int(max(3, n)); return n + (n % 2 == 0)
    def _mov_stats(xx, win):
        k = np.ones(win, float); s1 = np.convolve(xx, k, mode='same'); s2 = np.convolve(xx*xx, k, mode='same')
        m = s1 / win; v = s2 / win - m*m; v[v < 0] = 0.0; return m, np.sqrt(v)

    # 0) 초기 퍼센타일 바닥선
    w0 = _odd(int(round(per_win_s * fs)))
    dc = np.median(x[np.isfinite(x)]); x0 = x - dc
    b0 = percentile_filter(x0, percentile=int(per_q), size=w0, mode='nearest')

    # 1) QRS-aware + 변화점 보호
    if qrs_mask is not None:
        base_mask = qrs_mask.astype(bool, copy=False)
    else:
        if qrs_aware:
            try:
                if r_idx is None:
                    info = nk.ecg_peaks(x, sampling_rate=fs)[1]
                    r_idx = np.array(info.get("ECG_R_Peaks", []), dtype=int)
                base_mask = np.ones_like(x, dtype=bool)
                if r_idx.size > 0:
                    pad = int(round(0.12 * fs))
                    for r in r_idx:
                        lo = max(0, r - pad); hi = min(N, r + pad + 1)
                        base_mask[lo:hi] = False
                    t_s = int(round(0.08 * fs)); t_e = int(round(0.30 * fs))
                    for r in r_idx:
                        lo = max(0, r + t_s); hi = min(N, r + t_e + 1)
                        base_mask[lo:hi] = False
            except Exception:
                base_mask = np.ones_like(x, bool)
        else:
            base_mask = np.ones_like(x, bool)

    brks = _find_breaks(x, fs, k=6.5, min_gap_s=0.25)
    if brks:
        prot = np.zeros_like(x, bool); prot[np.asarray(brks, int)] = True
        prot = _dilate_mask(prot, fs, pad_s=max(0.35, float(break_pad_s)))
        base_mask &= (~prot)

    # 2) 위치별 λ 설계
    grad = np.gradient(x)
    g_ref = np.quantile(np.abs(grad), 0.95) + 1e-6
    z_grad = np.clip(np.abs(grad) / g_ref, 0.0, 6.0)
    lam_grad = asls_lam / (1.0 + 8.0 * z_grad)

    vw = _odd(int(round(vol_win_s * fs)))
    _, rs = _mov_stats(x, vw)
    rs_ref = np.quantile(rs, 0.90) + 1e-9
    z_vol = np.clip(rs / rs_ref, 0.0, 10.0)
    lam_vol = asls_lam / (1.0 + float(vol_gain) * z_vol)

    lam_local = np.minimum(lam_grad, lam_vol)
    lam_local = np.maximum(lam_local, asls_lam * max(5e-4, float(lam_floor_ratio)))

    if brks:
        tw = int(round(0.6 * fs))
        for b in brks:
            lo = max(0, b - tw); hi = min(N, b + tw + 1)
            lam_local[lo:hi] = np.minimum(lam_local[lo:hi],
                                          asls_lam * max(5e-4, float(lam_floor_ratio)*0.5))

    # 3) 세그먼트 피팅
    def _segments_from_lambda(lam_arr, fs_, brks):
        lam_eps = 1e-12; L = np.log(lam_arr + lam_eps)
        q_lo, q_hi = np.quantile(L, [0.05, 0.95]); q_hi = q_hi if q_hi > q_lo else q_lo + 1e-6
        bins = np.linspace(q_lo, q_hi, int(max(2, lam_bins)))
        idx = np.clip(np.digitize(L, bins, right=False), 0, len(bins))
        cuts = [0] + [int(b) for b in brks] + [N]; segs = []
        for s0, e0 in zip(cuts[:-1], cuts[1:]):
            if e0 - s0 <= 0: continue
            run_id = idx[s0:e0];
            if run_id.size == 0: continue
            a = s0; cur = run_id[0]
            for i in range(s0+1, e0):
                if idx[i] != cur: segs.append((a, i, cur)); a, cur = i, idx[i]
            segs.append((a, e0, cur))
        min_len = int(round(float(min_seg_s) * fs_)); merged = []
        for s, e, kbin in segs:
            if not merged: merged.append([s, e, kbin]); continue
            ms, me, mk = merged[-1]
            if (e - s) < min_len and mk == kbin: merged[-1][1] = e
            else:
                if (me - ms) < min_len and kbin != mk: merged[-1][1] = e
                else: merged.append([s, e, kbin])
        out = []; max_len = int(round(float(max_seg_s) * fs_))
        for s, e, kbin in merged:
            Lseg = e - s
            if Lseg <= max_len: out.append((s, e))
            else:
                step = max_len
                for a in range(s, e, step):
                    b = min(e, a + step)
                    if b - a > 5: out.append((a, b))
        out2 = []; last = -1
        for s, e in sorted(out):
            if s < last: s = last
            if e > s: out2.append((s, e)); last = e
        return out2

    b1 = np.zeros_like(x)
    segs = _segments_from_lambda(lam_local, fs, brks if hard_cut else [])
    if not segs: segs = [(0, N)]
    for s, e in segs:
        if (e - s) < max(5, int(0.20 * fs)): continue
        lam_i = float(np.median(lam_local[s:e]))
        seg = x0[s:e] - b0[s:e]
        mask_i = None if not qrs_aware else base_mask[s:e]
        b1_seg = baseline_asls_masked(
            seg, lam=max(3e4, lam_i), p=asls_p, niter=10,
            mask=mask_i, decim_for_baseline=max(1, int(asls_decim))
        )
        b1[s:e] = b1_seg

    # 4) very-low stabilization + offset control + RR cap
    b = b0 + b1
    b_slow = highpass_zero_drift(b, fs, fc=max(verylow_fc, 0.15))
    clamp_w = int(round(clamp_win_s * fs)); clamp_w += (clamp_w % 2 == 0)
    sg_win  = max(int(fs * 1.5), clamp_w); sg_win += (sg_win % 2 == 0)
    resid   = x - b_slow
    off     = savgol_filter(resid, window_length=sg_win, polyorder=2, mode='interp')
    off    -= np.median(off)
    off     = highpass_zero_drift(off, fs, fc=0.15)
    b_final = b_slow + off

    if rr_cap_enable:
        iso = rr_isoelectric_clamp(x - b_final, fs, t0_ms=rr_t0_ms, t1_ms=rr_t1_ms)
        iso -= np.median(iso)
        err = (b_final - b_slow) - iso
        err = np.clip(err, -float(rr_eps_dn), float(rr_eps_up))
        smw = max(3, int(round(0.12 * fs))); smw += (smw % 2 == 0)
        err = uniform_filter1d(err, size=smw, mode='nearest')
        b_final = b_slow + iso + err

    y_corr = x - b_final
    return y_corr, b_final

# =========================
# Masks
# =========================
@profiled()
def suppress_negative_sag(
    y, fs, win_sec=1.0, q_floor=20, k_neg=3.5, min_dur_s=0.25, pad_s=0.25,
    protect_qrs=True, r_idx=None, qrs_mask=None, use_fast_filter=True
):
    y = np.asarray(y, float); N = y.size
    if N < 10: return np.zeros(N, bool)
    w = max(3, int(round(win_sec * fs))); w += (w % 2 == 0)
    min_len = int(round(min_dur_s * fs)); pad_n = int(round(pad_s * fs))
    if use_fast_filter:
        m = uniform_filter1d(y, size=w, mode='nearest')
        m2 = uniform_filter1d(y * y, size=w, mode='nearest')
        v = m2 - m * m; v[v < 0] = 0.0; s = np.sqrt(v)
        zq = abs(0.01 * (50 - q_floor)) * 0.1
        floor = m - zq * s; median = m
    else:
        floor = percentile_filter(y, percentile=q_floor, size=w, mode='nearest')
        median = percentile_filter(y, percentile=50, size=w, mode='nearest')
    r = y - median; neg = np.minimum(r, 0.0)
    med = np.median(neg); mad = np.median(np.abs(neg - med)) + 1e-12
    zneg = (neg - med) / (1.4826 * mad)
    mask = (zneg < -abs(k_neg)) & (y < floor)
    if protect_qrs:
        if qrs_mask is not None: prot = qrs_mask.astype(bool, copy=False)
        else:
            if r_idx is None:
                try:
                    info = nk.ecg_peaks(y, sampling_rate=fs)[1]
                    r_idx = np.array(info.get("ECG_R_Peaks", []), dtype=int)
                except Exception:
                    r_idx = np.array([], int)
            prot = np.zeros(N, bool)
            if r_idx.size > 0:
                pad = int(round(0.12 * fs))
                for r0 in r_idx:
                    lo = max(0, r0 - pad); hi = min(N, r0 + pad + 1)
                    prot[lo:hi] = True
        mask &= (~prot)
    if not np.any(mask): return mask
    diff = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.flatnonzero(diff == 1); ends = np.flatnonzero(diff == -1)
    if starts.size == 0: return mask
    dur = ends - starts; long_idx = np.where(dur >= min_len)[0]
    out = np.zeros_like(mask)
    for i in long_idx:
        lo = max(0, starts[i] - pad_n); hi = min(N, ends[i] + pad_n)
        out[lo:hi] = True
    return out

@profiled()
def fix_downward_steps_mask(
    y, fs, pre_s=0.5, post_s=0.5, gap_s=0.08,
    amp_sigma=5.0, amp_abs=None, min_hold_s=0.45,
    refractory_s=0.80, protect_qrs=True,
    r_idx=None, qrs_mask=None, smooth_ms=120, hop_ms=10
):
    y = np.asarray(y, float); N = y.size
    if N < 10: return np.zeros(N, bool)
    if smooth_ms and smooth_ms > 0:
        m_win = max(3, int(round((smooth_ms/1000.0) * fs))); m_win += (m_win % 2 == 0)
        y_s = uniform_filter1d(y, size=m_win, mode='nearest')
    else: y_s = y
    med = np.median(y_s); mad = np.median(np.abs(y_s - med)) + 1e-12
    thr = amp_sigma * 1.4826 * mad
    if amp_abs is not None: thr = max(thr, float(amp_abs))
    S = np.concatenate(([0.0], np.cumsum(y_s, dtype=float)))
    def box_mean(start_idx, L): a = start_idx; b = start_idx + L; return (S[b] - S[a]) / float(L)
    pre   = int(round(pre_s  * fs)); post  = int(round(post_s * fs))
    gap   = int(round(gap_s  * fs)); hold  = int(round(min_hold_s * fs))
    refr  = int(round(refractory_s * fs))
    if pre < 1 or post < 1 or hold < 1: return np.zeros(N, bool)
    hop = max(1, int(round((hop_ms/1000.0) * fs)))
    i_min = pre; i_max = N - (gap + post + hold) - 1
    if i_max <= i_min: return np.zeros(N, bool)
    centers = np.arange(i_min, i_max + 1, hop, dtype=int)
    pre_starts = centers - pre; m1 = box_mean(pre_starts, pre)
    cpos = centers + gap; m2 = box_mean(cpos, post); m_hold = box_mean(cpos, hold)
    drop = m1 - m2
    cand = (drop > thr) & ((m1 - m_hold) >= (0.6 * drop))
    if not np.any(cand): return np.zeros(N, bool)
    prot = np.zeros(N, bool)
    if protect_qrs:
        if qrs_mask is not None: prot = qrs_mask.astype(bool, copy=False)
        else:
            if r_idx is None:
                try:
                    info = nk.ecg_peaks(y_s, sampling_rate=fs)[1]
                    r_idx = np.array(info.get("ECG_R_Peaks", []), dtype=int)
                except Exception:
                    r_idx = np.array([], int)
            if r_idx.size > 0:
                p = int(round(0.12 * fs))
                for r in r_idx:
                    lo = max(0, r - p); hi = min(N, r + p + 1)
                    prot[lo:hi] = True
    cand_idx = centers[cand]
    if protect_qrs and prot.any(): cand_idx = cand_idx[~prot[cand_idx]]
    if cand_idx.size == 0: return np.zeros(N, bool)
    mask = np.zeros(N, bool); last_end = -10**9
    order = np.argsort(-drop[cand])
    for j in order:
        i = centers[j]
        if not cand[j]: continue
        start = cpos[j]; end = start + hold
        if start - last_end < refr: continue
        if protect_qrs and prot.any():
            seg = prot[start:end]
            if seg.size and seg.mean() > 0.5: continue
        mask[start:end] = True; last_end = end
    return mask

@profiled()
def smooth_corners_mask(y, fs, L_ms=140, k_sigma=5.5,
                        protect_qrs=True, r_idx=None, qrs_mask=None, smooth_ms=20, use_float32=True):
    y = np.asarray(y, np.float32 if use_float32 else np.float64); N = y.size
    if N < 10: return np.zeros(N, bool)
    if smooth_ms > 0:
        win = max(3, int(round((smooth_ms / 1000.0) * fs)))
        y_s = uniform_filter1d(y, size=win, mode='nearest')
    else: y_s = y
    d1 = np.gradient(y_s); d2 = np.gradient(d1)
    med = np.median(d2); mad = np.median(np.abs(d2 - med)) + 1e-12
    z = (d2 - med) / (1.4826 * mad); cand = np.abs(z) > float(k_sigma)
    if not np.any(cand): return np.zeros(N, bool)
    if protect_qrs:
        if qrs_mask is not None: prot = qrs_mask.astype(bool, copy=False)
        else:
            if r_idx is None:
                try:
                    info = nk.ecg_peaks(y_s, sampling_rate=fs)[1]
                    r_idx = np.array(info.get("ECG_R_Peaks", []), dtype=int)
                except Exception:
                    r_idx = np.array([], int)
            prot = np.zeros(N, bool)
            if r_idx.size > 0:
                pad = int(round(0.12 * fs))
                for r in r_idx:
                    lo = max(0, r - pad); hi = min(N, r + pad + 1)
                    prot[lo:hi] = True
        cand &= (~prot)
    idx = np.flatnonzero(cand)
    if idx.size == 0: return np.zeros(N, bool)
    L = max(3, int(round((L_ms / 1000.0) * fs)))
    out = np.zeros(N, bool)
    gaps = np.diff(idx, prepend=idx[0]); starts = np.flatnonzero(gaps > L); starts = np.append(starts, len(idx))
    prev = 0
    for s in starts:
        seg_idx = idx[prev:s]
        if seg_idx.size == 0: continue
        a = max(0, seg_idx[0] - L); b = min(N, seg_idx[-1] + L)
        out[a:b] = True; prev = s
    return out

@profiled()
def burst_mask(
    y, fs, win_ms=140, k_diff=7.5, k_std=3.5, pad_ms=80,
    protect_qrs=True, r_idx=None, qrs_mask=None, pre_smooth_ms=0, use_float32=True
):
    x = np.asarray(y, np.float32 if use_float32 else np.float64); N = x.size
    if N < 10: return np.zeros(N, dtype=bool)
    if pre_smooth_ms and pre_smooth_ms > 0:
        sw = max(3, int(round((pre_smooth_ms/1000.0) * fs))); sw += (sw % 2 == 0)
        x = uniform_filter1d(x, size=sw, mode='nearest')
    dy = np.gradient(x); d_med = float(np.median(dy)); d_mad = float(np.median(np.abs(dy - d_med)) + 1e-12)
    z_diff = (dy - d_med) / (1.4826 * d_mad)
    w = max(3, int(round((win_ms/1000.0) * fs))); w += (w % 2 == 0)
    m  = uniform_filter1d(x,   size=w, mode='nearest')
    m2 = uniform_filter1d(x*x, size=w, mode='nearest')
    v = m2 - m*m; np.maximum(v, 0.0, out=v); rs = np.sqrt(v, dtype=x.dtype)
    r_med = float(np.median(rs)); r_mad = float(np.median(np.abs(rs - r_med)) + 1e-12)
    z_std = (rs - r_med) / (1.4826 * r_mad)
    cand = (np.abs(z_diff) > float(k_diff)) & (z_std > float(k_std))
    if not np.any(cand): return np.zeros(N, dtype=bool)
    if protect_qrs:
        if qrs_mask is not None: prot = qrs_mask.astype(bool, copy=False)
        else:
            if r_idx is None:
                try:
                    info = nk.ecg_peaks(x, sampling_rate=fs)[1]
                    r_idx = np.array(info.get("ECG_R_Peaks", []), dtype=int)
                except Exception:
                    r_idx = np.array([], dtype=int)
            prot = np.zeros(N, dtype=bool)
            if r_idx.size > 0:
                pad_r = int(round(0.12 * fs))
                for r in r_idx:
                    lo = max(0, r - pad_r); hi = min(N, r + pad_r + 1)
                    prot[lo:hi] = True
        cand &= (~prot)
    if not np.any(cand): return cand
    pad = int(round((pad_ms/1000.0) * fs))
    if pad > 0:
        st = np.ones(pad*2 + 1, dtype=bool)
        cand = binary_dilation(cand, structure=st)
    return cand

@profiled()
def high_variance_mask(
    y: np.ndarray, win=2000, k_sigma=5.0, pad=125, mode: str = "grid", hop_ms: int = 32
):
    x = np.asarray(y, np.float32); n = int(x.size)
    if n == 0:
        stats = {"threshold": 0.0, "removed_samples": 0, "kept_samples": 0, "compression_ratio": 1.0}
        return np.zeros(0, dtype=bool), stats
    w = int(max(2, win)); w += (w % 2 == 0); half = w // 2
    hop = max(1, int(round((hop_ms / 1000.0) * 250.0)))
    centers = np.arange(0, n, hop, dtype=int)
    starts = np.clip(centers - half, 0, n - 1); ends = np.clip(centers + half + 1, 0, n)
    S1 = np.concatenate(([0.0], np.cumsum(x,  dtype=np.float64)))
    S2 = np.concatenate(([0.0], np.cumsum(x*x, dtype=np.float64)))
    Ls = (ends - starts).astype(np.int64)
    sum1 = S1[ends] - S1[starts]; sum2 = S2[ends] - S2[starts]
    m  = sum1 / np.maximum(1, Ls); m2 = sum2 / np.maximum(1, Ls)
    v = m2 - m*m; v[v < 0.0] = 0.0
    rs_grid = np.sqrt(v, dtype=np.float64)
    rs_med = float(np.median(rs_grid)); rs_mad = float(np.median(np.abs(rs_grid - rs_med)) + 1e-12)
    thr = rs_med + 1.4826 * rs_mad * float(k_sigma)
    idx_full = np.arange(n, dtype=np.float64); idx_cent = centers.astype(np.float64)
    rs = np.interp(idx_full, idx_cent, rs_grid).astype(np.float32, copy=False)
    mask = rs > thr
    if pad and pad > 0 and mask.any():
        st = np.ones(int(pad) * 2 + 1, dtype=bool)
        mask = binary_dilation(mask, structure=st)
    kept = int((~mask).sum())
    stats = {"threshold": float(thr), "removed_samples": int(mask.sum()),
             "kept_samples": kept, "compression_ratio": float(kept / n)}
    return mask, stats

# =========================
# Wavelet denoise (QRS-aware 블렌딩)
# =========================
@profiled()
def qrs_aware_wavelet_denoise(y, fs, wavelet='db6', level=None, sigma_scale=2.8, blend_ms=80):
    y = np.asarray(y, float); N = y.size
    try: mask = make_qrs_mask(y, fs=fs)
    except Exception: mask = np.ones_like(y, dtype=bool)
    alpha = _smooth_binary(mask, fs, blend_ms=blend_ms)
    try:
        if level is None: level = min(5, max(2, int(np.log2(fs/8.0))))
        coeffs = pywt.wavedec(y, wavelet=wavelet, level=level, mode='symmetric')
        cA, details = coeffs[0], coeffs[1:]
        sigma = np.median(np.abs(details[-1])) / 0.6745 + 1e-12
        thr = float(sigma_scale) * sigma
        details_d = [pywt.threshold(c, thr, mode='soft') for c in details]
        y_w = pywt.waverec([cA] + details_d, wavelet=wavelet, mode='symmetric')
        if y_w.size != N: y_w = y_w[:N]
    except Exception:
        win = max(5, int(round(0.05 * fs)));  win += (win % 2 == 0)
        y_w = savgol_filter(y, window_length=win, polyorder=2, mode='interp')
    return alpha * y_w + (1.0 - alpha) * y, alpha

# =========================
# Custom X-only stretch zoom ViewBox
# =========================
class XZoomViewBox(pg.ViewBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, enableMenu=False, **kwargs)
        self.setMouseEnabled(x=True, y=True)
        self.setLimits(yMin=-1e12, yMax=1e12)
    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == QtCore.Qt.LeftButton and (ev.modifiers() & QtCore.Qt.ShiftModifier):
            ev.accept(); pos = ev.pos(); last = ev.lastPos()
            dx = pos.x() - last.x(); s = np.exp(-dx * 0.005); s = float(np.clip(s, 1e-3, 1e3))
            center = self.mapSceneToView(pos); self.scaleBy((s, 1.0), center=center)
        else:
            super().mouseDragEvent(ev, axis=axis)

# =========================
# Qt Viewer
# =========================
class ECGViewer(QtWidgets.QWidget):
    def __init__(self, t, y_raw, parent=None):
        super().__init__(parent)
        self.t = t; self.y_raw = y_raw
        self._recompute_timer = None

        root = QtWidgets.QVBoxLayout(self)

        # View toggles
        tg = QtWidgets.QHBoxLayout()
        self.cb_raw   = QtWidgets.QCheckBox("원본 신호");       self.cb_raw.setChecked(True)
        self.cb_corr  = QtWidgets.QCheckBox("가공(보정) 신호"); self.cb_corr.setChecked(True)
        self.cb_mask  = QtWidgets.QCheckBox("마스크 패널");     self.cb_mask.setChecked(True)
        self.cb_base  = QtWidgets.QCheckBox("Baseline 표시");   self.cb_base.setChecked(False)
        for cb in (self.cb_raw, self.cb_corr, self.cb_mask, self.cb_base): tg.addWidget(cb)
        tg.addStretch(1); root.addLayout(tg)

        # Baseline options
        bl = QtWidgets.QHBoxLayout()
        self.cb_qrsaware = QtWidgets.QCheckBox("QRS-aware"); self.cb_qrsaware.setChecked(True)
        self.cb_break_cut = QtWidgets.QCheckBox("Hard cut at breaks"); self.cb_break_cut.setChecked(True)
        self.cb_res_refit = QtWidgets.QCheckBox("Residual refit"); self.cb_res_refit.setChecked(True)  # 현재 비활성 코드
        self.cb_rrcap = QtWidgets.QCheckBox("RR cap"); self.cb_rrcap.setChecked(True)
        for w in [self.cb_qrsaware, self.cb_break_cut, self.cb_res_refit, self.cb_rrcap]: bl.addWidget(w)
        bl.addStretch(1); root.addLayout(bl)

        # Mask toggles
        row2 = QtWidgets.QHBoxLayout()
        self.cb_sag    = QtWidgets.QCheckBox("Sag");     self.cb_sag.setChecked(True)
        self.cb_step   = QtWidgets.QCheckBox("Step");    self.cb_step.setChecked(True)
        self.cb_corner = QtWidgets.QCheckBox("Corner");  self.cb_corner.setChecked(True)
        self.cb_burst  = QtWidgets.QCheckBox("Burst");   self.cb_burst.setChecked(True)
        self.cb_wave   = QtWidgets.QCheckBox("Wavelet"); self.cb_wave.setChecked(False)
        for cb in [self.cb_sag, self.cb_step, self.cb_corner, self.cb_burst, self.cb_wave]: row2.addWidget(cb)
        row2.addStretch(1); root.addLayout(row2)

        # R/Valley overlay
        row3 = QtWidgets.QHBoxLayout()
        self.cb_rpeaks = QtWidgets.QCheckBox("R-peak(보정후) 표시"); self.cb_rpeaks.setChecked(True); row3.addWidget(self.cb_rpeaks)
        self.cb_valleys = QtWidgets.QCheckBox("Valley(보정후) 표시"); self.cb_valleys.setChecked(True); row3.addWidget(self.cb_valleys)
        row3.addStretch(1); root.addLayout(row3)

        # Plots
        self.win_plot = pg.GraphicsLayoutWidget(); root.addWidget(self.win_plot)
        self.plot = self.win_plot.addPlot(row=0, col=0, viewBox=XZoomViewBox())
        self.plot.setLabel('bottom', 'Time (s)'); self.plot.setLabel('left', 'Amplitude'); self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.overview = self.win_plot.addPlot(row=1, col=0); self.overview.setMaximumHeight(150); self.overview.showGrid(x=True, y=True, alpha=0.2)
        self.region = pg.LinearRegionItem(); self.region.setZValue(10); self.overview.addItem(self.region)
        self.region.sigRegionChanged.connect(self.update_region)

        pen_raw  = pg.mkPen(color=(150,150,150), width=1)
        pen_corr = pg.mkPen(color=(255,215,0),   width=1.6)
        pen_base = pg.mkPen(color=(0,200,255),   width=1, style=QtCore.Qt.DashLine)

        self.curve_raw  = self.plot.plot([], [], pen=pen_raw)
        self.curve_corr = self.plot.plot([], [], pen=pen_corr)
        self.curve_base = self.plot.plot([], [], pen=pen_base); self.curve_base.setVisible(False)

        self.scatter_r_after = pg.ScatterPlotItem(size=6, pen=pg.mkPen(None), brush=pg.mkBrush(255,0,0,200))
        self.plot.addItem(self.scatter_r_after); self.scatter_r_after.setZValue(10); self.scatter_r_after.setVisible(self.cb_rpeaks.isChecked())

        self.scatter_v_after = pg.ScatterPlotItem(size=6, pen=pg.mkPen(None), brush=pg.mkBrush(0,180,255,200))
        self.plot.addItem(self.scatter_v_after); self.scatter_v_after.setZValue(10); self.scatter_v_after.setVisible(self.cb_valleys.isChecked())

        self.ov_curve = self.overview.plot([], [], pen=pg.mkPen(width=1))

        self.mask_plot = self.win_plot.addPlot(row=2, col=0)
        self.mask_plot.setMaximumHeight(130)
        self.mask_plot.setLabel('left', 'Masks'); self.mask_plot.setLabel('bottom', 'Time (s)')
        self.mask_plot.showGrid(x=True, y=True, alpha=0.2)
        self.hv_curve     = self.mask_plot.plot([], [], pen=pg.mkPen(width=1))
        self.sag_curve    = self.mask_plot.plot([], [], pen=pg.mkPen(width=1, style=QtCore.Qt.DotLine))
        self.step_curve   = self.mask_plot.plot([], [], pen=pg.mkPen(width=1, style=QtCore.Qt.DashLine))
        self.corner_curve = self.mask_plot.plot([], [], pen=pg.mkPen(width=1, style=QtCore.Qt.SolidLine))
        self.burst_curve  = self.mask_plot.plot([], [], pen=pg.mkPen(width=1, style=QtCore.Qt.DashDotLine))
        self.wave_curve   = self.mask_plot.plot([], [], pen=pg.mkPen(width=1, style=QtCore.Qt.DashDotDotLine))
        self.resrefit_curve = self.mask_plot.plot([], [], pen=pg.mkPen(width=1, style=QtCore.Qt.DashLine))

        # 이벤트 연결
        def connect_toggle(w, slot):
            if isinstance(w, QtWidgets.QCheckBox): w.toggled.connect(slot)
        for w in [self.cb_qrsaware, self.cb_break_cut, self.cb_res_refit, self.cb_rrcap,
                  self.cb_sag, self.cb_step, self.cb_corner, self.cb_burst, self.cb_wave,
                  self.cb_rpeaks, self.cb_valleys]:
            connect_toggle(w, self.schedule_recompute)
        for cb in (self.cb_raw, self.cb_corr, self.cb_mask, self.cb_base):
            cb.toggled.connect(self.update_visibility)

        self.set_data(t, y_raw)

        def dblclick(ev):
            if ev.double(): self.plot.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        self.plot.scene().sigMouseClicked.connect(dblclick)

    @profiled()
    def schedule_recompute(self):
        if self._recompute_timer is None:
            self._recompute_timer = QtCore.QTimer(self); self._recompute_timer.setSingleShot(True)
            self._recompute_timer.timeout.connect(self.recompute)
        self._recompute_timer.start(600)

    @profiled()
    def set_data(self, t, y):
        y_centered = np.asarray(y, float)
        if y_centered.size > 0: y_centered = y_centered - float(np.nanmean(y_centered))
        self.t = np.asarray(t, float); self.y_raw = y_centered
        self.curve_raw.setData(self.t, self.y_raw)
        self.ov_curve.setData(self.t, self.y_raw)
        end_t = min(self.t[0]+40.0, self.t[-1]) if self.t.size>1 else 0.0
        self.region.setRegion([self.t[0], end_t])
        self.recompute()

    def update_visibility(self):
        self.curve_raw.setVisible(self.cb_raw.isChecked())
        self.curve_corr.setVisible(self.cb_corr.isChecked())
        self.mask_plot.setVisible(self.cb_mask.isChecked())
        self.curve_base.setVisible(self.cb_base.isChecked())
        self.scatter_r_after.setVisible(self.cb_rpeaks.isChecked())
        self.scatter_v_after.setVisible(self.cb_valleys.isChecked())

    @profiled()
    def recompute(self):
        D = DEFAULTS
        # 1) Baseline — Hybrid BL++ (+ RR cap)
        y_src = self.y_raw.copy()
        y_corr, base = baseline_hybrid_plus_adaptive(
            y_src, FS,
            per_win_s=D["PER_WIN_S"], per_q=D["PER_Q"],
            asls_lam=D["ASLS_LAM"], asls_p=D["ASLS_P"], asls_decim=D["ASLS_DECIM"],
            qrs_aware=self.cb_qrsaware.isChecked(), verylow_fc=D["LPF_FC"],
            clamp_win_s=6.0, vol_win_s=D["VOL_WIN"], vol_gain=D["VOL_GAIN"],
            lam_floor_ratio=D["LAM_FLOOR_PERCENT"]/100.0,
            hard_cut=self.cb_break_cut.isChecked(), break_pad_s=D["BREAK_PAD_S"],
            rr_cap_enable=False,#self.cb_rrcap.isChecked(),
            rr_eps_up=D["RR_EPS_UP"], rr_eps_dn=D["RR_EPS_DN"],
            rr_t0_ms=D["RR_T0_MS"], rr_t1_ms=D["RR_T1_MS"],
        )

        # 1.5) Residual selective refit — 현재 비활성 (원본 주석 유지)
        resrefit_mask = np.zeros_like(y_corr, dtype=bool)
        # if self.cb_res_refit.isChecked():
        #     y_corr2, base2, resrefit_mask = selective_residual_refit(...)
        #     y_corr, base = y_corr2, base2

        # === No AGC / No Glitch ===
        y_corr_eq = y_corr

        # 추가 평활(저주파 보존, R-peak 보존 경향)
        y_corr_eq = smooth_preserve_r(y_corr_eq, target_fs=50)

        # 1초 메디안 베이스라인 제거로 잔류 오프셋 정리
        baseline = signal.medfilt(y_corr_eq, kernel_size=251)
        y_corr_eq = y_corr_eq - baseline

        # 2) R-peak & valley 검출
        detector = rPeakDetector(fs=int(FS))
        r_after = np.asarray(detector.rPeakDetection(y_corr_eq), int)
        v_after = np.asarray(detector.vValleyDetection(y_corr_eq), int)
        # 보호 구간(P/ST) 생성
        wins_protect = build_protect_windows(r_after, FS, y_corr.size,
                                             p_win=(-220, -100), st_win=(+120, +260))

        # === QVRi (PQ 앵커 + ROI 보호) ===
        y_before_qvri = y_corr.copy()  # 원형 저장
        y_corr_qvri, base_qvri = _qvri_residual_isoelectric(
            y_corr, FS, r_after,
            t0_ms=-240, t1_ms=-100, stride=2, lam=2000.0, pin_strength=1e9,
            protect_windows=wins_protect, w_protect=1e-6
        )

        # --- (A) QVRi baseline을 ROI에서 '경계보존 평탄'으로 교체 ---
        roi_wins = build_roi_windows(r_after, FS, y_corr.size,
                                     p_win=(-220, -100), st_win=(+120, +260))
        base_qvri_roi = roi_pchip_fill_baseline(base_qvri, roi_wins)
        y_qvri_edge = y_corr - base_qvri_roi  # 경계값 일치한 QVRi 출력

        # --- (B) 메디안 짧게(≈0.4~0.6s)로 만든 'P/ST 보존' 출력 생성 ---
        y_med_base = signal.medfilt(y_corr, kernel_size=101)  # FS=250 기준 ≈0.4s 권장
        y_med_out = y_corr - y_med_base

        # --- (C) ROI에서 두 결과의 적응 혼합(기본 γ=0.5, 자동 상향) ---
        y_corr_eq = roi_adaptive_mix(y_qvri_out=y_qvri_edge,
                                     y_med_out=y_med_out,
                                     wins=roi_wins, fs=FS,
                                     gamma=0.5, corr_min=0.15)

        # # Q/ST 국소 affine 복구 (원신호 기준)
        # y_corr_eq = affine_restore_roi(
        #     y_before_qvri, y_corr_qvri, wins_protect,
        #     gmin=0.95, gmax=1.12, off_cap=0.25,
        #     blend_ms=60, fs=FS, corr_min=0.15, skip_if_negative=False
        # )
        #

        # y_corr_eq = y_corr_qvri

        # 오버레이 표시
        if self.cb_rpeaks.isChecked() and r_after.size > 0:
            valid = (r_after >= 0) & (r_after < self.t.size)
            r_after = r_after[valid]
            self.scatter_r_after.setData(self.t[r_after], y_corr_eq[r_after])
        else:
            self.scatter_r_after.clear()

        if self.cb_valleys.isChecked() and v_after.size > 0:
            valid = (v_after >= 0) & (v_after < self.t.size)
            v_after = v_after[valid]
            self.scatter_v_after.setData(self.t[v_after], y_corr_eq[v_after])
        else:
            self.scatter_v_after.clear()

        # 4) Masks
        sag_mask = suppress_negative_sag(
            y_corr_eq, FS, win_sec=D["SAG_WIN_S"], q_floor=D["SAG_Q"], k_neg=D["SAG_K"],
            min_dur_s=D["SAG_MINDUR_S"], pad_s=D["SAG_PAD_S"], protect_qrs=True
        ) if self.cb_sag.isChecked() else np.zeros_like(y_corr_eq, bool)

        step_mask = fix_downward_steps_mask(
            y_corr_eq, FS,
            amp_sigma=D["STEP_SIGMA"],
            amp_abs=(None if D["STEP_ABS"] <= 0 else D["STEP_ABS"]),
            min_hold_s=D["STEP_HOLD_S"], protect_qrs=True
        ) if self.cb_step.isChecked() else np.zeros_like(y_corr_eq, bool)

        corner_mask = smooth_corners_mask(
            y_corr_eq, FS, L_ms=D["CORNER_L_MS"], k_sigma=D["CORNER_K"], protect_qrs=True
        ) if self.cb_corner.isChecked() else np.zeros_like(y_corr_eq, bool)

        b_mask = burst_mask(
            y_corr_eq, FS, win_ms=D["BURST_WIN_MS"], k_diff=D["BURST_KD"], k_std=D["BURST_KS"],
            pad_ms=D["BURST_PAD_MS"], protect_qrs=True
        ) if self.cb_burst.isChecked() else np.zeros_like(y_corr_eq, bool)

        alpha_w = np.zeros_like(y_corr_eq)
        if self.cb_wave.isChecked():
            _, alpha_w = qrs_aware_wavelet_denoise(
                y_corr_eq, FS, sigma_scale=D["WAVE_SIGMA"], blend_ms=D["WAVE_BLEND_MS"]
            )

        hv_mask, hv_stats = high_variance_mask(
            y_corr_eq, win=D["HV_WIN"], k_sigma=D["HV_KSIGMA"], pad=D["HV_PAD"]
        )

        # Plot 업데이트
        self.curve_base.setData(self.t, base)
        self.curve_corr.setData(self.t, y_corr_eq)
        self.curve_raw.setData(self.t, self.y_raw)

        self.hv_curve.setData(self.t, hv_mask.astype(int))
        self.sag_curve.setData(self.t, sag_mask.astype(int))
        self.step_curve.setData(self.t, step_mask.astype(int))
        self.corner_curve.setData(self.t, corner_mask.astype(int))
        self.burst_curve.setData(self.t, b_mask.astype(int))
        self.wave_curve.setData(self.t, (alpha_w > 0.5).astype(int))
        self.resrefit_curve.setData(self.t, resrefit_mask.astype(int))

        txt = (
            f"HV removed={int(hv_mask.sum())} ({100*hv_mask.mean():.2f}%) | "
            f"kept={len(y_corr_eq)-int(hv_mask.sum())} | ratio={(1-hv_mask.mean()):.3f}"
        )
        self.mask_plot.setTitle(txt)

        self.update_visibility()

        # X/Y 범위
        lo, hi = self.region.getRegion()
        self.plot.setXRange(lo, hi, padding=0)
        vis_idx = (self.t >= lo) & (self.t <= hi)
        if np.any(vis_idx):
            y_sub = self.y_raw[vis_idx]
            ymin, ymax = float(np.min(y_sub)), float(np.max(y_sub))
            if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
                margin = 0.1 * (ymax - ymin) if (ymax - ymin) > 0 else 1.0
                self.plot.setYRange(ymin - margin, ymax + margin, padding=0)

        profiler_report(topn=25)

    def update_region(self):
        lo, hi = self.region.getRegion(); self.plot.setXRange(lo, hi, padding=0)
        vis_idx = (self.t >= lo) & (self.t <= hi)
        if np.any(vis_idx):
            y_sub = self.y_raw[vis_idx]; y_min, y_max = np.min(y_sub), np.max(y_sub)
            if np.isfinite(y_min) and np.isfinite(y_max) and (y_max > y_min):
                margin = 0.1 * (y_max - y_min)
                self.plot.setYRange(float(y_min - margin), float(y_max + margin), padding=0)

# =========================
# Main
# =========================
def main():
    with FILE_PATH.open('r', encoding='utf-8') as f:
        data = json.load(f)
    ecg_raw = extract_ecg(data); assert ecg_raw is not None and ecg_raw.size > 0
    ecg = decimate_if_needed(ecg_raw, DECIM)#[100000:200000]
    t = np.arange(ecg.size) / FS

    app = QtWidgets.QApplication(sys.argv)
    QtWidgets.QApplication.setStyle('Fusion')
    w = QtWidgets.QMainWindow()
    viewer = ECGViewer(t, ecg)
    w.setWindowTitle(f"ECG Viewer — {int(FS_RAW)}→{int(FS)} Hz | Hybrid BL++ (AGC/Glitch 없음) | RR-cap | Masks on processed signal | No interpolation")
    w.setCentralWidget(viewer); w.resize(1480, 930); w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
