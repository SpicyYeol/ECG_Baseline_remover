# -*- coding: utf-8 -*-
# ECG Viewer — 1000→250 Hz | Hybrid BL++ (adaptive λ, variance-aware, hard-cut) + Residual Refit
# (AGC & Glitch 제거 버전)
# Masks(Sag/Step/Corner/Burst/Wave/HV)는 PROCESSED 신호(y_corr_eq=y_corr) 기준. 보간 없음.

import json
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
# =========================
# Lightweight Profiler
# =========================
from time import perf_counter

import neurokit2 as nk
import pyqtgraph as pg
import pywt
from PyQt5 import QtWidgets, QtCore
from scipy.linalg import solveh_banded
from scipy.ndimage import binary_dilation
from scipy.ndimage import median_filter as _mf
from scipy.signal import lfilter, lfilter_zi, decimate
from scipy.signal import savgol_filter

# =========================
# Defaults (수치 파라미터 고정값)
# =========================
DEFAULTS = dict(
    # Baseline Hybrid BL++
    PER_WIN_S=3.2, PER_Q=8, ASLS_LAM=8e7, ASLS_P=0.01, ASLS_DECIM=8,
    LPF_FC=0.55, VOL_WIN=0.8, VOL_GAIN=2.0, LAM_FLOOR_PERCENT=0.5, BREAK_PAD_S=0.30,
    # Residual refit
    RES_K=2.8, RES_WIN_S=0.5, RES_PAD_S=0.20,
    # RR cap
    RR_EPS_UP=6.0, RR_EPS_DN=8.0, RR_T0_MS=80, RR_T1_MS=320,
    # Masks
    SAG_WIN_S=1.0, SAG_Q=20, SAG_K=3.5, SAG_MINDUR_S=0.25, SAG_PAD_S=0.25,
    STEP_SIGMA=5.0, STEP_ABS=0.0, STEP_HOLD_S=0.45,
    CORNER_L_MS=140, CORNER_K=5.5,
    BURST_WIN_MS=140, BURST_KD=6.0, BURST_KS=3.0, BURST_PAD_MS=140,
    WAVE_SIGMA=2.8, WAVE_BLEND_MS=80,
    HV_WIN=2000, HV_KSIGMA=4.0, HV_PAD=200
)

_PROF = defaultdict(lambda: {"calls": 0, "total": 0.0})

import numpy as np

def bilateral_filter_1d(signal, sigma_s=5, sigma_r=0.2):
    n = len(signal)
    out = np.zeros_like(signal)
    for i in range(n):
        start = max(i - 3*sigma_s, 0)
        end = min(i + 3*sigma_s, n)
        idx = np.arange(start, end)
        spatial = np.exp(-0.5 * ((idx - i) / sigma_s) ** 2)
        range_ = np.exp(-0.5 * ((signal[idx] - signal[i]) / sigma_r) ** 2)
        weights = spatial * range_
        weights /= np.sum(weights)
        out[i] = np.sum(signal[idx] * weights)
    return out



def _prof_add(name: str, dt: float):
    d = _PROF[name]
    d["calls"] += 1
    d["total"] += float(dt)

class time_block:
    """with time_block('label'): ...  형태의 구간 측정용"""
    def __init__(self, name: str):
        self.name = name
        self.t0 = None
    def __enter__(self):
        self.t0 = perf_counter()
        return self
    def __exit__(self, exc_type, exc, tb):
        _prof_add(self.name, perf_counter() - self.t0)

def profiled(name: str = None):
    """함수/메서드에 붙이는 데코레이터"""
    def deco(fn):
        label = name or fn.__name__
        def wrapped(*args, **kwargs):
            t0 = perf_counter()
            try:
                return fn(*args, **kwargs)
            finally:
                _prof_add(label, perf_counter() - t0)
        wrapped.__name__ = fn.__name__
        wrapped.__doc__ = fn.__doc__
        return wrapped
    return deco

def profiler_report(topn: int = 30):
    """콘솔로 요약 출력 (총시간 내림차순)"""
    rows = []
    for k, v in _PROF.items():
        calls = v["calls"] or 1
        total = v["total"]
        avg = total / calls
        rows.append((k, calls, total, avg))
    rows.sort(key=lambda r: r[2], reverse=True)
    print("\n[Profiler]  function | calls | total_ms | avg_ms")
    for k, c, tot, avg in rows[:topn]:
        print(f"[Profiler] {k:>20} | {c:5d} | {tot*1000:8.2f} | {avg*1000:7.20f}")
    return rows

# =========================
# Config
# =========================
FILE_PATH = Path('11646C1011258_test5_20250825T112545inPlainText.json')
FS_RAW = 250.0
FS = 250.0
DECIM = int(round(FS_RAW / FS)) if FS > 0 else 1
if DECIM < 1: DECIM = 1

# =========================
# IO & Utils
# =========================
import numpy as np
from scipy.signal import find_peaks, medfilt

import numpy as np
def adaptive_kalman(y, init_Q=1e-5, init_R=1e-2, adapt_factor=0.98):
    x = y[0]
    P = 1
    Q = init_Q
    R = init_R
    out = [x]
    for k in range(1, len(y)):
        # 예측
        x_pred, P_pred = x, P + Q

        # 잔차
        e = y[k] - x_pred
        R = adapt_factor * R + (1 - adapt_factor) * e**2  # adaptive R update

        # 갱신
        K = P_pred / (P_pred + R)
        x = x_pred + K * e
        P = (1 - K) * P_pred

        out.append(x)
    return np.array(out)

def kalman_filter_motion_artifact(y, process_var=1e-5, meas_var=1e-2):
    """
    ✅ 모션 아티팩트 억제용 1D 칼만 필터
    process_var: 신호가 변화할 수 있는 정도 (Q)
    meas_var: 측정 노이즈 세기 (R)
    """
    n = len(y)
    x_est = np.zeros(n)
    P = np.zeros(n)

    # 초기값 설정
    x_est[0] = y[0]
    P[0] = 1.0

    for k in range(1, n):
        # --- Prediction ---
        x_pred = x_est[k-1]
        P_pred = P[k-1] + process_var

        # --- Update ---
        K = P_pred / (P_pred + meas_var)
        x_est[k] = x_pred + K * (y[k] - x_pred)
        P[k] = (1 - K) * P_pred

    return x_est


def lift_negative_sags_local(y, fs, win_s=1.0, sag_thr_q=15, smooth_ms=200, blend_ms=150):
    """
    ✅ QRS 피크 보존 + 하강부(local sag)만 위로 끌어올림
    - baseline: local median-filter 기반
    - sag detection: 하위 quantile 기반
    """
    x = np.asarray(y, float)
    N = len(x)
    t = np.arange(N) / fs

    # --- (1) 기본 baseline: 완만한 local median
    k = int(win_s * fs)
    if k % 2 == 0: k += 1
    base = medfilt(x, kernel_size=k)

    # --- (2) sag 구간 감지 (하위 quantile)
    thr = np.percentile(x - base, sag_thr_q)
    sag_mask = (x - base) < thr

    # --- (3) QRS 보호: 피크 ±100ms 제외
    distance = int(0.25 * fs)
    peaks, _ = find_peaks(np.abs(x), distance=distance, height=np.std(x) * 2)
    qrs_mask = np.zeros_like(x, bool)
    w = int(0.1 * fs)
    for p in peaks:
        s = max(0, p - w)
        e = min(N, p + w)
        qrs_mask[s:e] = True

    sag_mask &= ~qrs_mask  # QRS 제외

    # --- (4) sag 구간만 baseline upward shift
    sag_idx = np.where(sag_mask)[0]
    if len(sag_idx) > 0:
        shift_val = np.median(base[sag_idx] - x[sag_idx])
        base[sag_mask] -= shift_val * 0.8  # 하강부만 완화 보정

    # --- (5) 블렌딩 smoothing
    from scipy.ndimage import gaussian_filter1d
    base_smooth = gaussian_filter1d(base, sigma=(smooth_ms/1000)*fs/6)

    # --- (6) 최종 결과
    y_corr = x - (base_smooth - np.median(base_smooth))

    return y_corr, sag_mask, base_smooth

import numpy as np
from scipy.signal import find_peaks, medfilt

def lift_sag_to_neighbors(y, fs, sag_thr_q=15, win_s=0.5, pad_ms=100):
    """
    ✅ 하강부를 주변 평균 수준까지 끌어올림 (Local Leveling)
    - sag_thr_q: dip 감지 민감도 (낮을수록 더 많이 잡음)
    - win_s: baseline 추정 윈도우 (초)
    - pad_ms: 보정 양쪽 블렌딩 범위 (밀리초)
    """
    x = np.asarray(y, float)
    N = len(x)
    pad = int((pad_ms/1000)*fs)
    k = int(win_s*fs)
    if k % 2 == 0: k += 1
    base = medfilt(x, kernel_size=k)

    # sag 구간 탐지
    diff = x - base
    thr = np.percentile(diff, sag_thr_q)
    sag_mask = diff < thr

    # QRS 보호
    peaks, _ = find_peaks(np.abs(x), distance=int(0.25*fs), height=np.std(x)*2)
    protect = np.zeros_like(x, bool)
    w = int(0.1*fs)
    for p in peaks:
        protect[max(0,p-w):min(N,p+w)] = True
    sag_mask &= ~protect

    # sag 구간별로 주변 평균 기준으로 보정
    y_corr = x.copy()
    idx = np.where(sag_mask)[0]
    if len(idx) > 0:
        starts = np.where(np.diff(np.concatenate(([0], sag_mask.view(np.int8), [0]))) == 1)[0]
        ends   = np.where(np.diff(np.concatenate(([0], sag_mask.view(np.int8), [0]))) == -1)[0]
        for s, e in zip(starts, ends):
            left = max(0, s - pad)
            right = min(N, e + pad)
            left_mean = np.median(x[left:s]) if s > 0 else x[s]
            right_mean = np.median(x[e:right]) if e < N else x[e]
            target_level = (left_mean + right_mean) / 2
            seg_mean = np.median(x[s:e])
            delta = target_level - seg_mean
            y_corr[s:e] += delta  # 주변 평균 수준까지 상향 이동

            # 부드러운 경계 블렌딩
            blend_len = pad
            if s > 0:
                ramp = np.linspace(0, delta, blend_len)
                y_corr[max(0, s-blend_len):s] += ramp[:min(blend_len, s)]
            if e < N:
                ramp = np.linspace(delta, 0, blend_len)
                y_corr[e:min(N, e+blend_len)] += ramp[:min(blend_len, N-e)]

    return y_corr, sag_mask


@profiled()
def lift_negative_sags_asls(
    y, fs,
    sag_win_s=1.0, sag_q=20, sag_k=3.5, sag_min_dur_s=0.20, sag_pad_s=0.20,
    lam=3e5, p=0.002, niter=8, decim=6, blend_ms=80
):
    """
    음의 sag(아래로 꺼진) 구간만 강조해서 ASLS baseline을 추정해 빼줌으로써
    해당 구간의 파형을 '위로 끌어올리는' 효과를 냅니다.
    - sag 마스크: suppress_negative_sag 재사용
    - ASLS: mask=True 구간에 큰 가중(=baseline이 그 구간을 따라가도록)
    - 적용: 마스크 부근만 블렌딩 적용해 과보정을 방지
    """
    x = np.asarray(y, float)
    if x.size < 10:
        return x, np.zeros_like(x, bool), np.zeros_like(x, float)

    sag_mask = suppress_negative_sag(
        x, fs, win_sec=sag_win_s, q_floor=sag_q, k_neg=sag_k,
        min_dur_s=sag_min_dur_s, pad_s=sag_pad_s, protect_qrs=True
    )
    if not np.any(sag_mask):
        return x, sag_mask, np.zeros_like(x, float)

    # 음의 sag 구간에 가중을 크게(=baseline이 그 구간을 적극 추종)
    base_sag = baseline_asls_masked(
        x, lam=float(lam), p=float(p), niter=int(niter),
        mask=sag_mask, decim_for_baseline=max(1, int(decim))
    )

    # 블렌딩으로 마스크 경계 부드럽게
    alpha = _smooth_binary(sag_mask, fs, blend_ms=int(blend_ms))  # 0~1
    y_lift = (1.0 - alpha) * x + alpha * (x - base_sag)

    return y_lift, sag_mask, base_sag


@profiled()
def rr_segment_affine_normalize(y, fs,
                                qrs_pad_ms=80,
                                scale_clip=(0.85, 1.15),
                                ema_beta=0.1):
    """
    RR 구간별로 (QRS 제외) 중앙값/스케일(IQR)을 약하게 정렬.
    - 모폴로지 보존, 과도 평탄화 방지(스케일 클립)
    """
    import numpy as np
    x = np.asarray(y, float).copy()
    N = x.size
    try:
        r = np.array(nk.ecg_peaks(x, sampling_rate=fs)[1].get("ECG_R_Peaks", []), int)
    except Exception:
        r = np.array([], int)
    if r.size < 2:
        return x

    pad = int(round(qrs_pad_ms/1000.0 * fs))
    tgt_med = 0.0
    tgt_iqr = None

    for i in range(len(r)-1):
        a = r[i] + pad
        b = r[i+1] - pad
        if b - a < max(5, int(0.12*fs)):
            continue
        seg = x[a:b]
        q1, q2, q3 = np.percentile(seg, [25, 50, 75])
        iqr = max(1e-9, q3 - q1)

        # 목표 통계(느린 EMA)
        tgt_med = (1-ema_beta)*tgt_med + ema_beta*q2
        if tgt_iqr is None:
            tgt_iqr = iqr
        else:
            tgt_iqr = (1-ema_beta)*tgt_iqr + ema_beta*iqr

        s = np.clip(tgt_iqr/iqr, scale_clip[0], scale_clip[1])
        x[a:b] = (seg - q2) * s + tgt_med
    return x


@profiled()
def lowband_dynamic_equalizer(y, fs,
                              f_split=5.0,
                              ratio=3.0, knee=0.2,
                              atk_ms=60, rel_ms=400,
                              gmin=0.70, gmax=1.0):
    """
    5Hz 이하 저역 성분만 soft-knee 컴프레싱 → 드리프트/들뜸만 줄임.
    QRS 등의 중고역 성분은 보존.
    """
    import numpy as np
    from scipy.signal import butter, filtfilt
    from scipy.ndimage import uniform_filter1d

    x = np.asarray(y, float)
    if x.size == 0: return x

    # band-split
    b, a = butter(2, f_split/(fs*0.5), btype='low')
    l = filtfilt(b, a, x)          # 저역
    h = x - l                       # 중고역(모폴로지)

    # 저역의 에너지로 컴프레서 제어
    win = max(3, int(round(0.20*fs)))
    env = uniform_filter1d(np.abs(l), size=win, mode='nearest') + 1e-12
    T = float(np.percentile(env, 90))

    Tl, Th = T*(1.0-knee), T*(1.0+knee)
    s = env.copy()
    g = np.ones_like(s)
    high = s >= Th
    mid  = (s > Tl) & (~high)
    g_high = (T + (s - T)/ratio) / s
    g[high] = g_high[high]
    a_mid = (s[mid] - Tl) / max(1e-12, (Th - Tl))
    g[mid] = (1.0 - a_mid) * 1.0 + a_mid * g_high[mid]

    # attack/release 평활
    a_atk = np.exp(-1.0/max(1, int(atk_ms/1000.0*fs)))
    a_rel = np.exp(-1.0/max(1, int(rel_ms/1000.0*fs)))
    gg = np.empty_like(g); gg[0] = 1.0
    for n in range(1, g.size):
        a_ = a_atk if g[n] < gg[n-1] else a_rel
        gg[n] = a_*gg[n-1] + (1-a_)*g[n]
    gg = np.clip(gg, float(gmin), float(gmax))

    l_comp = l * gg
    return l_comp + h


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
    try:
        return decimate_fir_zero_phase(x, decim)
    except Exception:
        n = (len(x)//decim)*decim
        return x[:n].reshape(-1, decim).mean(axis=1)
@profiled()
def _onepole(sig, fc, fs, zero_phase=False, use_float32=True):
    """
    1차 저역통과(One-pole) — 고속/안정 버전
    y[n] = (1-α) * x[n] + α * y[n-1],  α = exp(-2π fc / fs)
    """
    x = np.asarray(sig, np.float32 if use_float32 else np.float64)
    N = x.size
    if N == 0 or fc <= 0.0:
        return x.astype(np.float64, copy=False)
    if fs <= 0.0:
        raise ValueError("fs must be > 0")

    alpha = float(np.exp(-2.0 * np.pi * float(fc) / float(fs)))
    b0 = 1.0 - alpha
    a1 = alpha

    try:
        if zero_phase:
            b = [b0]
            a = [1.0, -a1]
            padlen = min(3 * (max(len(a), len(b)) - 1), max(0, N - 1))
            y = filtfilt(b, a, x, padlen=padlen) if padlen > 0 else filtfilt(b, a, x)
            return y.astype(np.float64, copy=False)

        b = [b0]
        a = [1.0, -a1]
        zi = lfilter_zi(b, a) * x[0]
        y, _ = lfilter(b, a, x, zi=zi)
        return y.astype(np.float64, copy=False)

    except Exception:
        y = np.empty_like(x, dtype=x.dtype)
        y[0] = x[0]
        for i in range(1, N):
            y[i] = a1 * y[i-1] + b0 * x[i]
        return y.astype(np.float64, copy=False)


def replace_with_bandlimited(y, fs, mask, fc=12.0):
    """마스크 구간만 저역통과 재구성한 신호로 치환 후 페이드."""
    b,a = butter(3, fc/(fs/2.0), btype='low')
    y_lp = filtfilt(b, a, y)
    # 경계 페이드
    win = int(0.10*fs)  # 100 ms
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


def burst_gate_dampen(y, fs,
                      win_ms=140, k_diff=6.0, k_std=3.0, pad_ms=120,
                      limit_ratio=0.6, alpha=1.2, atk_ms=60, rel_ms=300,
                      protect_qrs=True):
    """
    급변(z_diff) + 분산(z_std) 동시 초과 구간만 가변 이득 g(t)로 감쇠.
    """
    x = np.asarray(y, float)
    N = x.size
    if N < 10: return x, np.zeros(N, bool), np.ones(N, float)

    w = max(3, int(round((win_ms/1000.0)*fs)));  w += (w % 2 == 0)
    dx  = np.gradient(x)
    dmed= float(np.median(dx)); dmad= float(np.median(np.abs(dx-dmed)) + 1e-12)
    zdf = (dx - dmed) / (1.4826*dmad)

    m  = uniform_filter1d(x,   size=w, mode='nearest')
    m2 = uniform_filter1d(x*x, size=w, mode='nearest')
    v  = np.maximum(m2 - m*m, 0.0)
    rs = np.sqrt(v)

    rs_med = float(np.median(rs)); rs_mad = float(np.median(np.abs(rs-rs_med)) + 1e-12)
    thr_std = rs_med + 1.4826*rs_mad*float(k_std)
    cand = (np.abs(zdf) > float(k_diff)) & (rs > thr_std)

    pad = int(round((pad_ms/1000.0)*fs))
    if pad > 0 and cand.any():
        st = np.ones(pad*2+1, dtype=bool)
        from scipy.ndimage import binary_dilation
        cand = binary_dilation(cand, structure=st)

    eps = 1e-12
    g_raw = np.ones_like(x)
    idx = rs > thr_std
    g_raw[idx] = np.minimum(1.0, (thr_std / (rs[idx] + eps))**float(alpha))
    g_raw = np.maximum(g_raw, float(limit_ratio))

    g_target = np.where(cand, g_raw, 1.0)

    def one_pole(env, atk, rel):
        out = np.empty_like(env)
        a_atk = np.exp(-1.0/max(1, int(atk*fs)))
        a_rel = np.exp(-1.0/max(1, int(rel*fs)))
        y0 = 1.0; out[0] = y0
        for n in range(1, env.size):
            a = a_atk if env[n] < out[n-1] else a_rel
            out[n] = a*out[n-1] + (1-a)*env[n]
        return out

    g = one_pole(g_target, atk_ms/1000.0, rel_ms/1000.0)

    y_out = x * g
    return y_out, cand, g


# --- add: robust high-pass for DC drift removal ---
from scipy.signal import butter, filtfilt

def highpass_zero_drift(x, fs, fc=0.3, order=2):
    """Remove DC/very-low drift without morphology loss."""
    if fc <= 0:
        return x - np.median(x)
    b, a = butter(order, fc/(fs/2.0), btype='high')
    y = filtfilt(b, a, np.asarray(x, float))
    return y - np.median(y)

import numpy as np
from scipy.ndimage import uniform_filter1d, percentile_filter

def wvg_flatten(y, fs,
                win_s=0.45,
                q_lo=25, q_hi=75,
                spread_thr=8.0,
                std_thr=6.0,
                blend_s=0.20):

    x = np.asarray(y, float)
    if x.size == 0: return x, np.zeros_like(x, bool)

    w = max(3, int(round(win_s * fs)))
    if w % 2 == 0: w += 1

    lo = percentile_filter(x, percentile=q_lo, size=w, mode='nearest')
    hi = percentile_filter(x, percentile=q_hi, size=w, mode='nearest')
    med = percentile_filter(x, percentile=50,   size=w, mode='nearest')

    m  = uniform_filter1d(x,   size=w, mode='nearest')
    m2 = uniform_filter1d(x*x, size=w, mode='nearest')
    v  = np.maximum(m2 - m*m, 0.0)
    sd = np.sqrt(v)

    spread = hi - lo
    quiet  = (spread <= float(spread_thr)) & (sd <= float(std_thr))

    if blend_s and quiet.any():
        L = max(3, int(round(blend_s * fs)))
        if L % 2 == 0: L += 1
        win = np.hanning(L); win /= win.sum()
        alpha = np.convolve(quiet.astype(float), win, mode='same')
    else:
        alpha = quiet.astype(float)

    y_flat = x * (1.0 - alpha) + med * alpha
    return y_flat, quiet


# =========================
# Baseline core deps
# =========================
@profiled()
def baseline_asls_masked(y, lam=1e6, p=0.008, niter=10, mask=None,
                         cg_tol=1e-3, cg_maxiter=200, decim_for_baseline=1,
                         use_float32=True):
    """
    ASLS(비대칭 가중 최소제곱) - 고속화
    """

    y = np.asarray(y, np.float32 if use_float32 else np.float64)
    N = y.size
    if N < 3:
        return np.zeros_like(y)

    if decim_for_baseline > 1:
        q = int(decim_for_baseline)
        n = (N // q) * q
        if n < q:  # 👈 이 한 줄 추가!
            return np.zeros_like(y)
        y_head = y[:n]
        y_ds = y_head.reshape(-1, q).mean(axis=1)
        z_ds = baseline_asls_masked(y_ds, lam=lam, p=p, niter=niter, mask=None,
                                    decim_for_baseline=1, use_float32=use_float32)
        idx = np.repeat(np.arange(z_ds.size), q)
        z_coarse = z_ds[idx]
        if z_coarse.size < N:
            z = np.empty(N, y.dtype)
            z[:z_coarse.size] = z_coarse
            z[z_coarse.size:] = z_coarse[-1]
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
    if N < 0.5 * 250:
        base_niter = min(base_niter, 5)
    if N < 0.25 * 250:
        base_niter = min(base_niter, 4)

    w = np.ones(N, dtype=y.dtype)
    z = np.zeros(N, dtype=y.dtype)

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
            if last_obj is not None and abs(last_obj - obj) <= 1e-5 * max(1.0, obj):
                break
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

# 변화점 탐지/마스크 팽창
@profiled()
def _find_breaks(y, fs, k=7.0, min_gap_s=0.30):
    dy = np.diff(y, prepend=y[0])
    med = np.median(dy); mad = np.median(np.abs(dy - med)) + 1e-12
    z = np.abs(dy - med) / (1.4826 * mad)
    idx = np.flatnonzero(z > float(k))
    if idx.size == 0: return []
    gap = int(round(min_gap_s * fs))
    breaks = [int(idx[0])]
    for i in idx[1:]:
        if i - breaks[-1] > gap:
            breaks.append(int(i))
    return breaks
@profiled()
def _dilate_mask(mask, fs, pad_s=0.45):
    pad = int(round(pad_s * fs))
    if pad <= 0: return mask
    k = np.ones(pad*2+1, dtype=int)
    return (np.convolve(mask.astype(int), k, mode='same') > 0)

# Hybrid BL++ (adaptive λ, variance-aware, hard-cut, local refit)
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
    """
    Hybrid BL++ (adaptive λ, variance-aware, hard-cut, local refit) — Optimized
    + RR isoelectric cap(option): QRS 사이 baseline 들림 억제
    """

    x = np.asarray(y, float)
    N = x.size
    if N < 8:
        return np.zeros_like(x), np.zeros_like(x)

    def _odd(n):
        n = int(max(3, n))
        return n + (n % 2 == 0)

    def _mov_stats(xx, win):
        k = np.ones(win, float)
        s1 = np.convolve(xx, k, mode='same')
        s2 = np.convolve(xx*xx, k, mode='same')
        m = s1 / win
        v = s2 / win - m*m
        v[v < 0] = 0.0
        return m, np.sqrt(v)

    def _segments_from_lambda(lam_arr, fs_, brks):
        lam_eps = 1e-12
        L = np.log(lam_arr + lam_eps)
        q_lo, q_hi = np.quantile(L, [0.05, 0.95])
        if q_hi <= q_lo:
            q_hi = q_lo + 1e-6
        bins = np.linspace(q_lo, q_hi, int(max(2, lam_bins)))
        idx = np.clip(np.digitize(L, bins, right=False), 0, len(bins))

        cuts = [0] + [int(b) for b in brks] + [N]
        segs = []
        for s0, e0 in zip(cuts[:-1], cuts[1:]):
            if e0 - s0 <= 0:
                continue
            run_id = idx[s0:e0]
            if run_id.size == 0:
                continue
            a = s0
            cur = run_id[0]
            for i in range(s0+1, e0):
                if idx[i] != cur:
                    segs.append((a, i, cur))
                    a, cur = i, idx[i]
            segs.append((a, e0, cur))

        min_len = int(round(float(min_seg_s) * fs_))
        merged = []
        for s, e, kbin in segs:
            if not merged:
                merged.append([s, e, kbin])
                continue
            ms, me, mk = merged[-1]
            if (e - s) < min_len and mk == kbin:
                merged[-1][1] = e
            else:
                if (me - ms) < min_len and kbin != mk:
                    merged[-1][1] = e
                else:
                    merged.append([s, e, kbin])

        out = []
        max_len = int(round(float(max_seg_s) * fs_))
        for s, e, kbin in merged:
            Lseg = e - s
            if Lseg <= max_len:
                out.append((s, e))
            else:
                step = max_len
                for a in range(s, e, step):
                    b = min(e, a + step)
                    if b - a > 5:
                        out.append((a, b))
        out2 = []
        last = -1
        for s, e in sorted(out):
            if s < last:
                s = last
            if e > s:
                out2.append((s, e))
                last = e
        return out2

    # ---------- 0) 초기 퍼센타일 바닥선 ----------
    w0 = _odd(int(round(per_win_s * fs)))
    dc = np.median(x[np.isfinite(x)])
    x0 = x - dc
    b0 = percentile_filter(x0, percentile=int(per_q), size=w0, mode='nearest')

    # ---------- 1) QRS-aware + 변화점 보호 ----------
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
        prot = np.zeros_like(x, bool)
        prot[np.asarray(brks, int)] = True
        prot = _dilate_mask(prot, fs, pad_s=max(0.35, float(break_pad_s)))
        base_mask &= (~prot)

    # ---------- 2) 위치별 λ 설계 ----------
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

    # ---------- 3) 세그먼트 피팅 ----------
    b1 = np.zeros_like(x)
    segs = _segments_from_lambda(lam_local, fs, brks if hard_cut else [])
    if not segs:
        segs = [(0, N)]

    for s, e in segs:
        if (e - s) < max(5, int(0.20 * fs)):
            continue
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

    clamp_w = _odd(int(round(clamp_win_s * fs)))
    sg_win  = max(_odd(int(fs * 1.5)), clamp_w)
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
        smw = max(3, int(round(0.12 * fs)));  smw += (smw % 2 == 0)
        err = uniform_filter1d(err, size=smw, mode='nearest')
        b_final = b_slow + iso + err

    y_corr = x - b_final
    return y_corr, b_final
@profiled()
def qrs_aware_soft_compressor(
    y, fs,
    win_s=0.6,          # 로컬 에너지(절대값 평균) 윈도우
    hi_q=92,            # 임계 기준을 잡을 전역 퍼센타일(큰 값일수록 덜 눌림)
    ratio=4.0,          # 임계 이상에서의 압축비(커질수록 더 강하게 줄임)
    knee=0.20,          # soft-knee 폭(임계 부근 완만한 곡선, 비율=임계의 20%)
    atk_ms=50, rel_ms=400,  # attack/release 시간
    gmin=0.35, gmax=1.0,    # 이득 하한/상한
    qrs_soft=0.5            # QRS 보호 강도(0=보호없음, 1=완전보호)
):
    """
    QRS-aware Soft Compressor
    - 로컬 에너지(|y|의 이동평균) 기반으로 임계 초과분만 부드럽게 압축
    - soft-knee + attack/release 로 펌핑/링잉 억제
    - QRS 근처는 과압축 완화(qrs_soft)
    반환: (y_comp, g)  # g(t)=적용 이득
    """
    import numpy as _np
    from scipy.ndimage import uniform_filter1d as _uf1d

    x = _np.asarray(y, float)
    N = x.size
    if N == 0:
        return x, _np.ones(0, float)

    # 1) 로컬 에너지(절대값 이동평균)
    w = max(3, int(round(win_s * fs)))
    if w % 2 == 0: w += 1
    env = _uf1d(_np.abs(x), size=w, mode='nearest') + 1e-12

    # 2) 전역 임계 T (퍼센타일 기반, outlier에 강건)
    T = float(_np.percentile(env, hi_q))
    if T <= 0:
        return x, _np.ones_like(x)

    # 3) soft-knee 압축 곡선 (임계 근방 부드럽게)
    #    s<=Tl : g=1,   s>=Th : g=(T+(s-T)/ratio)/s
    #    Tl=T*(1-knee), Th=T*(1+knee)
    Tl = T * (1.0 - float(knee))
    Th = T * (1.0 + float(knee))
    s = env

    g_raw = _np.ones_like(s)
    # 선형 보간 soft-knee
    # knee 내부: Th쪽 공식으로 선형 전이
    mask_low  = s <= Tl
    mask_high = s >= Th
    mask_mid  = (~mask_low) & (~mask_high)

    g_high = (T + (s - T) / float(ratio)) / s
    g_raw[mask_high] = g_high[mask_high]

    # knee 구간 보간
    if _np.any(mask_mid):
        a = (s[mask_mid] - Tl) / max(1e-12, (Th - Tl))
        g_mid = (1.0 - a) * 1.0 + a * g_high[mask_mid]
        g_raw[mask_mid] = g_mid

    # 4) QRS 보호: QRS 근처에서는 (1.0 ↔ g_raw) 사이로 보수적
    try:
        qmask = make_qrs_mask(x, fs=fs)  # True=비-QRS, False=QRS부
        # 비-QRS(=True)일수록 'g_raw' 적용, QRS는 1.0에 가깝게
        alpha = _uf1d(qmask.astype(float), size=max(3, int(0.08*fs)), mode='nearest')
        g_raw = (qrs_soft) * 1.0 + (1.0 - qrs_soft) * (alpha * g_raw + (1.0 - alpha) * 1.0)
    except Exception:
        pass

    # 5) attack / release 평활
    def one_pole(env, atk_t, rel_t):
        out = _np.empty_like(env)
        a_atk = _np.exp(-1.0/max(1, int(atk_t*fs)))
        a_rel = _np.exp(-1.0/max(1, int(rel_t*fs)))
        y0 = 1.0; out[0] = y0
        for n in range(1, env.size):
            a = a_atk if env[n] < out[n-1] else a_rel
            out[n] = a*out[n-1] + (1-a)*env[n]
        return out

    g = one_pole(_np.clip(g_raw, float(gmin), float(gmax)), atk_ms/1000.0, rel_ms/1000.0)

    # 6) 적용
    y_out = x * g
    return y_out, g

def soft_agc_qrs_aware(
    y, fs,
    win_s=0.8,
    method="mad",
    target_q=70,
    alpha=1.0,
    gmin=0.35, gmax=1.0,
    smooth_s=0.6,
    qrs_soft=0.35
):
    """
    QRS-aware soft AGC
    """
    x = np.asarray(y, float)
    N = x.size
    if N == 0:
        return x

    try:
        qmask = make_qrs_mask(x, fs=fs)
        alpha_qrs = qrs_soft + (qmask.astype(float)) * (1.0 - qrs_soft)
    except Exception:
        alpha_qrs = np.ones_like(x)

    win = max(3, int(round(win_s * fs)))
    if win % 2 == 0: win += 1
    if method == "rms":
        m  = uniform_filter1d(x,   size=win, mode='nearest')
        m2 = uniform_filter1d(x*x, size=win, mode='nearest')
        v  = np.maximum(m2 - m*m, 0.0)
        s  = np.sqrt(v + 1e-12)
    else:
        med = percentile_filter(x, percentile=50, size=win, mode='nearest')
        r   = x - med
        m1  = uniform_filter1d(np.abs(r), size=win, mode='nearest')
        s   = 1.4826 * m1 + 1e-12

    s_ref = float(np.percentile(s, target_q))
    g = (s_ref / (s + 1e-12)) ** float(alpha)
    g = np.clip(g, float(gmin), float(gmax))

    smw = max(3, int(round(smooth_s * fs)))
    if smw % 2 == 0: smw += 1
    g = uniform_filter1d(g, size=smw, mode='nearest')

    w = alpha_qrs
    y_eq = x * (w + (1.0 - w) * g)
    return y_eq


def rr_isoelectric_clamp(y, fs, r_idx=None, t0_ms=80, t1_ms=300):
    """RR 사이 등전위(PR/T) median을 스플라인처럼 연결한 baseline"""
    x = np.asarray(y, float)
    if r_idx is None or len(r_idx) < 2:
        try:
            info = nk.ecg_peaks(x, sampling_rate=fs)[1]
            r_idx = np.array(info.get("ECG_R_Peaks", []), int)
        except Exception:
            r_idx = np.array([], int)
    if r_idx.size < 2:
        return np.zeros_like(x)

    t0 = int(round(t0_ms * 1e-3 * fs))
    t1 = int(round(t1_ms * 1e-3 * fs))

    pts_x, pts_y = [], []
    N = x.size
    for r in r_idx[:-1]:
        a = max(0, r + t0); b = min(N, r + t1)
        if b - a < max(5, int(0.04 * fs)):
            continue
        m = float(np.median(x[a:b]))
        pts_x.append((a + b) // 2); pts_y.append(m)
    if len(pts_x) < 2:
        return np.zeros_like(x)

    xs = np.arange(N, dtype=float)
    baseline_rr = np.interp(xs, np.array(pts_x, float), np.array(pts_y, float))
    baseline_rr -= np.median(baseline_rr)
    return baseline_rr

# =========================
# Residual-based selective refit
# =========================
@profiled()
def selective_residual_refit(
    y_src, base_in, fs,
    k_sigma=3.2, win_s=0.5, pad_s=0.20,
    asls_lam=5e4, asls_p=0.01, asls_decim=6,
    grid_ms=32, topk_per_5s=1, min_gap_s=0.20, max_asls_blk_s=3.0,
    parallel_workers=0, use_float32=True
):
    """
    선택적 잔차 리핏 — ASLS 전용
    """

    import numpy as _np
    from scipy.ndimage import binary_dilation as _bin_dil

    x  = _np.asarray(y_src, _np.float32 if use_float32 else _np.float64)
    bb = _np.asarray(base_in, _np.float32 if use_float32 else _np.float64).copy()
    N = x.size
    if N < 10:
        return (x - bb).astype(_np.float64, copy=False), bb.astype(_np.float64, copy=False), _np.zeros(N, bool)

    resid = x - bb
    med = float(_np.median(resid))
    mad = float(_np.median(_np.abs(resid - med)) + 1e-12)
    z = _np.abs((resid - med) / (1.4826 * mad))
    cand = z > float(k_sigma)

    pad_n = int(round(pad_s * fs))
    if pad_n > 0 and cand.any():
        st = _np.ones(pad_n * 2 + 1, dtype=bool)
        cand = _bin_dil(cand, structure=st)

    if not cand.any():
        return (x - bb).astype(_np.float64, copy=False), bb.astype(_np.float64, copy=False), _np.zeros(N, bool)

    diff = _np.diff(cand.astype(_np.int8), prepend=0, append=0)
    starts = _np.flatnonzero(diff == 1)
    ends   = _np.flatnonzero(diff == -1)

    min_len = max(5, int(0.20 * fs))
    keep = _np.where((ends - starts) >= min_len)[0]
    if keep.size == 0:
        return (x - bb).astype(_np.float64, copy=False), bb.astype(_np.float64, copy=False), _np.zeros(N, bool)
    starts = starts[keep]; ends = ends[keep]

    hop = max(1, int(round((grid_ms / 1000.0) * fs)))
    scores = []
    for a, b in zip(starts, ends):
        zz = z[a:b:hop]
        L  = max(1, b - a)
        scores.append(float(zz.mean() if zz.size else 0.0) * (_np.sqrt(L)))
    scores = _np.asarray(scores)

    T = N / float(fs)
    K = max(3, int(_np.ceil(T / 5.0) * max(1, int(topk_per_5s))))
    if scores.size > K:
        ord_idx = _np.argsort(scores)[::-1][:K]
        starts = starts[ord_idx]; ends = ends[ord_idx]

    refit_mask = _np.zeros(N, dtype=bool)

    max_blk = int(round(max_asls_blk_s * fs))

    def _taper(L):
        if L <= 8: return _np.ones(L, float)
        tlen = min(L // 3, max(3, int(0.06 * fs)))
        if tlen <= 0: return _np.ones(L, float)
        w = _np.hanning(2 * tlen)
        t = _np.ones(L, float); t[:tlen] = w[:tlen]; t[-tlen:] = w[-tlen:]
        return t

    def fit_one(a, b):
        L = int(b - a)
        if L <= 0:
            return None
        out = _np.zeros(L, dtype=_np.float32)
        i = 0
        while i < L:
            j = min(L, i + max_blk)
            seg_ctx = (x[a+i:a+j] - bb[a+i:a+j]).astype(_np.float64, copy=False)
            b_loc = baseline_asls_masked(
                seg_ctx, lam=float(asls_lam), p=float(asls_p),
                niter=8, mask=None, decim_for_baseline=max(1, int(asls_decim)),
            ).astype(_np.float32, copy=False)
            n = b_loc.size
            if i > 0:
                ov = max(3, int(0.10 * fs))
                ov = min(ov, n // 2)
                if ov > 0:
                    w = _np.hanning(2 * ov)
                    out[i:i+ov]   = out[i:i+ov]   * w[:ov] + b_loc[:ov] * w[ov:]
                    out[i+ov:i+n] = b_loc[ov:]
                else:
                    out[i:i+n] = b_loc
            else:
                out[i:i+n] = b_loc
            i += max_blk
        if L > 8:
            from scipy.ndimage import median_filter as _mf
            out = _mf(out, size=max(3, int(0.10 * fs)), mode='nearest')
            out *= _taper(L)
        return (a, b, out)

    if parallel_workers and len(starts) > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=int(parallel_workers)) as ex:
            futs = [ex.submit(fit_one, int(a), int(b)) for a, b in zip(starts, ends)]
            for fu in as_completed(futs):
                res = fu.result()
                if res is None: continue
                a, b, vec = res
                bb[a:b] += vec.astype(bb.dtype, copy=False)
                refit_mask[a:b] = True
    else:
        for a, b in zip(starts, ends):
            res = fit_one(int(a), int(b))
            if res is None: continue
            a, b, vec = res
            bb[a:b] += vec.astype(bb.dtype, copy=False)
            refit_mask[a:b] = True

    y_corr2 = (x - bb).astype(_np.float64, copy=False)
    return y_corr2, bb.astype(_np.float64, copy=False), refit_mask



# =========================
# Masks (computed on processed signal)
# =========================
@profiled()
def suppress_negative_sag(
    y, fs, win_sec=1.0, q_floor=20, k_neg=3.5, min_dur_s=0.25, pad_s=0.25,
    protect_qrs=True, r_idx=None, qrs_mask=None, use_fast_filter=True
):
    """
    suppress_negative_sag (고속화 버전)
    """

    y = np.asarray(y, float)
    N = y.size
    if N < 10:
        return np.zeros(N, bool)

    w = max(3, int(round(win_sec * fs)))
    w += (w % 2 == 0)
    min_len = int(round(min_dur_s * fs))
    pad_n = int(round(pad_s * fs))

    if use_fast_filter:
        m = uniform_filter1d(y, size=w, mode='nearest')
        m2 = uniform_filter1d(y * y, size=w, mode='nearest')
        v = m2 - m * m
        v[v < 0] = 0.0
        s = np.sqrt(v)
        zq = abs(0.01 * (50 - q_floor)) * 0.1
        floor = m - zq * s
        median = m
    else:
        floor = percentile_filter(y, percentile=q_floor, size=w, mode='nearest')
        median = percentile_filter(y, percentile=50, size=w, mode='nearest')

    r = y - median
    neg = np.minimum(r, 0.0)
    med = np.median(neg)
    mad = np.median(np.abs(neg - med)) + 1e-12
    zneg = (neg - med) / (1.4826 * mad)
    mask = (zneg < -abs(k_neg)) & (y < floor)

    if protect_qrs:
        if qrs_mask is not None:
            prot = qrs_mask.astype(bool, copy=False)
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
                    lo = max(0, r0 - pad)
                    hi = min(N, r0 + pad + 1)
                    prot[lo:hi] = True
        mask &= (~prot)

    if not np.any(mask):
        return mask

    diff = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)

    if starts.size == 0:
        return mask

    dur = ends - starts
    long_idx = np.where(dur >= min_len)[0]
    out = np.zeros_like(mask)
    for i in long_idx:
        lo = max(0, starts[i] - pad_n)
        hi = min(N, ends[i] + pad_n)
        out[lo:hi] = True

    return out

@profiled()
def fix_downward_steps_mask(
    y, fs,
    pre_s=0.5, post_s=0.5, gap_s=0.08,
    amp_sigma=5.0, amp_abs=None, min_hold_s=0.45,
    refractory_s=0.80, protect_qrs=True,
    r_idx=None, qrs_mask=None,
    smooth_ms=120,
    hop_ms=10
):
    """
    Downward step 검출(고속화)
    """

    y = np.asarray(y, float)
    N = y.size
    if N < 10:
        return np.zeros(N, bool)

    if smooth_ms and smooth_ms > 0:
        m_win = max(3, int(round((smooth_ms/1000.0) * fs)))
        if m_win % 2 == 0: m_win += 1
        y_s = uniform_filter1d(y, size=m_win, mode='nearest')
    else:
        y_s = y

    med = np.median(y_s)
    mad = np.median(np.abs(y_s - med)) + 1e-12
    thr = amp_sigma * 1.4826 * mad
    if amp_abs is not None:
        thr = max(thr, float(amp_abs))

    S = np.concatenate(([0.0], np.cumsum(y_s, dtype=float)))

    def box_mean(start_idx, L):
        a = start_idx
        b = start_idx + L
        return (S[b] - S[a]) / float(L)

    pre   = int(round(pre_s  * fs))
    post  = int(round(post_s * fs))
    gap   = int(round(gap_s  * fs))
    hold  = int(round(min_hold_s * fs))
    refr  = int(round(refractory_s * fs))

    if pre < 1 or post < 1 or hold < 1:
        return np.zeros(N, bool)

    hop = max(1, int(round((hop_ms/1000.0) * fs)))
    i_min = pre
    i_max = N - (gap + post + hold) - 1
    if i_max <= i_min:
        return np.zeros(N, bool)
    centers = np.arange(i_min, i_max + 1, hop, dtype=int)

    pre_starts = centers - pre
    m1 = box_mean(pre_starts, pre)

    cpos = centers + gap
    m2 = box_mean(cpos, post)

    m_hold = box_mean(cpos, hold)

    drop = m1 - m2
    cond_drop = drop > thr
    cond_hold = (m1 - m_hold) >= (0.6 * drop)

    cand = cond_drop & cond_hold
    if not np.any(cand):
        return np.zeros(N, bool)

    prot = np.zeros(N, bool)
    if protect_qrs:
        if qrs_mask is not None:
            prot = qrs_mask.astype(bool, copy=False)
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
    if protect_qrs and prot.any():
        cand_idx = cand_idx[~prot[cand_idx]]

    if cand_idx.size == 0:
        return np.zeros(N, bool)

    mask = np.zeros(N, bool)
    last_end = -10**9

    order = np.argsort(-drop[cand])
    for j in order:
        i = centers[j]
        if not cand[j]:
            continue
        start = cpos[j]
        end   = start + hold
        if start - last_end < refr:
            continue
        if protect_qrs and prot.any():
            seg = prot[start:end]
            if seg.size and seg.mean() > 0.5:
                continue
        mask[start:end] = True
        last_end = end

    return mask

@profiled()
def smooth_corners_mask(
    y, fs, L_ms=140, k_sigma=5.5,
    protect_qrs=True, r_idx=None, qrs_mask=None, smooth_ms=20, use_float32=True
):
    """
    빠른 corner 검출용 마스크 (고속/강건)
    """

    y = np.asarray(y, np.float32 if use_float32 else np.float64)
    N = y.size
    if N < 10:
        return np.zeros(N, bool)

    if smooth_ms > 0:
        win = max(3, int(round((smooth_ms / 1000.0) * fs)))
        y_s = uniform_filter1d(y, size=win, mode='nearest')
    else:
        y_s = y

    d1 = np.gradient(y_s)
    d2 = np.gradient(d1)

    med = np.median(d2)
    mad = np.median(np.abs(d2 - med)) + 1e-12
    z = (d2 - med) / (1.4826 * mad)

    cand = np.abs(z) > float(k_sigma)
    if not np.any(cand):
        return np.zeros(N, bool)

    if protect_qrs:
        if qrs_mask is not None:
            prot = qrs_mask.astype(bool, copy=False)
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
                    lo = max(0, r - pad)
                    hi = min(N, r + pad + 1)
                    prot[lo:hi] = True
        cand &= (~prot)

    idx = np.flatnonzero(cand)
    if idx.size == 0:
        return np.zeros(N, bool)

    L = max(3, int(round((L_ms / 1000.0) * fs)))
    out = np.zeros(N, bool)

    gaps = np.diff(idx, prepend=idx[0])
    starts = np.flatnonzero(gaps > L)
    starts = np.append(starts, len(idx))

    prev = 0
    for s in starts:
        seg_idx = idx[prev:s]
        if seg_idx.size == 0:
            continue
        a = max(0, seg_idx[0] - L)
        b = min(N, seg_idx[-1] + L)
        out[a:b] = True
        prev = s

    return out

@profiled()
def rolling_std_fast(y: np.ndarray, w: int) -> np.ndarray:
    y = y.astype(float); k = np.ones(int(w), float)
    s1 = np.convolve(y, k, mode='same'); s2 = np.convolve(y*y, k, mode='same')
    m = s1 / int(w); v = s2 / w - m*m; v[v < 0] = 0.0
    return np.sqrt(v)

@profiled()
def high_variance_mask(
    y: np.ndarray, win=2000, k_sigma=5.0, pad=125,
    mode: str = "grid", hop_ms: int = 32, block_s: float = 1.0
):
    """
    고분산(HV) 구간 마스크 — 초고속 버전
    """

    x = np.asarray(y, np.float32)
    n = int(x.size)
    if n == 0:
        stats = {"threshold": 0.0, "removed_samples": 0, "kept_samples": 0, "compression_ratio": 1.0}
        return np.zeros(0, dtype=bool), stats

    w = int(max(2, win))
    if w % 2 == 0:
        w += 1
    half = w // 2

    if mode == "full":
        m  = uniform_filter1d(x,   size=w, mode='nearest', origin=0)
        m2 = uniform_filter1d(x*x, size=w, mode='nearest', origin=0)
        v = m2 - m*m
        np.maximum(v, 0.0, out=v)
        rs = np.sqrt(v, dtype=np.float32)

        rs_med = float(np.median(rs))
        rs_mad = float(np.median(np.abs(rs - rs_med)) + 1e-12)
        thr = rs_med + 1.4826 * rs_mad * float(k_sigma)

    elif mode == "block":
        B = max(w, 512)
        nb = (n + B - 1) // B
        rs = np.empty(n, dtype=np.float32)
        for b in range(nb):
            s = b * B
            e = min(n, s + B)
            seg = x[s:e]
            sd = float(seg.std(ddof=0))
            rs[s:e] = sd
        rs_med = float(np.median(rs))
        rs_mad = float(np.median(np.abs(rs - rs_med)) + 1e-12)
        thr = rs_med + 1.4826 * rs_mad * float(k_sigma)

    else:
        hop = max(1, int(round((hop_ms / 1000.0) * 250.0)))
        centers = np.arange(0, n, hop, dtype=int)
        starts = np.clip(centers - half, 0, n - 1)
        ends   = np.clip(centers + half + 1, 0, n)

        S1 = np.concatenate(([0.0], np.cumsum(x,  dtype=np.float64)))
        S2 = np.concatenate(([0.0], np.cumsum(x*x, dtype=np.float64)))
        Ls = (ends - starts).astype(np.int64)

        sum1 = S1[ends] - S1[starts]
        sum2 = S2[ends] - S2[starts]
        m  = sum1 / np.maximum(1, Ls)
        m2 = sum2 / np.maximum(1, Ls)
        v = m2 - m*m
        v[v < 0.0] = 0.0
        rs_grid = np.sqrt(v, dtype=np.float64)

        rs_med = float(np.median(rs_grid))
        rs_mad = float(np.median(np.abs(rs_grid - rs_med)) + 1e-12)
        thr = rs_med + 1.4826 * rs_mad * float(k_sigma)

        idx_full = np.arange(n, dtype=np.float64)
        idx_cent = centers.astype(np.float64)
        rs = np.interp(idx_full, idx_cent, rs_grid).astype(np.float32, copy=False)

    mask = rs > thr

    if pad and pad > 0 and mask.any():
        st = np.ones(int(pad) * 2 + 1, dtype=bool)
        mask = binary_dilation(mask, structure=st)

    kept = int((~mask).sum())
    stats = {
        "threshold": float(thr),
        "removed_samples": int(mask.sum()),
        "kept_samples": kept,
        "compression_ratio": float(kept / n)
    }
    return mask, stats


@profiled()
def _smooth_binary(mask: np.ndarray, fs: float, blend_ms: int = 80) -> np.ndarray:
    L = max(3, int(round(blend_ms/1000.0 * fs)))
    if L % 2 == 0: L += 1
    win = np.hanning(L); win = win / win.sum()
    return np.convolve(mask.astype(float), win, mode='same')
@profiled()
def qrs_aware_wavelet_denoise(y, fs, wavelet='db6', level=None, sigma_scale=2.8, blend_ms=80):
    y = np.asarray(y, float); N = y.size
    try:
        mask = make_qrs_mask(y, fs=fs)
    except Exception:
        mask = np.ones_like(y, dtype=bool)
    alpha = _smooth_binary(mask, fs, blend_ms=blend_ms)
    try:
        if level is None:
            level = min(5, max(2, int(np.log2(fs/8.0))))
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


from scipy.signal import butter, filtfilt

@profiled()
def qrs_aware_lowpass(y, fs, fc=35.0, order=3, blend_ms=100):
    """
    비-QRS 구간(=mask True)에만 저역통과를 강하게 적용하고,
    QRS 근처는 원파형 보존. 짜글거림(고주파)을 제거.
    """
    x = np.asarray(y, float)
    if x.size == 0:
        return x, np.zeros_like(x, float)

    # 비-QRS 블렌딩 계수 (mask=True -> alpha≈1 -> 더 많이 필터)
    try:
        mask = make_qrs_mask(x, fs=fs)   # True=비-QRS, False=QRS/T 근처
    except Exception:
        mask = np.ones_like(x, bool)
    alpha = _smooth_binary(mask, fs, blend_ms=blend_ms)

    # zero-phase low-pass (영위상)
    b, a = butter(order, fc / (fs * 0.5), btype='low')
    y_lp = filtfilt(b, a, x)

    # 비-QRS(α→1)에서는 y_lp, QRS(α→0)에서는 원신호 보존
    y_out = alpha * y_lp + (1.0 - alpha) * x
    return y_out, alpha


@profiled()
def burst_mask(
    y, fs, win_ms=140, k_diff=7.5, k_std=3.5, pad_ms=80,
    protect_qrs=True, r_idx=None, qrs_mask=None, pre_smooth_ms=0, use_float32=True
):
    """
    버스트(급변+분산상승) 마스크 — 고속화 버전
    """

    x = np.asarray(y, np.float32 if use_float32 else np.float64)
    N = x.size
    if N < 10:
        return np.zeros(N, dtype=bool)

    if pre_smooth_ms and pre_smooth_ms > 0:
        sw = max(3, int(round((pre_smooth_ms/1000.0) * fs)))
        if sw % 2 == 0: sw += 1
        x = uniform_filter1d(x, size=sw, mode='nearest')

    dy = np.gradient(x)
    d_med = float(np.median(dy))
    d_mad = float(np.median(np.abs(dy - d_med)) + 1e-12)
    z_diff = (dy - d_med) / (1.4826 * d_mad)

    w = max(3, int(round((win_ms/1000.0) * fs)))
    if w % 2 == 0: w += 1
    m  = uniform_filter1d(x,   size=w, mode='nearest')
    m2 = uniform_filter1d(x*x, size=w, mode='nearest')
    v = m2 - m*m
    np.maximum(v, 0.0, out=v)
    rs = np.sqrt(v, dtype=x.dtype)

    r_med = float(np.median(rs))
    r_mad = float(np.median(np.abs(rs - r_med)) + 1e-12)
    z_std = (rs - r_med) / (1.4826 * r_mad)

    cand = (np.abs(z_diff) > float(k_diff)) & (z_std > float(k_std))
    if not np.any(cand):
        return np.zeros(N, dtype=bool)

    if protect_qrs:
        if qrs_mask is not None:
            prot = qrs_mask.astype(bool, copy=False)
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

    if not np.any(cand):
        return cand

    pad = int(round((pad_ms/1000.0) * fs))
    if pad > 0:
        st = np.ones(pad*2 + 1, dtype=bool)
        cand = binary_dilation(cand, structure=st)

    return cand

# =========================
# Custom X-only stretch zoom ViewBox (Shift+좌클릭 드래그)
# =========================
class XZoomViewBox(pg.ViewBox):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, enableMenu=False, **kwargs)
        self.setMouseEnabled(x=True, y=True)
        self.setLimits(yMin=-1e12, yMax=1e12)

    def mouseDragEvent(self, ev, axis=None):
        if ev.button() == QtCore.Qt.LeftButton and (ev.modifiers() & QtCore.Qt.ShiftModifier):
            ev.accept()
            pos = ev.pos()
            last = ev.lastPos()
            dx = pos.x() - last.x()
            s = np.exp(-dx * 0.005)
            s = float(np.clip(s, 1e-3, 1e3))
            center = self.mapSceneToView(pos)
            self.scaleBy((s, 1.0), center=center)
        else:
            super().mouseDragEvent(ev, axis=axis)

# =========================
# Qt Viewer (체크박스만 남긴 최소 옵션 UI)
# =========================
class ECGViewer(QtWidgets.QWidget):
    def __init__(self, t, y_raw, parent=None):
        super().__init__(parent)
        self.t = t; self.y_raw = y_raw
        self._recompute_timer = None

        root = QtWidgets.QVBoxLayout(self)

        # ====== View Toggles (체크박스 유지) ======
        tg = QtWidgets.QHBoxLayout()
        self.cb_raw   = QtWidgets.QCheckBox("원본 신호");       self.cb_raw.setChecked(True)
        self.cb_corr  = QtWidgets.QCheckBox("가공(보정) 신호"); self.cb_corr.setChecked(True)
        self.cb_mask  = QtWidgets.QCheckBox("마스크 패널");     self.cb_mask.setChecked(True)
        self.cb_base  = QtWidgets.QCheckBox("Baseline 표시");   self.cb_base.setChecked(False)
        for cb in (self.cb_raw, self.cb_corr, self.cb_mask, self.cb_base):
            tg.addWidget(cb)
        tg.addStretch(1)
        root.addLayout(tg)

        # ====== Baseline (체크박스만) ======
        bl = QtWidgets.QHBoxLayout()
        self.cb_qrsaware = QtWidgets.QCheckBox("QRS-aware"); self.cb_qrsaware.setChecked(True)
        self.cb_break_cut = QtWidgets.QCheckBox("Hard cut at breaks"); self.cb_break_cut.setChecked(True)
        self.cb_res_refit = QtWidgets.QCheckBox("Residual refit"); self.cb_res_refit.setChecked(True)
        self.cb_rrcap = QtWidgets.QCheckBox("RR cap"); self.cb_rrcap.setChecked(True)
        for w in [self.cb_qrsaware, self.cb_break_cut, self.cb_res_refit, self.cb_rrcap]:
            bl.addWidget(w)
        bl.addStretch(1)
        root.addLayout(bl)

        # ====== Mask toggles (체크박스만) ======
        row2 = QtWidgets.QHBoxLayout()
        self.cb_sag    = QtWidgets.QCheckBox("Sag");     self.cb_sag.setChecked(True)
        self.cb_step   = QtWidgets.QCheckBox("Step");    self.cb_step.setChecked(True)
        self.cb_corner = QtWidgets.QCheckBox("Corner");  self.cb_corner.setChecked(True)
        self.cb_burst  = QtWidgets.QCheckBox("Burst");   self.cb_burst.setChecked(True)
        self.cb_wave   = QtWidgets.QCheckBox("Wavelet"); self.cb_wave.setChecked(False)
        for cb in [self.cb_sag, self.cb_step, self.cb_corner, self.cb_burst, self.cb_wave]:
            row2.addWidget(cb)
        row2.addStretch(1)
        root.addLayout(row2)

        # ====== Plots ======
        self.win_plot = pg.GraphicsLayoutWidget(); root.addWidget(self.win_plot)

        self.plot = self.win_plot.addPlot(row=0, col=0, viewBox=XZoomViewBox())
        self.plot.getViewBox().setMouseEnabled(x=True, y=True)
        self.plot.setLabel('bottom','Time (s)'); self.plot.setLabel('left','Amplitude')
        self.plot.showGrid(x=True,y=True,alpha=0.3)

        self.overview = self.win_plot.addPlot(row=1, col=0); self.overview.setMaximumHeight(150); self.overview.showGrid(x=True,y=True,alpha=0.2)
        self.region = pg.LinearRegionItem(); self.region.setZValue(10); self.overview.addItem(self.region); self.region.sigRegionChanged.connect(self.update_region)

        pen_raw  = pg.mkPen(color=(150, 150, 150), width=1)
        pen_corr = pg.mkPen(color=(255, 215, 0),   width=1.6)  # Yellow (Gold)
        pen_base = pg.mkPen(color=(0, 200, 255),   width=1, style=QtCore.Qt.DashLine)

        self.curve_raw  = self.plot.plot([], [], pen=pen_raw)
        self.curve_corr = self.plot.plot([], [], pen=pen_corr)
        self.curve_base = self.plot.plot([], [], pen=pen_base); self.curve_base.setVisible(False)
        self.curve_corr.setZValue(5); self.curve_raw.setZValue(3); self.curve_base.setZValue(2)

        self.ov_curve = self.overview.plot([], [], pen=pg.mkPen(width=1))

        self.mask_plot = self.win_plot.addPlot(row=2, col=0); self.mask_plot.setMaximumHeight(130)
        self.mask_plot.setLabel('left','Masks'); self.mask_plot.setLabel('bottom','Time (s)')
        self.mask_plot.showGrid(x=True,y=True,alpha=0.2)
        self.hv_curve    = self.mask_plot.plot([], [], pen=pg.mkPen(width=1))
        self.sag_curve   = self.mask_plot.plot([], [], pen=pg.mkPen(width=1, style=QtCore.Qt.DotLine))
        self.step_curve  = self.mask_plot.plot([], [], pen=pg.mkPen(width=1, style=QtCore.Qt.DashLine))
        self.corner_curve= self.mask_plot.plot([], [], pen=pg.mkPen(width=1, style=QtCore.Qt.SolidLine))
        self.burst_curve = self.mask_plot.plot([], [], pen=pg.mkPen(width=1, style=QtCore.Qt.DashDotLine))
        self.wave_curve  = self.mask_plot.plot([], [], pen=pg.mkPen(width=1, style=QtCore.Qt.DashDotDotLine))
        self.resrefit_curve = self.mask_plot.plot([], [], pen=pg.mkPen(width=1, style=QtCore.Qt.DashLine))

        # 이벤트 연결 (체크박스만)
        def connect_toggle(w, slot):
            if isinstance(w, QtWidgets.QCheckBox):
                w.toggled.connect(slot)

        for w in [self.cb_qrsaware, self.cb_break_cut, self.cb_res_refit, self.cb_rrcap,
                  self.cb_sag, self.cb_step, self.cb_corner, self.cb_burst, self.cb_wave]:
            connect_toggle(w, self.schedule_recompute)

        for cb in (self.cb_raw, self.cb_corr, self.cb_mask, self.cb_base):
            cb.toggled.connect(self.update_visibility)

        self.set_data(t, y_raw)

        def dblclick(ev):
            if ev.double():
                self.plot.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        self.plot.scene().sigMouseClicked.connect(dblclick)

    @profiled()
    def schedule_recompute(self):
        if self._recompute_timer is None:
            self._recompute_timer = QtCore.QTimer(self)
            self._recompute_timer.setSingleShot(True)
            self._recompute_timer.timeout.connect(self.recompute)
        self._recompute_timer.start(600)

    @profiled()
    def set_data(self, t, y):
        y_centered = np.asarray(y, float)
        if y_centered.size > 0:
            y_centered = y_centered - float(np.nanmean(y_centered))

        self.t = np.asarray(t, float)
        self.y_raw = y_centered

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

    @profiled()
    def recompute(self):
        D = DEFAULTS  # shorthand

        # 1) Baseline — Hybrid BL++ (+ RR cap)
        y_src = self.y_raw.copy()
        y_corr, base = baseline_hybrid_plus_adaptive(
            y_src, FS,
            per_win_s=D["PER_WIN_S"],
            per_q=D["PER_Q"],
            asls_lam=D["ASLS_LAM"],
            asls_p=D["ASLS_P"],
            asls_decim=D["ASLS_DECIM"],
            qrs_aware=self.cb_qrsaware.isChecked(),
            verylow_fc=D["LPF_FC"],
            clamp_win_s=6.0,
            vol_win_s=D["VOL_WIN"],
            vol_gain=D["VOL_GAIN"],
            lam_floor_ratio=D["LAM_FLOOR_PERCENT"]/100.0,
            hard_cut=self.cb_break_cut.isChecked(),
            break_pad_s=D["BREAK_PAD_S"],
            rr_cap_enable=self.cb_rrcap.isChecked(),
            rr_eps_up=D["RR_EPS_UP"],
            rr_eps_dn=D["RR_EPS_DN"],
            rr_t0_ms=D["RR_T0_MS"],
            rr_t1_ms=D["RR_T1_MS"],
        )

        # 1.5) Residual selective refit
        resrefit_mask = np.zeros_like(y_corr, dtype=bool)
        if self.cb_res_refit.isChecked():
            y_corr2, base2, resrefit_mask = selective_residual_refit(
                y_src, base, FS,
                k_sigma=D["RES_K"], win_s=D["RES_WIN_S"], pad_s=D["RES_PAD_S"],
                asls_lam=1e5, asls_p=0.02, asls_decim=8
            )
            y_corr, base = y_corr2, base2




        # === No AGC / No Glitch ===
        y_corr_eq = y_corr

        y_flat, quiet_mask = wvg_flatten(
            y_corr_eq, FS, win_s=0.45, q_lo=25, q_hi=75,
            spread_thr=8.0, std_thr=6.0, blend_s=0.20
        )
        y_corr_eq = y_flat

        y_burst, burst_mask_bin, gain = burst_gate_dampen(
            y_corr_eq, FS, win_ms=140, k_diff=6.0, k_std=3.0, pad_ms=140,
            limit_ratio=0.6, alpha=1.2, atk_ms=60, rel_ms=300
        )
        y_corr_eq = y_burst

        if burst_mask_bin.any():
            y_corr_eq = replace_with_bandlimited(y_corr_eq, FS, burst_mask_bin, fc=12.0)

        y_corr_eq = bilateral_filter_1d(y_corr_eq, sigma_s=6, sigma_r=0.2 * np.std(y_corr_eq))

        # --- 비-QRS 구간만 저역통과하여 짜글거림 제거 ---
        y_corr_eq, _alpha_lp = qrs_aware_lowpass(y_corr_eq, FS, fc=32.0, order=3, blend_ms=100)

        import scipy.signal as sps

        baseline = sps.medfilt(y_corr_eq, kernel_size=251)  # 1초 정도 윈도우 (fs=250Hz)
        y_corr_eq = y_corr_eq - baseline

        # --- (3) 미세 진동 평활화 ---
        # 이동평균 또는 Savitzky-Golay smoothing
        # y_corr_eq = sps.savgol_filter(y_corr_eq, window_length=9, polyorder=2)
        # 원 신호


        # Masks (체크박스 on/off + 고정 파라미터)
        sag_mask = suppress_negative_sag(
            y_corr_eq, FS,
            win_sec=D["SAG_WIN_S"], q_floor=D["SAG_Q"], k_neg=D["SAG_K"],
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

        b_mask = np.zeros_like(y_corr_eq, bool)
        if self.cb_burst.isChecked():
            b_mask = burst_mask(
                y_corr_eq, FS, win_ms=D["BURST_WIN_MS"], k_diff=D["BURST_KD"], k_std=D["BURST_KS"],
                pad_ms=D["BURST_PAD_MS"], protect_qrs=True
            )

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

        # X 범위 및 자동 Y 스케일
        lo, hi = self.region.getRegion()
        self.plot.setXRange(lo, hi, padding=0)

        vis_idx = (self.t >= lo) & (self.t <= hi)
        if np.any(vis_idx):
            y_sub = self.y_raw[vis_idx]
            ymin, ymax = float(np.min(y_sub)), float(np.max(y_sub))
            if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
                margin = 0.1 * (ymax - ymin) if (ymax - ymin) > 0 else 1.0
                self.plot.setYRange(ymin - margin, ymax + margin, padding=0)

        rows = profiler_report(topn=25)

    def update_region(self):
        lo, hi = self.region.getRegion()
        self.plot.setXRange(lo, hi, padding=0)

        vis_idx = (self.t >= lo) & (self.t <= hi)
        if np.any(vis_idx):
            y_sub = self.y_raw[vis_idx]
            y_min, y_max = np.min(y_sub), np.max(y_sub)
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
    ecg = decimate_if_needed(ecg_raw, DECIM)
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
