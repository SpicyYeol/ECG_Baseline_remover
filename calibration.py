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
import numpy as np
import pyqtgraph as pg
import pywt
from PyQt5 import QtWidgets, QtCore
from scipy.linalg import solveh_banded
from scipy.ndimage import binary_dilation
from scipy.ndimage import median_filter as _mf
from scipy.ndimage import percentile_filter, median_filter
from scipy.ndimage import uniform_filter1d
from scipy.signal import lfilter, lfilter_zi, filtfilt, decimate
from scipy.signal import savgol_filter

_PROF = defaultdict(lambda: {"calls": 0, "total": 0.0})

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
        print(f"[Profiler] {k:>20} | {c:5d} | {tot*1000:8.2f} | {avg*1000:7.2f}")
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

    - scipy.signal.lfilter / filtfilt(C 구현) 사용 → 파이썬 루프 제거
    - α 지수형식 사용(연속시간 RC 정확 이산화) → 작은 fc에서도 수치안정
    - zero_phase=True이면 filtfilt(영위상, 2차 통과)로 지연 제거
    """
    x = np.asarray(sig, np.float32 if use_float32 else np.float64)
    N = x.size
    if N == 0 or fc <= 0.0:
        return x.astype(np.float64, copy=False)
    if fs <= 0.0:
        raise ValueError("fs must be > 0")

    # 안정한 계수 (α in (0,1))
    alpha = float(np.exp(-2.0 * np.pi * float(fc) / float(fs)))
    b0 = 1.0 - alpha
    a1 = alpha

    # SciPy 경로 (가장 빠름)
    try:

        if zero_phase:
            # 영위상: 1차 필터를 전후방 통과(유효 차수 2)
            b = [b0]
            a = [1.0, -a1]
            # padlen은 신호 길이에 맞게 자동, 짧은 신호 보호
            padlen = min(3 * (max(len(a), len(b)) - 1), max(0, N - 1))
            y = filtfilt(b, a, x, padlen=padlen) if padlen > 0 else filtfilt(b, a, x)
            return y.astype(np.float64, copy=False)

        # causal: 초기조건을 x[0]에 맞춰 세팅(원래 구현 y[0]=x[0]에 최대한 근접)
        b = [b0]
        a = [1.0, -a1]
        zi = lfilter_zi(b, a) * x[0]   # step 입력 x[0]에 대한 정상상태 IC
        y, _ = lfilter(b, a, x, zi=zi)
        return y.astype(np.float64, copy=False)

    except Exception:
        # SciPy 없음 → Numpy fallback (여전히 빠르진 않지만 안전)
        y = np.empty_like(x, dtype=x.dtype)
        y[0] = x[0]
        # y[n] = a1*y[n-1] + b0*x[n]
        # (가능하면 여기도 numba jit로 감싸면 10~20배↑)
        for i in range(1, N):
            y[i] = a1 * y[i-1] + b0 * x[i]
        return y.astype(np.float64, copy=False)

from scipy.signal import butter, filtfilt

def replace_with_bandlimited(y, fs, mask, fc=12.0):
    """마스크 구간만 저역통과 재구성한 신호로 치환 후 페이드."""
    b,a = butter(3, fc/(fs/2.0), btype='low')
    y_lp = filtfilt(b, a, y)
    # 경계 페이드
    win = int(0.10*fs)  # 100 ms
    w = np.ones_like(y, float)
    # 앞/뒤 경계에서 선형 페이드
    d = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.flatnonzero(d==1); ends = np.flatnonzero(d==-1)
    for s,e in zip(starts, ends):
        a0 = max(0, s-win); b0 = min(len(y), s+win)
        a1 = max(0, e-win); b1 = min(len(y), e+win)
        # 들어갈 때 페이드
        if b0-a0 > 1: w[a0:b0] *= np.linspace(1, 0, b0-a0)
        # 나올 때 페이드
        if b1-a1 > 1: w[a1:b1] *= np.linspace(0, 1, b1-a1)
        w[s:e] = 0.0
    return y*w + y_lp*(1.0-w)


def burst_gate_dampen(y, fs,
                      win_ms=140, k_diff=6.0, k_std=3.0, pad_ms=120,
                      limit_ratio=0.6,     # rs가 임계치의 1/limit_ratio 배수까지 내려가도록
                      alpha=1.2,           # 감쇠 곡률
                      atk_ms=60, rel_ms=300,
                      protect_qrs=True):
    """
    급변(z_diff) + 분산(z_std) 동시 초과 구간만 가변 이득 g(t)로 감쇠.
    g(t) = min(1, (thr / rs)^alpha) 를 attack/release로 평활해 링잉/펌핑 방지.
    """
    x = np.asarray(y, float)
    N = x.size
    if N < 10: return x, np.zeros(N, bool), np.ones(N, float)

    # --- 1) 급변 + 분산 계산(= burst_mask와 동일 메커니즘)
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

    # pad 확장
    pad = int(round((pad_ms/1000.0)*fs))
    if pad > 0 and cand.any():
        st = np.ones(pad*2+1, dtype=bool)
        from scipy.ndimage import binary_dilation
        cand = binary_dilation(cand, structure=st)

    # --- 2) 이득 곡선 g_raw
    # rs가 커질수록 더 많이 줄임. limit_ratio로 과도 감쇠 방지.
    eps = 1e-12
    g_raw = np.ones_like(x)
    idx = rs > thr_std
    g_raw[idx] = np.minimum(1.0, (thr_std / (rs[idx] + eps))**float(alpha))
    g_raw = np.maximum(g_raw, float(limit_ratio))   # 하한

    # cand 영역만 적용(나머지는 g=1)
    g_target = np.where(cand, g_raw, 1.0)

    # --- 3) Attack/Release 평활 (one-pole)
    def one_pole(env, atk, rel):
        out = np.empty_like(env)
        a_atk = np.exp(-1.0/max(1, int(atk*fs)))   # 빠르게 내려가고
        a_rel = np.exp(-1.0/max(1, int(rel*fs)))   # 천천히 올라온다
        y0 = 1.0; out[0] = y0
        for n in range(1, env.size):
            a = a_atk if env[n] < out[n-1] else a_rel
            out[n] = a*out[n-1] + (1-a)*env[n]
        return out

    g = one_pole(g_target, atk_ms/1000.0, rel_ms/1000.0)

    # --- 4) 적용
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
    # hard zero anchor
    return y - np.median(y)

import numpy as np
from scipy.ndimage import uniform_filter1d, percentile_filter

def wvg_flatten(y, fs,
                win_s=0.45,          # 고정 창(초)
                q_lo=25, q_hi=75,    # 상·하 분위선
                spread_thr=8.0,      # 분위 폭 임계(µV 등 신호 단위)
                std_thr=6.0,         # 표준편차 임계
                blend_s=0.20):       # 가장자리 부드럽게

    x = np.asarray(y, float)
    if x.size == 0: return x, np.zeros_like(x, bool)

    w = max(3, int(round(win_s * fs)))
    if w % 2 == 0: w += 1

    # 1) 국소 상·하 분위와 중앙값
    lo = percentile_filter(x, percentile=q_lo, size=w, mode='nearest')
    hi = percentile_filter(x, percentile=q_hi, size=w, mode='nearest')
    med = percentile_filter(x, percentile=50,   size=w, mode='nearest')

    # 2) 국소 표준편차(1-pass)
    m  = uniform_filter1d(x,   size=w, mode='nearest')
    m2 = uniform_filter1d(x*x, size=w, mode='nearest')
    v  = np.maximum(m2 - m*m, 0.0)
    sd = np.sqrt(v)

    # 3) “조용한 구간” 마스크: 상·하 분산이 모두 작을 때만 평탄화
    spread = hi - lo
    quiet  = (spread <= float(spread_thr)) & (sd <= float(std_thr))

    # 4) 경계 블렌딩(마스크 가장자리 링잉 방지)
    if blend_s and quiet.any():
        L = max(3, int(round(blend_s * fs)))
        if L % 2 == 0: L += 1
        win = np.hanning(L); win /= win.sum()
        alpha = np.convolve(quiet.astype(float), win, mode='same')
    else:
        alpha = quiet.astype(float)

    # 5) 조용한 구간만 로컬 중앙값으로 당김
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
    ASLS(비대칭 가중 최소제곱) - 고속화:
      * SPD 밴드행렬 → scipy.linalg.solveh_banded(Cholesky)
      * 오프대각 캐시, 주대각만 반복마다 갱신
      * 세그 길이 기반 적응 반복 + 조기 종료
      * (옵션) float32 경로
    """

    y = np.asarray(y, np.float32 if use_float32 else np.float64)
    N = y.size
    if N < 3:
        return np.zeros_like(y)

    if decim_for_baseline > 1:
        q = int(decim_for_baseline)
        n = (N // q) * q
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

    # --- 상부밴드(ab_u) 캐시 구성 (solveh_banded는 상/하 밴드 중 하나 선택)
    #   대각수=3(±0, ±1, ±2) → 상부밴드 shape=(3, N)
    #   ab_u[0]=+2대각(λ*1), ab_u[1]=+1대각(λ*-4), ab_u[2]=주대각(λ*6 + wg)
    ab_u = np.zeros((3, N), dtype=y.dtype)
    ab_u[0, 2:] = lam * 1.0
    ab_u[1, 1:] = lam * (-4.0)
    ab_u[2, :]  = lam * 6.0  # wg는 반복마다 더함

    # 적응 반복 수 (세그먼트가 짧으면 반복 줄이기)
    # 250 Hz 기준 0.5s만 되어도 충분히 수렴하는 편
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

        # 주대각 갱신 (in-place)
        ab_u[2, :] = lam * 6.0 + wg

        b = wg * y
        # SPD 해법
        z = solveh_banded(ab_u, b, lower=False, overwrite_ab=False,
                          overwrite_b=True, check_finite=False)

        # 가중치 갱신
        w = p * (y > z) + (1.0 - p) * (y < z)

        # 조기 종료: 목적함수 근사 수렴
        if it >= 1:
            r = (y - z)
            data_term = float(np.dot((wg * r).astype(np.float64), r.astype(np.float64)))
            # 2차차분 근사
            d2 = np.diff(z.astype(np.float64), n=2, prepend=float(z[0]), append=float(z[-1]))
            reg_term = float(lam) * float(np.dot(d2, d2))
            obj = data_term + reg_term
            if last_obj is not None and abs(last_obj - obj) <= 1e-5 * max(1.0, obj):
                break
            last_obj = obj

    return z.astype(np.float64, copy=False)  # 호출자 일관성(상위는 float64)


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
    qrs_aware=True, verylow_fc=0.03, clamp_win_s=6.0,
    vol_win_s=0.6, vol_gain=6.0, lam_floor_ratio=0.03,
    hard_cut=True, break_pad_s=0.30,
    # --- 새로 추가된 최적화 옵션 ---
    r_idx=None,            # (옵션) 미리 계산된 R-피크 인덱스 전달 시 neurokit2 호출 생략
    qrs_mask=None,         # (옵션) 미리 계산된 QRS 보호 마스크 전달
    lam_bins=6,            # λ 지역화 양자화 bin 수 (로그 스케일)
    min_seg_s=0.50,        # 너무 짧은 세그먼트는 병합
    max_seg_s=6.0          # 너무 긴 세그먼트는 하위로 분절(안정성/메모리 관점)
):
    """
    Hybrid BL++ (adaptive λ, variance-aware, hard-cut, local refit) — Optimized
    - 고정 스텝 분할 제거 → λ 양자화+런 기반 분할로 baseline_asls_masked 호출 횟수 최소화
    - 이동통계 컨볼루션 커널 공유
    - QRS 마스크/피크 외부 주입 가능
    """


    x = np.asarray(y, float)
    N = x.size
    if N < 8:
        return np.zeros_like(x), np.zeros_like(x)

    # ---------- 유틸 ----------
    def _odd(n):
        n = int(max(3, n))
        return n + (n % 2 == 0)

    # 이동창 컨볼루션 커널(공유)
    def _mov_stats(xx, win):
        # 평균/표준편차 빠른 계산 (convolution)
        k = np.ones(win, float)
        s1 = np.convolve(xx, k, mode='same')
        s2 = np.convolve(xx*xx, k, mode='same')
        m = s1 / win
        v = s2 / win - m*m
        v[v < 0] = 0.0
        return m, np.sqrt(v)

    # λ 런-세그먼트 생성: 로그-스케일 양자화 후 연속 구간 추출
    def _segments_from_lambda(lam_arr, fs_, brks):
        # 로그-스케일 양자화
        lam_eps = 1e-12
        L = np.log(lam_arr + lam_eps)
        q_lo, q_hi = np.quantile(L, [0.05, 0.95])
        if q_hi <= q_lo:  # 이상치 방지
            q_hi = q_lo + 1e-6
        bins = np.linspace(q_lo, q_hi, int(max(2, lam_bins)))
        idx = np.clip(np.digitize(L, bins, right=False), 0, len(bins))

        # 변화점(brks)을 경계로 강제 분할
        cuts = [0] + [int(b) for b in brks] + [N]
        # 각 구간 안에서 run-length
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

        # 너무 짧은 세그먼트 병합
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
                # 앞 세그가 너무 짧으면 강제로 병합
                if (me - ms) < min_len and kbin != mk:
                    merged[-1][1] = e
                else:
                    merged.append([s, e, kbin])

        # 너무 긴 세그먼트는 분절 (메모리/수렴 안정)
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
        # 오름차순/겹침 제거
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
                    # T-wave 보호
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

    # ---------- 2) 위치별 λ 설계 (gradient + volatility) ----------
    grad = np.gradient(x)
    g_ref = np.quantile(np.abs(grad), 0.95) + 1e-6
    z_grad = np.clip(np.abs(grad) / g_ref, 0.0, 6.0)
    lam_grad = asls_lam / (1.0 + 8.0 * z_grad)

    vw = _odd(int(round(vol_win_s * fs)))
    _, rs = _mov_stats(x, vw)  # 공유 컨볼루션
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

    # ---------- 3) 세그먼트 피팅: λ-런 기반 최소 호출 ----------
    #   - brks 포함 경계로 분할
    #   - 각 세그에 대해 lam_i = median(lam_local[seg])
    b1 = np.zeros_like(x)
    segs = _segments_from_lambda(lam_local, fs, brks if hard_cut else [])
    if not segs:
        segs = [(0, N)]

    for s, e in segs:
        # 너무 짧은 세그는 건너뜀
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

    # 4) very-low stabilization + smooth offset clamp
    b = b0 + b1
    # replace one-pole LP with zero-phase high-pass (0.3 Hz) to kill drift
    b_slow = highpass_zero_drift(b, fs, fc=0.55)

    # residual clamp: Savitzky–Golay instead of median to avoid ripples
    clamp_w = _odd(int(round(clamp_win_s * fs)))
    # ensure window is large enough and odd
    sg_win = max(_odd(int(fs * 1.5)), clamp_w)
    resid = x - b_slow
    off = savgol_filter(resid, window_length=sg_win, polyorder=2, mode='interp')

    off = highpass_zero_drift(off, fs, fc=0.15)

    b_final = b_slow + off
    b_final += rr_isoelectric_clamp(x - b_final, fs)
    y_corr = x - b_final
    return y_corr, b_final
def soft_agc_qrs_aware(
    y, fs,
    win_s=0.8,           # 국소 스케일 창(초) — 0.6~1.2 추천
    method="mad",        # "mad" | "rms"
    target_q=70,         # s_ref = 국소 스케일의 q 분위수(50~80 추천)
    alpha=1.0,           # 이득 곡률(0.7~1.2)
    gmin=0.35, gmax=1.0, # 과도 증폭 방지(커진 구간만 누르므로 1.0 상한 권장)
    smooth_s=0.6,        # gain 저역 스무딩(초)
    qrs_soft=0.35        # QRS 부근에서 AGC 세기를 줄이는 혼합가중(0.2~0.5)
):
    """
    QRS-aware soft AGC: 급격히 커진 구간을 부드럽게 눌러 균질화.
    - 곱셈형 스케일만 적용 → 모폴로지 보존
    - gain은 저주파로만 변화 → 구간경계 아티팩트 최소화
    - QRS는 덜 눌러 스파이크 왜곡 방지
    """
    x = np.asarray(y, float)
    N = x.size
    if N == 0:
        return x

    # 1) QRS 보호 가중(0~1)
    try:
        qmask = make_qrs_mask(x, fs=fs)  # True=비QRS, False=QRS/T 보호구간
        # QRS 안에서는 가중을 낮춰(=AGC 약화): alpha_qrs ∈ [qrs_soft, 1]
        alpha_qrs = qrs_soft + (qmask.astype(float)) * (1.0 - qrs_soft)
    except Exception:
        alpha_qrs = np.ones_like(x)

    # 2) 국소 스케일 s(t)
    win = max(3, int(round(win_s * fs)))
    if win % 2 == 0: win += 1
    if method == "rms":
        m  = uniform_filter1d(x,   size=win, mode='nearest')
        m2 = uniform_filter1d(x*x, size=win, mode='nearest')
        v  = np.maximum(m2 - m*m, 0.0)
        s  = np.sqrt(v + 1e-12)
    else:  # robust MAD of residual about local median
        med = percentile_filter(x, percentile=50, size=win, mode='nearest')
        r   = x - med
        # L1-근사로 빠르게
        m1  = uniform_filter1d(np.abs(r), size=win, mode='nearest')
        s   = 1.4826 * m1 + 1e-12

    # 3) 기준 스케일 & 이득
    s_ref = float(np.percentile(s, target_q))
    g = (s_ref / (s + 1e-12)) ** float(alpha)
    g = np.clip(g, float(gmin), float(gmax))

    # 4) gain 스무딩
    smw = max(3, int(round(smooth_s * fs)))
    if smw % 2 == 0: smw += 1
    g = uniform_filter1d(g, size=smw, mode='nearest')

    # 5) QRS-aware 혼합 적용
    #   QRS에서는 y 그대로(가중 alpha_qrs), 비QRS에서는 g 적용(가중 1-alpha_qrs)
    #   y_eq = y * [ alpha_qrs*1 + (1-alpha_qrs)*g ]
    w = alpha_qrs
    y_eq = x * (w + (1.0 - w) * g)
    return y_eq


def rr_isoelectric_clamp(y, fs, r_idx=None, t0_ms=80, t1_ms=300):
    """RR 사이 등전위(PR/T) 구간 median을 스플라인으로 이어 baseline으로 사용."""
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
        if b - a < max(5, int(0.04 * fs)):   # 너무 짧으면 skip
            continue
        m = float(np.median(x[a:b]))
        pts_x.append((a + b) // 2); pts_y.append(m)
    if len(pts_x) < 2:
        return np.zeros_like(x)

    # 부드러운 선형 보간(충분히 효과적). 필요 시 UnivariateSpline로 교체 가능
    xs = np.arange(N, dtype=float)
    baseline_rr = np.interp(xs, np.array(pts_x, float), np.array(pts_y, float))
    # zero-mean 보정(절대치가 아니라 기울기만 제거하고 싶을 때)
    baseline_rr -= np.median(baseline_rr)
    return baseline_rr

# =========================
# Residual-based selective refit
# =========================
@profiled()
def selective_residual_refit(
    y_src, base_in, fs,
    k_sigma=3.2,               # 잔차 z-임계
    win_s=0.5,                 # 로컬 컨텍스트 윈도
    pad_s=0.20,                # 후보 pad
    method='approx',           # 'approx' | 'percentile' | 'asls'
    per_q=20,                  # percentile 목표
    asls_lam=5e4, asls_p=0.01, asls_decim=6,
    # ==== 추가 가속 파라미터 ====
    grid_ms=32,                # 후보 스코어링 다운샘플 간격
    topk_per_5s=1,             # 5초당 상위 K개만 리핏 (최소 3개 보장)
    min_gap_s=0.20,            # 세그 사이 병합 최소 간격
    max_asls_blk_s=3.0,        # ASLS 블록 크기(초)
    parallel_workers=0,        # 0=단일, >0=스레드
    use_float32=True
):
    """
    선택적 잔차 리핏 — 고속/저비용 버전
    - 후보: robust z > k, binary_dilation으로 pad 확장
    - Top-K만 처리(길이/강도 스코어)
    - method='approx': 전구간 이동통계로 floor 사전계산 → 세그마다 슬라이스만 적용
    - method='percentile': 격자 분위수(np.partition) 계산 후 선형보간(전구간 1회)
    - method='asls': 세그먼트 내부 OLA + 병렬
    """


    x  = np.asarray(y_src, np.float32 if use_float32 else np.float64)
    bb = np.asarray(base_in, np.float32 if use_float32 else np.float64).copy()
    N = x.size
    if N < 10:
        return (x - bb).astype(np.float64, copy=False), bb.astype(np.float64, copy=False), np.zeros(N, bool)

    # -------- 0) 잔차/후보 마스크 --------
    resid = x - bb
    med = float(np.median(resid))
    mad = float(np.median(np.abs(resid - med)) + 1e-12)
    z = np.abs((resid - med) / (1.4826 * mad))
    cand = z > float(k_sigma)

    # pad 확장
    pad_n = int(round(pad_s * fs))
    if pad_n > 0 and cand.any():
        st = np.ones(pad_n * 2 + 1, dtype=bool)
        cand = binary_dilation(cand, structure=st)

    if not cand.any():
        return (x - bb).astype(np.float64, copy=False), bb.astype(np.float64, copy=False), np.zeros(N, bool)

    # 연속 구간 + 근접 병합
    diff = np.diff(cand.astype(np.int8), prepend=0, append=0)
    starts = np.flatnonzero(diff == 1)
    ends   = np.flatnonzero(diff == -1)

    # 인접 구간 병합 (min_gap_s 이하 간격은 하나로)
    min_gap = int(round(min_gap_s * fs))
    merged_s = []
    merged_e = []
    if starts.size:
        s = int(starts[0]); e = int(ends[0])
        for i in range(1, len(starts)):
            if int(starts[i]) - e <= min_gap:
                e = int(ends[i])
            else:
                merged_s.append(s); merged_e.append(e)
                s = int(starts[i]); e = int(ends[i])
        merged_s.append(s); merged_e.append(e)
    starts = np.asarray(merged_s, int); ends = np.asarray(merged_e, int)

    # 너무 짧은 구간 제거(원 기준 0.2s)
    min_len = max(5, int(0.20 * fs))
    keep = np.where((ends - starts) >= min_len)[0]
    if keep.size == 0:
        return (x - bb).astype(np.float64, copy=False), bb.astype(np.float64, copy=False), np.zeros(N, bool)
    starts = starts[keep]; ends = ends[keep]

    # -------- 1) Top-K 세그먼트 선별 (다운샘플 그리드 스코어) --------
    hop = max(1, int(round((grid_ms / 1000.0) * fs)))
    # 세그 점수: 평균 z * sqrt(L) (길이 가중)
    scores = []
    for a, b in zip(starts, ends):
        zz = z[a:b:hop]
        L  = max(1, b - a)
        scores.append(float(zz.mean() if zz.size else 0.0) * np.sqrt(L))
    scores = np.asarray(scores)

    T = N / float(fs)
    K = max(3, int(np.ceil(T / 5.0) * max(1, int(topk_per_5s))))
    if scores.size > K:
        ord_idx = np.argsort(scores)[::-1][:K]
        starts = starts[ord_idx]; ends = ends[ord_idx]

    refit_mask = np.zeros(N, dtype=bool)

    # 공통 파라
    wloc = max(3, int(round(win_s * fs)))
    if wloc % 2 == 0: wloc += 1

    def _taper(L):
        if L <= 8: return np.ones(L, float)
        tlen = min(L // 3, max(3, int(0.06 * fs)))
        if tlen <= 0: return np.ones(L, float)
        w = np.hanning(2 * tlen)
        t = np.ones(L, float); t[:tlen] = w[:tlen]; t[-tlen:] = w[-tlen:]
        return t

    # -------- 2) 보정량 전구간 사전계산 (approx / percentile) --------
    loc_full = None
    if method == 'approx':
        # mean - c*std (q 하위 분위수 근사). q=20 → c≈0.6
        c = max(0.0, (50.0 - float(per_q))) * 0.02
        m  = uniform_filter1d(resid, size=wloc, mode='nearest')
        m2 = uniform_filter1d(resid * resid, size=wloc, mode='nearest')
        v  = m2 - m * m; np.maximum(v, 0.0, out=v)
        std = np.sqrt(v, dtype=resid.dtype)
        loc_full = (m - c * std).astype(np.float32, copy=False)

    elif method == 'percentile':
        # 격자에서만 np.partition로 정확 분위수 계산 → 선형보간
        qs = float(per_q) / 100.0
        k_idx = int(qs * (wloc - 1))
        half = wloc // 2
        centers = np.arange(0, N, hop, dtype=int)
        starts_c = np.clip(centers - half, 0, N - 1)
        ends_c   = np.clip(centers + half + 1, 0, N)
        p_samps = np.empty_like(centers, dtype=np.float32)
        for i, (a, b) in enumerate(zip(starts_c, ends_c)):
            win = resid[a:b]
            if win.size:
                kth = np.partition(win, k_idx)
                p_samps[i] = kth[k_idx]
            else:
                p_samps[i] = resid[centers[i]]
        loc_full = np.interp(np.arange(N, dtype=float), centers.astype(float), p_samps).astype(np.float32)

    # -------- 3) 세그먼트 적용 --------
    if method in ('approx', 'percentile'):
        for a, b in zip(starts, ends):
            L = int(b - a)
            loc = loc_full[a:b].astype(np.float64, copy=False)
            loc *= _taper(L)
            bb[a:b] += loc.astype(bb.dtype, copy=False)
            refit_mask[a:b] = True

    else:  # 'asls'
        max_blk = int(round(max_asls_blk_s * fs))

        def fit_one(a, b):

            L = int(b - a)
            if L <= 0:
                return None
            out = np.zeros(L, dtype=np.float32)
            i = 0
            while i < L:
                j = min(L, i + max_blk)
                seg_ctx = (x[a+i:a+j] - bb[a+i:a+j]).astype(np.float64, copy=False)
                b_loc = baseline_asls_masked(
                    seg_ctx, lam=float(asls_lam), p=float(asls_p),
                    niter=8, mask=None, decim_for_baseline=max(1, int(asls_decim)),
                ).astype(np.float32, copy=False)
                n = b_loc.size
                if i > 0:
                    ov = max(3, int(0.10 * fs))
                    ov = min(ov, n // 2)
                    if ov > 0:
                        w = np.hanning(2 * ov)
                        out[i:i+ov]   = out[i:i+ov]   * w[:ov] + b_loc[:ov] * w[ov:]
                        out[i+ov:i+n] = b_loc[ov:]
                    else:
                        out[i:i+n] = b_loc
                else:
                    out[i:i+n] = b_loc
                i += max_blk
            # 부드럽게
            if L > 8:
                out = _mf(out, size=max(3, int(0.10 * fs)), mode='nearest')
                out *= _taper(L)
            return (a, b, out)

        if parallel_workers and len(starts) > 1:

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

    y_corr2 = (x - bb).astype(np.float64, copy=False)
    return y_corr2, bb.astype(np.float64, copy=False), refit_mask



# =========================
# Masks (computed on processed signal)
# =========================
@profiled()
def suppress_negative_sag(
    y,
    fs,
    win_sec=1.0,
    q_floor=20,
    k_neg=3.5,
    min_dur_s=0.25,
    pad_s=0.25,
    protect_qrs=True,
    r_idx=None,
    qrs_mask=None,
    use_fast_filter=True
):
    """
    suppress_negative_sag (고속화 버전)
    -----------------------------------
    - baseline 하강(sag) 구간을 검출하여 mask 반환
    - percentile_filter → uniform_filter1d 기반 근사 분위수로 대체 (수십 배 빠름)
    - QRS 보호는 r_idx/qrs_mask 전달 시 neurokit2 호출 생략
    - while 루프 제거 (벡터 연산 기반)
    """


    y = np.asarray(y, float)
    N = y.size
    if N < 10:
        return np.zeros(N, bool)

    # --- 이동창 크기 및 파라미터
    w = max(3, int(round(win_sec * fs)))
    w += (w % 2 == 0)
    min_len = int(round(min_dur_s * fs))
    pad_n = int(round(pad_s * fs))

    # --- (1) 이동 분위수 근사 (빠른 방법)
    if use_fast_filter:
        # 근사 분위수 (percentile 대신 빠른 평균-중앙 근사)
        # 분위수 q_floor와 중앙값(50%)를 근사: 이동 평균 ± 이동 표준편차 비율로 조정
        # 평균과 분산은 uniform_filter1d로 O(N)
        m = uniform_filter1d(y, size=w, mode='nearest')
        m2 = uniform_filter1d(y * y, size=w, mode='nearest')
        v = m2 - m * m
        v[v < 0] = 0.0
        s = np.sqrt(v)
        # 분위수 근사: 평균 - z * 표준편차
        zq = abs(0.01 * (50 - q_floor)) * 0.1  # 근사 계수 (q_floor=20 → 약 3σ 하단)
        floor = m - zq * s
        median = m
    else:

        floor = percentile_filter(y, percentile=q_floor, size=w, mode='nearest')
        median = percentile_filter(y, percentile=50, size=w, mode='nearest')

    # --- (2) 음의 편차 검출
    r = y - median
    neg = np.minimum(r, 0.0)
    med = np.median(neg)
    mad = np.median(np.abs(neg - med)) + 1e-12
    zneg = (neg - med) / (1.4826 * mad)
    mask = (zneg < -abs(k_neg)) & (y < floor)

    # --- (3) QRS 보호
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

    # --- (4) 연결 구간 확장 (벡터 방식)
    if not np.any(mask):
        return mask

    diff = np.diff(mask.astype(int), prepend=0, append=0)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)

    if starts.size == 0:
        return mask

    # 지속시간 조건 + pad 확장
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
    smooth_ms=120,            # 사전 평활(강건화). 0이면 생략
    hop_ms=10                 # 후보 평가 간격(다운샘플링). 5~20ms 권장
):
    """
    Downward step 검출(고속화):
      - 미리 한 번 평활(y_s)
      - 누적합 기반 box-mean으로 pre/post/hold 평균을 벡터화 계산
      - drop 검증 + hold 안정성 조건
      - QRS 보호 + 불응기 적용
    반환: bool mask (step 이후 hold 구간 True)
    """


    y = np.asarray(y, float)
    N = y.size
    if N < 10:
        return np.zeros(N, bool)

    # ---------- 0) 강건 평활(옵션) ----------
    if smooth_ms and smooth_ms > 0:
        m_win = max(3, int(round((smooth_ms/1000.0) * fs)))
        if m_win % 2 == 0: m_win += 1
        # 평균 평활(속도) → 필요시 median_filter로 교체 가능
        y_s = uniform_filter1d(y, size=m_win, mode='nearest')
    else:
        y_s = y

    # ---------- 1) 전역 임계값(강건 스케일) ----------
    med = np.median(y_s)
    mad = np.median(np.abs(y_s - med)) + 1e-12
    thr = amp_sigma * 1.4826 * mad
    if amp_abs is not None:
        thr = max(thr, float(amp_abs))

    # ---------- 2) 박스 평균을 위한 누적합 ----------
    # box-mean(i, L) = (S[i+L] - S[i]) / L, where S is cumsum prepend 0
    S = np.concatenate(([0.0], np.cumsum(y_s, dtype=float)))

    def box_mean(start_idx, L):
        # start_idx: 배열 (동일 길이), L: 정수
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

    # ---------- 3) 후보 중심 인덱스(다운샘플 평가) ----------
    hop = max(1, int(round((hop_ms/1000.0) * fs)))
    # 유효 범위: [pre, N - (gap + post + hold)]
    i_min = pre
    i_max = N - (gap + post + hold) - 1
    if i_max <= i_min:
        return np.zeros(N, bool)
    centers = np.arange(i_min, i_max + 1, hop, dtype=int)

    # pre 평균: [i-pre, i)
    pre_starts = centers - pre
    m1 = box_mean(pre_starts, pre)

    # post 평균: [c, c+post), where c = i+gap
    cpos = centers + gap
    m2 = box_mean(cpos, post)

    # hold 평균: [c, c+hold)
    m_hold = box_mean(cpos, hold)

    drop = m1 - m2
    cond_drop = drop > thr
    # step 이후 구간이 충분히 유지되는지(원 구현 조건) 확인
    cond_hold = (m1 - m_hold) >= (0.6 * drop)

    cand = cond_drop & cond_hold
    if not np.any(cand):
        return np.zeros(N, bool)

    # ---------- 4) QRS 보호 ----------
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

    # 후보 중심 → 실제 step 시작/마킹 구간 정의
    cand_idx = centers[cand]
    # QRS 보호: 시작점이 보호 구간이면 제외
    if protect_qrs and prot.any():
        cand_idx = cand_idx[~prot[cand_idx]]

    if cand_idx.size == 0:
        return np.zeros(N, bool)

    # ---------- 5) 불응기 & 최종 마스크 빌드 ----------
    # 각 이벤트에 대해 mark: [cpos, cpos+hold)
    mask = np.zeros(N, bool)
    last_end = -10**9

    # drop이 큰 순서로 우선(충돌 시 의미 있는 이벤트 먼저)
    order = np.argsort(-drop[cand])  # 내림차순
    for j in order:
        i = centers[j]
        if not cand[j]:
            continue
        start = cpos[j]
        end   = start + hold
        if start - last_end < refr:
            continue
        # QRS 보호: 구간 내 보호가 너무 많으면 skip(선택)
        if protect_qrs and prot.any():
            seg = prot[start:end]
            # 보호구간 비율이 과도하면 스킵 (옵션, 50%)
            if seg.size and seg.mean() > 0.5:
                continue
        mask[start:end] = True
        last_end = end

    return mask

@profiled()
def smooth_corners_mask(
    y,
    fs,
    L_ms=140,
    k_sigma=5.5,
    protect_qrs=True,
    r_idx=None,
    qrs_mask=None,
    smooth_ms=20,         # 미세 평활 (기본 20ms)
    use_float32=True
):
    """
    빠른 corner 검출용 마스크 (고속/강건)
    ------------------------------------
    - numpy.diff 기반 이차차분을 벡터화하여 계산
    - numba 없이 pure-NumPy에서 약 10~30배 빠름
    - QRS 보호는 외부 mask나 r_idx 제공 시 neurokit2 호출 제거
    - 2차 미분 잡음 완화용 pre-smoothing(20ms)
    """

    y = np.asarray(y, np.float32 if use_float32 else np.float64)
    N = y.size
    if N < 10:
        return np.zeros(N, bool)

    # --- 1) 미세 평활(잡음 완화)
    if smooth_ms > 0:
        win = max(3, int(round((smooth_ms / 1000.0) * fs)))
        y_s = uniform_filter1d(y, size=win, mode='nearest')
    else:
        y_s = y

    # --- 2) 2차 미분(중심차분) 벡터화 계산
    # np.gradient는 C구현이라 빠르고 안전함
    d1 = np.gradient(y_s)
    d2 = np.gradient(d1)

    # --- 3) 강건 표준화
    med = np.median(d2)
    mad = np.median(np.abs(d2 - med)) + 1e-12
    z = (d2 - med) / (1.4826 * mad)

    # --- 4) 과도한 변화 검출
    cand = np.abs(z) > float(k_sigma)
    if not np.any(cand):
        return np.zeros(N, bool)

    # --- 5) QRS 보호
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

    # --- 6) 인접 이벤트 병합 및 마스크 확장
    idx = np.flatnonzero(cand)
    if idx.size == 0:
        return np.zeros(N, bool)

    L = max(3, int(round((L_ms / 1000.0) * fs)))
    out = np.zeros(N, bool)

    # 간격 기반 병합: idx 차이가 L보다 크면 새 segment
    gaps = np.diff(idx, prepend=idx[0])
    starts = np.flatnonzero(gaps > L)
    starts = np.append(starts, len(idx))

    # 이전 커서부터 segment 단위 병합
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
    y: np.ndarray,
    win=2000,
    k_sigma=5.0,
    pad=125,
    mode: str = "grid",      # "grid" | "full" | "block"
    hop_ms: int = 32,        # grid 모드에서 격자 간격
    block_s: float = 1.0     # block 모드에서 블록 길이(초)
):
    """
    고분산(HV) 구간 마스크 — 초고속 버전
    - mode="grid": 누적합으로 격자에서만 정확 롤링표준편차 계산 → 선형보간(O(N/hop))
    - mode="full": uniform_filter1d로 전구간 롤링표준편차(O(N))
    - mode="block": 블록별 표준편차 상수(초고속, 거친 필터)

    공통:
    - 임계: median(rs) + 1.4826 * MAD(rs) * k_sigma  (격자 표본 기반으로도 충분히 안정적)
    - pad 확장: scipy.ndimage.binary_dilation (C 구현)
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
        # ----- 전구간 롤링 표준편차 (기존 고속화 버전) -----
        m  = uniform_filter1d(x,   size=w, mode='nearest', origin=0)
        m2 = uniform_filter1d(x*x, size=w, mode='nearest', origin=0)
        v = m2 - m*m
        np.maximum(v, 0.0, out=v)
        rs = np.sqrt(v, dtype=np.float32)

        rs_med = float(np.median(rs))
        rs_mad = float(np.median(np.abs(rs - rs_med)) + 1e-12)
        thr = rs_med + 1.4826 * rs_mad * float(k_sigma)

    elif mode == "block":
        # ----- 블록 상수 표준편차 (초고속, 거친 필터링) -----
        # block_s(초)는 호출부에서 FS를 모르면 대략적 샘플 수로 직접 설정해야 합니다.
        # 여기선 win과 동일 스케일을 쓰도록, 블록 길이를 win에 동기화
        B = max(w, 512)  # 최소 블록 512 샘플
        nb = (n + B - 1) // B
        rs = np.empty(n, dtype=np.float32)
        for b in range(nb):
            s = b * B
            e = min(n, s + B)
            seg = x[s:e]
            sd = float(seg.std(ddof=0))
            rs[s:e] = sd
        # 임계 (전구간 rs 기반)
        rs_med = float(np.median(rs))
        rs_mad = float(np.median(np.abs(rs - rs_med)) + 1e-12)
        thr = rs_med + 1.4826 * rs_mad * float(k_sigma)

    else:
        # ----- GRID: 격자에서만 정확 롤링표준편차 → 보간 -----
        hop = max(1, int(round((hop_ms / 1000.0) * 250.0)))  # FS 모르면 250Hz 가정
        # FS를 아신다면 외부에서 hop을 직접 지정하세요. (예: hop=int(FS*0.032))
        centers = np.arange(0, n, hop, dtype=int)
        starts = np.clip(centers - half, 0, n - 1)
        ends   = np.clip(centers + half + 1, 0, n)

        # 누적합으로 창 평균/제곱평균
        S1 = np.concatenate(([0.0], np.cumsum(x,  dtype=np.float64)))
        S2 = np.concatenate(([0.0], np.cumsum(x*x, dtype=np.float64)))
        Ls = (ends - starts).astype(np.int64)

        sum1 = S1[ends] - S1[starts]
        sum2 = S2[ends] - S2[starts]
        m  = sum1 / np.maximum(1, Ls)
        m2 = sum2 / np.maximum(1, Ls)
        v = m2 - m*m
        v[v < 0.0] = 0.0
        rs_grid = np.sqrt(v, dtype=np.float64)  # 격자상의 std

        # 임계치는 격자 표본으로 계산
        rs_med = float(np.median(rs_grid))
        rs_mad = float(np.median(np.abs(rs_grid - rs_med)) + 1e-12)
        thr = rs_med + 1.4826 * rs_mad * float(k_sigma)

        # 전체 길이로 선형 보간
        idx_full = np.arange(n, dtype=np.float64)
        idx_cent = centers.astype(np.float64)
        rs = np.interp(idx_full, idx_cent, rs_grid).astype(np.float32, copy=False)

    mask = rs > thr

    # pad 확장
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
@profiled()
def burst_mask(
    y,
    fs,
    win_ms=140,          # 분산 창
    k_diff=7.5,          # 1차차분 z-스코어 임계
    k_std=3.5,           # 롤링 표준편차 z-스코어 임계
    pad_ms=80,
    protect_qrs=True,
    r_idx=None,
    qrs_mask=None,
    pre_smooth_ms=0,     # 0이면 생략, 10~20ms 권장(노이즈 심할 때)
    use_float32=True
):
    """
    버스트(급변+분산상승) 마스크 — 고속화 버전
    - dy = np.gradient(y) (C구현)로 급변 검출
    - uniform_filter1d로 E[x], E[x^2] → std를 O(N) 1-pass로 계산
    - 임계는 robust 통계(Median/MAD) 기반
    - pad 확장은 binary_dilation 사용
    """

    x = np.asarray(y, np.float32 if use_float32 else np.float64)
    N = x.size
    if N < 10:
        return np.zeros(N, dtype=bool)

    # 0) 선택적 사전 평활
    if pre_smooth_ms and pre_smooth_ms > 0:
        sw = max(3, int(round((pre_smooth_ms/1000.0) * fs)))
        if sw % 2 == 0: sw += 1
        x = uniform_filter1d(x, size=sw, mode='nearest')

    # 1) 급변 지표: 1차차분(gradient)
    dy = np.gradient(x)
    d_med = float(np.median(dy))
    d_mad = float(np.median(np.abs(dy - d_med)) + 1e-12)
    z_diff = (dy - d_med) / (1.4826 * d_mad)

    # 2) 분산 지표: 롤링 표준편차
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

    # 3) 동시 조건: 급변 & 분산 상승
    cand = (np.abs(z_diff) > float(k_diff)) & (z_std > float(k_std))
    if not np.any(cand):
        return np.zeros(N, dtype=bool)

    # 4) QRS 보호
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

    # 5) pad 확장 (binary_dilation)
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
            self.scaleBy((s, 1.0), center=center)  # X만 확대/축소
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

        # ====== View Toggles ======
        tg = QtWidgets.QHBoxLayout()
        self.cb_raw   = QtWidgets.QCheckBox("원본 신호");       self.cb_raw.setChecked(True)
        self.cb_corr  = QtWidgets.QCheckBox("가공(보정) 신호"); self.cb_corr.setChecked(True)
        self.cb_mask  = QtWidgets.QCheckBox("마스크 패널");     self.cb_mask.setChecked(True)
        self.cb_base  = QtWidgets.QCheckBox("Baseline 표시");   self.cb_base.setChecked(False)
        for cb in (self.cb_raw, self.cb_corr, self.cb_mask, self.cb_base):
            tg.addWidget(cb)
        tg.addStretch(1)
        root.addLayout(tg)

        # ====== Baseline (Hybrid BL++ only) ======
        bl = QtWidgets.QHBoxLayout()
        self.cb_qrsaware = QtWidgets.QCheckBox("QRS-aware"); self.cb_qrsaware.setChecked(True)
        self.sb_asls_lam = QtWidgets.QDoubleSpinBox(); self.sb_asls_lam.setRange(0.0,1e9); self.sb_asls_lam.setDecimals(0); self.sb_asls_lam.setValue(8e7)
        self.sb_asls_p = QtWidgets.QDoubleSpinBox();   self.sb_asls_p.setRange(0.001,0.2); self.sb_asls_p.setSingleStep(0.001); self.sb_asls_p.setValue(0.01)
        self.sb_per_win = QtWidgets.QDoubleSpinBox();  self.sb_per_win.setRange(0.5,10.0); self.sb_per_win.setValue(3.2); self.sb_per_win.setSingleStep(0.1)
        self.sb_per_q = QtWidgets.QSpinBox();          self.sb_per_q.setRange(1,49); self.sb_per_q.setValue(8)
        self.sb_asls_decim = QtWidgets.QSpinBox();     self.sb_asls_decim.setRange(1,64); self.sb_asls_decim.setValue(8)
        self.sb_lpf_fc = QtWidgets.QDoubleSpinBox();   self.sb_lpf_fc.setRange(0.005,0.5); self.sb_lpf_fc.setSingleStep(0.005); self.sb_lpf_fc.setValue(0.3)
        self.sb_vol_win = QtWidgets.QDoubleSpinBox();  self.sb_vol_win.setRange(0.1,5.0); self.sb_vol_win.setSingleStep(0.05); self.sb_vol_win.setValue(0.8)
        self.sb_vol_gain = QtWidgets.QDoubleSpinBox(); self.sb_vol_gain.setRange(0.1,50.0); self.sb_vol_gain.setSingleStep(0.1); self.sb_vol_gain.setValue(2.0)
        self.sb_lam_floor = QtWidgets.QDoubleSpinBox(); self.sb_lam_floor.setRange(0.1,50.0); self.sb_lam_floor.setSingleStep(0.1); self.sb_lam_floor.setValue(0.5)
        self.cb_break_cut = QtWidgets.QCheckBox("Hard cut at breaks"); self.cb_break_cut.setChecked(True)
        self.sb_break_pad = QtWidgets.QDoubleSpinBox();  self.sb_break_pad.setRange(0.0, 2.0); self.sb_break_pad.setSingleStep(0.05); self.sb_break_pad.setValue(0.30)
        self.cb_res_refit = QtWidgets.QCheckBox("Residual refit"); self.cb_res_refit.setChecked(True)
        self.cmb_res_method = QtWidgets.QComboBox(); self.cmb_res_method.addItems(["percentile", "asls"])
        self.sb_res_k = QtWidgets.QDoubleSpinBox(); self.sb_res_k.setRange(1.0, 10.0); self.sb_res_k.setSingleStep(0.1); self.sb_res_k.setValue(2.8)
        self.sb_res_win = QtWidgets.QDoubleSpinBox(); self.sb_res_win.setRange(0.05, 3.0); self.sb_res_win.setSingleStep(0.05); self.sb_res_win.setValue(0.5)
        self.sb_res_pad = QtWidgets.QDoubleSpinBox(); self.sb_res_pad.setRange(0.0, 1.5); self.sb_res_pad.setSingleStep(0.05); self.sb_res_pad.setValue(0.20)

        for lbl, w in [
            ("QRS", self.cb_qrsaware),
            ("λ", self.sb_asls_lam), ("p", self.sb_asls_p),
            ("PerWin(s)", self.sb_per_win), ("PerQ", self.sb_per_q),
            ("AsLS decim", self.sb_asls_decim), ("LPF fc", self.sb_lpf_fc),
            ("VOL win(s)", self.sb_vol_win), ("VOL gain", self.sb_vol_gain), ("λ floor(%)", self.sb_lam_floor),
            ("Break pad(s)", self.sb_break_pad), ("", self.cb_break_cut),
            ("Residual refit", self.cb_res_refit), ("mode", self.cmb_res_method),
            ("kσ", self.sb_res_k), ("win(s)", self.sb_res_win), ("pad(s)", self.sb_res_pad),
        ]:
            bl.addWidget(QtWidgets.QLabel(lbl) if lbl else QtWidgets.QLabel()); bl.addWidget(w)
        root.addLayout(bl)

        # ====== Mask params ======
        row2 = QtWidgets.QHBoxLayout()
        self.cb_sag = QtWidgets.QCheckBox("Sag"); self.cb_sag.setChecked(True)
        self.sb_sag_win = QtWidgets.QDoubleSpinBox(); self.sb_sag_win.setRange(0.2,5.0); self.sb_sag_win.setValue(1.0)
        self.sb_sag_q = QtWidgets.QSpinBox(); self.sb_sag_q.setRange(1,49); self.sb_sag_q.setValue(20)
        self.sb_sag_k = QtWidgets.QDoubleSpinBox(); self.sb_sag_k.setRange(0.5,10.0); self.sb_sag_k.setValue(3.5)
        self.sb_sag_mindur = QtWidgets.QDoubleSpinBox(); self.sb_sag_mindur.setRange(0.05,2.0); self.sb_sag_mindur.setValue(0.25)
        self.sb_sag_pad = QtWidgets.QDoubleSpinBox(); self.sb_sag_pad.setRange(0.0,1.0); self.sb_sag_pad.setValue(0.25)

        self.cb_step = QtWidgets.QCheckBox("Step"); self.cb_step.setChecked(True)
        self.sb_step_sigma = QtWidgets.QDoubleSpinBox(); self.sb_step_sigma.setRange(1.0,15.0); self.sb_step_sigma.setValue(5.0)
        self.sb_step_abs = QtWidgets.QDoubleSpinBox(); self.sb_step_abs.setRange(0.0,500.0); self.sb_step_abs.setValue(0.0)
        self.sb_step_hold = QtWidgets.QDoubleSpinBox(); self.sb_step_hold.setRange(0.1,2.0); self.sb_step_hold.setValue(0.45)

        self.cb_corner = QtWidgets.QCheckBox("Corner"); self.cb_corner.setChecked(True)
        self.sb_corner_L = QtWidgets.QSpinBox(); self.sb_corner_L.setRange(20,400); self.sb_corner_L.setValue(140)
        self.sb_corner_k = QtWidgets.QDoubleSpinBox(); self.sb_corner_k.setRange(1.0,15.0); self.sb_corner_k.setValue(5.5)

        self.cb_burst = QtWidgets.QCheckBox("Burst"); self.cb_burst.setChecked(True)
        self.sb_burst_win = QtWidgets.QSpinBox(); self.sb_burst_win.setRange(20,400); self.sb_burst_win.setValue(140)
        self.sb_burst_kd = QtWidgets.QDoubleSpinBox(); self.sb_burst_kd.setRange(1.0,20.0); self.sb_burst_kd.setValue(6.0)
        self.sb_burst_ks = QtWidgets.QDoubleSpinBox(); self.sb_burst_ks.setRange(1.0,20.0); self.sb_burst_ks.setValue(3.0)
        self.sb_burst_pad = QtWidgets.QSpinBox(); self.sb_burst_pad.setRange(0,400); self.sb_burst_pad.setValue(140)

        self.cb_wave = QtWidgets.QCheckBox("Wavelet"); self.cb_wave.setChecked(False)
        self.sb_wave_sigma = QtWidgets.QDoubleSpinBox(); self.sb_wave_sigma.setRange(1.0,6.0); self.sb_wave_sigma.setValue(2.8)
        self.sb_wave_blend = QtWidgets.QSpinBox(); self.sb_wave_blend.setRange(20,200); self.sb_wave_blend.setValue(80)

        self.win_sb = QtWidgets.QSpinBox(); self.win_sb.setRange(10,50000); self.win_sb.setValue(2000)
        self.kd_sb = QtWidgets.QDoubleSpinBox(); self.kd_sb.setRange(0.1,50.0); self.kd_sb.setSingleStep(0.1); self.kd_sb.setValue(4.0)
        self.pad_sb = QtWidgets.QSpinBox(); self.pad_sb.setRange(0,20000); self.pad_sb.setValue(200)

        for lbl, w in [
            ("Sag", self.cb_sag), ("Win(s)", self.sb_sag_win), ("q", self.sb_sag_q), ("k", self.sb_sag_k),
            ("minDur(s)", self.sb_sag_mindur), ("Pad(s)", self.sb_sag_pad),
            ("Step", self.cb_step), ("σ", self.sb_step_sigma), ("Abs", self.sb_step_abs), ("Hold(s)", self.sb_step_hold),
            ("Corner", self.cb_corner), ("L(ms)", self.sb_corner_L), ("kσ", self.sb_corner_k),
            ("Burst", self.cb_burst), ("Win(ms)", self.sb_burst_win), ("kΔ", self.sb_burst_kd), ("kstd", self.sb_burst_ks), ("Pad(ms)", self.sb_burst_pad),
            ("Wave", self.cb_wave), ("σ", self.sb_wave_sigma), ("Blend(ms)", self.sb_wave_blend),
            ("HV WIN", self.win_sb), ("HV Kσ", self.kd_sb), ("HV PAD", self.pad_sb),
        ]:
            row2.addWidget(QtWidgets.QLabel(lbl)); row2.addWidget(w)
        root.addLayout(row2)

        # ====== Y축 수동/자동 제어 UI ======
        yctl = QtWidgets.QHBoxLayout()
        self.btn_auto_y = QtWidgets.QPushButton("Auto Y-Scale: ON")
        self.btn_auto_y.setCheckable(True); self.btn_auto_y.setChecked(True)
        self.ymin_spin = QtWidgets.QDoubleSpinBox(); self.ymin_spin.setRange(-1e6, 1e6); self.ymin_spin.setDecimals(6); self.ymin_spin.setValue(-1.0)
        self.ymax_spin = QtWidgets.QDoubleSpinBox(); self.ymax_spin.setRange(-1e6, 1e6); self.ymax_spin.setDecimals(6); self.ymax_spin.setValue(1.0)
        self.ymin_spin.setEnabled(False); self.ymax_spin.setEnabled(False)
        yctl.addWidget(self.btn_auto_y)
        yctl.addWidget(QtWidgets.QLabel("Ymin")); yctl.addWidget(self.ymin_spin)
        yctl.addWidget(QtWidgets.QLabel("Ymax")); yctl.addWidget(self.ymax_spin)
        yctl.addStretch(1)
        root.addLayout(yctl)

        # ====== Plots ======
        self.win_plot = pg.GraphicsLayoutWidget(); root.addWidget(self.win_plot)

        self.plot = self.win_plot.addPlot(row=0, col=0, viewBox=XZoomViewBox())
        self.plot.getViewBox().setMouseEnabled(x=True, y=True)
        self.plot.setLabel('bottom','Time (s)'); self.plot.setLabel('left','Amplitude')
        self.plot.showGrid(x=True,y=True,alpha=0.3)

        self.overview = self.win_plot.addPlot(row=1, col=0); self.overview.setMaximumHeight(150); self.overview.showGrid(x=True,y=True,alpha=0.2)
        self.region = pg.LinearRegionItem(); self.region.setZValue(10); self.overview.addItem(self.region); self.region.sigRegionChanged.connect(self.update_region)

        # Colors: raw(gray), corrected(yellow), baseline(cyan dashed)
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

        # ---- 이벤트 연결 ----
        def connect_change(w, slot):
            if isinstance(w, (QtWidgets.QDoubleSpinBox, QtWidgets.QSpinBox)):
                w.valueChanged.connect(slot)
            elif isinstance(w, QtWidgets.QCheckBox):
                w.toggled.connect(slot)
            elif isinstance(w, QtWidgets.QPushButton):
                w.clicked.connect(slot)
            elif isinstance(w, QtWidgets.QComboBox):
                w.currentIndexChanged.connect(lambda _=None: slot())

        for w in [
            self.cb_qrsaware, self.sb_asls_lam, self.sb_asls_p, self.sb_per_win, self.sb_per_q,
            self.sb_asls_decim, self.sb_lpf_fc,
            self.sb_vol_win, self.sb_vol_gain, self.sb_lam_floor,
            self.cb_break_cut, self.sb_break_pad,
            self.cb_res_refit, self.cmb_res_method, self.sb_res_k, self.sb_res_win, self.sb_res_pad,
            self.cb_sag, self.sb_sag_win, self.sb_sag_q, self.sb_sag_k, self.sb_sag_mindur, self.sb_sag_pad,
            self.cb_step, self.sb_step_sigma, self.sb_step_abs, self.sb_step_hold,
            self.cb_corner, self.sb_corner_L, self.sb_corner_k,
            self.cb_burst, self.sb_burst_win, self.sb_burst_kd, self.sb_burst_ks, self.sb_burst_pad,
            self.cb_wave, self.sb_wave_sigma, self.sb_wave_blend,
            self.win_sb, self.kd_sb, self.pad_sb,
        ]:
            connect_change(w, self.schedule_recompute)

        for cb in (self.cb_raw, self.cb_corr, self.cb_mask, self.cb_base):
            cb.toggled.connect(self.update_visibility)

        # Y축 제어 핸들러
        self.btn_auto_y.clicked.connect(self._toggle_y_auto)
        self.ymin_spin.valueChanged.connect(self._apply_y_range_from_spins)
        self.ymax_spin.valueChanged.connect(self._apply_y_range_from_spins)

        # Data
        self.set_data(t, y_raw)

        def dblclick(ev):
            if ev.double():
                self.plot.enableAutoRange(axis=pg.ViewBox.XYAxes, enable=True)
        self.plot.scene().sigMouseClicked.connect(dblclick)

    # ---- Y축 제어 메서드 ----
    def _toggle_y_auto(self):
        state = self.btn_auto_y.isChecked()
        self.plot.enableAutoRange('y', state)
        self.btn_auto_y.setText(f"Auto Y-Scale: {'ON' if state else 'OFF'}")
        self.ymin_spin.setEnabled(not state)
        self.ymax_spin.setEnabled(not state)
        if not state:
            self._apply_y_range_from_spins()

    def _apply_y_range_from_spins(self):
        if self.btn_auto_y.isChecked():
            return
        ylo = self.ymin_spin.value()
        yhi = self.ymax_spin.value()
        if yhi <= ylo:
            yhi = ylo + 1e-9
            self.ymax_spin.setValue(yhi)
        self.plot.setYRange(ylo, yhi, padding=0)

    # ---- 디바운스 재계산 ----
    @profiled()
    def schedule_recompute(self):
        if self._recompute_timer is None:
            self._recompute_timer = QtCore.QTimer(self)
            self._recompute_timer.setSingleShot(True)
            self._recompute_timer.timeout.connect(self.recompute)
        self._recompute_timer.start(600)

    @profiled()
    def set_data(self, t, y):
        # 평균 제거(0 기준 중심화)
        y_centered = np.asarray(y, float)
        if y_centered.size > 0:
            y_centered = y_centered - float(np.nanmean(y_centered))

        self.t = np.asarray(t, float)
        self.y_raw = y_centered

        # 플롯 초기 세팅
        self.curve_raw.setData(self.t, self.y_raw)
        self.ov_curve.setData(self.t, self.y_raw)

        # 초기 영역
        end_t = min(self.t[0]+40.0, self.t[-1]) if self.t.size>1 else 0.0
        self.region.setRegion([self.t[0], end_t])

        # Y 스핀 초기값 갱신(데이터 기반)
        if self.y_raw.size > 0:
            y_min, y_max = float(np.min(self.y_raw)), float(np.max(self.y_raw))
            if np.isfinite(y_min) and np.isfinite(y_max) and y_max > y_min:
                m = 0.1 * (y_max - y_min)
                self.ymin_spin.blockSignals(True); self.ymax_spin.blockSignals(True)
                self.ymin_spin.setValue(y_min - m)
                self.ymax_spin.setValue(y_max + m)
                self.ymin_spin.blockSignals(False); self.ymax_spin.blockSignals(False)

        self.recompute()

    def update_visibility(self):
        self.curve_raw.setVisible(self.cb_raw.isChecked())
        self.curve_corr.setVisible(self.cb_corr.isChecked())
        self.mask_plot.setVisible(self.cb_mask.isChecked())
        self.curve_base.setVisible(self.cb_base.isChecked())

    @profiled()
    def recompute(self):
        # 1) Baseline — Hybrid BL++
        y_src = self.y_raw.copy()
        y_corr, base = baseline_hybrid_plus_adaptive(
            y_src, FS,
            per_win_s=float(self.sb_per_win.value()),
            per_q=int(self.sb_per_q.value()),
            asls_lam=float(self.sb_asls_lam.value()),
            asls_p=float(self.sb_asls_p.value()),
            asls_decim=int(self.sb_asls_decim.value()),
            qrs_aware=self.cb_qrsaware.isChecked(),
            verylow_fc=float(self.sb_lpf_fc.value()),
            clamp_win_s=6.0,
            vol_win_s=float(self.sb_vol_win.value()),
            vol_gain=float(self.sb_vol_gain.value()),
            lam_floor_ratio=float(self.sb_lam_floor.value())/100.0,
            hard_cut=self.cb_break_cut.isChecked(),
            break_pad_s=float(self.sb_break_pad.value())
        )

        # 1.5) Residual selective refit
        resrefit_mask = np.zeros_like(y_corr, dtype=bool)
        if self.cb_res_refit.isChecked():
            mode = self.cmb_res_method.currentText()
            y_corr2, base2, resrefit_mask = selective_residual_refit(
                y_src, base, FS,
                k_sigma=float(self.sb_res_k.value()),
                win_s=float(self.sb_res_win.value()),
                pad_s=float(self.sb_res_pad.value()),
                method=mode,
                per_q=20, asls_lam=1e5,  # ↑ lam 완화
                asls_p=0.02,  # 약간 완화
                asls_decim=8  # 연산량↓ + 안정성
            )
            y_corr, base = y_corr2, base2

        # === No AGC / No Glitch ===
        y_corr_eq = y_corr  # 처리 신호는 순수 BL++(+선택적 리핏) 결과

        # # Soft-AGC: 커진 구간만 완만히 누르되 모폴로지는 유지
        # y_corr_eq = soft_agc_qrs_aware(
        #     y_corr, FS,
        #     win_s=0.8, method="mad", target_q=70,
        #     alpha=1.0, gmin=0.35, gmax=1.0,
        #     smooth_s=0.6, qrs_soft=0.35
        # )
        y_flat, quiet_mask = wvg_flatten(
            y_corr_eq, FS,
            win_s=0.45, q_lo=25, q_hi=75,
            spread_thr=8.0, std_thr=6.0, blend_s=0.20
        )
        y_corr_eq = y_flat

        # 버스트 게이트 적용
        y_burst, burst_mask_bin, gain = burst_gate_dampen(
            y_corr_eq, FS,
            win_ms=140, k_diff=6.0, k_std=3.0, pad_ms=140,
            limit_ratio=0.6, alpha=1.2, atk_ms=60, rel_ms=300
        )
        y_corr_eq = y_burst

        if burst_mask_bin.any():
            y_corr_eq = replace_with_bandlimited(y_corr_eq, FS, burst_mask_bin, fc=12.0)

        # 2) Masks on processed signal
        sag_mask = suppress_negative_sag(
            y_corr_eq, FS, win_sec=float(self.sb_sag_win.value()), q_floor=int(self.sb_sag_q.value()),
            k_neg=float(self.sb_sag_k.value()), min_dur_s=float(self.sb_sag_mindur.value()),
            pad_s=float(self.sb_sag_pad.value()), protect_qrs=True) if self.cb_sag.isChecked() else np.zeros_like(y_corr_eq, bool)

        step_mask = fix_downward_steps_mask(
            y_corr_eq, FS, amp_sigma=float(self.sb_step_sigma.value()),
            amp_abs=(None if float(self.sb_step_abs.value()) <= 0 else float(self.sb_step_abs.value())),
            min_hold_s=float(self.sb_step_hold.value()),
            protect_qrs=True) if self.cb_step.isChecked() else np.zeros_like(y_corr_eq, bool)

        corner_mask = smooth_corners_mask(
            y_corr_eq, FS, L_ms=int(self.sb_corner_L.value()),
            k_sigma=float(self.sb_corner_k.value()), protect_qrs=True) if self.cb_corner.isChecked() else np.zeros_like(y_corr_eq, bool)

        b_mask = np.zeros_like(y_corr_eq, bool)
        if self.cb_burst.isChecked():
            b_mask = burst_mask(
                y_corr_eq, FS, win_ms=int(self.sb_burst_win.value()),
                k_diff=float(self.sb_burst_kd.value()), k_std=float(self.sb_burst_ks.value()),
                pad_ms=int(self.sb_burst_pad.value()), protect_qrs=True)

        alpha_w = np.zeros_like(y_corr_eq)
        if self.cb_wave.isChecked():
            _, alpha_w = qrs_aware_wavelet_denoise(
                y_corr_eq, FS, sigma_scale=float(self.sb_wave_sigma.value()),
                blend_ms=int(self.sb_wave_blend.value()))

        hv_mask, hv_stats = high_variance_mask(
            y_corr_eq, win=int(self.win_sb.value()),
            k_sigma=float(self.kd_sb.value()), pad=int(self.pad_sb.value()))

        # 표시 업데이트
        self.curve_base.setData(self.t, base)
        self.curve_corr.setData(self.t, y_corr_eq)
        self.curve_raw.setData(self.t, self.y_raw)

        # 마스크 패널
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

        lo, hi = self.region.getRegion()
        self.plot.setXRange(lo, hi, padding=0)

        # 자동 Y스케일이면 가시 구간 기반으로 margin 포함하여 설정
        if self.btn_auto_y.isChecked():
            vis_idx = (self.t >= lo) & (self.t <= hi)
            if np.any(vis_idx):
                y_sub = self.y_raw[vis_idx]
                ymin, ymax = float(np.min(y_sub)), float(np.max(y_sub))
                if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
                    margin = 0.1 * (ymax - ymin) if (ymax - ymin) > 0 else 1.0
                    self.plot.setYRange(ymin - margin, ymax + margin, padding=0)

        rows = profiler_report(topn=25)  # 콘솔 출력

    def update_region(self):
        lo, hi = self.region.getRegion()
        self.plot.setXRange(lo, hi, padding=0)

        if self.btn_auto_y.isChecked():
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
    w.setWindowTitle(f"ECG Viewer — {int(FS_RAW)}→{int(FS)} Hz | Hybrid BL++ (AGC/Glitch 없음) | Masks on processed signal | No interpolation")
    w.setCentralWidget(viewer); w.resize(1480, 930); w.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
