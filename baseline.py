# -*- coding: utf-8 -*-
"""
Minimal, memory-efficient ECG processor — JSON → processed ECG (no UI)

- 입력: JSON 파일 경로, 원본 샘플레이트(fs_raw)
- 출력: 가공 신호 (옵션: 시간축 포함)
- 주요 최적화:
  * dtype 일괄 float32 고정
  * np.subtract(..., out=...), in-place 연산
  * 필터/행렬/커널 캐시(ECGWorkspace)
  * interp1d 제거 → np.interp (선형)
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List
import numpy as np

from scipy import signal
from scipy.signal import decimate, butter, filtfilt, savgol_filter
from scipy.interpolate import PchipInterpolator
from scipy.linalg import solveh_banded
from scipy.ndimage import uniform_filter1d, percentile_filter

from calibration import profiler_report, profiled

try:
    import neurokit2 as nk  # type: ignore[import]
except Exception:
    nk = None


# -----------------------------
# Lightweight profiler (no-op safe)
# -----------------------------
_PROF = {}


from dataclasses import dataclass

@dataclass
class ECGWorkspace:
    fs: float
    dtype: np.dtype = np.float32
    # caches
    bp_ba: Optional[Tuple[np.ndarray, np.ndarray]] = None
    hp_ba_015: Optional[Tuple[np.ndarray, np.ndarray]] = None
    hp_ba_var: dict = None  # fc -> (b,a)
    ab_u: dict = None       # N -> ab_u tri-diagonal band
    k_ones: dict = None     # win -> ones kernel

    def __post_init__(self):
        self.hp_ba_var = {}
        self.ab_u = {}
        self.k_ones = {}

    @profiled()
    def butter_band(self, lo, hi, order=2):
        if self.bp_ba is None:
            ny = 0.5 * self.fs
            b, a = butter(order, [lo/ny, hi/ny], btype='band')
            self.bp_ba = (b.astype(self.dtype), a.astype(self.dtype))
        return self.bp_ba

    @profiled()
    def highpass(self, fc, order=2):
        key = (fc, order)
        if key not in self.hp_ba_var:
            ny = 0.5 * self.fs
            b, a = butter(order, fc/ny, btype='high')
            self.hp_ba_var[key] = (b.astype(self.dtype), a.astype(self.dtype))
        return self.hp_ba_var[key]

    @profiled()
    def ones(self, win: int):
        if win not in self.k_ones:
            self.k_ones[win] = np.ones(win, dtype=self.dtype)
        return self.k_ones[win]

    @profiled()
    def band_matrix(self, N: int, lam: float, dtype=None):
        key = (N, lam, self.dtype if dtype is None else dtype)
        if key not in self.ab_u:
            dt = self.dtype if dtype is None else dtype
            ab = np.zeros((3, N), dtype=dt)
            ab[0, 2:] = lam
            ab[1, 1:] = -4.0 * lam
            ab[2, :]  =  6.0 * lam  # diag, 나중에 wg 더함
            self.ab_u[key] = ab
        return self.ab_u[key].copy()  # 구조는 공유, 값은 수정되므로 사본

# -----------------------------
# Workspace: caches for filters, band-mats, kernels
# -----------------------------
@dataclass
class ECGWorkspace:
    fs: float
    dtype: np.dtype = np.float32
    bp_cache: dict = None
    hp_cache: dict = None
    ab_u_cache: dict = None
    ones_cache: dict = None

    def __post_init__(self):
        self.bp_cache = {}
        self.hp_cache = {}
        self.ab_u_cache = {}
        self.ones_cache = {}

    @profiled()
    def butter_band(self, lo_hz, hi_hz, order=2):
        key = ("bp", order, lo_hz, hi_hz)
        if key not in self.bp_cache:
            ny = 0.5 * self.fs
            b, a = butter(order, [lo_hz/ny, hi_hz/ny], btype='band')
            self.bp_cache[key] = (b.astype(self.dtype), a.astype(self.dtype))
        return self.bp_cache[key]

    @profiled()
    def highpass(self, fc_hz, order=2):
        key = ("hp", order, fc_hz)
        if key not in self.hp_cache:
            ny = 0.5 * self.fs
            b, a = butter(order, fc_hz/ny, btype='high')
            self.hp_cache[key] = (b.astype(self.dtype), a.astype(self.dtype))
        return self.hp_cache[key]

    @profiled()
    def ones(self, win: int):
        if win not in self.ones_cache:
            self.ones_cache[win] = np.ones(win, dtype=self.dtype)
        return self.ones_cache[win]

    @profiled()
    def band_matrix(self, N: int, lam: float, dtype=None):
        dt = self.dtype if dtype is None else dtype
        key = ("ab_u", N, float(lam), dt)
        if key not in self.ab_u_cache:
            ab = np.zeros((3, N), dtype=dt)
            ab[0, 2:] = lam
            ab[1, 1:] = -4.0 * lam
            ab[2, :]  =  6.0 * lam   # diag, 매 반복 시 wg 더해 수정
            self.ab_u_cache[key] = ab
        # 구조 공유하지만 대각을 수정해야 하므로 사본 반환
        return self.ab_u_cache[key].copy()


# -----------------------------
# IO
# -----------------------------
@profiled()
def extract_ecg(obj):
    if isinstance(obj, dict):
        if 'ECG' in obj and isinstance(obj['ECG'], list):
            return np.array(obj['ECG'], dtype=float)
        for v in obj.values():
            hit = extract_ecg(v)
            if hit is not None:
                return hit
    elif isinstance(obj, list):
        for it in obj:
            hit = extract_ecg(it)
            if hit is not None:
                return hit
    return None

@profiled()
def decimate_fir_zero_phase(x, q=4):
    return decimate(x, q, ftype='fir', zero_phase=True)

@profiled()
def decimate_if_needed(x, decim: int):
    if decim <= 1:
        return x
    try:
        return decimate_fir_zero_phase(x, decim)
    except Exception:
        n = (len(x) // decim) * decim
        return x[:n].reshape(-1, decim).mean(axis=1)


# -----------------------------
# Helpers (memory-conscious)
# -----------------------------
@profiled()
def highpass_zero_drift(x, fs, fc=0.3, order=2, ws: Optional[ECGWorkspace]=None, out=None):
    if fc <= 0:
        m = np.median(x)
        if out is None:
            return x - m
        np.subtract(x, m, out=out); return out
    b, a = (butter(order, fc/(fs/2), btype='high') if ws is None
            else ws.highpass(fc, order))
    y = filtfilt(b, a, np.asarray(x))
    m = np.median(y)
    if out is None:
        y -= m
        return y
    np.subtract(y, m, out=out)
    return out


def _odd(n: int) -> int:
    n = int(max(3, n))
    return n + (n % 2 == 0)
@profiled()
def smooth_preserve_r(ecg, fs=250, target_fs=100, ws: Optional[ECGWorkspace]=None, out=None):
    """
    bandpass(0.5~35Hz) → FIR decimate → 등간격 선형보간(수동 구현, out 지원)
    """
    # 1) bandpass (계수 캐시 활용)
    if ws is None:
        b, a = butter(2, [0.5/(fs/2), 35/(fs/2)], btype='band')
    else:
        b, a = ws.butter_band(0.5, 35, order=2)
    padlen = min(3 * max(len(a), len(b)), max(0, ecg.size - 1))
    x = filtfilt(b, a, ecg, padlen=padlen)

    # 2) decimate
    decim = int(round(fs / target_fs))
    x_d = decimate(x, decim, ftype='fir', zero_phase=True)

    # 3) 등간격 선형보간 (np.interp(out=...) 미지원 → 수동 구현)
    N  = ecg.size
    Nd = x_d.size
    if out is None:
        out = np.empty(N, dtype=ecg.dtype)

    if N <= 1:
        out[...] = x_d[0]
        return out

    # target i ∈ [0..N-1] → source pos ∈ [0..Nd-1]
    # pos = i * (Nd-1) / (N-1)
    # i0 = floor(pos), i1 = min(i0+1, Nd-1), frac = pos - i0
    pos  = (np.arange(N, dtype=np.float64) * (Nd - 1)) / (N - 1)
    i0   = np.floor(pos).astype(np.int64)
    i1   = np.minimum(i0 + 1, Nd - 1)
    frac = pos - i0

    # out = x_d[i0] + frac * (x_d[i1] - x_d[i0])
    np.subtract(x_d[i1], x_d[i0], out=out)    # out = x_d[i1] - x_d[i0]
    out *= frac.astype(out.dtype, copy=False) # out *= frac
    out += x_d[i0]                             # out += x_d[i0]
    return out




# -----------------------------
# R/Valley detector (compact)
# -----------------------------
class rPeakDetector:
    def __init__(self, fs: int = 250, min_rr_ms: int = 250,
                 use_bandpass: bool = True, bp_lo: float = 5.0, bp_hi: float = 20.0):
        self.fs = int(fs)
        self._refractory = max(1, int((min_rr_ms / 1000.0) * self.fs))
        self.use_bandpass = use_bandpass
        self.bp_lo, self.bp_hi = float(bp_lo), float(bp_hi)

    @profiled()
    def _bandpass(self, x):
        ny = 0.5 * self.fs
        lo = max(0.5, self.bp_lo) / ny
        hi = min(self.bp_hi, 0.45 * self.fs) / ny
        b, a = signal.butter(3, [lo, hi], btype="band")
        padlen = min(3 * max(len(a), len(b)), max(0, x.size - 1))
        return signal.filtfilt(b, a, x, padlen=padlen)

    @profiled()
    def _auto_polarity(self, x):
        q_hi, q_lo = np.percentile(x, [99, 1])
        return x if abs(q_hi) >= abs(q_lo) else -x

    @profiled()
    def _dedup(self, ecg: np.ndarray, idxs: np.ndarray) -> np.ndarray:
        if idxs.size == 0: return idxs
        kept = [int(idxs[0])]
        for cur in idxs[1:]:
            last = kept[-1]
            if cur - last < self._refractory:
                if ecg[cur] > ecg[last]: kept[-1] = int(cur)
            else:
                kept.append(int(cur))
        return np.asarray(kept, int)

    @profiled()
    def detect_extrema(self, ecg: np.ndarray, mode: str = "peak") -> np.ndarray:
        x = np.asarray(ecg, float)
        if self.use_bandpass:
            x = self._bandpass(x)
        if mode == "peak":
            x = self._auto_polarity(x)
            thr = 0.55 * float(np.nanmax(x))
            p, _ = signal.find_peaks(x, distance=self._refractory, height=thr)
            return self._dedup(x, p.astype(int))
        else:
            thr = 0.55 * float(-np.nanmin(x))
            v, _ = signal.find_peaks(-x, distance=self._refractory, height=thr)
            return self._dedup(x, v.astype(int))

    @profiled()
    def rPeakDetection(self, cur_ecg: np.ndarray) -> np.ndarray:
        return self.detect_extrema(cur_ecg, mode="peak")

    @profiled()
    def vValleyDetection(self, cur_ecg: np.ndarray) -> np.ndarray:
        return self.detect_extrema(cur_ecg, mode="valley")


# -----------------------------
# ROI helpers
# -----------------------------
@profiled()
def _roi_bounds(idx: int, fs: float, pre_ms: float, post_ms: float, total_len: int):
    start = max(0, int(idx + pre_ms * 1e-3 * fs))
    end = min(total_len, int(idx + post_ms * 1e-3 * fs))
    min_len = max(5, int(0.04 * fs))
    if end - start < min_len:
        return None, None
    return start, end
@profiled()
def build_protect_windows(r_idx, fs, N, p_win=(-220, -100), st_win=(+120, +260)):
    wins = []
    for r in r_idx:
        a, b = _roi_bounds(r, fs, p_win[0], p_win[1], N)
        if a is not None and b is not None:
            wins.append((a, b))
        a, b = _roi_bounds(r, fs, st_win[0], st_win[1], N)
        if a is not None and b is not None:
            wins.append((a, b))
    return wins

def build_roi_windows(r_idx, fs, N, p_win=(-220, -100), st_win=(+120, +260)):
    return build_protect_windows(r_idx, fs, N, p_win, st_win)
@profiled()
def roi_pchip_fill_baseline(b, wins):
    b = np.asarray(b, float).copy()
    N = b.size
    for (a, bnd) in wins:
        a0 = max(0, a - 1)
        b0 = min(N - 1, bnd)
        if b0 - a0 < 3:
            continue
        xs = np.array([a0, b0], dtype=float)
        ys = np.array([b[a0], b[b0]], dtype=float)
        f = PchipInterpolator(xs, ys, extrapolate=True)
        b[a:bnd] = f(np.arange(a, bnd))
    return b
@profiled()
def roi_adaptive_mix(y_qvri_out, y_med_out, wins, fs, gamma=0.5, corr_min=0.15):
    yq = y_qvri_out  # no copy
    ym = y_med_out
    N = yq.size
    for (a, b) in wins:
        a = max(0, int(a)); b = min(N, int(b))
        if b - a < 5:
            continue
        s_q = yq[a:b]
        s_m = ym[a:b]
        m_q = s_q.mean(); m_m = s_m.mean()
        v_q = np.var(s_q) + 1e-9
        cov = float(np.dot(s_q - m_q, s_m - m_m)) / max(1.0, (b - a))
        rho = float(cov / (np.sqrt(v_q) * (np.std(s_m) + 1e-9)))
        g = gamma if (rho >= corr_min and cov >= 0) else min(0.85, max(gamma, 0.65))
        L = b - a
        Lw = max(3, int(round(0.06 * fs)));  Lw += (Lw % 2 == 0)
        if 2 * Lw < L:
            edge = np.concatenate([np.linspace(0,1,Lw,False),
                                   np.ones(L-2*Lw,dtype=s_q.dtype),
                                   np.linspace(1,0,Lw,False)])
        else:
            edge = np.linspace(0,1,L,False, dtype=s_q.dtype)
        # in-place blend
        mix = (1 - g) * s_q + g * s_m
        s_q *= (1 - edge)
        s_q += edge * mix
    return yq



# -----------------------------
# Baseline core (Hybrid BL++)
# -----------------------------
@profiled()
def baseline_asls_masked(y, lam=1e6, p=0.008, niter=10, mask=None,
                         decim_for_baseline=1, use_float32=True, ws: Optional[ECGWorkspace]=None):
    dt = np.float32 if use_float32 else np.float64
    y = np.asarray(y, dtype=dt)
    N = y.size
    if N < 3:
        return np.zeros_like(y)

    if decim_for_baseline > 1:
        q = int(decim_for_baseline)
        n = (N // q) * q
        if n < q:
            return np.zeros_like(y)
        y_head = y[:n]
        # 평균 다운샘플(뷰→reshape은 카피 없이 가능)
        y_ds = y_head.reshape(-1, q).mean(axis=1, dtype=dt)
        z_ds = baseline_asls_masked(y_ds, lam=lam, p=p, niter=niter, mask=None,
                                    decim_for_baseline=1, use_float32=use_float32, ws=ws)
        # 업샘플: repeat view 활용
        idx = np.repeat(np.arange(z_ds.size), q)
        z_coarse = z_ds[idx]
        if z_coarse.size < N:
            z = np.empty(N, dtype=dt)
            z[:z_coarse.size] = z_coarse
            z[z_coarse.size:] = z_coarse[-1]
        else:
            z = z_coarse[:N]
        return z.astype(np.float64, copy=False)

    g = np.ones(N, dtype=dt) if mask is None else np.where(mask, 1.0, 1e-3).astype(dt)
    lam = dt(lam)
    ab_u = (ws.band_matrix(N, lam, dtype=dt) if ws is not None else
            np.vstack([np.r_[np.zeros(2, dt), lam*np.ones(N-2, dt)],
                       np.r_[np.zeros(1, dt), -4*lam*np.ones(N-1, dt)],
                       6*lam*np.ones(N, dt)]))
    w  = np.ones(N, dtype=dt)
    z  = np.zeros(N, dtype=dt)
    last_obj = None

    for _ in range(int(niter)):
        wg = w * g
        ab_u[2, :] = 6.0*lam + wg  # diag 수정 (in-place)
        b = wg * y                 # rhs
        z = solveh_banded(ab_u, b, lower=False, overwrite_ab=False,
                          overwrite_b=True, check_finite=False)
        # 가중치 업데이트 (in-place 불가 → 재할당 최소화)
        w = p * (y > z) + (1.0 - p) * (y <= z)

        # 수렴 체크(저비용)
        r = y - z
        data_term = float((wg * r).dot(r))
        d2 = np.diff(z, n=2, prepend=float(z[0]), append=float(z[-1]))
        reg_term = float(lam) * float(d2.dot(d2))
        obj = data_term + reg_term
        if last_obj is not None and abs(last_obj - obj) <= 1e-5 * max(1.0, obj):
            break
        last_obj = obj

    return z.astype(np.float64, copy=False)

@profiled()
def rr_isoelectric_clamp(y, fs, r_idx=None, t0_ms=80, t1_ms=300):
    x = np.asarray(y, float)
    if r_idx is None or len(r_idx) < 2:
        if nk is not None:
            try:
                info = nk.ecg_peaks(x, sampling_rate=fs)[1]
                r_idx = np.array(info.get("ECG_R_Peaks", []), int)
            except Exception:
                r_idx = np.array([], int)
        else:
            r_idx = np.array([], int)
    if r_idx.size < 2: return np.zeros_like(x)
    t0 = int(round(t0_ms * 1e-3 * fs))
    t1 = int(round(t1_ms * 1e-3 * fs))
    pts_x, pts_y = [], []
    N = x.size
    for r in r_idx[:-1]:
        a = max(0, r + t0); b = min(N, r + t1)
        if b - a < max(5, int(0.04 * fs)): continue
        m = float(np.median(x[a:b]))
        pts_x.append((a + b) // 2); pts_y.append(m)
    if len(pts_x) < 2: return np.zeros_like(x)
    xs = np.arange(N, dtype=float)
    baseline_rr = np.interp(xs, np.array(pts_x, float), np.array(pts_y, float))
    baseline_rr -= np.median(baseline_rr)
    return baseline_rr
@profiled()
def baseline_hybrid_plus_adaptive(
        y, fs,
        per_win_s=3.2, per_q=8,
        asls_lam=8e7, asls_p=0.01, asls_decim=8,
        qrs_aware=True, verylow_fc=0.55,
        vol_win_s=0.8, vol_gain=2.0, lam_floor_ratio=0.5/100.0,
        hard_cut=True, break_pad_s=0.30,
        rr_cap_enable=False, rr_eps_up=6.0, rr_eps_dn=8.0, rr_t0_ms=80, rr_t1_ms=320,
        r_idx=None, qrs_mask=None, lam_bins=6, min_seg_s=0.50, max_seg_s=6.0,
        ws: Optional[ECGWorkspace]=None
):
    x = np.asarray(y, float)
    N = x.size
    if N < 8: return np.zeros_like(x), np.zeros_like(x)

    # 0) 초기 퍼센타일 바닥선
    w0 = _odd(int(round(per_win_s * fs)))
    dc = np.median(x[np.isfinite(x)])
    x0 = x - dc
    b0 = percentile_filter(x0, percentile=int(per_q), size=w0, mode='nearest')

    # 1) QRS-aware 마스크
    if qrs_mask is not None:
        base_mask = qrs_mask.astype(bool, copy=False)
    else:
        if qrs_aware and nk is not None:
            try:
                info = nk.ecg_peaks(x, sampling_rate=fs)[1]
                r_idx = np.array(info.get("ECG_R_Peaks", []), dtype=int)
            except Exception:
                r_idx = np.array([], dtype=int)
        base_mask = np.ones_like(x, dtype=bool)
        if r_idx is not None and r_idx.size > 0:
            pad = int(round(0.12 * fs))
            for r in r_idx:
                lo = max(0, r - pad); hi = min(N, r + pad + 1)
                base_mask[lo:hi] = False
            t_s = int(round(0.08 * fs)); t_e = int(round(0.30 * fs))
            for r in r_idx:
                lo = max(0, r + t_s); hi = min(N, r + t_e + 1)
                base_mask[lo:hi] = False

    # 2) 위치별 λ
    grad = np.gradient(x)
    g_ref = np.quantile(np.abs(grad), 0.95) + 1e-6
    z_grad = np.clip(np.abs(grad) / g_ref, 0.0, 6.0)
    lam_grad = asls_lam / (1.0 + 8.0 * z_grad)

    vw = _odd(int(round(vol_win_s * fs)))
    # 이동 표준편차
    k = np.ones(vw, float)
    s1 = np.convolve(x, k, mode='same')
    s2 = np.convolve(x * x, k, mode='same')
    m = s1 / vw
    v = s2 / vw - m * m
    v[v < 0] = 0.0
    rs = np.sqrt(v)
    rs_ref = np.quantile(rs, 0.90) + 1e-9
    z_vol = np.clip(rs / rs_ref, 0.0, 10.0)
    lam_vol = asls_lam / (1.0 + float(vol_gain) * z_vol)

    lam_local = np.minimum(lam_grad, lam_vol)
    lam_local = np.maximum(lam_local, asls_lam * max(5e-4, float(lam_floor_ratio)))

    # 3) 단일 세그먼트(간소화) ASLS
    b1 = np.zeros_like(x)
    lam_i = float(np.median(lam_local))
    seg = x0 - b0
    mask_i = base_mask
    b1_seg = baseline_asls_masked(
        seg, lam=max(3e4, lam_i), p=asls_p, niter=10,
        mask=mask_i, decim_for_baseline=max(1, int(asls_decim)), ws=ws
    )
    b1[:] = b1_seg

    # 4) very-low stabilization + offset control
    b = b0 + b1
    b_slow = highpass_zero_drift(b, fs, fc=max(verylow_fc, 0.15), ws=ws)
    sg_win = _odd(max(int(fs * 1.5), int(round(6.0 * fs))))
    resid = x - b_slow
    off = savgol_filter(resid, window_length=sg_win, polyorder=2, mode='interp')
    off -= np.median(off)
    off = highpass_zero_drift(off, fs, fc=0.15, ws=ws)
    b_final = b_slow + off

    if rr_cap_enable and r_idx is not None and len(r_idx) >= 2:
        iso = rr_isoelectric_clamp(x - b_final, fs, t0_ms=rr_t0_ms, t1_ms=rr_t1_ms)
        iso -= np.median(iso)
        err = (b_final - b_slow) - iso
        err = np.clip(err, -float(rr_eps_dn), float(rr_eps_up))
        smw = _odd(int(round(0.12 * fs)))
        err = uniform_filter1d(err, size=smw, mode='nearest')
        b_final = b_slow + iso + err

    y_corr = x - b_final
    return y_corr, b_final


# -----------------------------
# QVRi (Residual Isoelectric with protection)
# -----------------------------
@profiled()
def _qvri_residual_isoelectric(
        y: np.ndarray, fs: float, r_idx: np.ndarray,
        t0_ms: int = -240, t1_ms: int = -100, stride: int = 2,
        lam: float = 2e3, pin_strength: float = 1e9,
        protect_windows: Optional[List[Tuple[int, int]]] = None, w_protect: float = 1e-6
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(y, float)
    N = x.size
    if N < 10 or r_idx is None or len(r_idx) < 2:
        return x, np.zeros_like(x)

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
        idx_knot.append(int((a + b) // 2)); val_knot.append(m)
    if len(idx_knot) == 0:
        return x, np.zeros_like(x)
    idx_knot = np.asarray(idx_knot, dtype=int)
    val_knot = np.asarray(val_knot, dtype=float)

    w = np.ones(N, dtype=float)
    if protect_windows:
        for (a, b) in protect_windows:
            a = max(0, int(a)); b = min(N, int(b))
            if b > a: w[a:b] = float(w_protect)
    w[idx_knot] = float(pin_strength)

    y0 = x.copy()
    y0[idx_knot] = val_knot

    ab_u = np.zeros((3, N), dtype=float)
    ab_u[0, 2:] = lam * 1.0
    ab_u[1, 1:] = lam * (-4.0)
    ab_u[2, :] = lam * 6.0
    ab_u[2, :] += w
    bvec = w * y0
    z = solveh_banded(ab_u, bvec, lower=False, overwrite_ab=False,
                      overwrite_b=True, check_finite=False)
    baseline_qvri = z - np.median(z)
    y_qvri = (x - baseline_qvri) - np.median(x - baseline_qvri)
    return y_qvri, baseline_qvri


# -----------------------------
# PUBLIC API
# -----------------------------
@profiled()
def process_ecg_from_json(json_path: str,
                          fs_raw: float,
                          fs_target: float = 250.0,
                          return_time: bool = False):
    ws = ECGWorkspace(fs=fs_target, dtype=np.float32)

    data = json.loads(Path(json_path).read_text(encoding='utf-8'))
    ecg_raw = extract_ecg(data)
    assert ecg_raw is not None and ecg_raw.size > 0, "JSON에서 ECG 배열을 찾지 못했습니다."

    # dtype 고정 (한 번만)
    ecg_raw = np.asarray(ecg_raw, dtype=np.float32)

    # 리샘플
    decim = max(1, int(round(float(fs_raw) / float(fs_target))))
    ecg = decimate_if_needed(ecg_raw, decim).astype(np.float32, copy=False)
    fs = float(fs_target)
    if return_time:
        t = np.arange(ecg.size, dtype=np.float32) / fs

    # 1) Hybrid BL++ (캐시 사용)
    y_corr, base = baseline_hybrid_plus_adaptive(
        ecg, fs,
        per_win_s=3.2, per_q=8,
        asls_lam=8e7, asls_p=0.01, asls_decim=8,
        qrs_aware=True, verylow_fc=0.55,
        vol_win_s=0.8, vol_gain=2.0, lam_floor_ratio=0.5/100.0,
        hard_cut=True, break_pad_s=0.30,
        rr_cap_enable=False, ws=ws
    )

    # 1.5) 잔류 오프셋 제거 (out 사용)
    baseline_short = signal.medfilt(y_corr, kernel_size=101).astype(np.float32, copy=False)
    y_corr_eq = np.empty_like(y_corr, dtype=np.float32)
    np.subtract(y_corr, baseline_short, out=y_corr_eq)

    # 2) R-peak
    detector = rPeakDetector(fs=int(fs))
    r_after = detector.rPeakDetection(y_corr_eq)

    # 3) QVRi
    wins_protect = build_protect_windows(r_after, fs, y_corr.size,
                                         p_win=(-220, -100), st_win=(+120, +260))
    y_corr_qvri, base_qvri = _qvri_residual_isoelectric(
        y_corr, fs, r_after,
        t0_ms=-240, t1_ms=-100, stride=2, lam=2000.0, pin_strength=1e9,
        protect_windows=wins_protect, w_protect=1e-6
    )

    # 4) ROI 보정 + 적응혼합 (in-place)
    roi_wins = build_roi_windows(r_after, fs, y_corr.size,
                                 p_win=(-220, -100), st_win=(+120, +260))
    base_qvri_roi = roi_pchip_fill_baseline(base_qvri, roi_wins).astype(np.float32, copy=False)

    # y_qvri_edge = y_corr - base_qvri_roi (out 이용)
    y_qvri_edge = np.empty_like(y_corr, dtype=np.float32)
    np.subtract(y_corr, base_qvri_roi, out=y_qvri_edge)

    y_med_base = signal.medfilt(y_corr, kernel_size=101).astype(np.float32, copy=False)
    y_med_out  = np.empty_like(y_corr, dtype=np.float32)
    np.subtract(y_corr, y_med_base, out=y_med_out)

    roi_adaptive_mix(y_qvri_out=y_qvri_edge, y_med_out=y_med_out, wins=roi_wins, fs=fs, gamma=0.5, corr_min=0.15)

    # 5) R 보존 평활 (out 활용)
    y_final = smooth_preserve_r(y_qvri_edge, fs=fs, target_fs=100, ws=ws, out=None).astype(np.float32, copy=False)

    return (t, y_final) if return_time else y_final



# -----------------------------
# CLI example
# -----------------------------
if __name__ == "__main__":
    # 예시: 동일 샘플레이트(250→250)
    y = process_ecg_from_json(
        "11646C1011258_test5_20250825T112545inPlainText.json",
        fs_raw=250.0, fs_target=250.0, return_time=False
    )
    print(y.shape, float(np.mean(y)), float(np.std(y)))
    profiler_report(topn=25)
