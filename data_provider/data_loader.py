import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
from data_provider.m4 import M4Dataset, M4Meta
from data_provider.uea import subsample, interpolate_missing, Normalizer
from sktime.datasets import load_from_tsfile_to_dataframe
import warnings
from utils.augmentation import run_augmentation_single
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import math
warnings.filterwarnings('ignore')

HUGGINGFACE_REPO = "thuml/Time-Series-Library"
import math
import numpy as np
from torch.utils.data import Dataset

class Dataset_Synthetic(Dataset):
    """
    Generates synthetic time-series *on the fly*.

    NEW (per your spec):
      - Difficulty-conditioned sampling: d in [0,1] with easy/medium/hard mixtures.
      - Bundle-level composition (ST/NR/LM/VE) using variance-allocation + Dirichlet.
      - Optional latent-factor wrapper to induce cross-feature correlations.
      - Careful distribution sampling (no naive uniform over everything).
    """

    # -------------------------------------------------------------------------
    # 0) Small RNG helpers (distribution sampling done carefully)
    # -------------------------------------------------------------------------
    @staticmethod
    def _clip01(x: float) -> float:
        return float(min(1.0, max(0.0, x)))

    @staticmethod
    def _trunc_normal(rng, mu: float, sigma: float, lo: float, hi: float, max_tries: int = 50) -> float:
        """
        Rejection-sampled truncated Normal. Falls back to clipping after max_tries.
        """
        for _ in range(max_tries):
            x = rng.normal(mu, sigma)
            if lo <= x <= hi:
                return float(x)
        return float(np.clip(rng.normal(mu, sigma), lo, hi))

    @staticmethod
    def _zscore(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        m = x.mean()
        s = x.std() + eps
        return (x - m) / s

    @staticmethod
    def _safe_unit_var(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        # Center and scale to unit variance (per-component) for variance-allocation mixing.
        x = x - x.mean()
        v = x.var() + eps
        return x / np.sqrt(v)

    @staticmethod
    def _dirichlet_concentration(d: float, conc_easy: float = 80.0, conc_hard: float = 6.0) -> float:
        d = Dataset_Synthetic._clip01(d)
        return (1.0 - d) * conc_easy + d * conc_hard

    
    def _difficulty_profile(self, d: float) -> str:
        d = self._clip01(d)
        if d < 0.30:
            return "easy"
        if d < 0.70:
            return "medium"
        return "hard"

    def _dirichlet_from_targets(self, rng, targets, conc: float) -> np.ndarray:
        """
        targets: array-like that sums to 1, interpreted as desired variance fractions.
        conc: higher => closer to targets (less randomness).
        """
        t = np.asarray(targets, dtype=np.float32)
        t = np.maximum(t, 1e-6)
        t = t / t.sum()
        alpha = t * float(conc)
        return rng.dirichlet(alpha).astype(np.float32)


    # -------------------------------------------------------------------------
    # 1) Default primitive generators (unchanged signatures for compatibility)
    # -------------------------------------------------------------------------
    @staticmethod
    def _white_noise(rng, T):
        return rng.normal(scale=rng.uniform(0.1, 1.0), size=T)

    @staticmethod
    def _random_walk(rng, T):
        steps = rng.normal(scale=rng.uniform(0.05, 0.3), size=T)
        return np.cumsum(steps)

    @staticmethod
    def _gbm(rng, T):
        mu = rng.uniform(-0.01, 0.01)
        sigma = rng.uniform(0.02, 0.12)
        dt = 1.0
        W = rng.normal(scale=np.sqrt(dt), size=T).cumsum()
        t = np.arange(T) * dt
        return np.exp((mu - 0.5 * sigma**2) * t + sigma * W)

    @staticmethod
    def _linear_trend(rng, T):
        slope = rng.uniform(-0.2, 0.2)
        intercept = rng.uniform(-1, 1)
        return intercept + slope * np.arange(T)

    @staticmethod
    def _poly_trend(rng, T):
        coeffs = rng.uniform(-1e-4, 1e-4, size=3)
        t = np.arange(T)
        return coeffs[0] * t**2 + coeffs[1] * t + coeffs[2]

    @staticmethod
    def _exp_trend(rng, T):
        a = rng.uniform(0.5, 1.5)
        b = rng.uniform(-0.01, 0.01)
        return a * np.exp(b * np.arange(T))

    @staticmethod
    def _sinusoid(rng, T):
        amp = rng.uniform(0.5, 2.0)
        phase = rng.uniform(0, 2 * math.pi)
        freq = rng.uniform(0.01, 0.2)
        return amp * np.sin(2 * math.pi * freq * np.arange(T) + phase)

    @staticmethod
    def _fourier_composite(rng, T):
        n_freq = rng.integers(2, 6)
        sig = np.zeros(T)
        for _ in range(n_freq):
            sig += Dataset_Synthetic._sinusoid(rng, T)
        return sig / max(1, n_freq)

    @staticmethod
    def _ar1(rng, T):
        phi = rng.uniform(-0.9, 0.9)
        sigma = rng.uniform(0.05, 0.5)
        eps = rng.normal(scale=sigma, size=T)
        x = np.zeros(T)
        for i in range(1, T):
            x[i] = phi * x[i - 1] + eps[i]
        return x

    @staticmethod
    def _arma(rng, T):
        while True:
            ar = rng.uniform(-0.6, 0.6, size=2)
            if np.sum(np.abs(ar)) < 0.95:
                break
        while True:
            ma = rng.uniform(-0.6, 0.6, size=2)
            if np.sum(np.abs(ma)) < 0.95:
                break

        burn = 100
        N = T + burn + 2
        eps = rng.normal(size=N).astype(np.float32)
        x   = np.zeros(N, dtype=np.float32)

        for t in range(2, N):
            x[t] = (eps[t]
                    + ar[0]*x[t-1] + ar[1]*x[t-2]
                    + ma[0]*eps[t-1] + ma[1]*eps[t-2])

        series = x[burn:burn+T]
        rms = np.sqrt((series**2).mean()) + 1e-6
        series = series / rms
        return np.clip(series, -10, 10)

    @staticmethod
    def _garch(rng, T):
        while True:
            alpha1 = rng.uniform(0.05, 0.30)
            beta1  = rng.uniform(0.30, 0.90)
            if alpha1 + beta1 < 0.97:
                break
        alpha0 = rng.uniform(0.01, 0.05) * (1 - alpha1 - beta1)

        var  = np.ones(T) * alpha0 / (1 - alpha1 - beta1)
        eps  = np.zeros(T, dtype=np.float32)
        z    = rng.normal(size=T)

        for t in range(1, T):
            var[t] = alpha0 + alpha1 * eps[t-1] ** 2 + beta1 * var[t-1]
            if var[t] < 1e-12:
                var[t] = 1e-12
            eps[t] = z[t] * np.sqrt(var[t])

        return eps.astype(np.float32)

    @staticmethod
    def _ou(rng, T):
        theta = rng.uniform(0.1, 0.8)
        mu = rng.uniform(-1, 1)
        sigma = rng.uniform(0.05, 0.5)
        x = np.zeros(T)
        for t in range(1, T):
            x[t] = x[t - 1] + theta * (mu - x[t - 1]) + sigma * rng.normal()
        return x

    @staticmethod
    def _regime_switch(rng, T):
        means = rng.uniform(-2, 2, size=2)
        phi = rng.uniform(0.4, 0.9, size=2)
        trans = np.array([[0.95, 0.05], [0.05, 0.95]])
        state = 0
        x = np.zeros(T)
        for t in range(1, T):
            if rng.random() > trans[state, state]:
                state = 1 - state
            eps = rng.normal(scale=0.3)
            x[t] = means[state] + phi[state] * x[t - 1] + eps
        return x

    @staticmethod
    def _hawkes(rng, T):
        lam0 = rng.uniform(0.1, 0.3)
        alpha = rng.uniform(0.05, 0.2)
        beta = rng.uniform(0.5, 1.5)
        t, events = 0.0, []
        while t < T:
            lam_t = lam0 + alpha * sum(np.exp(-beta * (t - np.array(events)))) if events else lam0
            w = -math.log(max(1e-12, rng.random())) / lam_t
            t += w
            lam_new = lam0 + alpha * sum(np.exp(-beta * (t - np.array(events)))) if events else lam0
            if rng.random() <= lam_new / lam_t:
                events.append(t)

        series = np.zeros(T)
        for e in events:
            idx = int(e)
            if idx < T:
                series[idx] += 1
        return series

    @staticmethod
    def _latent_factor(rng, T):
        factor = Dataset_Synthetic._ar1(rng, T)
        weight = rng.uniform(0.5, 1.5)
        noise = rng.normal(scale=0.1, size=T)
        return weight * factor + noise

    @staticmethod
    def _chaotic_logistic_map(rng, T):
        r = rng.uniform(3.7, 3.99)
        x = rng.random()
        out = np.zeros(T)
        for i in range(T):
            x = r * x * (1 - x)
            out[i] = x
        return out

    @staticmethod
    def _fbm(rng, T):
        H = rng.uniform(0.1, 0.9)
        g  = rng.normal(size=T) + 1j*rng.normal(size=T)
        f  = np.fft.fft(g)
        k  = np.arange(T)
        cov = 0.5 * (np.abs(k-1)**(2*H) - 2*np.abs(k)**(2*H) + np.abs(k+1)**(2*H))
        lam = np.real(np.fft.fft(np.concatenate([cov, cov[-2:0:-1]])))[:T]
        W   = np.fft.ifft(np.sqrt(np.maximum(lam, 0)) * f).cumsum().real
        return W / (np.std(W) + 1e-6)

    @staticmethod
    def _sarima(rng, T):
        s  = rng.integers(6, 24)
        def sample_phi():
            while True:
                phi = rng.uniform(-0.6, 0.6)
                if abs(phi) < 0.95:
                    return phi
        def sample_theta():
            while True:
                th = rng.uniform(-0.6, 0.6)
                if abs(th) < 0.95:
                    return th

        phi   = sample_phi()
        theta = sample_theta()
        Phi   = sample_phi()
        Theta = sample_theta()

        burn  = 5*s + 200
        N     = T + burn + 5
        eps   = rng.normal(size=N).astype(np.float32)
        x     = np.zeros(N, dtype=np.float32)

        for t in range(max(s,1), N):
            ar  = phi  * x[t-1]
            sar = Phi  * x[t-s] if t-s >= 0 else 0.0
            ma  = theta * eps[t-1]
            sma = Theta * eps[t-s] if t-s >= 0 else 0.0
            x[t] = eps[t] + ar + sar + ma + sma

        return x[burn:burn+T]

    @staticmethod
    def _arfima(rng, T):
        d  = rng.uniform(-0.45, 0.45)
        N  = T + 200
        w  = rng.normal(size=N)
        k  = np.arange(N)
        # NOTE: This is a rough coefficient proxy; keep as-is if it works for you.
        coeff = np.exp(np.log(np.abs((d - k + 1)) + 1e-12) - np.log(k+1))
        series = np.convolve(w, coeff)[:N]
        return series[-T:]

    @staticmethod
    def _egarch(rng, T):
        omega  = rng.uniform(-0.3, 0.0)
        alpha  = rng.uniform(0.05, 0.3)
        gamma  = rng.uniform(-0.5, 0.5)
        beta   = rng.uniform(0.6, 0.95)
        z  = rng.normal(size=T)
        logh = np.zeros(T)
        for t in range(1,T):
            logh[t] = omega + beta*logh[t-1] + alpha*(np.abs(z[t-1]) - np.sqrt(2/np.pi)) + gamma*z[t-1]
        return np.exp(0.5*logh) * z

    @staticmethod
    def _neg_binom(rng, T):
        mean  = rng.uniform(1 , 10)
        r     = rng.uniform(1 , 5)
        p     = r / (mean + r)
        return rng.negative_binomial(r, p, size=T).astype(float)

    # -------------------------------------------------------------------------
    # 2) Difficulty sampling (THIS is the key change)
    # -------------------------------------------------------------------------
    def _sample_difficulty(self, rng) -> float:
        """
        Returns d in [0,1].

        Default: 3-component mixture (easy/medium/hard) using truncated normals:
          easy   ~ TN(0.20, 0.08; [0.00,0.35])
          medium ~ TN(0.50, 0.10; [0.35,0.65])
          hard   ~ TN(0.80, 0.08; [0.65,1.00])
        with mixture probs [0.30, 0.40, 0.30].

        If user provides difficulty_sampler (callable), use it.
        If user provides difficulty_mode (string), use predefined distributions.
        """
        # Check for callable sampler first (legacy support)
        if self.difficulty_sampler is not None:
            d = float(self.difficulty_sampler(rng))
            return self._clip01(d)

        # Check for difficulty_mode string (pickle-safe)
        if self.difficulty_mode is not None:
            mode = self.difficulty_mode.lower()
            if mode == 'uniform':
                return float(rng.uniform(0, 1))
            elif mode == 'easy':
                return float(rng.beta(2, 5))  # skewed toward 0
            elif mode == 'medium':
                return float(rng.beta(2, 2))  # centered around 0.5
            elif mode == 'hard':
                return float(rng.beta(5, 2))  # skewed toward 1

        # Default: 3-component mixture
        u = rng.random()
        if u < 0.30:  # easy
            return self._trunc_normal(rng, 0.20, 0.08, 0.00, 0.35)
        elif u < 0.70:  # medium
            return self._trunc_normal(rng, 0.50, 0.10, 0.35, 0.65)
        else:  # hard
            return self._trunc_normal(rng, 0.80, 0.08, 0.65, 1.00)

    # -------------------------------------------------------------------------
    # 3) Difficulty-conditioned parameter sampling for components
    #    (We DO NOT alter primitive methods; we add bundle-specific samplers.)
    # -------------------------------------------------------------------------
    def _gen_sine_easy(
        self,
        rng,
        T: int,
        d: float,
        allowed_periods=(24, 48, 96, 168, 336),
    ) -> np.ndarray:
        """
        Ultra-easy deterministic seasonality: single discrete-period sinusoid.
        Designed for sanity checks (any decent model should overfit quickly).
        """
        d = self._clip01(d)

        # For seq_len=96 sanity: restrict to learnable periods only
        # (You can loosen later. For now: make it trivial.)
        allowed = (24, 48, 96)

        P = int(rng.choice(allowed))

        t = np.arange(T, dtype=np.float32)

        # Keep amplitude very stable and not too small.
        # Optional: vary slightly with d, but keep easy.
        A = float(rng.uniform(0.8, 1.2))

        # Phase is constant for the whole window (no drift).
        phase = float(rng.uniform(0.0, 2.0 * math.pi))

        y = A * np.sin(2.0 * math.pi * t / P + phase)

        # No mean shift for sanity
        return y.astype(np.float32)

    def _gen_st_curriculum(self, rng, T: int, d: float, stage: int,
                        allowed_periods=(24,48,96,168,336)) -> np.ndarray:
        d = self._clip01(d)
        t = np.arange(T, dtype=np.float32)

        # keep P learnable early
        P = int(rng.choice((24, 48, 96) if stage <= 3 else allowed_periods))
        phase = float(rng.uniform(0, 2*math.pi))
        A0 = float(rng.uniform(0.8, 1.2))

        # Stage 0/1 base sine
        y = A0 * np.sin(2*np.pi * t / P + phase).astype(np.float32)

        # Stage 2: harmonics (same P)
        if stage >= 2:
            K = 1 + int(np.round(1 + 4*d))     # 2..6 approx
            alpha = 1.8 - 1.0*d                # harder: slower decay
            for k in range(2, K+1):
                ak = A0 / (k**alpha)
                phk = float(rng.uniform(0, 2*math.pi))
                y += ak * np.sin(2*np.pi * k * t / P + phk).astype(np.float32)
            y /= float(K)

        # Stage 3: mild trend
        if stage >= 3:
            slope = rng.uniform(-0.002 - 0.010*d, 0.002 + 0.010*d)
            y += (slope * t).astype(np.float32)

        # Stage 4: amplitude modulation (slow)
        if stage >= 4:
            Pm = int(rng.choice((4*P, 8*P)))
            a = 0.02 + 0.25*d
            y *= (1.0 + a * np.sin(2*np.pi * t / Pm + rng.uniform(0, 2*math.pi))).astype(np.float32)

        # Stage 5: phase drift (small random walk in phase)
        if stage >= 5:
            # cumulative phase noise
            phase_rw = np.cumsum(rng.normal(scale=(0.0005 + 0.004*d), size=T)).astype(np.float32)
            y = A0 * np.sin(2*np.pi * t / P + phase + phase_rw).astype(np.float32)

        # Stage 6: AR residual (controlled sigma)
        if stage >= 6:
            phi = rng.uniform(0.2, 0.6 + 0.3*d)
            sig = 0.02 + 0.25*d
            eps = rng.normal(scale=sig, size=T).astype(np.float32)
            ar = np.zeros(T, dtype=np.float32)
            for i in range(1, T):
                ar[i] = phi*ar[i-1] + eps[i]
            # keep AR a fraction of signal energy
            y = y + 0.3*ar

        # Stage 1+: observation noise (monotone)
        if stage >= 1:
            sigma_obs = 0.01 + 0.20*d
            y += rng.normal(scale=sigma_obs, size=T).astype(np.float32)

        return y.astype(np.float32)


    def _noise_sigma(self, d: float, sigma_min: float = 0.10, sigma_max: float = 1.00) -> float:
        d = self._clip01(d)
        return sigma_min + d * (sigma_max - sigma_min)

    def _sample_period(self, rng, allowed=(24, 48, 96, 168, 336)) -> int:
        return int(rng.choice(np.asarray(allowed, dtype=int)))

    def _gen_fourier(self, rng, T: int, d: float, allowed_periods=(24, 48, 96, 168, 336)) -> np.ndarray:
        d = self._clip01(d)
        prof = self._difficulty_profile(d)
        t = np.arange(T, dtype=np.float32)

        # Period choice: keep learnable early
        if prof == "easy":
            P = int(rng.choice((24, 48, 96)))
            K = 1
            A0_lo, A0_hi = 0.8, 1.2
            amp_drift_std = 0.0
            alpha_decay = 2.0
        elif prof == "medium":
            P = int(rng.choice((24, 48, 96, 168)))
            K = int(rng.integers(1, 4))  # 1..3
            A0_lo, A0_hi = 0.7, 1.4
            amp_drift_std = 0.001
            alpha_decay = 1.7
        else:
            P = int(rng.choice(tuple(allowed_periods)))
            K = int(rng.integers(2, 7))  # 2..6
            A0_lo, A0_hi = 0.6, 1.8
            amp_drift_std = 0.003
            alpha_decay = 1.3

        A0 = float(rng.uniform(A0_lo, A0_hi))
        base_phase = float(rng.uniform(0.0, 2.0 * math.pi))

        sig = np.zeros(T, dtype=np.float32)

        # All harmonics share same base period P (learnable), but have independent phases.
        for k in range(1, K + 1):
            ak = A0 / (k ** alpha_decay)
            ph = base_phase if k == 1 else float(rng.uniform(0.0, 2.0 * math.pi))

            if amp_drift_std > 0.0:
                drift = np.cumsum(rng.normal(scale=amp_drift_std, size=T)).astype(np.float32)
                a_t = ak * np.exp(drift)
            else:
                a_t = ak

            sig += (a_t * np.sin(2.0 * math.pi * (k * t / P) + ph)).astype(np.float32)

        sig /= float(max(1, K))
        return sig

    def _gen_trend(self, rng, T: int, d: float) -> np.ndarray:
        """
        Trend component: linear/poly/exp with difficulty-controlled magnitude.
        """
        d = self._clip01(d)
        prof = self._difficulty_profile(d)
        if prof == "easy":
            if rng.random() < 0.85:
                return np.zeros(T, dtype=np.float32)
        elif prof == "medium":
            if rng.random() < 0.50:
                return np.zeros(T, dtype=np.float32)
        # hard: always present


        u = rng.random()
        t = np.arange(T, dtype=np.float32)

        if u < 0.50:
            # linear: slope grows with difficulty
            m = rng.uniform(-0.01 - 0.05*d, 0.01 + 0.05*d)
            b = rng.uniform(-1.0, 1.0)
            return (b + m * t).astype(np.float32)

        if u < 0.80:
            # poly degree 2 or 3 with small coefficients that grow with difficulty
            deg = int(rng.choice([2, 3]))
            # keep coefficients small, but allow larger curvature for hard
            c_scale = 0.002 + 0.018*d
            c2 = rng.uniform(-c_scale, c_scale)
            c1 = rng.uniform(-0.05 - 0.10*d, 0.05 + 0.10*d)
            c0 = rng.uniform(-1.0, 1.0)
            if deg == 2:
                return (c2 * (t**2) + c1 * t + c0).astype(np.float32)
            # cubic
            c3 = rng.uniform(-c_scale*0.3, c_scale*0.3)
            return (c3 * (t**3) + c2 * (t**2) + c1 * t + c0).astype(np.float32)

        # exp: growth/decay rate grows with difficulty but remains mild
        a = rng.uniform(0.5, 1.5)
        r = rng.uniform(-0.005 - 0.02*d, 0.005 + 0.02*d)
        return (a * np.exp(r * t)).astype(np.float32)

    def _gen_ar_residual(self, rng, T: int, d: float) -> np.ndarray:
        """
        Residual: AR1/ARMA with persistence increasing with difficulty.
        """

        d = self._clip01(d)
        prof = self._difficulty_profile(d)
        if prof == "easy" and rng.random() < 0.85:
            return np.zeros(T, dtype=np.float32)
        if prof == "medium" and rng.random() < 0.40:
            return np.zeros(T, dtype=np.float32)

        if rng.random() < 0.7:
            phi = rng.uniform(0.3 + 0.2*d, 0.7 + 0.2*d) * (1 if rng.random() < 0.85 else -1)
            sigma = rng.uniform(0.02, 0.08 + 0.20*d)
            eps = rng.normal(scale=sigma, size=T).astype(np.float32)
            x = np.zeros(T, dtype=np.float32)
            for i in range(1, T):
                x[i] = phi * x[i-1] + eps[i]
            return x
        else:
            # use your existing ARMA then scale by a difficulty-dependent factor
            x = self._arma(rng, T).astype(np.float32)
            return x * (0.15 + 0.45*d)  # smaller


    def _gen_regime(self, rng, T: int, d: float) -> np.ndarray:
        """
        Regime component: difficulty controls changepoint frequency and shift sizes.
        """
        d = self._clip01(d)
        M = int(rng.choice([2, 3, 4]))
        # p_stay decreases with d => more frequent regime changes
        p_center = 0.995 - 0.095 * d   # maps d=0 -> 0.995, d=1 -> 0.900
        p_band   = 0.002 + 0.010 * d   # wider uncertainty when hard
        prof = self._difficulty_profile(d)
        if prof == "easy":
            eps_scale = 0.03
        elif prof == "medium":
            eps_scale = 0.08
        else:
            eps_scale = 0.20
        


        low  = np.clip(p_center - p_band, 0.85, 0.999)
        high = np.clip(p_center + p_band, 0.85, 0.999)
        p_stay = rng.uniform(low, high)
        #p_stay = float(np.clip(p_stay, 0.85, 0.999))

        # regime means and slopes
        means = rng.uniform(-1.0, 1.0, size=M).astype(np.float32)
        means += rng.normal(scale=(0.5 + 1.5*d), size=M).astype(np.float32)

        slopes = rng.uniform(-0.01, 0.01, size=M).astype(np.float32)
        slopes += rng.normal(scale=(0.02 + 0.08*d), size=M).astype(np.float32)

        # within-regime AR persistence
        phi = rng.uniform(0.2, 0.8, size=M).astype(np.float32)

        state = int(rng.integers(0, M))
        x = np.zeros(T, dtype=np.float32)
        for t in range(1, T):
            if rng.random() > p_stay:
                state = int(rng.integers(0, M))
            eps = rng.normal(scale=eps_scale)
            x[t] = (means[state] + slopes[state]*t) + phi[state]*x[t-1] + eps
        return x

    def _gen_stoch_trend(self, rng, T: int, d: float) -> np.ndarray:
        """
        Stochastic trend: RW / GBM / OU with difficulty affecting step/volatility.
        """
        d = self._clip01(d)
        u = rng.random()
        if u < 0.40:
            step = rng.normal(scale=rng.uniform(0.01, 0.30 + 0.50*d), size=T).astype(np.float32)
            return np.cumsum(step)
        if u < 0.80:
            mu = rng.uniform(-0.02, 0.02)
            sigma = rng.uniform(0.05, 0.30 + 0.30*d)
            W = rng.normal(scale=1.0, size=T).cumsum().astype(np.float32)
            t = np.arange(T, dtype=np.float32)
            return np.exp((mu - 0.5*sigma**2) * t + sigma * W).astype(np.float32)
        # OU: stronger perturbations with d; keep mean reversion moderate
        theta = rng.uniform(0.05, 0.5)
        mu0 = rng.uniform(-1.0, 1.0)
        sigma = rng.uniform(0.05, 0.30 + 0.40*d)
        x = np.zeros(T, dtype=np.float32)
        for t in range(1, T):
            x[t] = x[t-1] + theta*(mu0 - x[t-1]) + sigma*rng.normal()
        return x

    def _gen_long_memory(self, rng, T: int, d: float) -> np.ndarray:
        """
        Long memory: ARFIMA and/or fBM with persistence increasing in d.
        """
        d = self._clip01(d)
        if rng.random() < 0.60:
            # ARFIMA differencing parameter in [0.1, 0.2+0.3d]
            # reuse your arfima but override its 'd' indirectly by post-warping:
            # simplest: generate arfima and scale its smoothness by adding integrated noise
            x = self._arfima(rng, T).astype(np.float32)
            # mild additional persistence for hard
            x = x + (0.15 + 0.35*d) * np.cumsum(rng.normal(scale=0.02, size=T)).astype(np.float32)
            return x
        else:
            # fBM Hurst in [0.6, 0.7+0.2d]
            # your fbm currently samples H uniformly; to control it, re-implement lightweight:
            H = rng.uniform(0.6, 0.7 + 0.2*d)
            # simple approximate fBM: integrate fractional Gaussian-ish noise proxy
            # (keep it cheap; you can swap in a better fBM later)
            z = rng.normal(size=T).astype(np.float32)
            # AR(1) low-pass to imitate persistence; strength grows with H
            phi = 0.6 + 0.35*(H - 0.6)/0.3
            y = np.zeros(T, dtype=np.float32)
            for t in range(1, T):
                y[t] = phi*y[t-1] + z[t]
            return y

    def _gen_volatility_events(self, rng, T: int, d: float) -> np.ndarray:
        """
        VE: mean AR + volatility modulation + Hawkes spikes.
        Difficulty controls GARCH persistence, event rate, and spike amplitude.
        """
        d = self._clip01(d)

        # mean AR(1)
        phi = rng.uniform(0.2, 0.8)
        base_sigma = rng.uniform(0.10, 0.40)
        eps = rng.normal(scale=base_sigma, size=T).astype(np.float32)
        prof = self._difficulty_profile(d)
        if prof == "easy":
            lam0 = rng.uniform(0.0005, 0.005)
            spike_amp = rng.uniform(0.5, 1.5)
        elif prof == "medium":
            lam0 = rng.uniform(0.002, 0.02)
            spike_amp = rng.uniform(1.0, 3.0)
        else:
            lam0 = rng.uniform(0.01, 0.08)
            spike_amp = rng.uniform(2.0, 7.0)

        m = np.zeros(T, dtype=np.float32)
        for t in range(1, T):
            m[t] = phi*m[t-1] + eps[t]

        # volatility series via GARCH-like recursion (difficulty increases persistence)
        alpha = rng.uniform(0.05, 0.20)
        beta  = rng.uniform(0.60 + 0.20*d, 0.98)
        # enforce alpha+beta < 0.99
        beta = min(beta, 0.99 - alpha - 1e-3)

        omega = rng.uniform(0.01, 0.05) * (1 - alpha - beta)
        h = np.ones(T, dtype=np.float32) * (omega / max(1e-6, (1 - alpha - beta)))
        z = rng.normal(size=T).astype(np.float32)
        r = np.zeros(T, dtype=np.float32)
        for t in range(1, T):
            h[t] = omega + alpha*(r[t-1]**2) + beta*h[t-1]
            h[t] = max(h[t], 1e-8)
            r[t] = z[t] * np.sqrt(h[t]).astype(np.float32)

        # Hawkes spikes: difficulty increases baseline intensity and amplitude
        alpha_h = rng.uniform(0.10, 0.80)
        beta_h  = rng.uniform(0.50, 2.00)
        # generate discrete-time Hawkes approx:
        # intensity_t = lam0 + sum_{ti< t} alpha_h * exp(-beta_h*(t-ti))
        events = np.zeros(T, dtype=np.float32)
        intensity = lam0
        for t in range(T):
            # thinning in discrete time:
            if rng.random() < min(0.95, intensity):
                events[t] = 1.0
            # update intensity (exponential decay + excitation)
            intensity = lam0 + intensity * np.exp(-beta_h) + alpha_h * events[t]

        # spike kernel + amplitude
        kernel = np.exp(-np.arange(10, dtype=np.float32) / 2.0)  # short impulse response
        e = np.convolve(events, kernel, mode="same").astype(np.float32) * spike_amp

        return (m + r + e).astype(np.float32)

    # -------------------------------------------------------------------------
    # 4) Bundle mixers (ST / NR / LM / VE) using variance-allocation + Dirichlet
    # -------------------------------------------------------------------------
    def _mix_components(self, comps: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
        """
        comps: list of 1D arrays length T
        weights: Dirichlet weights that sum to 1 (interpreted as variance fractions)
        Returns: combined 1D array length T
        """
        # standardize each component to unit variance so weights are meaningful
        comps_u = [self._safe_unit_var(c.astype(np.float32)) for c in comps]
        y = np.zeros_like(comps_u[0], dtype=np.float32)
        for w, c in zip(weights, comps_u):
            y += (np.sqrt(max(1e-12, float(w))) * c).astype(np.float32)
        return y

    def _draw_bundle_1d(self, rng, T: int, bundle: str, d: float, allowed_periods=(24, 48, 96, 168, 336)) -> np.ndarray:
        """
        Draw ONE feature (1D) from a bundle distribution.
        """
        bundle = bundle.upper()
        d = self._clip01(d)

        if bundle == "STC":
            y = self._gen_st_curriculum(rng, T, d, stage=6, allowed_periods=allowed_periods)
            return y

        if bundle == "ST":
            prof = self._difficulty_profile(d)

            s  = self._gen_fourier(rng, T, d, allowed_periods=allowed_periods)
            tr = self._gen_trend(rng, T, d)
            ar = self._gen_ar_residual(rng, T, d)

            # Variance targets (sum to 1): [seasonality, trend, residual]
            if prof == "easy":
                targets = [0.92, 0.06, 0.02]
                conc = 120.0
                sigma_obs = self._noise_sigma(d, 0.003, 0.03)
            elif prof == "medium":
                targets = [0.70, 0.18, 0.12]
                conc = 35.0
                sigma_obs = self._noise_sigma(d, 0.01, 0.08)
            else:
                targets = [0.50, 0.20, 0.30]
                conc = 10.0
                sigma_obs = self._noise_sigma(d, 0.03, 0.20)

            w = self._dirichlet_from_targets(rng, targets, conc)
            y = self._mix_components([s, tr, ar], w)

            # Measurement noise (small at easy)
            y += rng.normal(scale=sigma_obs, size=T).astype(np.float32)
            return y


        if bundle == "NR":
            prof = self._difficulty_profile(d)

            g  = self._gen_regime(rng, T, d)
            st = self._gen_stoch_trend(rng, T, d)
            ar = self._gen_ar_residual(rng, T, d)

            if prof == "easy":
                targets = [0.55, 0.35, 0.10]   # regime not dominant
                conc = 80.0
                sigma_obs = self._noise_sigma(d, 0.01, 0.05)
            elif prof == "medium":
                targets = [0.50, 0.30, 0.20]
                conc = 25.0
                sigma_obs = self._noise_sigma(d, 0.03, 0.12)
            else:
                targets = [0.45, 0.25, 0.30]
                conc = 8.0
                sigma_obs = self._noise_sigma(d, 0.06, 0.35)

            w = self._dirichlet_from_targets(rng, targets, conc)
            y = self._mix_components([g, st, ar], w)
            y += rng.normal(scale=sigma_obs, size=T).astype(np.float32)
            return y

        if bundle == "LM":
            prof = self._difficulty_profile(d)

            lm = self._gen_long_memory(rng, T, d)
            comps = [lm]

            # Seasonality: rare at easy, moderate at medium, common at hard
            p_seas = 0.05 if prof == "easy" else (0.25 if prof == "medium" else 0.45)
            if rng.random() < p_seas:
                seas = self._gen_fourier(rng, T, 0.3*d, allowed_periods=allowed_periods)
                comps.append(seas)

            # Residual noise-like component (small at easy)
            if prof == "easy":
                res = rng.normal(scale=self._noise_sigma(d, 0.01, 0.05), size=T).astype(np.float32)
                targets = [0.90, 0.10] if len(comps) == 1 else [0.80, 0.10, 0.10]
                conc = 90.0
                sigma_obs = self._noise_sigma(d, 0.003, 0.03)
            elif prof == "medium":
                res = rng.normal(scale=self._noise_sigma(d, 0.03, 0.12), size=T).astype(np.float32)
                targets = [0.75, 0.10, 0.15] if len(comps) == 2 else [0.70, 0.12, 0.18]
                conc = 25.0
                sigma_obs = self._noise_sigma(d, 0.01, 0.10)
            else:
                res = rng.normal(scale=self._noise_sigma(d, 0.06, 0.25), size=T).astype(np.float32)
                targets = [0.60, 0.15, 0.25] if len(comps) == 2 else [0.55, 0.15, 0.30]
                conc = 8.0
                sigma_obs = self._noise_sigma(d, 0.03, 0.25)

            comps.append(res)

            w = self._dirichlet_from_targets(rng, targets, conc)
            y = self._mix_components(comps, w)
            y += rng.normal(scale=sigma_obs, size=T).astype(np.float32)
            return y


        if bundle == "VE":
            prof = self._difficulty_profile(d)

            y = self._gen_volatility_events(rng, T, d)

            if prof == "easy":
                sigma_obs = self._noise_sigma(d, 0.005, 0.03)
            elif prof == "medium":
                sigma_obs = self._noise_sigma(d, 0.01, 0.08)
            else:
                sigma_obs = self._noise_sigma(d, 0.03, 0.20)

            y += rng.normal(scale=sigma_obs, size=T).astype(np.float32)
            return y


        raise ValueError(f"Unknown bundle '{bundle}'. Expected one of: ST, NR, LM, VE.")

    def _apply_latent_factor_wrapper(self, rng, X: np.ndarray, d: float) -> np.ndarray:
        """
        Option A: correlate already-generated independent features via a random mixing matrix A.

        X: (T, D)
        Returns: (T, D) with induced cross-feature correlations.
        """
        T, D = X.shape
        if D <= 1:
            return X

        # target correlation increases mildly with difficulty
        rho = rng.uniform(0.20, 0.70) if d is None else rng.uniform(0.20, 0.50 + 0.20*d)

        # random mixing matrix with controlled off-diagonal energy
        A = rng.normal(size=(D, D)).astype(np.float32)
        # normalize rows
        A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-6)

        # blend between identity and random mix to control correlation strength
        A = ((1.0 - rho) * np.eye(D, dtype=np.float32) + rho * A).astype(np.float32)

        Y = (X @ A.T).astype(np.float32)
        return Y

    def _gen_var(self, rng, T: int, D: int, sparsity: float = 0.4) -> np.ndarray:
        """Generate a VAR(1) process with controlled sparsity and spectral radius.

        Each channel depends on a sparse random subset of channels at the previous
        timestep, producing genuine cross-lagged dependencies rather than
        instantaneous mixing.

        Args:
            rng: numpy Generator
            T: series length
            D: number of channels
            sparsity: fraction of A coefficients set to zero (0 = dense, 1 = all zero)

        Returns:
            (T, D) float32 array
        """
        # Sample coefficient matrix
        A = rng.standard_normal((D, D)).astype(np.float32) * 0.5

        # Apply sparsity mask
        if sparsity > 0:
            mask = rng.random((D, D)) < sparsity
            A[mask] = 0.0

        # Scale to ensure spectral radius < target (stability)
        eigvals = np.linalg.eigvals(A)
        spec_radius = np.max(np.abs(eigvals))
        if spec_radius > 0.9:
            A = A * (0.85 / spec_radius)

        # Innovation noise: mild heteroscedasticity across channels
        sigma = rng.uniform(0.05, 0.3, size=D).astype(np.float32)

        Y = np.zeros((T, D), dtype=np.float32)
        for t in range(1, T):
            Y[t] = A @ Y[t - 1] + rng.standard_normal(D).astype(np.float32) * sigma

        return Y

    def _gen_shared_latent(self, rng, T: int, D: int, n_factors: int | None = None) -> np.ndarray:
        """Generate D channels from 2-4 shared latent bundle processes.

        Each channel is a Dirichlet-weighted sum of latent factor processes plus
        idiosyncratic noise. Factors are drawn from different bundles so channels
        inherit heterogeneous statistical properties while sharing a common driver.

        Args:
            rng: numpy Generator
            T: series length
            D: number of channels
            n_factors: number of latent factors (default: min(D, random 2-4))

        Returns:
            (T, D) float32 array
        """
        if n_factors is None:
            n_factors = int(rng.integers(2, min(5, D + 1)))

        d = self._sample_difficulty(rng)

        # One latent process per factor, each from a different bundle
        factor_bundles = rng.choice(self.bundles, size=n_factors, replace=True)
        factors = np.stack(
            [self._draw_bundle_1d(rng, T, b, d, allowed_periods=self.allowed_periods)
             for b in factor_bundles],
            axis=1
        ).astype(np.float32)  # (T, n_factors)

        # Factor loadings: each channel mixes the latent factors differently
        # Dirichlet ensures non-negative loadings summing to 1 per channel
        alpha = np.ones(n_factors) * 2.0  # concentration = 2 → spread, not too sparse
        loadings = rng.dirichlet(alpha, size=D).astype(np.float32)  # (D, n_factors)

        # Shared component
        shared = (factors @ loadings.T).astype(np.float32)  # (T, D)

        # Idiosyncratic noise (different bundle per channel for heterogeneity)
        idio_bundles = rng.choice(self.bundles, size=D, replace=True)
        idio = np.stack(
            [self._draw_bundle_1d(rng, T, b, d, allowed_periods=self.allowed_periods)
             for b in idio_bundles],
            axis=1
        ).astype(np.float32)  # (T, D)

        # Mix: strength of common component vs idiosyncratic
        common_weight = rng.uniform(0.4, 0.8)
        Y = (common_weight * shared + (1.0 - common_weight) * idio).astype(np.float32)
        return Y

    # -------------------------------------------------------------------------
    # 5) Init / draw logic
    # -------------------------------------------------------------------------
    def __init__(
        self,
        root_path=None,
        flag='train',
        size=None,
        features='M',
        num_samples=10000,
        generators=None,                      # primitives list (legacy)
        generators_distribution=None,          # weights for primitives (legacy)
        channels=7,
        d_sampler=1,                           # legacy (kept; not used for difficulty)
        scale=0,
        seed=42,
        # NEW:
        mode="bundle",                    # "primitives" | "bundle"
        bundle=None,                          # "ST" | "NR" | "LM" | "VE" (or list)
        bundle_distribution=None,             # weights over bundles if bundle is list
        difficulty_sampler=None,              # callable rng -> d in [0,1] (legacy, prefer difficulty_mode)
        difficulty_mode=None,                 # 'uniform' | 'easy' | 'medium' | 'hard' (pickle-safe)
        p_latent_factor=0.0,                  # probability of applying LF wrapper
        allowed_periods=(24, 48, 96, 168, 336),
        multivariate_mode="independent",      # "independent" | "var" | "latent_factor"
        timeenc=0,                            # time encoding: 0=manual, 1=timeF
        freq='h',                             # frequency for time features
        cache_size=500,                       # 0 = on-the-fly, N = fixed pre-generated cache of N samples
        **_
    ):
        self.scale = scale
        if size is None:
            self.seq_len, self.label_len, self.pred_len = 96, 48, 96
        else:
            self.seq_len, self.label_len, self.pred_len = size
        self.T_total = self.seq_len + self.pred_len
        self.timeenc = timeenc
        self.freq = freq

        # Determine number of time mark columns based on freq
        # For hourly data: 4 features (HourOfDay, DayOfWeek, DayOfMonth, DayOfYear)
        # We use 4 as default to match common real datasets
        freq_to_mark_dim = {
            'h': 4,   # hour: HourOfDay, DayOfWeek, DayOfMonth, DayOfYear
            't': 5,   # minute: MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear
            's': 6,   # second: SecondOfMinute, MinuteOfHour, HourOfDay, DayOfWeek, DayOfMonth, DayOfYear
            'd': 3,   # day: DayOfWeek, DayOfMonth, DayOfYear
            'b': 3,   # business day: same as day
            'w': 2,   # week: DayOfMonth, WeekOfYear
            'm': 1,   # month: MonthOfYear
        }
        self.mark_dim = freq_to_mark_dim.get(freq.lower()[:1], 4)
        self.num_samples = num_samples
        self.features = features
        self.base_rng = np.random.default_rng(seed)
        self._seed = seed

        # NEW controls
        self.mode = str(mode).lower()
        self.allowed_periods = tuple(int(x) for x in allowed_periods)
        self.p_latent_factor = float(p_latent_factor)
        _valid_mv_modes = ("independent", "var", "latent_factor")
        if multivariate_mode not in _valid_mv_modes:
            raise ValueError(f"multivariate_mode must be one of {_valid_mv_modes}, got {multivariate_mode!r}")
        self.multivariate_mode = multivariate_mode

        # Handle difficulty sampling - prefer difficulty_mode (pickle-safe) over difficulty_sampler
        self.difficulty_mode = difficulty_mode
        if difficulty_sampler is not None:
            self.difficulty_sampler = difficulty_sampler
        else:
            self.difficulty_sampler = None  # Will be created per-call in _sample_difficulty

        # legacy fields (kept)
        self.d_sampler = d_sampler
        self.channels  = channels

        # primitive registry (legacy path)
        methods = {
            'white_noise': self._white_noise,
            'random_walk': self._random_walk,
            'gbm': self._gbm,
            'linear_trend': self._linear_trend,
            'poly_trend': self._poly_trend,
            'exp_trend': self._exp_trend,
            'sinusoid': self._sinusoid,
            'fourier_composite': self._fourier_composite,
            'ar1': self._ar1,
            'arma': self._arma,
            'garch': self._garch,
            'ou': self._ou,
            'regime_switch': self._regime_switch,
            'hawkes': self._hawkes,
            'latent_factor': self._latent_factor,
            'chaotic_logistic_map': self._chaotic_logistic_map,
            'fbm': self._fbm,
            'sarima': self._sarima,
            'arfima': self._arfima,
            'egarch': self._egarch,
            'neg_binom': self._neg_binom,
        }

        # primitives selection
        if generators is None:
            generators = list(methods.keys())
        elif isinstance(generators, str):
            generators = [generators]

        try:
            self.generators = [methods[name] for name in generators]
        except KeyError as e:
            raise ValueError(f'Unknown generator "{e}"')

        if generators_distribution is None:
            w = np.ones(len(self.generators), dtype=float)
        else:
            if len(generators_distribution) != len(self.generators):
                raise ValueError("`generators_distribution` length must match `generators`")
            w = np.asarray(generators_distribution, dtype=float)
            if (w < 0).any():
                raise ValueError("All weights must be ≥ 0")
        w_sum = float(w.sum())
        if w_sum <= 0:
            raise ValueError("At least one primitive weight must be positive")
        self.weights = (w / w_sum).astype(np.float64)

        # bundle selection
        if bundle is None:
            self.bundles = ["ST", "NR", "LM", "VE"]
        elif isinstance(bundle, str):
            self.bundles = [bundle.upper()]
        else:
            self.bundles = [str(b).upper() for b in bundle]

        if bundle_distribution is None:
            bw = np.ones(len(self.bundles), dtype=float)
        else:
            if len(bundle_distribution) != len(self.bundles):
                raise ValueError("`bundle_distribution` length must match `bundle`")
            bw = np.asarray(bundle_distribution, dtype=float)
            if (bw < 0).any():
                raise ValueError("All bundle weights must be ≥ 0")
        bw_sum = float(bw.sum())
        if bw_sum <= 0:
            raise ValueError("At least one bundle weight must be positive")
        self.bundle_weights = (bw / bw_sum).astype(np.float64)

        D = 1 if features == 'S' else channels
        if cache_size == 0:
            # On-the-fly mode: generate a fresh sample per __getitem__ call
            print(f"On-the-fly synthetic generation enabled (D={D}, T={self.T_total}, num_samples={num_samples})")
            self._cache = None
        else:
            # Pre-generate a capped number of unique samples; __getitem__ cycles via modulo.
            num_cache = min(num_samples, cache_size)
            print(f"Pre-generating {num_cache} synthetic samples (D={D}, T={self.T_total}, total_requested={num_samples})...")
            self._cache = np.empty((num_cache, self.T_total, D), dtype=np.float32)
            for i in range(num_cache):
                rng = np.random.default_rng(seed + i)
                if self.mode == "bundle":
                    self._cache[i] = self._draw_series_bundle(rng, D)
                else:
                    self._cache[i] = self._draw_series_primitives(rng, D)
            print(f"Pre-generation done. Cache size: {self._cache.nbytes / 1e6:.1f} MB")
        self._cache_epoch = 0

    def resample(self, epoch: int):
        """Regenerate the cache with new seeds for this epoch (preserves diversity across epochs)."""
        if self._cache is None or epoch == self._cache_epoch:
            return
        num_cache = len(self._cache)
        D = self._cache.shape[2]
        epoch_seed = self.base_rng.integers(0, 2**31) + epoch * num_cache
        for i in range(num_cache):
            rng = np.random.default_rng(epoch_seed + i)
            if self.mode == "bundle":
                self._cache[i] = self._draw_series_bundle(rng, D)
            else:
                self._cache[i] = self._draw_series_primitives(rng, D)
        self._cache_epoch = epoch

    def __len__(self):
        return self.num_samples

    # -------------------------------------------------------------------------
    # 6) Draw series: primitives path OR bundle-mix path
    # -------------------------------------------------------------------------
    def _draw_series_primitives(self, rng, D: int) -> np.ndarray:
        gens = rng.choice(self.generators, size=D, p=self.weights)
        return np.stack([g(rng, self.T_total) for g in gens], axis=1).astype(np.float32)

    def _draw_series_bundle(self, rng, D: int) -> np.ndarray:
        """Generate a D-channel sample using the configured multivariate_mode.

        Modes:
          "independent"  — each channel sampled independently from its bundle;
                           optional linear mixing via p_latent_factor (legacy behaviour).
          "var"          — VAR(1) residuals mixed with per-channel bundle envelopes.
          "latent_factor"— D channels driven by 2-4 shared latent bundle processes
                           plus idiosyncratic components.
        """
        if D <= 1 or self.multivariate_mode == "independent":
            # Original independent-channels path
            d = self._sample_difficulty(rng)
            chosen = rng.choice(self.bundles, size=D, p=self.bundle_weights)
            X = np.stack(
                [self._draw_bundle_1d(rng, self.T_total, b, d, allowed_periods=self.allowed_periods)
                 for b in chosen],
                axis=1
            ).astype(np.float32)
            if (D > 1) and (self.p_latent_factor > 0) and (rng.random() < self.p_latent_factor):
                X = self._apply_latent_factor_wrapper(rng, X, d)

        elif self.multivariate_mode == "var":
            # VAR(1) process shaped by per-channel bundle envelopes.
            # Generate independent bundle channels first, then add VAR cross-lag structure.
            d = self._sample_difficulty(rng)
            chosen = rng.choice(self.bundles, size=D, p=self.bundle_weights)
            envelope = np.stack(
                [self._draw_bundle_1d(rng, self.T_total, b, d, allowed_periods=self.allowed_periods)
                 for b in chosen],
                axis=1
            ).astype(np.float32)  # (T, D) — captures marginal statistics
            var_signal = self._gen_var(rng, self.T_total, D)  # (T, D) — cross-lag structure
            # Blend: envelope preserves realistic marginals; VAR adds cross-channel dynamics
            blend = rng.uniform(0.3, 0.7)
            X = (blend * envelope + (1.0 - blend) * var_signal).astype(np.float32)

        elif self.multivariate_mode == "latent_factor":
            X = self._gen_shared_latent(rng, self.T_total, D)

        else:
            raise ValueError(f"Unknown multivariate_mode: {self.multivariate_mode!r}")

        if self.scale:
            X = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-6)

        return X

    def __getitem__(self, idx):
        if self._cache is None:
            rng = np.random.default_rng(self._seed + idx)
            D = 1 if self.features == 'S' else self.channels
            if self.mode == "bundle":
                series = self._draw_series_bundle(rng, D)
            else:
                series = self._draw_series_primitives(rng, D)
        else:
            series = self._cache[idx % len(self._cache)]

        seq_x = series[: self.seq_len]
        seq_y = series[self.seq_len - self.label_len : self.seq_len + self.pred_len]

        t = np.linspace(0, 1, self.T_total, dtype=np.float32)
        marks = np.zeros((self.T_total, self.mark_dim), dtype=np.float32)
        for i in range(self.mark_dim):
            phase = i * 0.25
            marks[:, i] = np.sin(2 * np.pi * (t + phase) * (i + 1)) * 0.5
        seq_x_mark = marks[: self.seq_len]
        seq_y_mark = marks[self.seq_len - self.label_len : self.seq_len + self.pred_len]

        return seq_x, seq_y, seq_x_mark, seq_y_mark


class Dataset_Mixed(Dataset):
    """
    Wraps both a real dataset and a synthetic dataset.

    synth_ratio controls how much synthetic data to ADD on top of real data:
    - synth_ratio=0.0: no synthetic data, only real
    - synth_ratio=0.5: add 50% more data as synthetic (total = 1.5x real)
    - synth_ratio=1.0: add 100% more data as synthetic (total = 2x real, i.e., 50-50 split)

    The total dataset length is: len(real) + len(synth)
    where len(synth) = len(real) * synth_ratio

    Returns the same interface: (seq_x, seq_y, seq_x_mark, seq_y_mark)
    """

    def __init__(self, real_dataset, synth_dataset, synth_ratio=0.5, seed=42):
        """
        Args:
            real_dataset: Dataset returning (seq_x, seq_y, seq_x_mark, seq_y_mark)
            synth_dataset: Dataset returning (seq_x, seq_y, seq_x_mark, seq_y_mark)
                           Should have num_samples = len(real_dataset) * synth_ratio
            synth_ratio: Ratio of synthetic data to add (0.0 = none, 1.0 = same as real)
            seed: Random seed for reproducibility
        """
        self.real = real_dataset
        self.synth = synth_dataset
        self.synth_ratio = synth_ratio
        self.rng = np.random.default_rng(seed)
        self._update_sampling_prob()

    def _update_sampling_prob(self):
        """Calculate the probability of sampling synthetic based on dataset sizes."""
        real_len = len(self.real)
        synth_len = len(self.synth)
        total_len = real_len + synth_len
        # Probability of sampling synthetic = synth_len / total_len
        self._synth_prob = synth_len / total_len if total_len > 0 else 0.0

    @property
    def scale(self):
        """Return scale from real dataset for inverse transform compatibility."""
        return getattr(self.real, 'scale', False)

    @property
    def scaler(self):
        """Return scaler from real dataset for inverse transform compatibility."""
        return getattr(self.real, 'scaler', None)

    def inverse_transform(self, data):
        """Inverse transform using the real dataset's scaler."""
        if hasattr(self.real, 'inverse_transform'):
            return self.real.inverse_transform(data)
        elif self.scaler is not None:
            return self.scaler.inverse_transform(data)
        return data

    def __len__(self):
        # Total length is real + synthetic (not max)
        return len(self.real) + len(self.synth)

    def __getitem__(self, idx):
        # Sample based on the proportion of each dataset
        if self.rng.random() < self._synth_prob:
            return self.synth[idx % len(self.synth)]
        return self.real[idx % len(self.real)]

    def set_synth_ratio(self, ratio):
        """Update synth_ratio for gradual annealing.

        Note: This only updates the stored ratio. The actual synthetic dataset
        size is fixed at creation time. For annealing, this affects the
        sampling probability.
        """
        self.synth_ratio = max(0.0, ratio)
        # Recalculate sampling probability based on actual dataset sizes
        self._update_sampling_prob()


class Dataset_ETT_hour(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()

        local_fp = os.path.join(self.root_path, self.data_path)
        cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]

        if os.path.exists(local_fp):
            df_raw = pd.read_csv(local_fp)
        else:
            ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
            df_raw = ds["train"].to_pandas()
            
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0) 

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTm1.csv',
                 target='OT', scale=True, timeenc=0, freq='t', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        
        local_fp = os.path.join(self.root_path, self.data_path)
        cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]

        if os.path.exists(local_fp):
            df_raw = pd.read_csv(local_fp)
        else:
            ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
            df_raw = ds["train"].to_pandas()

        border1s = [0, 12 * 30 * 24 * 4 - self.seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len]
        border2s = [12 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, args, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h', seasonal_patterns=None):
        # size [seq_len, label_len, pred_len]
        self.args = args
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        local_fp = os.path.join(self.root_path, self.data_path)
        cfg_name = os.path.splitext(os.path.basename(self.data_path))[0]

        if os.path.exists(local_fp):
            df_raw = pd.read_csv(local_fp)
        else:
            ds = load_dataset(HUGGINGFACE_REPO, name=cfg_name)
            split_name = "train" if "train" in ds else list(ds.keys())[0]
            df_raw = ds[split_name].to_pandas()

        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

        if self.set_type == 0 and self.args.augmentation_ratio > 0:
            self.data_x, self.data_y, augmentation_tags = run_augmentation_single(self.data_x, self.data_y, self.args)

        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_M4(Dataset):
    def __init__(self, args, root_path, flag='pred', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=False, inverse=False, timeenc=0, freq='15min',
                 seasonal_patterns='Yearly'):
        # size [seq_len, label_len, pred_len]
        # init
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.root_path = root_path

        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]

        self.seasonal_patterns = seasonal_patterns
        self.history_size = M4Meta.history_size[seasonal_patterns]
        self.window_sampling_limit = int(self.history_size * self.pred_len)
        self.flag = flag

        self.__read_data__()

    def __read_data__(self):
        # M4Dataset.initialize()
        if self.flag == 'train':
            dataset = M4Dataset.load(training=True, dataset_file=self.root_path)
        else:
            dataset = M4Dataset.load(training=False, dataset_file=self.root_path)
        training_values = np.array(
            [v[~np.isnan(v)] for v in
             dataset.values[dataset.groups == self.seasonal_patterns]])  # split different frequencies
        self.ids = np.array([i for i in dataset.ids[dataset.groups == self.seasonal_patterns]])
        self.timeseries = [ts for ts in training_values]

    def __getitem__(self, index):
        insample = np.zeros((self.seq_len, 1))
        insample_mask = np.zeros((self.seq_len, 1))
        outsample = np.zeros((self.pred_len + self.label_len, 1))
        outsample_mask = np.zeros((self.pred_len + self.label_len, 1))  # m4 dataset

        sampled_timeseries = self.timeseries[index]
        cut_point = np.random.randint(low=max(1, len(sampled_timeseries) - self.window_sampling_limit),
                                      high=len(sampled_timeseries),
                                      size=1)[0]

        insample_window = sampled_timeseries[max(0, cut_point - self.seq_len):cut_point]
        insample[-len(insample_window):, 0] = insample_window
        insample_mask[-len(insample_window):, 0] = 1.0
        outsample_window = sampled_timeseries[
                           max(0, cut_point - self.label_len):min(len(sampled_timeseries), cut_point + self.pred_len)]
        outsample[:len(outsample_window), 0] = outsample_window
        outsample_mask[:len(outsample_window), 0] = 1.0
        return insample, outsample, insample_mask, outsample_mask

    def __len__(self):
        return len(self.timeseries)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def last_insample_window(self):
        """
        The last window of insample size of all timeseries.
        This function does not support batching and does not reshuffle timeseries.

        :return: Last insample window of all timeseries. Shape "timeseries, insample size"
        """
        insample = np.zeros((len(self.timeseries), self.seq_len))
        insample_mask = np.zeros((len(self.timeseries), self.seq_len))
        for i, ts in enumerate(self.timeseries):
            ts_last_window = ts[-self.seq_len:]
            insample[i, -len(ts):] = ts_last_window
            insample_mask[i, -len(ts):] = 1.0
        return insample, insample_mask


class PSMSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        train_path = os.path.join(root_path, "train.csv")
        test_path = os.path.join(root_path, "test.csv")
        label_path = os.path.join(root_path, "test_label.csv")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_df      = pd.read_csv(train_path)
            test_df       = pd.read_csv(test_path)
            test_label_df = pd.read_csv(label_path)
        else:
            ds_data  = load_dataset(HUGGINGFACE_REPO, name="PSM-data")
            ds_label = load_dataset(HUGGINGFACE_REPO, name="PSM-label")
            train_df      = ds_data["train"].to_pandas()
            test_df       = ds_data["test"].to_pandas()
            test_label_df = ds_label[next(iter(ds_label))].to_pandas()

        data = train_df.values[:, 1:]
        data = np.nan_to_num(data)
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        
        test_data = test_df.values[:, 1:]
        test_data = np.nan_to_num(test_data)
        self.test = self.scaler.transform(test_data)
        
        self.train = data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = test_label_df.values[:, 1:]
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class MSLSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        train_path = os.path.join(root_path, "MSL_train.npy")
        test_path  = os.path.join(root_path, "MSL_test.npy")
        label_path = os.path.join(root_path, "MSL_test_label.npy")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_data = np.load(train_path)
            test_data  = np.load(test_path)
            test_label = np.load(label_path)
        else:
            train_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="MSL/MSL_train.npy",repo_type="dataset")
            test_path  = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="MSL/MSL_test.npy",repo_type="dataset")
            label_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="MSL/MSL_test_label.npy",repo_type="dataset")

            train_data  = np.load(train_path)
            test_data   = np.load(test_path)
            test_label  = np.load(label_path)

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data  = self.scaler.transform(test_data)

        self.train = train_data
        self.test  = test_data
        self.test_labels = test_label

        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMAPSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        train_path = os.path.join(root_path, "SMAP_train.npy")
        test_path  = os.path.join(root_path, "SMAP_test.npy")
        label_path = os.path.join(root_path, "SMAP_test_label.npy")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_data = np.load(train_path)
            test_data  = np.load(test_path)
            test_label = np.load(label_path)
        else:
            train_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMAP/SMAP_train.npy",repo_type="dataset")
            test_path  = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMAP/SMAP_test.npy",repo_type="dataset")
            label_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMAP/SMAP_test_label.npy",repo_type="dataset")

            train_data  = np.load(train_path)
            test_data   = np.load(test_path)
            test_label = np.load(label_path)

        # 标准化
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data  = self.scaler.transform(test_data)

        self.train = train_data
        self.test  = test_data
        self.test_labels = test_label

        data_len = len(self.train)
        self.val = self.train[int(data_len * 0.8):]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SMDSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=100, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        
        train_path = os.path.join(root_path, "SMD_train.npy")
        test_path  = os.path.join(root_path, "SMD_test.npy")
        label_path = os.path.join(root_path, "SMD_test_label.npy")

        if all(os.path.exists(p) for p in [train_path, test_path, label_path]):
            train_data = np.load(train_path)
            test_data  = np.load(test_path)
            test_label = np.load(label_path)
        else:
            train_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMD/SMD_train.npy",repo_type="dataset")
            test_path  = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMD/SMD_test.npy",repo_type="dataset")
            label_path = hf_hub_download(repo_id=HUGGINGFACE_REPO, filename="SMD/SMD_test_label.npy",repo_type="dataset")

            train_data  = np.load(train_path)
            test_data   = np.load(test_path)
            test_label = np.load(label_path)
            
        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = test_label
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class SWATSegLoader(Dataset):
    def __init__(self, args, root_path, win_size, step=1, flag="train"):
        self.flag = flag
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()

        train2_path = os.path.join(root_path, "swat_train2.csv")
        test_path   = os.path.join(root_path, "swat2.csv")
        if all(os.path.exists(p) for p in [train2_path, test_path]):
            train_data = pd.read_csv(train2_path)
            test_data   = pd.read_csv(test_path)
        else:
            ds = load_dataset(HUGGINGFACE_REPO, name="SWaT")
            train_data = ds["train"].to_pandas()
            test_data  = ds["test"].to_pandas()
        labels = test_data.values[:, -1:]
        train_data = train_data.values[:, :-1]
        test_data = test_data.values[:, :-1]

        self.scaler.fit(train_data)
        train_data = self.scaler.transform(train_data)
        test_data = self.scaler.transform(test_data)
        self.train = train_data
        self.test = test_data
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = labels
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.flag == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.flag == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        index = index * self.step
        if self.flag == "train":
            return np.float32(self.train[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'val'):
            return np.float32(self.val[index:index + self.win_size]), np.float32(self.test_labels[0:self.win_size])
        elif (self.flag == 'test'):
            return np.float32(self.test[index:index + self.win_size]), np.float32(
                self.test_labels[index:index + self.win_size])
        else:
            return np.float32(self.test[
                              index // self.step * self.win_size:index // self.step * self.win_size + self.win_size]), np.float32(
                self.test_labels[index // self.step * self.win_size:index // self.step * self.win_size + self.win_size])


class UEAloader(Dataset):
    """
    Dataset class for datasets included in:
        Time Series Classification Archive (www.timeseriesclassification.com)
    Argument:
        limit_size: float in (0, 1) for debug
    Attributes:
        all_df: (num_samples * seq_len, num_columns) dataframe indexed by integer indices, with multiple rows corresponding to the same index (sample).
            Each row is a time step; Each column contains either metadata (e.g. timestamp) or a feature.
        feature_df: (num_samples * seq_len, feat_dim) dataframe; contains the subset of columns of `all_df` which correspond to selected features
        feature_names: names of columns contained in `feature_df` (same as feature_df.columns)
        all_IDs: (num_samples,) series of IDs contained in `all_df`/`feature_df` (same as all_df.index.unique() )
        labels_df: (num_samples, num_labels) pd.DataFrame of label(s) for each sample
        max_seq_len: maximum sequence (time series) length. If None, script argument `max_seq_len` will be used.
            (Moreover, script argument overrides this attribute)
    """

    def __init__(self, args, root_path, file_list=None, limit_size=None, flag=None):
        self.args = args
        self.root_path = root_path
        self.flag = flag
        self.all_df, self.labels_df = self.load_all(root_path, file_list=file_list, flag=flag)
        self.all_IDs = self.all_df.index.unique()  # all sample IDs (integer indices 0 ... num_samples-1)

        if limit_size is not None:
            if limit_size > 1:
                limit_size = int(limit_size)
            else:  # interpret as proportion if in (0, 1]
                limit_size = int(limit_size * len(self.all_IDs))
            self.all_IDs = self.all_IDs[:limit_size]
            self.all_df = self.all_df.loc[self.all_IDs]

        # use all features
        self.feature_names = self.all_df.columns
        self.feature_df = self.all_df

        # pre_process
        normalizer = Normalizer()
        self.feature_df = normalizer.normalize(self.feature_df)
        print(len(self.all_IDs))

    def _resolve_ts_path(self, root_path, dataset_name, flag):
        split = "TRAIN" if "train" in str(flag).lower() else "TEST"
        fname = f"{dataset_name}_{split}.ts"
        local = os.path.join(root_path, fname)
        if os.path.exists(local):
            return local
        return hf_hub_download(HUGGINGFACE_REPO, filename=f"{dataset_name}/{fname}", repo_type="dataset")

    def load_all(self, root_path, file_list=None, flag=None):
        """
        Loads datasets from ts files contained in `root_path` into a dataframe, optionally choosing from `pattern`
        Args:
            root_path: directory containing all individual .ts files
            file_list: optionally, provide a list of file paths within `root_path` to consider.
                Otherwise, entire `root_path` contents will be used.
        Returns:
            all_df: a single (possibly concatenated) dataframe with all data corresponding to specified files
            labels_df: dataframe containing label(s) for each sample
        """
        # Select paths for training and evaluation
        dataset_name = self.args.model_id
        ts_path = self._resolve_ts_path(root_path, dataset_name, flag or "train")

        all_df, labels_df = self.load_single(ts_path)
        return all_df, labels_df

    def load_single(self, filepath):
        df, labels = load_from_tsfile_to_dataframe(filepath, return_separate_X_and_y=True,
                                                             replace_missing_vals_with='NaN')
        labels = pd.Series(labels, dtype="category")
        self.class_names = labels.cat.categories
        labels_df = pd.DataFrame(labels.cat.codes,
                                 dtype=np.int8)  # int8-32 gives an error when using nn.CrossEntropyLoss

        lengths = df.applymap(
            lambda x: len(x)).values  # (num_samples, num_dimensions) array containing the length of each series

        horiz_diffs = np.abs(lengths - np.expand_dims(lengths[:, 0], -1))

        if np.sum(horiz_diffs) > 0:  # if any row (sample) has varying length across dimensions
            df = df.applymap(subsample)

        lengths = df.applymap(lambda x: len(x)).values
        vert_diffs = np.abs(lengths - np.expand_dims(lengths[0, :], 0))
        if np.sum(vert_diffs) > 0:  # if any column (dimension) has varying length across samples
            self.max_seq_len = int(np.max(lengths[:, 0]))
        else:
            self.max_seq_len = lengths[0, 0]

        # First create a (seq_len, feat_dim) dataframe for each sample, indexed by a single integer ("ID" of the sample)
        # Then concatenate into a (num_samples * seq_len, feat_dim) dataframe, with multiple rows corresponding to the
        # sample index (i.e. the same scheme as all datasets in this project)

        df = pd.concat((pd.DataFrame({col: df.loc[row, col] for col in df.columns}).reset_index(drop=True).set_index(
            pd.Series(lengths[row, 0] * [row])) for row in range(df.shape[0])), axis=0)

        # Replace NaN values
        grp = df.groupby(by=df.index)
        df = grp.transform(interpolate_missing)

        return df, labels_df

    def instance_norm(self, case):
        if self.root_path.count('EthanolConcentration') > 0:  # special process for numerical stability
            mean = case.mean(0, keepdim=True)
            case = case - mean
            stdev = torch.sqrt(torch.var(case, dim=1, keepdim=True, unbiased=False) + 1e-5)
            case /= stdev
            return case
        else:
            return case

    def __getitem__(self, ind):
        batch_x = self.feature_df.loc[self.all_IDs[ind]].values
        labels = self.labels_df.loc[self.all_IDs[ind]].values
        if self.flag == "TRAIN" and self.args.augmentation_ratio > 0:
            num_samples = len(self.all_IDs)
            num_columns = self.feature_df.shape[1]
            seq_len = int(self.feature_df.shape[0] / num_samples)
            batch_x = batch_x.reshape((1, seq_len, num_columns))
            batch_x, labels, augmentation_tags = run_augmentation_single(batch_x, labels, self.args)

            batch_x = batch_x.reshape((1 * seq_len, num_columns))

        return self.instance_norm(torch.from_numpy(batch_x)), \
               torch.from_numpy(labels)

    def __len__(self):
        return len(self.all_IDs)
