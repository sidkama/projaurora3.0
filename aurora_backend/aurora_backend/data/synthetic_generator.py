from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class SyntheticEEGGenerator:
    """Generate synthetic EEG-like and biometric signals.

    This is not intended to be physiologically exact, but to provide:
    - Distinct spectral patterns per task label
    - Simple heart-rate and EDA-like channels
    so the Aurora pipeline has meaningful structure to learn.
    """

    sample_rate_hz: int
    window_seconds: float
    eeg_channels: int
    bio_channels: int = 2
    random_state: int | None = None

    def __post_init__(self) -> None:
        self.window_samples = int(self.sample_rate_hz * self.window_seconds)
        self.rng = np.random.default_rng(self.random_state)
        self.task_labels: List[str] = [
            "rest",
            "motor_imagery",
            "visuospatial",
            "working_memory",
        ]

    # ------------------------------------------------------------------
    # Task profiles
    # ------------------------------------------------------------------
    def _task_profile(self, task_label: str) -> dict:
        """Return simple parameters for the given task.

        alpha ~ 10 Hz, beta ~ 20 Hz, gamma ~ 40 Hz.
        """
        if task_label == "rest":
            return dict(
                alpha_amp=1.0,
                beta_amp=0.3,
                gamma_amp=0.1,
                hr_base=70.0,
                hr_var=1.0,
                eda_trend=0.1,
                eda_spike_prob=0.01,
            )
        if task_label == "motor_imagery":
            return dict(
                alpha_amp=0.8,
                beta_amp=1.2,
                gamma_amp=0.3,
                hr_base=80.0,
                hr_var=2.5,
                eda_trend=0.15,
                eda_spike_prob=0.03,
            )
        if task_label == "visuospatial":
            return dict(
                alpha_amp=0.7,
                beta_amp=0.8,
                gamma_amp=1.2,
                hr_base=78.0,
                hr_var=2.0,
                eda_trend=0.2,
                eda_spike_prob=0.04,
            )
        if task_label == "working_memory":
            return dict(
                alpha_amp=0.6,
                beta_amp=0.9,
                gamma_amp=0.9,
                hr_base=82.0,
                hr_var=3.0,
                eda_trend=0.25,
                eda_spike_prob=0.05,
            )

        # Unknown label → rest-like
        return dict(
            alpha_amp=1.0,
            beta_amp=0.3,
            gamma_amp=0.1,
            hr_base=70.0,
            hr_var=1.0,
            eda_trend=0.1,
            eda_spike_prob=0.01,
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample_window(
        self, task_label: str | None = None
    ) -> Tuple[np.ndarray, np.ndarray, str]:
        """Sample one window of synthetic data.

        Returns
        -------
        eeg : np.ndarray, shape (eeg_channels, T)
        bio : np.ndarray, shape (bio_channels, T)
        task_label : str
        """
        if task_label is None:
            task_label = str(self.rng.choice(self.task_labels))

        profile = self._task_profile(task_label)
        T = self.window_samples
        t = np.arange(T, dtype=np.float32) / float(self.sample_rate_hz)

        eeg_channels = []
        for ch in range(self.eeg_channels):
            phase_alpha = self.rng.uniform(0, 2 * np.pi)
            phase_beta = self.rng.uniform(0, 2 * np.pi)
            phase_gamma = self.rng.uniform(0, 2 * np.pi)

            alpha = profile["alpha_amp"] * np.sin(2 * np.pi * 10.0 * t + phase_alpha)
            beta = profile["beta_amp"] * np.sin(2 * np.pi * 20.0 * t + phase_beta)
            gamma = profile["gamma_amp"] * np.sin(2 * np.pi * 40.0 * t + phase_gamma)

            noise = 0.3 * self.rng.standard_normal(T)
            eeg_channels.append(alpha + beta + gamma + noise)

        eeg = np.stack(eeg_channels, axis=0)

        # Heart rate: base + slow sinusoid + noise
        hr_t = np.linspace(0.0, self.window_seconds, T, dtype=np.float32)
        hr = (
            profile["hr_base"]
            + profile["hr_var"] * np.sin(2 * np.pi * hr_t / self.window_seconds)
            + 0.5 * self.rng.standard_normal(T)
        )

        # EDA: slow drift + occasional spikes
        eda = (
            profile["eda_trend"]
            + 0.05
            * np.cumsum(self.rng.standard_normal(T)).astype(np.float32)
            / np.sqrt(max(T, 1))
        )
        spikes = (self.rng.random(T) < profile["eda_spike_prob"]).astype(np.float32)
        eda = eda + 0.5 * spikes

        if self.bio_channels == 1:
            bio = hr[None, :]
        else:
            bio = np.stack([hr, eda], axis=0)

        return eeg.astype(np.float32), bio.astype(np.float32), task_label

    def generate_dataset(
        self, n_per_task: int = 50
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """Generate a labeled dataset for all tasks.

        Returns
        -------
        eeg_data : (N, C_eeg, T)
        bio_data : (N, C_bio, T)
        labels : list[str] of length N
        """
        eeg_list = []
        bio_list = []
        labels: List[str] = []
        for label in self.task_labels:
            for _ in range(n_per_task):
                eeg, bio, _ = self.sample_window(label)
                eeg_list.append(eeg)
                bio_list.append(bio)
                labels.append(label)

        eeg_data = np.stack(eeg_list, axis=0)
        bio_data = np.stack(bio_list, axis=0)
        return eeg_data, bio_data, labels
