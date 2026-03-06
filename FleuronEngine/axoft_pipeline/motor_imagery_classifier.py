"""
Motor Imagery Classifier for BCI Pipeline
================================================

Classifies left vs right hand motor imagery using mu rhythm (8-12Hz) suppression.

Motor imagery detection relies on the phenomenon of Event-Related Desynchronization (ERD):
- When imagining LEFT hand movement: C4 (right motor cortex) shows mu suppression
- When imagining RIGHT hand movement: C3 (left motor cortex) shows mu suppression
- At rest: Both C3 and C4 show normal mu rhythm power

This classifier uses the asymmetry between C3 and C4 mu band power to classify
motor imagery in real-time.

Reference: Pfurtscheller, G., & Lopes da Silva, F. H. (1999).
"""

import numpy as np
from typing import Dict, Optional, Tuple


def extract_mu_power(signal: np.ndarray, sample_rate: float = 160.0) -> float:
    """
    Extract 8-12Hz mu band power from single channel.

    Uses cascaded IIR bandpass filtering for O(n) complexity.

    Args:
        signal: EEG signal from one channel
        sample_rate: Sampling rate in Hz

    Returns:
        Power (variance) in the mu band
    """
    # Highpass at 8Hz
    RC_hp = 1.0 / (2.0 * np.pi * 8.0)
    dt = 1.0 / sample_rate
    alpha_hp = RC_hp / (RC_hp + dt)

    hp = np.zeros(len(signal), dtype=np.float32)
    hp[0] = signal[0]
    for i in range(1, len(signal)):
        hp[i] = alpha_hp * (hp[i-1] + signal[i] - signal[i-1])

    # Lowpass at 12Hz
    RC_lp = 1.0 / (2.0 * np.pi * 12.0)
    alpha_lp = dt / (RC_lp + dt)

    lp = np.zeros(len(hp), dtype=np.float32)
    lp[0] = hp[0]
    for i in range(1, len(hp)):
        lp[i] = alpha_lp * hp[i] + (1 - alpha_lp) * lp[i-1]

    # Return power (variance)
    return float(np.var(lp))


class MotorImageryClassifier:
    """
    Simple mu-suppression based motor imagery classifier.

    Uses C3/C4 asymmetry in the 8-12Hz mu band to classify
    left vs right hand motor imagery.

    The classifier works by:
    1. Extracting mu band (8-12Hz) power from C3 and C4
    2. Computing the asymmetry: (C3 - C4) / (C3 + C4)
    3. Classifying based on threshold:
       - Positive asymmetry (C3 > C4) -> RIGHT_HAND (left cortex suppressed)
       - Negative asymmetry (C3 < C4) -> LEFT_HAND (right cortex suppressed)
       - Near zero -> REST
    """

    def __init__(self, sample_rate: float = 160.0):
        """
        Initialize the classifier.

        Args:
            sample_rate: EEG sampling rate in Hz
        """
        self.sample_rate = sample_rate
        self.baseline_c3 = None
        self.baseline_c4 = None
        self.threshold = 0.15  # Asymmetry threshold for classification
        self.is_calibrated = False

        # History for smoothing predictions
        self.history = []
        self.history_length = 5

    def calibrate(self, baseline_c3: np.ndarray, baseline_c4: np.ndarray):
        """
        Calibrate baseline mu power from resting state data.

        Args:
            baseline_c3: Resting state C3 signal (>= 1 second recommended)
            baseline_c4: Resting state C4 signal (>= 1 second recommended)
        """
        self.baseline_c3 = extract_mu_power(baseline_c3, self.sample_rate)
        self.baseline_c4 = extract_mu_power(baseline_c4, self.sample_rate)
        self.is_calibrated = True

    def classify(self, signal_c3: np.ndarray, signal_c4: np.ndarray) -> Dict:
        """
        Classify motor imagery from C3/C4 signals.

        Args:
            signal_c3: EEG signal from C3 electrode (left motor cortex)
            signal_c4: EEG signal from C4 electrode (right motor cortex)

        Returns:
            dict with keys:
            - prediction: 'LEFT_HAND', 'RIGHT_HAND', or 'REST'
            - confidence: 0.0 to 1.0
            - c3_mu_power: Mu band power at C3
            - c4_mu_power: Mu band power at C4
            - c3_suppression: Suppression relative to baseline (if calibrated)
            - c4_suppression: Suppression relative to baseline (if calibrated)
            - asymmetry: (C3 - C4) / (C3 + C4)
        """
        # Extract mu band power
        mu_c3 = extract_mu_power(signal_c3, self.sample_rate)
        mu_c4 = extract_mu_power(signal_c4, self.sample_rate)

        # Calculate suppression relative to baseline (if calibrated)
        if self.is_calibrated and self.baseline_c3 > 0 and self.baseline_c4 > 0:
            c3_suppression = (self.baseline_c3 - mu_c3) / self.baseline_c3
            c4_suppression = (self.baseline_c4 - mu_c4) / self.baseline_c4
        else:
            c3_suppression = 0.0
            c4_suppression = 0.0

        # Calculate asymmetry (key classification feature)
        # Avoid division by zero
        total_power = mu_c3 + mu_c4
        if total_power < 1e-10:
            asymmetry = 0.0
        else:
            asymmetry = (mu_c3 - mu_c4) / total_power

        # Classify based on asymmetry
        if asymmetry > self.threshold:
            # C3 > C4: Left motor cortex less suppressed
            # This means RIGHT motor cortex is more suppressed
            # ERD in right cortex -> LEFT hand imagination
            prediction = "LEFT_HAND"
            confidence = min(abs(asymmetry) / 0.3, 1.0)
        elif asymmetry < -self.threshold:
            # C4 > C3: Right motor cortex less suppressed
            # This means LEFT motor cortex is more suppressed
            # ERD in left cortex -> RIGHT hand imagination
            prediction = "RIGHT_HAND"
            confidence = min(abs(asymmetry) / 0.3, 1.0)
        else:
            prediction = "REST"
            confidence = 1.0 - abs(asymmetry) / self.threshold

        # Add to history for temporal smoothing
        self.history.append(prediction)
        if len(self.history) > self.history_length:
            self.history.pop(0)

        # Get smoothed prediction (majority vote)
        if len(self.history) >= 3:
            from collections import Counter
            vote_count = Counter(self.history)
            smoothed_prediction = vote_count.most_common(1)[0][0]
        else:
            smoothed_prediction = prediction

        return {
            'prediction': prediction,
            'smoothed_prediction': smoothed_prediction,
            'confidence': confidence,
            'c3_mu_power': mu_c3,
            'c4_mu_power': mu_c4,
            'c3_suppression': c3_suppression,
            'c4_suppression': c4_suppression,
            'asymmetry': asymmetry,
            'is_calibrated': self.is_calibrated
        }

    def reset(self):
        """Reset classifier state (but keep calibration)."""
        self.history = []

    def set_threshold(self, threshold: float):
        """
        Adjust classification threshold.

        Args:
            threshold: New asymmetry threshold (0.1 to 0.3 typical)
        """
        self.threshold = max(0.05, min(0.5, threshold))


# Test function
def test_classifier():
    """Quick test of the motor imagery classifier."""
    np.random.seed(42)

    # Create synthetic C3/C4 signals
    # Simulate RIGHT hand imagery: suppress LEFT motor cortex (C3)
    t = np.linspace(0, 1, 160)  # 1 second at 160Hz

    # C3: Suppressed mu (low amplitude 10Hz)
    c3_signal = 5 * np.sin(2 * np.pi * 10 * t) + np.random.randn(160) * 2

    # C4: Normal mu (higher amplitude 10Hz)
    c4_signal = 15 * np.sin(2 * np.pi * 10 * t) + np.random.randn(160) * 2

    classifier = MotorImageryClassifier(sample_rate=160.0)
    result = classifier.classify(c3_signal, c4_signal)

    print("Motor Imagery Classification Test")
    print("=" * 40)
    print(f"Prediction: {result['prediction']}")
    print(f"Confidence: {result['confidence']:.2%}")
    print(f"C3 mu power: {result['c3_mu_power']:.2f}")
    print(f"C4 mu power: {result['c4_mu_power']:.2f}")
    print(f"Asymmetry: {result['asymmetry']:.3f}")
    print(f"(Expected: RIGHT_HAND because C3 < C4)")


if __name__ == "__main__":
    test_classifier()
