import numpy as np
import matplotlib.pyplot as plt
from typing import List

MODULATOR_FREQUENCY = 1_000.0
CARRIER_FREQUENCY = 10_000.0
MODULATION_INDEX = 1.0

SAMPLES = 384_000.0


def low_pass_filter(signal: List[float], tau: float) -> List[float]:
    output = np.zeros_like(signal)

    output[0] = tau * signal[0]
    for i in range(1, len(signal)):
        output[i] = output[i - 1] + tau * (signal[i] - output[i - 1])

    return output


def diode(signal: List[float]) -> List[float]:
    return [abs(s) for s in signal]

def fm_signal() -> List[float]:
    time = np.arange(SAMPLES) / SAMPLES
    time = time / 20

    modulator = np.sin(2.0 * np.pi * MODULATOR_FREQUENCY * time) * MODULATION_INDEX
    carrier = np.sin(2.0 * np.pi * CARRIER_FREQUENCY * time)

    modulated = np.zeros_like(modulator)

    for i, t in enumerate(time):
        modulated[i] = np.sin(2.0 * np.pi * (CARRIER_FREQUENCY * t + modulator[i]))

    return modulated[50_000:100_000]

def fm_demodulate() -> List[float]:
    # Create an array of SAMPLES elements with their value constantly increasing
    time = np.arange(SAMPLES) / SAMPLES
    time = time / 20

    modulator = np.sin(2.0 * np.pi * MODULATOR_FREQUENCY * time) * MODULATION_INDEX
    carrier = np.sin(2.0 * np.pi * CARRIER_FREQUENCY * time)

    modulated = np.zeros_like(modulator)

    for i, t in enumerate(time):
        modulated[i] = np.sin(2.0 * np.pi * (CARRIER_FREQUENCY * t + modulator[i]))

    tau = 1 / (2 * np.pi * CARRIER_FREQUENCY * 0.7)
    filtered = low_pass_filter(modulated, tau)

    positive = np.zeros_like(filtered)

    for i, value in enumerate(filtered):
        positive[i] = abs(value)

    tau = 1 / (2 * np.pi * MODULATOR_FREQUENCY * 3)
    envelope = low_pass_filter(positive, tau)

    envelope -= 0.00233
    envelope *= 0.82 * 10**4

    return envelope[50_000:100_000]
