import smbus
import time
import math
import numpy as np
from scipy.fft import fft, fftfreq

# --- ADXL345 I2C Setup ---
bus = smbus.SMBus(1)
address = 0x53
bus.write_byte_data(address, 0x2D, 0x08)  # Set to measurement mode

# --- Sampling Config ---
SAMPLE_RATE = 100  # Hz
DURATION = 1.0     # seconds
N_SAMPLES = int(SAMPLE_RATE * DURATION)

# --- Read acceleration from ADXL345 ---
def read_axes():
    data = bus.read_i2c_block_data(address, 0x32, 6)

    def convert(lo, hi):
        value = (hi << 8) | lo
        if value & (1 << 15):
            value -= (1 << 16)
        return value * 0.0039  # Convert to g

    x = convert(data[0], data[1])
    y = convert(data[2], data[3])
    z = convert(data[4], data[5])
    return x, y, z

# --- Sample acceleration and return magnitude array ---
def sample_acceleration(n_samples, sample_rate):
    samples = []
    for _ in range(n_samples):
        x, y, z = read_axes()
        magnitude = math.sqrt(x**2 + y**2 + z**2)
        samples.append(magnitude)
        time.sleep(1.0 / sample_rate)
    return np.array(samples)

# --- Perform FFT and return frequencies and acceleration amplitudes ---
def compute_fft(acc_samples, sample_rate):
    N = len(acc_samples)
    acc_samples = acc_samples - np.mean(acc_samples)  # Remove DC offset
    yf = fft(acc_samples)
    xf = fftfreq(N, 1 / sample_rate)
    amplitudes = np.abs(yf[0:N // 2])
    frequencies = xf[0:N // 2]
    return frequencies, amplitudes

# --- Calculate velocity from acceleration amplitude using FFT result ---
def compute_velocity_mm_s(frequencies, acc_amplitudes):
    velocities = []
    for i, f in enumerate(frequencies):
        if f == 0:
            velocities.append(0)
            continue
        acc_ms2 = acc_amplitudes[i] * 9.81  # Convert g → m/s²
        v = acc_ms2 / (2 * math.pi * f)     # m/s
        velocities.append(v * 1000)         # mm/s
    return np.array(velocities)

# --- Print dominant frequency and velocity ---
def print_vibration_report(freqs, amps, vels):
    max_amp_idx = np.argmax(amps)
    dominant_freq = freqs[max_amp_idx]
    dominant_velocity = vels[max_amp_idx]

    print(f"--- Vibration Analysis ---")
    print(f"Dominant Frequency: {dominant_freq:.2f} Hz")
    print(f"Acceleration Amplitude: {amps[max_amp_idx]:.4f} g")
    print(f"Particle Velocity: {dominant_velocity:.2f} mm/s")

# --- Main Execution ---
if __name__ == "__main__":
    acc_data = sample_acceleration(N_SAMPLES, SAMPLE_RATE)
    freqs, amps = compute_fft(acc_data, SAMPLE_RATE)
    vels = compute_velocity_mm_s(freqs, amps)
    print_vibration_report(freqs, amps, vels)