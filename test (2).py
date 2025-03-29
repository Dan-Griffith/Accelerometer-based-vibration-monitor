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
DURATION = 60.0     # seconds
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

# --- Collect raw X, Y, Z data over time ---
def collect_data(n_samples, sample_rate):
    x_vals, y_vals, z_vals = [], [], []
    for _ in range(n_samples):
        x, y, z = read_axes()
        x_vals.append(x)
        y_vals.append(y)
        z_vals.append(z)
        time.sleep(1.0 / sample_rate)
    return np.array(x_vals), np.array(y_vals), np.array(z_vals)

# --- Compute normalized FFT and return frequency + amplitude ---
def compute_fft(signal, sample_rate):
    N = len(signal)
    signal = signal - np.mean(signal)  # Remove DC (offset/gravity)
    yf = fft(signal)
    xf = fftfreq(N, 1 / sample_rate)
    amplitudes = (2.0 / N) * np.abs(yf[0:N // 2])  # Normalized to g
    frequencies = xf[0:N // 2]
    return frequencies, amplitudes

# --- Convert acceleration amplitude to velocity (mm/s) ---
def compute_velocity_mm_s(frequencies, amplitudes):
    velocities = []
    for i, f in enumerate(frequencies):
        if f == 0:
            velocities.append(0)
            continue
        acc_ms2 = amplitudes[i] * 9.81  # g → m/s²
        v = acc_ms2 / (2 * math.pi * f)  # m/s
        velocities.append(v * 1000)      # mm/s
    return np.array(velocities)

# --- Report per-axis dominant frequency, amplitude, velocity ---
def report(axis, freqs, amps, vels):
    idx = np.argmax(amps)
    print(f"\n--- Axis: {axis} ---")
    print(f"Dominant Frequency: {freqs[idx]:.2f} Hz")
    print(f"Acceleration Amplitude: {amps[idx]:.4f} g")
    print(f"Particle Velocity: {vels[idx]:.2f} mm/s")

# --- Main Loop (Every Second) ---
if __name__ == "__main__":
    print("Running real-time vibration analysis...\nPress Ctrl+C to stop.\n")

    current_sample_rate = SAMPLE_RATE
    MIN_SAMPLE_RATE = 10
    MAX_SAMPLE_RATE = 1000
    margin = 1.2

    try:
        while True:
            n_samples = int(current_sample_rate * DURATION)
            print(f"\nSampling at {current_sample_rate:.2f} Hz (n_samples = {DURATION}")

            x_vals, y_vals, z_vals = collect_data(N_SAMPLES, current_sample_rate)

            max_dom_freq = 0
            for axis, data in zip(['X', 'Y', 'Z'], [x_vals, y_vals, z_vals]):
                freqs, amps = compute_fft(data, current_sample_rate)
                vels = compute_velocity_mm_s(freqs, amps)
                report(axis, freqs, amps, vels)
                idx = np.argmax(amps)
                max_dom_freq = max(max_dom_freq, freqs[idx]) # Dominant frequency across all axes
            
            if max_dom_freq == 0:
                desired_rate = current_sample_rate #no adjustment if no dominant frequency
            else:  
                desired_rate = 2 * max_dom_freq * margin
                desired_rate = max(MIN_SAMPLE_RATE, min(desired_rate, MAX_SAMPLE_RATE))

            #update with 80% of current rate, 20% of desired for smooth transition
            new_sample_rate = (0.8 * current_sample_rate) + (0.2 * desired_rate)
            print(f"\nMax dominant frequency: {max_dom_freq:.2f} Hz")
            print(f"Desired sample rate based on Nyquist: {desired_rate:.2f} Hz")
            print(f"Updating sample rate from {current_sample_rate:.2f} Hz to {new_sample_rate:.2f} Hz\n")

            current_sample_rate = new_sample_rate

            time.sleep(0.1)  # small pause before next scan
    except KeyboardInterrupt:
        print("\nStopped.")