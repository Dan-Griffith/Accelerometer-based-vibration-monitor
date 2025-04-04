#source ./adxl/bin/activate
#python3 accelerometer.py

import smbus
import time
import sys
import microcontroller.pin as pin
import busio
import adafruit_adxl34x
import math
import numpy as np
from scipy.fft import fft, fftfreq


i2c = busio.I2C(pin.SCL, pin.SDA)
bus = smbus.SMBus(1)
accelerometer = adafruit_adxl34x.ADXL345(i2c)


bus.write_byte_data(0x53, 0x2C, 0x0B)
value = bus.read_byte_data(0x53, 0x31)
value &= ~0x0F;
value |= 0x0B;
value |= 0x08;
bus.write_byte_data(0x53, 0x31, value)
bus.write_byte_data(0x53, 0x2D, 0x08)

accelerometer.enable_motion_detection(threshold=20)
accelerometer.enable_tap_detection(tap_count=1, threshold=3, duration=100, latency=20, window=255)


def getAxes():
	bytes = bus.read_i2c_block_data(0x53, 0x32, 6)
	
	print(hex(bytes[0]), hex(bytes[1]), hex(bytes[2]), hex(bytes[3]), hex(bytes[4]), hex(bytes[5]))
	
	
	
	x = (bytes[1] << 8) | bytes[0]
	y = (bytes[3] << 8) | bytes[2]
	z = (bytes[5] << 8) | bytes[4]
	
	if x & (1 << 15): x -= (1 << 16)
	if y & (1 << 15): y -= (1 << 16)
	if z & (1 << 15): z -= (1 << 16)
	return x * 0.0039,y *0.0039, z*0.0039
	
#while True:
	#x,y,z = getAxes()
	#print(f"X = {x}, Y = {y}, Z = {z}")
	#print(getAxes())
	#print("%f %f %f"%accelerometer.acceleration)
	#print(accelerometer.events['motion'])
	#print("Tapped: %s"%accelerometer.events['tap'])
	#time.sleep(0.5)
	
	
# --- Sampling Config ---
INITIAL_SAMPLE_RATE = 60  # Hz
DURATION = 30    # seconds, longer duration gives better sampling accuracy
MIN_SAMPLE_RATE = 10 # Hz
MAX_SAMPLE_RATE = 250 # Hz
margin = 1.2 #use margin to avoid sampling too fast
current_sample_rate = INITIAL_SAMPLE_RATE


def collect_data(n_samples, sample_rate):
    x_vals, y_vals, z_vals = [], [], []
    for _ in range(n_samples):
        x, y, z = getAxes()
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
        velocities.append(v*1000)      # mm/s
    return np.array(velocities)

# --- Report per-axis dominant frequency, amplitude, velocity ---
def report(axis, freqs, amps, vels):
	idx = np.argmax(amps)
	
	bytes = bus.read_i2c_block_data(0x53, 0x32, 6)
	
	#print(hex(bytes[0]), hex(bytes[1]), hex(bytes[2]), hex(bytes[3]), hex(bytes[4]), hex(bytes[5]))
	print(f"\n--- Axis: {axis} ---", int(time.time()))
	print(f"Dominant Frequency: {freqs[idx]:.2f} Hz")
	print(f"Acceleration Amplitude: {amps[idx]:.4f} g")
	print(f"Particle Velocity: {vels[idx]:.2f} mm/s")
	with open("output.txt", "a") as file:
		file.write(f"\n--- Axis: {axis} --- {int(time.time())}")
		file.write(f"\nDominant Frequency: {freqs[idx]:.2f} Hz")
		file.write(f"\nAcceleration Amplitude: {amps[idx]:.4f} g")
		file.write(f"\nParticle Velocity: {vels[idx]:.2f} mm/s")
		file.write(f"\nHex: {hex(bytes[0]), hex(bytes[1]), hex(bytes[2]), hex(bytes[3]), hex(bytes[4]), hex(bytes[5])}")
		
		time.sleep(0.5)

# --- Main Loop (Every Second) ---
if __name__ == "__main__":
    print("Running real-time vibration analysis...\nPress Ctrl+C to stop.\n")
    try:
        while True:
            n_samples = int(current_sample_rate * DURATION)
            print(f"\nSampling at {current_sample_rate:.2f} Hz (n_samples = {n_samples})")

            x_vals, y_vals, z_vals = collect_data(n_samples, current_sample_rate)

            max_dom_freq = 0
            for axis, data in zip(['X', 'Y', 'Z'], [x_vals, y_vals, z_vals]):
                freqs, amps = compute_fft(data, current_sample_rate)
                vels = compute_velocity_mm_s(freqs, amps)
                report(axis, freqs, amps, vels)
                idx = np.argmax(amps)
                max_dom_freq = max(max_dom_freq, freqs[idx])

            if max_dom_freq ==0:
                desired_rate = current_sample_rate #no change if no new dominant frequency
            else:
                desired_rate = 2 * max_dom_freq * margin #update with nyquist sampling rate
                desired_rate = max(MIN_SAMPLE_RATE, min(desired_rate, MAX_SAMPLE_RATE))

            new_sample_rate = (0.8 * current_sample_rate) + (0.2 * desired_rate)
            print(f"\nMax dominant frequency: {max_dom_freq:.2f} Hz")
            print(f"Desired sample rate (Nyquist with margin): {desired_rate:.2f} Hz")
            print(f"Updating sample rate from {current_sample_rate:.2f} Hz to {new_sample_rate:.2f} Hz\n")
            current_sample_rate = new_sample_rate

            time.sleep(0.1)  # small pause before next scan
    except KeyboardInterrupt:
        print("\nStopped.")
