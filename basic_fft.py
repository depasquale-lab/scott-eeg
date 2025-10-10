import numpy as np
import matplotlib.pyplot as plt

# Sampling parameters
fs = 1000        # Sampling frequency (Hz)
T = 1            # Duration (seconds)
t = np.linspace(0, T, int(fs*T), endpoint=False)

# Frequencies of the three sine waves (Hz)
f1 = 50
f2 = 120
f3 = 300

# Generate the three sine waves
x1 = np.sin(2 * np.pi * f1 * t)
x2 = np.sin(2 * np.pi * f2 * t)
x3 = np.sin(2 * np.pi * f3 * t)

# Sum the signals
x = x1 + x2 + x3

# Compute FFT
N = len(x)
X = np.fft.fft(x)
freqs = np.fft.fftfreq(N, d=1/fs)

# Only keep the positive frequencies
pos_mask = freqs >= 0
freqs = freqs[pos_mask]
power = np.abs(X[pos_mask])**2 / N  # Power spectrum

# Plot the time-domain signal
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(t, x)
plt.title("Sum of Three Sine Waves (Time Domain)")
plt.xlabel("Time [s]")
plt.ylabel("Amplitude")

# Plot the power spectrum
plt.subplot(1, 2, 2)
plt.plot(freqs, power)
plt.title("Power Spectrum")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Power")
plt.xlim(0, 400)  # Limit x-axis for clarity
plt.tight_layout()
plt.show()
