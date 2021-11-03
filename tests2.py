import numpy as np

import sol2

if __name__ == '__main__':
    signal = np.cos(np.array((0., 30., 45., 60., 90.)) * np.pi / 180.) + \
             np.cos(2 * np.array((0., 30., 45., 60., 90.)) * np.pi / 180.)
    # np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    print(f"signal: {signal}\n")
    fourier_signal = sol2.DFT(signal)
    print(f"fourier signal: {fourier_signal}\n")
    signal = sol2.IDFT(fourier_signal)
    print(f"signal: {signal}\n")

