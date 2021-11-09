import math

import numpy as np
import matplotlib.pyplot as plt
import sol2
import sol1

def test_dft_idft():
    signal = np.cos(np.array((0., 30., 45., 60., 90.)) * np.pi / 180.) + \
             np.cos(2 * np.array((0., 30., 45., 60., 90.)) * np.pi / 180.)
    # np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    print(f"signal: {signal}\n")
    fourier_signal = sol2.DFT(signal)
    print(f"fourier signal: {fourier_signal}\n")
    signal = sol2.IDFT(fourier_signal)
    print(f"signal: {signal}\n")

def test_dft2_idft2(image):
    fourier_image = sol2.DFT2(image)
    # plt.imshow(np.log(1+np.abs(fourier_image)), cmap='gray')
    # plt.show()

    back_image = sol2.IDFT2(fourier_image)
    plt.imshow(np.real(back_image), cmap='gray')  # from tips: np.real (or np.real_if_close)change toll
    plt.show()


def test_conv_der(image):
    der_im = sol2.conv_der(image)
    plt.imshow(der_im, cmap='gray')
    plt.show()

def test_fourier_der(image):
    der_im = sol2.fourier_der(image)
    plt.imshow(np.real(der_im), cmap='gray')
    plt.show()

if __name__ == '__main__':
    """
    – np.fft.fft2, np.fft.ifft2 2D discrete Fast Fourier Transform (and
    inverse). You can use these functions to check your results from section
    1.1 and 1.2.
    """
    monkey = "ex1_presubmit/presubmit_externals/monkey.jpg"
    small = "ex2_presubmit/external/small_image.jpg"
    image = sol2.read_image(monkey, 1)
    small_image = sol2.read_image(small, 1)
    # test_dft_idft()
    # test_dft2_idft2(image)

    aria = "ex2_presubmit/external/aria_4kHz.wav"
    long_wav = "ex2_presubmit/external/f2btrop6.0.wav"
    # sol2.change_rate(aria, ratio=4)

    # sol2.change_samples(aria, ratio=2)

    # test_conv_der(image)

    a = np.array([1, 2, 3])
    b = np.array([1, 2, 3, 4, 5, 6]). reshape(2,3)
    c = np.array([1, 2])
    print(np.multiply(a, b))
    print(np.multiply(c[:, np.newaxis], b))
    test_fourier_der(image)



