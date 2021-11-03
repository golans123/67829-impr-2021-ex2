import math

import numpy as np
import scipy.io.wavfile
# *********************** DFT ********************************


# *********************** 1D DFT *****************************
def dft_idft_helper(signal, exp_power_coefficient, transform_denominator):
    """

    :param signal:
    :param exp_power_coefficient: -2 for dft, 2 for idft
    :param transform_denominator: 1 for dft, (1/len(signal)) for idft
    :return:
    """
    signal = np.array(signal, dtype=np.complex128)  # = f(x) / F(u)
    N = len(signal)
    transform = np.array(signal, dtype=np.complex128)
    x = np.arange(N)  # [0,...,N-1]
    exp_power = (exp_power_coefficient * math.pi * 1j * x) / N
    for i in range(N):  # x / u
        exp_part = np.exp(exp_power * i)
        transform[i] = np.matmul(signal, exp_part) / transform_denominator
    return transform


def DFT(signal):
    """
    Discrete Fourier Transform. should be implemented without the use of loops.

    when the fourier_signal is transformed into a real signal you can expect
    IDFT to return real values as well, although it may return with a tiny
    imaginary part. You can ignore the imaginary part.

    :param signal: an array of dtype float64 with shape (N,) or (N,1)
    :return: the complex Fourier signal and complex signal, both of the same
    shape as the input. dtype complex 128
    """
    fourier_transform = dft_idft_helper(signal, -2, 1)
    return fourier_transform


def IDFT(fourier_signal):
    """
    Inverse DFT. should be implemented without the use of loops.
    :param fourier_signal: an array of dtype complex128 with shape (N,) or (N,1)
    :return: the original signal, of the same shape as an input's element.
    dtype complex 128
    """
    fourier_signal = np.array(fourier_signal)  # = F(u)
    N = len(fourier_signal)
    # -(2*pi*i*u*x)/N
    inverse_fourier_transform = dft_idft_helper(fourier_signal, 2, N)
    # signal = inverse_fourier_transform.real
    # from tips: np.real (or np.real_if_close)
    return np.real_if_close(inverse_fourier_transform)  # todo: which to use?


# *********************** 2D DFT *****************************
def DFT2(image):
    """
    DFT in 2D
    :param image: a grayscale image of dtype float64
    :return: fourier_image, shape should be the same as the shape of the input.
    """
    fourier_image = np.zeros(len(image))
    for i in range(len(image)):
        fourier_image[i] = DFT(image[i])
    for i in range(len(fourier_image[0])):
        fourier_image[i] = DFT(fourier_image[i])  # fixme: change to column-wise
    return fourier_image


def IDFT2(fourier_image):
    """
    Inverse DFT in 2D
    :param fourier_image: a 2D array of dtype complex128, both of shape
    (M,N) or (M,N,1).
    :return: image, the origin of fourier_image is a real image transformed
    with DFT2 you can expect the returned image to be real valued.
    """
    image = np.zeros(len(fourier_image))
    for i in range(len(fourier_image)):
        image[i] = IDFT(fourier_image[i])
    for i in range(len(image[0])):
        image[i] = DFT(image[i])  # fixme: change to column-wise
    return image


# ********************************  Speech Fast Forward  ********************
def change_rate(filename, ratio):
    """
     Fast forward by rate change. changes the duration of an audio file by
     keeping the same samples, but changing the sample rate written in the
     file header.
     This function saves the audio in a new file called change_rate.wav.
     You can use the functions read, write from scipy.io.wavfile.
    :param filename: a string representing the path to a WAV file
    :param ratio: a positive float64 representing the duration change. You
     may assume that 0.25 < ratio < 4 .
    :return: None
    """
    rate, data = scipy.io.wavfile.read(filename)
    scipy.io.wavfile.write(filename="change_rate.wav", rate=rate*ratio,
                           data=data)


def resize(data, ratio):
    """
    where: data is a 1D ndarray of dtype float64 or complex128(*) representing
    the original sample points, and  this function should call DFT (1) and IDFT
    (2). In Fourier representation, you can use np.fft.fftshift (and
    np.fft.ifftshift later) in order to shift the zero-frequency component to
    the center of the spectrum before clipping the high frequencies.
    :param data:
    :param ratio:
    :return: the returned value of resize is a 1D ndarray of the dtype of data
    representing the new sample points.
    """
    # this function should call DFT (1) and IDFT (2)
    fourier_signal = DFT(data)
    # shift the zero-frequency component to the center of the spectrum
    fourier_signal = np.fft.fftshift(fourier_signal)
    # clip the high frequencies.
    fourier_signal = fourier_signal[:]
    # shift back frequencies order
    fourier_signal = np.fft.ifftshift(fourier_signal)
    # convert back to a signal
    new_sample_points = IDFT(fourier_signal)
    return new_sample_points


def change_samples(filename, ratio):
    """
    that changes the duration of an audio file by reducing the number of
    samples using Fourier. This function does not change the sample rate of
    the given file.
    The result should be saved in a file called change_samples.wav.

    This function will call the function resize(data, ratio) to change the
    number of samples by the given ratio

    :param filename: a string representing the path to a WAV file
    :param ratio:  a positive float64 representing the duration change.
    You may assume that 0.25 < ratio < 4.
    :return: The function should return a 1D ndarray of dtype float64
    representing the new sample points
    """
    rate, data = scipy.io.wavfile.read(filename)
    new_sample_points = resize(data, ratio)
    scipy.io.wavfile.write(filename="change_rate.wav", rate=rate,
                           data=new_sample_points)
    return new_sample_points
