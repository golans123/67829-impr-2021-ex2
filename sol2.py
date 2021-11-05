import math

import numpy as np
import scipy.io.wavfile
# for ex2_helper
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
# *********************** 1. DFT ********************************


# *********************** 1.1 1D DFT *****************************
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


# *********************** 1.2 2D DFT *****************************
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


# ********************************  2. Speech Fast Forward  ********************
# 2.1
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


# 2.2
def resize(data, ratio):
    """
    data is a 1D ndarray of dtype float64 or complex128(*) representing
    the original sample points, and this function should call DFT (1) and IDFT
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
    # todo: clip the high frequencies.
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
    # todo:
    # Note 1: In case of slowing down, we add the needed amount of zeros at the
    # high Fourier frequencies.

    # Note 2: In case you need to pad with zeros and you have 2 unequal sides,
    # you may choose which side to pad with the one extra 0.
    # example: an array with size 25, and ratio=0.5. pad with 25 zeros, you may
    # choose which side has 13 and which one has 12.

    # Note 3: In case you end up with a non-integer, floor it.
    return new_sample_points


# 2.3
def resize_spectrogram(data, ratio):
    """
    a function that speeds up a WAV file, without changing the pitch, using
    spectrogram scaling. This is done by computing the spectrogram, changing
    the number of spectrogram columns, and creating back the audio.

    use the provided functions stft and istft in order to transfer the data to
    the spectrogram and back. use the default parameters for win_length and
    hop_length.

    Each row in the spectrogram can be resized using resize according to ratio.
    Notice that while each row in the spectrogram should be resized correctly
    according to the ratio, the size of the rescaled 1D array will not be
    precisely accurate due to the window size.

    :param data: a 1D ndarray of dtype float64 representing the original sample
    points.
    :param ratio: a positive float64 representing the rate change of the WAV
    file. 0.25 < ratio < 4.
    :return:  return the new sample points according to ratio with the same
    datatype as data.
    """
    # transfer the data to the spectrogram
    transposed_stft_matrix = stft(y=data)

    # resize each row in the spectrogram using resize according to ratio.
    new_sample_points = np.zeros(len(transposed_stft_matrix))
    for i in range(len(transposed_stft_matrix)):
        new_sample_points[i] = resize(data, ratio)

    # transfer the data back from a spectrogram
    y_rec = istft(stft_matrix=new_sample_points)
    return y_rec


# 2.4
def resize_vocoder(data, ratio):
    """
    a function that speedups a WAV file by phase vocoding its spectrogram.
    Phase vocoding is the process of scaling the spectrogram, but includes the
    correction of the phases of each frequency according to the shift of each
    window.

    You can use the supplied function phase_vocoder(spec, ratio), which scales
    the spectrogram spec by ratio and corrects the phases. You may also use the
    function phase_vocoder from librosa, which has a different interface

    :param data: a 1D ndarray of dtype float64 representing the original sample
    points
    :param ratio: a positive float64 representing the rate change of the WAV
    file. 0.25 < ratio < 4.
    :return: the given data rescaled according to ratio with the same datatype
    as data.
    """












































# ************************* from ex2_helper ********************************
def stft(y, win_length=640, hop_length=160):
    fft_window = signal.windows.hann(win_length, False)

    # Window the time series.
    n_frames = 1 + (len(y) - win_length) // hop_length
    frames = [y[s:s + win_length] for s in np.arange(n_frames) * hop_length]

    stft_matrix = np.fft.fft(fft_window * frames, axis=1)
    return stft_matrix.T


def istft(stft_matrix, win_length=640, hop_length=160):
    n_frames = stft_matrix.shape[1]
    y_rec = np.zeros(win_length + hop_length * (n_frames - 1), dtype=np.float)
    ifft_window_sum = np.zeros_like(y_rec)

    ifft_window = signal.windows.hann(win_length, False)[:, np.newaxis]
    win_sq = ifft_window.squeeze() ** 2

    # invert the block and apply the window function
    ytmp = ifft_window * np.fft.ifft(stft_matrix, axis=0).real

    for frame in range(n_frames):
        frame_start = frame * hop_length
        frame_end = frame_start + win_length
        y_rec[frame_start: frame_end] += ytmp[:, frame]
        ifft_window_sum[frame_start: frame_end] += win_sq

    # Normalize by sum of squared window
    y_rec[ifft_window_sum > 0] /= ifft_window_sum[ifft_window_sum > 0]
    return y_rec


def phase_vocoder(spec, ratio):
    num_timesteps = int(spec.shape[1] / ratio)
    time_steps = np.arange(num_timesteps) * ratio

    # interpolate magnitude
    yy = np.meshgrid(np.arange(time_steps.size), np.arange(spec.shape[0]))[1]
    xx = np.zeros_like(yy)
    coordiantes = [yy, time_steps + xx]
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect', order=1).astype(np.complex)

    # phase vocoder
    # Phase accumulator; initialize to the first sample
    spec_angle = np.pad(np.angle(spec), [(0, 0), (0, 1)], mode='constant')
    phase_acc = spec_angle[:, 0]

    for (t, step) in enumerate(np.floor(time_steps).astype(np.int)):
        # Store to output array
        warped_spec[:, t] *= np.exp(1j * phase_acc)

        # Compute phase advance
        dphase = (spec_angle[:, step + 1] - spec_angle[:, step])

        # Wrap to -pi:pi range
        dphase = np.mod(dphase - np.pi, 2 * np.pi) - np.pi

        # Accumulate phase
        phase_acc += dphase

    return warped_spec

