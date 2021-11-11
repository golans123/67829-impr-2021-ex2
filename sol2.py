import math
import numpy as np
import scipy.io.wavfile
# for ex2_helper
from scipy import signal
from scipy.ndimage.interpolation import map_coordinates
# for sol1 read_image
import imageio
import skimage.color
# NUM_SHADES = 256
MAX_SHADE_VAL = 255
GRAY_REPRESENTATION = 1
# RGB_REPRESENTATION = 2
# *********************** 1. DFT ********************************


# *********************** 1.1 1D DFT *****************************
def dft_idft_helper(signal, exp_power_coefficient, transform_denominator):
    """
    dtype float64 with shape (N,) or (N,1)
    :param signal:= f(x) / F(u)
    :param exp_power_coefficient: -2 for dft, 2 for idft
    :param transform_denominator: 1 for dft, (1/len(signal)) for idft
    :return:
    """
    # dtype float64 with shape (N,) or (N,1)
    signal = signal.reshape(signal.shape[0])
    N = len(signal)
    a = np.arange(N)
    b = a.reshape((N, 1))
    dft_matrix = np.exp(exp_power_coefficient * 1j * np.pi * b * a / N)
    transform = np.matmul(dft_matrix, signal) / transform_denominator
    return np.array(transform).astype(np.complex128)


def DFT(signal):
    """
    Discrete Fourier Transform. should be implemented without the use of loops.

    :param signal: an array of dtype float64 with shape (N,) or (N,1)
    :return: the complex Fourier signal and complex signal, both of the same
    shape as the input. dtype complex 128
    """
    # -(2*pi*i*u*x)/N
    fourier_transform = dft_idft_helper(signal, -2, 1)
    return fourier_transform.reshape(signal.shape)


def IDFT(fourier_signal):
    """
    Inverse DFT. should be implemented without the use of loops.

    when the fourier_signal is transformed into a real signal you can expect
    IDFT to return real values as well, although it may return with a tiny
    imaginary part. You can ignore the imaginary part.

    :param fourier_signal: an array of dtype complex128 with shape (N,) or
    (N,1)
    :return: the original signal, of the same shape as an input's element.
    dtype complex 128
    """
    N = len(fourier_signal)
    # (2*pi*i*u*x)/N
    inverse_fourier_transform = dft_idft_helper(fourier_signal, 2, N)
    return inverse_fourier_transform.reshape(fourier_signal.shape)


# *********************** 1.2 2D DFT *****************************
def dft2_idft2_helper(image, func):
    """
    if func = DFT does DFT2 if func = IDFT does IDFT2.
    :param image: image for DFT2/IDFT2
    :param func: DFT or IDFT
    :return: transformed image
    """
    image = image.reshape(image.shape[0], image.shape[1])
    transform = np.zeros((len(image), len(image[0])), dtype=np.complex128)
    for i in range(len(image)):
        transform[i, :] = func(image[i])
    for i in range(len(transform[0])):
        transform[:, i] = func(transform[:, i])  # column-wise
    return transform


def DFT2(image):
    """
    DFT in 2D
    :param image: a grayscale image of dtype float64
    :return: fourier_image, shape should be the same as the shape of the input.
    """
    return dft2_idft2_helper(image, DFT).reshape(image.shape)


def IDFT2(fourier_image):
    """
    Inverse DFT in 2D
    :param fourier_image: a 2D array of dtype complex128, both of shape
    (M,N) or (M,N,1).
    :return: image, the origin of fourier_image is a real image transformed
    with DFT2 you can expect the returned image to be real valued.
    """
    return dft2_idft2_helper(fourier_image, IDFT).reshape(fourier_image.shape)


# ********************************  2. Speech Fast Forward  *******************
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
    scipy.io.wavfile.write(filename="change_rate.wav", rate=np.int(
        np.around(rate*ratio)), data=data)


# 2.2
def clip_or_pad(fourier_signal, ratio):
    """
    clip or pad to correctly resize signal (avoid artifacts due to resampling).
    Note 1: In case of slowing down, we add the needed amount of zeros at the
    high Fourier frequencies.

    Note 2: In case you need to pad with zeros and you have 2 unequal sides,
    you may choose which side to pad with the one extra 0.
    example: an array with size 25, and ratio=0.5. pad with 25 zeros, you may
    choose which side has 13 and which one has 12.

    Note 3: In case you end up with a non-integer, floor it.
    :param fourier_signal: centered fourier signal
    :param ratio:
    :return:
    """
    fourier_signal_len = len(fourier_signal)
    if ratio > 1:  # clip
        clip_amount = fourier_signal_len - math.floor(fourier_signal_len/ratio)
        fourier_signal = fourier_signal[math.floor(clip_amount/2):
                                        -math.ceil(clip_amount/2)]
    elif ratio < 1:  # pad
        pad_amount = math.floor(fourier_signal_len/ratio - fourier_signal_len)
        fourier_signal = np.pad(fourier_signal, (math.floor(pad_amount/2),
                                                 math.ceil(pad_amount/2)),
                                "constant")
    return fourier_signal


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
    # clip the high frequencies / pad with zeros.
    # effectively does resampling in the frequency domain
    fourier_signal = clip_or_pad(fourier_signal, ratio)
    # shift back frequencies order
    fourier_signal = np.fft.ifftshift(fourier_signal)
    new_sample_points = IDFT(fourier_signal)
    return new_sample_points


  # todo: review all funcs' return types
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
    new_sample_points = np.real(resize(data, ratio))
    scipy.io.wavfile.write(filename="change_samples.wav", rate=rate,
                           data=new_sample_points)
    return new_sample_points.astype(np.float64)


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
    spectrogram = stft(y=data)
    # resize each row in the spectrogram using resize according to ratio.
    new_spectrogram = []
    for i in range(len(spectrogram)):
        new_spectrogram.insert(len(new_spectrogram), resize(spectrogram[i],
                                                            ratio))
    new_spectrogram = np.array(new_spectrogram)
    # transfer the data back from a spectrogram
    new_sample_points = istft(stft_matrix=new_spectrogram)
    return new_sample_points


# 2.4
def resize_vocoder(data, ratio):
    """
    a function that speedups a WAV file by phase vocoding its spectrogram.
    Phase vocoding is the process of scaling the spectrogram, but includes the
    correction of the phases of each frequency according to the shift of each
    window.

    You can use the supplied function phase_vocoder(spec, ratio), which scales
    the spectrogram spec by ratio and corrects the phases.

    :param data: a 1D ndarray of dtype float64 representing the original sample
    points
    :param ratio: a positive float64 representing the rate change of the WAV
    file. 0.25 < ratio < 4.
    :return: the given data rescaled according to ratio with the same datatype
    as data.
    """
    # transfer the data to the spectrogram
    spectrogram = stft(y=data)
    # scales the spectrogram spec by ratio and corrects the phases
    warped_spec = phase_vocoder(spec=spectrogram, ratio=ratio)
    rescaled_data = istft(stft_matrix=warped_spec)
    return rescaled_data


# ************************* 3 Image derivatives *****************************
# In both sections (3.1 and 3.2) you should not normalize the magnitude values
# to be in the range of [0,1].
# 3.1 Image derivatives in image space
def conv_der(im):
    """
    a function that computes the magnitude of image derivatives. It derives the
    image in each direction separately (vertical and horizontal) using simple
    convolution with [0.5, 0, −0.5] as a row and column vectors. Next, it uses
    these derivative images to compute the magnitude image.

    - scipy.signal.convolve2d 2D convolution – use the ’same’ option when you
    want the output to have the same size as the input

    – np.meshgrid used to create index maps, you can use np.arange instead,
    and perform the same operations via broadcasing

    - np.complex128 dtype of array with complex numbers.

    :param im: a grayscale images of type float64.
    :return: the magnitude of the derivative, with the same dtype and shape as
    the input.
    """
    horizonal_derivative_kernel = np.array([[0.5, 0, -0.5]])
    vertical_derivative_kernel = np.array([[0.5], [0], [-0.5]])
    # derive the image in each direction separately (vertical and horizontal)
    im_post_dx = scipy.signal.convolve2d(in1=im,
                                         in2=horizonal_derivative_kernel,
                                         mode='same')
    im_post_dy = scipy.signal.convolve2d(in1=im,
                                         in2=vertical_derivative_kernel,
                                         mode='same')
    # The output should be calculated in the following way:
    magnitude = np.sqrt(np.abs(im_post_dx)**2 + np.abs(im_post_dy)**2)
    return np.array(magnitude).astype(np.float64)


# 3.2 Image derivatives in Fourier space
def derive_image_by_axis(image, shape_index):
    """
    applying the equation from class for derivative calculation.
    :param image:
    :param shape_index:
    :return:
    """
    # dft2
    fourier_image = DFT2(image)
    # center the (U,V)=(0,0) frequency
    fourier_image = np.fft.fftshift(fourier_image)
    # derive dft
    N = fourier_image.shape[shape_index]
    # multiply the frequencies in the range [−N/2, ..., N/2]
    if shape_index == 0:
        u = np.arange(-(np.floor(N / 2)), (np.ceil(N / 2))).astype(np.int64)
        fourier_image = np.multiply(u[:, np.newaxis], fourier_image)
    else:  # shape_index == 1
        v = np.arange(-(np.floor(N / 2)), (np.ceil(N / 2))).astype(np.int64)
        fourier_image = np.multiply(v, fourier_image)
    # shifting back
    fourier_image = np.fft.ifftshift(fourier_image)
    # idft2
    axis_derived_image = ((2 * np.pi * 1j) / fourier_image.shape[1]) * \
                         IDFT2(fourier_image)
    return axis_derived_image


def fourier_der(im):
    """
    a function that computes the magnitude of the image derivatives using
    Fourier transform. Use DFT, IDFT, and the equations from class, to compute
    derivatives in the x and y directions. Use np.fft.fftshift in the frequency
    domain so that the (U,V)=(0,0) frequency will be at the center of the
    image, and multiply the frequencies in the range [−N/2, ..., N/2] before
    shifting back.
    note: You may not assume the image is square.
    :param im: a float64 grayscale image.
    :return: a float64 grayscale image.
    """
    # compute derivatives in the x and y directions (DFT, IDFT, and the
    # equations from class)
    x_derived_image = derive_image_by_axis(im, shape_index=0)
    # derive y
    y_derived_image = derive_image_by_axis(im, shape_index=1)
    magnitude = np.sqrt(np.abs(x_derived_image) ** 2 + np.abs(y_derived_image)
                        ** 2)
    return np.array(magnitude).astype(np.float64)


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
    warped_spec = map_coordinates(np.abs(spec), coordiantes, mode='reflect',
                                  order=1).astype(np.complex)

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


# ********************** from sol1 ****************************************
def read_image(filename, representation):
    """
    a function which reads an image file and converts it into a given
    representation.
    :param filename: the filename of an image on disk (could be grayscale or
    RGB).
    :param representation: a grayscale image (1) or an RGB image (2).
    :return: an image represented by a matrix of type np.float64 with
    intensities (either grayscale or RGB channel intensities)
    normalized to the range [0; 1].

    You will find the function rgb2gray from the module skimage.color useful,
    as well as imread from
    imageio. We won't ask you to convert a grayscale image to RGB.
    """
    image = imageio.imread(filename).astype(np.float64)
    if representation == GRAY_REPRESENTATION:
        image = skimage.color.rgb2gray(image)
    # normalize intensities
    return image / MAX_SHADE_VAL
