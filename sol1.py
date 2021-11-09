import numpy as np
import matplotlib.pyplot as plt
import imageio
import skimage.color


NUM_SHADES = 256
MAX_SHADE_VAL = 255
GRAY_REPRESENTATION = 1
RGB_REPRESENTATION = 2


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


def imdisplay(filename, representation):
    """
    a function to utilize read_image to display an image in a given representation.
    open a new figure and display the loaded image in the converted representation.
    :param filename: the filename of an image on disk (could be grayscale or RGB).
    :param representation: a grayscale image (1) or an RGB image (2).
    :return:
    """
    image = read_image(filename=filename, representation=representation)
    if representation == GRAY_REPRESENTATION:
        plt.imshow(image, cmap='gray')
    if representation == RGB_REPRESENTATION:
        plt.imshow(image)
    plt.show()


def rgb2yiq(imRGB):
    """
    the input is a heightXwidthX3 np.float64 matrix.

    In the RGB case,
    the red channel is encoded in imRGB[:,:,0],
    the green in imRGB[:,:,1],
    and the blue in imRGB[:,:,2].

    imRGB is in the [0; 1] range,

    :param imRGB:
    :return: an image having the same dimensions as the input.
    """
    rgb_to_yiq_mat = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321],
                               [0.212, -0.523, 0.311]])
    im_yiq = np.matmul(imRGB, rgb_to_yiq_mat.T)
    return im_yiq


def yiq2rgb(imYIQ):
    """
    the input is a heightXwidthX3 np.float64 matrix.

    for YIQ,
    imYIQ[:,:,0] encodes the luminance channel Y,
    imYIQ[:,:,1] encodes I,
    and imYIQ[:,:,2] encodes Q.

    while the Y channel is in the [0,1] range,
    the I and Q channels are in the [-1; 1] range
    (though they do not span it entirely).

    :param imYIQ:
    :return: an image having the same dimensions as the input.
    """
    rgb_to_yiq_mat = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321],
                               [0.212, -0.523, 0.311]])
    im_rgb = imYIQ @ np.linalg.inv(rgb_to_yiq_mat).T
    return im_rgb


def histogram_equalize_helper(im_to_eq):
    """
    Let C(k) be the cumulative histogram at intensity k and let m
    be the first gray level for which C(m) not 0,
    :param im_to_eq:
    :return:
    """
    # original histogram
    hist_orig, bins = np.histogram(a=np.uint8(np.round(
        im_to_eq * MAX_SHADE_VAL)), bins=NUM_SHADES, range=(0, MAX_SHADE_VAL))
    # lookup table
    hist_cumsum = np.cumsum(hist_orig)
    first_gray_level = np.nonzero(hist_cumsum)[0][0]
    lookup_table = np.round(MAX_SHADE_VAL * ((hist_cumsum -
                                              hist_cumsum[first_gray_level]) /
                                             (hist_cumsum[MAX_SHADE_VAL] -
                                              hist_cumsum[first_gray_level])))
    # equalized image
    im_eq = np.array(lookup_table[np.uint8(MAX_SHADE_VAL * im_to_eq)], np.float64)

    # equalized histogram, not normalized
    hist_eq, bins = np.histogram(a=np.uint8(np.round(im_eq)),
                                 bins=NUM_SHADES, range=(0, MAX_SHADE_VAL))
    im_eq = im_eq / MAX_SHADE_VAL
    return [im_eq, hist_orig, hist_eq]


def histogram_equalize(im_orig):
    """
    a function that performs histogram equalization of a given grayscale or RGB
     image.

    If an RGB image is given, the following equalization procedure should only
    operate on the Y channel of the corresponding YIQ image and then convert
    back from YIQ to RGB.

    Moreover, the outputs hist_orig and hist_eq should be the histogram of the
    Y channel only.

    The required intensity transformation is defined such that the gray levels
    should have an approximately uniform gray-level histogram (i.e. equalized
    histogram) stretched over the entire [0; 1] gray level range.

    Make sure you stretch the equalized histogram only if it covers partly that
    range. You may use the NumPy functions histogram and cumsum to perform the
    equalization.

    Also note that although im_orig is of type np.float64, you are required to
    internally perform the equalization using 256 bin histograms.

    NOTE: After equalizing histogram of Y and converting the image back to RGB
    there may be some values outside the range [0; 1]. You may normalize or
    clip the output for your own testing, but do not to do so for the
    submission.
    :param im_orig: the input grayscale or RGB float64 image with values in [0; 1].
    :return: a list [im_eq, hist_orig, hist_eq] where
    im_eq - is the equalized image. grayscale or RGB float64 image with values in [0; 1].
    hist_orig - is a 256 bin histogram of the original image (array with shape (256,) ).
    hist_eq - is a 256 bin histogram of the equalized image (array with shape (256,) ).
    """
    # returns none if image is neither grayscale nor RGB
    result = None
    # if the input is a grayscale image.
    if np.ndim(im_orig) == 2:
        # result = [im_eq, hist_orig, hist_eq]
        result = histogram_equalize_helper(im_orig)
    # if the input is an RGB image.
    if np.ndim(im_orig) == 3:
        im_yiq = rgb2yiq(im_orig)
        im_y = im_yiq[:, :, 0]
        result = histogram_equalize_helper(im_y)
        im_yiq[:, :, 0] = result[0]
        result[0] = yiq2rgb(im_yiq)
    return result


def z_initialization(hist_cumsum, n_quant):
    """
    z is an array with shape (n_quant + 1,)
    :param hist_cumsum:
    :param n_quant:
    :return:
    """
    z = np.zeros(n_quant + 1)
    num_pixels = hist_cumsum[-1]
    for i in range(n_quant + 1):
        where_array = np.where(hist_cumsum >= i * (num_pixels / n_quant))[0]
        z[i] = where_array[0]
    z[0] = -1  # a recommended solution for z_0, in order to start from g_0.
    z[-1] = MAX_SHADE_VAL  # last z is 255
    return z


def quantize_iteration_helper(z, i, hist_orig):
    """
    calculate parameters to apply the quantization formula.
    :param z:
    :param i:
    :param hist_orig:
    :return:
    """
    z_i = z.astype(int)[i] + 1
    z_i_one = (z.astype(int))[i + 1]
    # add +1 to z_i_one to include it
    hist_z_range = np.array(hist_orig[z_i: z_i_one + 1])
    z_range = np.arange(z_i, z_i_one + 1)
    return z_i, z_i_one, hist_z_range, z_range


def q_calculation(z, q, hist_orig, n_quant):
    for i in range(n_quant):
        z_i, z_i_one, hist_z_range, z_range = \
            quantize_iteration_helper(z, i, hist_orig)
        q[i] = (z_range @ hist_z_range) / hist_z_range.sum()


def error_calculation(z, q, hist_orig, n_quant, error):
    """

    :param z:
    :param q:
    :param hist_orig:
    :param n_quant:
    :param error:
    :return:
    """
    cur_error = 0
    for i in range(n_quant):
        z_i, z_i_one, hist_z_range, z_range = \
            quantize_iteration_helper(z, i, hist_orig)
        mse = np.square(q[i] - z_range)
        cur_error += np.array(mse @ hist_z_range).sum()
    error = np.append(error, cur_error)


def calculate_lookup_table(z, q, hist_orig, n_quant):
    lookup_table = np.arange(NUM_SHADES)
    for i in range(n_quant):
        z_i, z_i_one, hist_z_range, z_range = \
            quantize_iteration_helper(z, i, hist_orig)
        # z_i <= lookup_table <= z_i_one
        lookup_table[z_i: z_i_one + 1] = np.repeat(
            q[i], len(lookup_table[z_i: z_i_one + 1]))
        return lookup_table


def quantize_helper(im_orig, n_quant, n_iter):
    """
    quantize content
    :param im_orig: a single matrix, either a grayscale image or the y channel
    of an yiq image.
    :param n_quant:
    :param n_iter:
    :return:
    """
    # the initialization step (does not count as an iteration)
    hist_orig, bins = np.histogram(a=np.uint8(np.round(im_orig*MAX_SHADE_VAL)),
                                   bins=NUM_SHADES, range=(0, MAX_SHADE_VAL))
    hist_cumsum = hist_orig.cumsum()
    z = z_initialization(hist_cumsum, n_quant)
    # q is a one dimensional array, containing n_quant elements.
    q = np.zeros(n_quant)
    # perform the two steps n_iter times unless the process converges.
    error = np.array([])
    lookup_table = None
    im_quant = None
    # initial calculation of q. not counted as an iteration
    q_calculation(z, q, hist_orig, n_quant)
    new_z = np.copy(z)
    for k in range(n_iter):
        # update z, not including z[0] and z[-1]
        new_z[1:-1] = (q[:-1] + q[1:]) / 2
        # check convergence. Convergence is defined as a situation where the
        # values of z have not been changed during the last iteration.
        if np.allclose(z, new_z):
            break
        z = np.copy(new_z)
        # update q
        q_calculation(z, q, hist_orig, n_quant)
        # calculate the error for the current iteration for all quants
        error_calculation(z, q, hist_orig, n_quant, error)
    # calculate the lookup table
    lookup_table = calculate_lookup_table(z, q, hist_orig, n_quant)
    if lookup_table is not None:
        im_quant = np.array(
            lookup_table[np.uint8(np.round(MAX_SHADE_VAL * im_orig))],
            np.float64) / MAX_SHADE_VAL
    return [im_quant, error]


def quantize(im_orig, n_quant, n_iter):
    """
    performs optimal quantization of a given grayscale or RGB image.

    If an RGB image is given, the quantization procedure should only operate on
    the Y channel of the corresponding YIQ image and then convert back from
    YIQ to RGB.

    Each iteration of the quantization process is composed of the following
    two steps:
    1.Computing z - the borders which divide the histograms into segments. z is
    an array with shape (n_quant + 1,). The first and last elements are 0 and
    255 respectively.
    2. Computing q - the values to which each of the segments' intensities will
    map. q is also a one dimensional array, containing n_quant elements.

    :param im_orig: the input image to be quantized (float64 image with values
    in [0; 1]).
    :param n_quant: the number of intensities your output image should have.
    :param n_iter: the maximum number of iterations of the optimization
    procedure (may converge earlier.)
    :return: a list [im_quant, error] where
    im_quant: the quantized output image. (float64 image with values in
    [0; 1]).
    error:  an array with shape (n_iter,) (or less) of the total intensities
    error for each iteration of the quantization procedure.
    """
    # returns none if image is neither grayscale nor RGB
    result = None
    # if the input is a grayscale image.
    if np.ndim(im_orig) == 2:
        # result = [im_quant, error]
        result = quantize_helper(im_orig, n_quant, n_iter)
    # if the input is an RGB image.
    if np.ndim(im_orig) == 3:
        im_yiq = rgb2yiq(im_orig)
        im_y = im_yiq[:, :, 0]
        result = quantize_helper(im_y, n_quant, n_iter)
        im_yiq[:, :, 0] = result[0]
        result[0] = yiq2rgb(im_yiq)
    return result

