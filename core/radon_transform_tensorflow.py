import numpy as np
from math import exp
import tensorflow as tf
import time
from matplotlib import pyplot
from skimage.transform import radon, iradon
import SimpleITK as sitk
import os


def create_radon_kernel(window_size, degrees):
    def bandpass(signal, delta):
        return 1.0 * tf.maximum(1.0 - tf.abs(signal - delta), 0)

    space_indexs = []
    shift_index = tf.range(0, window_size, dtype=tf.float32) - window_size / 2
    if np.ndim(degrees) == 0:
        degrees = [degrees, ]
    for theta in np.deg2rad(-degrees):
        cos_theta = shift_index * tf.cast(tf.cos(theta), tf.float32)
        sin_theta = shift_index * tf.cast(tf.sin(theta), tf.float32)
        cos_theta, sin_theta = tf.expand_dims(cos_theta, 0), tf.expand_dims(sin_theta, -1)
        line_assign = (cos_theta + sin_theta + window_size / 2) % window_size
        index_slice = [bandpass(line_assign, idx) for idx in range(window_size)]
        space_index = tf.sparse.reshape(tf.sparse.from_dense(index_slice), (window_size, -1))
        space_indexs.append(space_index)
    space_indexs = tf.sparse.concat(0, space_indexs)
    return space_indexs


def radon_transform(image, transform_kernel, axis=0):
    image_shape = tf.shape(image)
    if axis == -1:
        axis = len(image_shape)-1
    new_axes = [axis, ] + [ax for ax in range(len(image_shape)) if ax != axis]
    image_trans = tf.transpose(image, new_axes)
    sinograms = tf.sparse.sparse_dense_matmul(tf.reshape(image_trans, (image_shape[axis], -1)), transform_kernel, adjoint_b=True)
    sinograms = tf.reshape(sinograms, (image_shape[axis], -1, image_shape[new_axes[-1]]))
    return sinograms


def iradon_transform(sinogram, transform_kernel, axis=0):
    image_shape = tf.shape(sinogram)
    if axis == -1:
        axis = len(image_shape) - 1
    new_axes = [axis, ] + [ax for ax in range(len(image_shape)) if ax != axis]
    image_trans = tf.transpose(sinogram, new_axes)
    radon_filtered = tf.cast(tf.reshape(image_trans, shape=(image_shape[0], -1)), tf.float32)
    back_image = tf.sparse.sparse_dense_matmul(transform_kernel, radon_filtered, adjoint_a=True, adjoint_b=True)
    back_image = tf.reshape(back_image, shape=[image_shape[new_axes[-1]], image_shape[new_axes[-1]], image_shape[axis]])
    return back_image * np.pi / (2 * angles_count)


def apply_fourier_filter(sinogram, fourier_filter):
    window_size = tf.shape(sinogram)[-1]
    projection_size_padded = tf.shape(fourier_filter)[-1]
    pad_width = projection_size_padded-window_size
    paddings = [[0, 0], ]*(len(tf.shape(sinogram))-1) + [[0, pad_width], ]
    sinogram_view = tf.signal.fft(tf.cast(tf.pad(sinogram, paddings), tf.complex64))
    radon_filtered = tf.signal.ifft(sinogram_view * fourier_filter)
    radon_filtered = tf.slice(radon_filtered, [0, ]*len(tf.shape(sinogram)), tf.shape(sinogram))
    return radon_filtered


def image_filter(sinogram, img_flt):
    window_size = tf.shape(sinogram)[-1]
    projection_size_padded = tf.shape(img_flt)[-1]
    pad_width = projection_size_padded-1
    paddings = [[0, 0], ]*(len(tf.shape(sinogram))-1) + [[0, pad_width], ]
    sinogram_view = tf.expand_dims(tf.pad(sinogram, paddings), axis=-1)
    sinogram_view = tf.nn.conv2d(sinogram_view, tf.reshape(img_flt, [1, -1, 1, 1]), strides=1, padding='VALID')
    # radon_filtered = tf.reshape(sinogram_views, shape=(-1, 1))
    radon_filtered = tf.squeeze(sinogram_view, axis=-1)
    return radon_filtered


if __name__ == '__main__':
    gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    image_path = 'G:/dataset/NACdata1/TCGA-LUAD/TCGA-50-5045/05-21-1995-ThoraxThoraxPETCT1-30440/2.000000-CT Spiral 5.0 B30s-86581.nii.gz'
    sitk_image = sitk.ReadImage(image_path)
    image = np.expand_dims(np.expand_dims(sitk.GetArrayFromImage(sitk_image), axis=-1), axis=0)/1000 + 1.0
    batch_size = image.shape[0]
    window_size = image.shape[1]
    channel_size = 1

    def get_fourier_filter(size):
        n = np.concatenate((np.arange(1, size / 2 + 1, 2, dtype=int),
                            np.arange(size / 2 - 1, 0, -2, dtype=int)))
        image_filter = np.zeros(size)
        image_filter[0] = 0.25
        image_filter[1::2] = -1 / (np.pi * n) ** 2
        fourier_filter = 2 * tf.signal.fft(image_filter)
        return fourier_filter, tf.convert_to_tensor(image_filter, tf.float32)

    time_start = time.time()
    angles_count = 30
    radon_transform_matrix = 'space_indexs_%d_%d.npz' % (angles_count, window_size)
    if not os.path.exists(radon_transform_matrix):
        space_indexs = create_radon_kernel(window_size, np.linspace(0, 180, angles_count, endpoint=False))
        np.savez(radon_transform_matrix, dense_shape=space_indexs.dense_shape, indices=space_indexs.indices, values=space_indexs.values)
    else:
        space_indexs = np.load(radon_transform_matrix, allow_pickle=True)
        space_indexs = tf.sparse.SparseTensor(indices=space_indexs['indices'], values=space_indexs['values'], dense_shape=space_indexs['dense_shape'])

    time_end = time.time()

    print(time_end-time_start)
    radon_trans = radon(image[0, 1, :, :, 0], np.linspace(0, 180, angles_count, endpoint=False))
    time_start = time.time()
    image_trans = [tf.unstack(im, axis=0) for im in tf.unstack(image, axis=-1)]
    sinogram_views = [radon_transform(image_tran, space_indexs, axis=0) for image_tran in image_trans]
    time_end = time.time()
    print(time_end - time_start)

    projection_size_padded = max(64, int(2 ** np.ceil(np.log2(2 * window_size))))

    # radon_filtered = [image_filter(sinogram_view, imf) for sinogram_view in sinogram_views]
    time_start = time.time()
    four_flt, imf = get_fourier_filter(window_size)
    radon_filtered = [apply_fourier_filter(sinogram_view, four_flt) for sinogram_view in sinogram_views]
    time_end = time.time()
    print((time_end - time_start))
    iradon_trans = iradon(radon_trans, np.linspace(0, 180, angles_count, endpoint=False))
    time_start = time.time()
    image_views = [iradon_transform(sinogram, space_indexs, axis=0) for sinogram in radon_filtered]
    time_end = time.time()
    print(time_end - time_start)

    pyplot.subplot(2, 2, 1)
    pyplot.imshow(image[10][:, :, 0])
    pyplot.subplot(2, 2, 2)
    pyplot.imshow(radon_trans)
    pyplot.subplot(2, 2, 3)
    pyplot.imshow(tf.transpose(sinogram_views[10][0]))
    pyplot.subplot(2, 2, 4)
    pyplot.imshow(image_views[10][:, :, 0])
    pyplot.show()



    # sinogram_views = tf.nn.conv1d(tf.expand_dims(tf.pad(sinogram_views, [[0, 0], [0, 199]], ), axis=-1), tf.reshape(image_filter, [-1, 1, 1]), stride=1, padding='VALID')
    # radon_filtered = tf.reshape(sinogram_views, shape=(-1, 1))

    # space_indexs = tf.sparse.SparseTensor(indices=space_indexs.indices, values=space_indexs.values**(0), dense_shape=space_indexs.dense_shape)



