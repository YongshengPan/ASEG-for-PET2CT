import os
from tensorflow import keras
import tensorflow as tf
import numpy as np
import time
from sklearn import metrics
from skimage.metrics import structural_similarity as ssim
from core import radon_transform_tensorflow as rtt


def cls_loss_with_logits(y_pred, y_true, model='categorical', from_logits=False):
    if model == 'categorical':
        return tf.reduce_mean(keras.metrics.categorical_crossentropy(y_pred=y_pred, y_true=y_true, from_logits=from_logits))
    elif model == 'binary':
        return tf.reduce_mean(keras.metrics.binary_crossentropy(y_pred=y_pred, y_true=y_true, from_logits=from_logits))
    else:
        return tf.reduce_mean(keras.metrics.binary_crossentropy(y_pred=y_pred, y_true=y_true, from_logits=from_logits))


def seg_loss(y_pred, y_true, model='dice'):
    ex_axis = [0, 1, 2, 3, 4]
    ex_axis = tuple(ex_axis[0: keras.backend.ndim(y_pred)-1])
    if model == 'dice':
        value = 1 - (2 * tf.reduce_sum(y_true * y_pred, axis=ex_axis) + 1) / (tf.reduce_sum(y_true, axis=ex_axis) + tf.reduce_sum(y_pred, axis=ex_axis) + 1)
    elif model == 'jaccard':
        value = 1 - (tf.reduce_sum(tf.minimum(y_true, y_pred), axis=ex_axis) + 1) / (tf.reduce_sum(tf.maximum(y_true, y_pred), axis=ex_axis) + 1)
    else:
        value = 1 - (2 * tf.reduce_sum(y_true * y_pred, axis=ex_axis) + 1) / (tf.reduce_sum(y_true, axis=ex_axis) + tf.reduce_sum(y_pred, axis=ex_axis) + 1)
    return tf.reduce_mean(value)


def mae_loss(con_feat, fake_feat, weight=1.0):
    # return tf.reduce_mean(tf.abs(con_feat - fake_feat)) * weight
    # weight = tf.tanh(tf.abs(con_feat - fake_feat)*1000)+0.001
    # weight = weight/tf.reduce_mean(weight)
    return tf.reduce_mean(tf.abs(con_feat - fake_feat) * weight)


def mse_loss(con_feat, fake_feat, weight=1):
    return tf.reduce_mean(tf.square(con_feat - fake_feat)) * weight


def mae_loss_with_weight(con_feat, fake_feat, weight):
    return tf.reduce_mean(tf.abs(con_feat - fake_feat) * weight)


def gaussian(window_size, sigma):
    loc_val = [(x - window_size // 2)/sigma for x in range(window_size)]
    gauss = tf.exp(-tf.convert_to_tensor(loc_val)**2/2)#, tf.float32)
    gauss = gauss / tf.reduce_sum(gauss)
    return gauss


def create_window(window_size, channel, dim=2):
    _1D_window = tf.expand_dims(gaussian(window_size, 1.5), axis=-1)
    window = tf.ones([1, 1])
    for dims in range(dim):
        window = tf.expand_dims(tf.tensordot(_1D_window, window, axes=(-1, -1)), axis=-1)
    window = tf.tensordot(window, tf.ones([channel, 1]), axes=(-1, -1))
    return window


def ssim_tf(img1, img2, window, size_average=True):
    dim = len(tf.shape(img1))
    if dim == 5:
        conv = tf.nn.conv3d
    elif dim == 4:
        conv = tf.nn.conv2d
    else:
        conv = tf.nn.conv1d
    mu1 = conv(img1, window, [1]*dim, padding='SAME')
    mu2 = conv(img2, window, [1]*dim, padding='SAME')

    mu1_sq = tf.pow(mu1, 2)
    mu2_sq = tf.pow(mu2, 2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = conv(img1 * img1, window, [1]*dim, padding='SAME') - mu1_sq
    sigma2_sq = conv(img2 * img2, window, [1]*dim, padding='SAME') - mu2_sq
    sigma12 = conv(img1 * img2, window, [1]*dim, padding='SAME') - mu1_mu2

    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    mcs_map = (2 * sigma12 + C2)/(sigma1_sq + sigma2_sq + C2)
    if size_average:
        return tf.reduce_mean(ssim_map), tf.reduce_mean(mcs_map)
    else:
        return tf.reduce_mean(ssim_map, axis=(1, 2, 3, 4))


def ms_ssim(img1, img2, window_size=11, size_average=True):
    # time_start = time.time()
    img1 = tf.convert_to_tensor(img1, tf.float32)
    img2 = tf.convert_to_tensor(img2, tf.float32)
    channel = tf.shape(img1)[-1]
    dim = len(tf.shape(img1))
    window = create_window(window_size, channel, dim=dim-2)
    weights = [0.0448, 0.2856, 0.3001, 0.2363, 0.1333]

    _ssim, _mcs = ssim_tf(img1, img2, window, size_average)
    ssims, mcses = [_ssim], [_mcs]
    dim = len(tf.shape(img1))-2
    if dim == 3:
        pool = tf.nn.avg_pool3d
    elif dim == 2:
        pool = tf.nn.avg_pool2d
    else:
        pool = tf.nn.avg_pool1d
    for stride in range(1, len(weights)):
        img1 = pool(img1, [2]*dim, [2]*dim, padding='SAME')
        img2 = pool(img2, [2]*dim, [2]*dim, padding='SAME')
        _ssim, _mcs = ssim_tf(img1, img2, window, size_average)
        ssims.append(_ssim)
        mcses.append(_mcs)
    output = (ssims[-1] ** weights[-1]) * (tf.convert_to_tensor(mcses) ** weights)
    # time_end = time.time()
    # print('time cost', time_end - time_start, 's')
    return tf.reduce_prod(output)


def ms_ssim_loss(y_pred, y_true, window_size=11):
    channel = tf.shape(y_true)[-1]
    dim = len(tf.shape(y_true))
    window = create_window(window_size, channel, dim=dim - 2)
    SSIM, _ = ssim_tf(y_pred, y_true, window=window, size_average=True)
    return SSIM
    #return 1-ms_ssim(y_pred, y_true, window_size=window_size, size_average=True)


def ct_segmentation_loss(y_pred, y_true, model='dice'):
    thres = -1100, -900, -400, -150, -100, -10, -100, 150, 250, 400, 20000
    scale, offset = 1000, -1000
    y_true = y_true*scale+offset
    y_pred = y_pred*scale+offset
    dice_value = 0
    for idx in range(len(thres)-1):
        # region = tf.minimum(tf.nn.relu(y_true - thres[idx]), 1) - tf.minimum(tf.nn.relu(y_true - thres[idx+1]), 1)
        # predict = tf.minimum(tf.nn.relu(y_pred - thres[idx]), 1) - tf.minimum(tf.nn.relu(y_pred - thres[idx+1]), 1)
        region = tf.nn.sigmoid(y_true - thres[idx]) - tf.nn.sigmoid(y_true - thres[idx + 1])
        predict = tf.nn.sigmoid(y_pred - thres[idx]) - tf.nn.sigmoid(y_pred - thres[idx + 1])
        dice_value = dice_value + seg_loss(region, predict, model=model)

    return dice_value

# region, predict = np.logical_and(tissue_thres < groundtruth, groundtruth < bone_thres), np.logical_and(
#     tissue_thres < prediction, prediction < bone_thres)
# Dice1 = (2 * np.sum(predict * region, axis=ex_axis) + 1e-3) / (
#             np.sum(predict, ex_axis) + np.sum(region, axis=ex_axis) + 1e-3)
#
# region, predict = groundtruth > bone_thres, prediction > bone_thres
# Dice2 = (2 * np.sum(predict * region, axis=ex_axis) + 1e-3) / (
#             np.sum(predict, ex_axis) + np.sum(region, axis=ex_axis) + 1e-3)


def wavelet_loss(y_pred, y_true):
    x_d = tf.reshape(tf.constant([1.0, 1.0], tf.float32), (2, 1, 1, 1, 1)), tf.reshape(tf.constant([-1.0, 1.0], tf.float32), (2, 1, 1, 1, 1))
    y_d = tf.reshape(tf.constant([1.0, 1.0], tf.float32), (1, 2, 1, 1, 1)), tf.reshape(tf.constant([-1.0, 1.0], tf.float32), (1, 2, 1, 1, 1))
    z_d = tf.reshape(tf.constant([1.0, 1.0], tf.float32), (1, 1, 2, 1, 1)), tf.reshape(tf.constant([-1.0, 1.0], tf.float32), (1, 1, 2, 1, 1))

    kernels = tf.concat([x_d[0]*y_d[0]*z_d[0], x_d[0]*y_d[0]*z_d[1], x_d[0]*y_d[1]*z_d[0], x_d[0]*y_d[1]*z_d[1],
               x_d[1]*y_d[0]*z_d[0], x_d[1]*y_d[0]*z_d[1], x_d[1]*y_d[1]*z_d[0], x_d[1]*y_d[1]*z_d[1]], axis=-1)

    wl_pred = tf.nn.conv3d(y_pred, kernels, [1, 1, 1, 1, 1], padding='VALID')
    wl_true = tf.nn.conv3d(y_true, kernels, [1, 1, 1, 1, 1], padding='VALID')
    return mae_loss(wl_pred, wl_true)



angles_count = 30
radon_transform_matrix = {}

def sinogram_loss(y_pred, y_true):
    diff_trans = tf.transpose(y_pred-y_true, (4, 0, 1, 2, 3))
    window_size = diff_trans.shape[-2]

    # time_start = time.time()
    radon_transform_matrix_index = 'space_indexs_%d_%d.npz' % (angles_count, window_size)
    if radon_transform_matrix_index in radon_transform_matrix:
        space_indexs = radon_transform_matrix[radon_transform_matrix_index]
    elif os.path.exists(radon_transform_matrix_index):
        space_indexs = np.load(radon_transform_matrix_index, allow_pickle=True)
        space_indexs = tf.sparse.SparseTensor(indices=space_indexs['indices'], values=space_indexs['values'],
                                              dense_shape=space_indexs['dense_shape'])
        radon_transform_matrix[radon_transform_matrix_index] = space_indexs
    else:
        space_indexs = rtt.create_radon_kernel(window_size, np.linspace(0, 180, angles_count, endpoint=False))
        np.savez(radon_transform_matrix_index, dense_shape=space_indexs.dense_shape, indices=space_indexs.indices,
                     values=space_indexs.values)
        radon_transform_matrix[radon_transform_matrix_index] = space_indexs

    diff_trans = tf.unstack(tf.reshape(diff_trans, shape=(-1, diff_trans.shape[-3], diff_trans.shape[-2], diff_trans.shape[-1])), axis=0)
    sinogram_diff = tf.reduce_mean([tf.reduce_mean(tf.abs(rtt.radon_transform(diff_trans[idx], space_indexs, axis=0))) for idx in range(len(diff_trans))])

    # time_end = time.time()
    # print(time_end - time_start)
    return sinogram_diff


# def maxproj_loss(y_pred, y_true):
#
#     x_d = mae_loss(tf.reduce_max(y_pred, 1), tf.reduce_max(y_true, 1))
#     y_d = mae_loss(tf.reduce_max(y_pred, 2), tf.reduce_max(y_true, 2))
#     z_d = mae_loss(tf.reduce_max(y_pred, 3), tf.reduce_max(y_true, 3))
#     return x_d + y_d + z_d
#
# np.linspace(0, 180, angles_count, endpoint=False, dtype=np.float32)
# rotation_matrix = tf.convert_to_tensor([[[np.cos(theta), -np.sin(theta)], [np.sin(theta), -np.cos(theta)]] for theta in np.linspace(0, 180, angles_count, endpoint=False, dtype=np.float32)])

def maxproj_loss(y_pred, y_true):
    # time_start = time.time()
    y_pred = tf.transpose(y_pred, (2, 3, 0, 1, 4))
    y_true = tf.transpose(y_true, (2, 3, 0, 1, 4))
    image_shape = tf.shape(y_true, out_type=tf.int32)
    label_prop_x = tf.range(0, image_shape[0], dtype=tf.float32) - tf.cast(image_shape[0], tf.float32) * 0.5 - 0.5
    label_prop_y = tf.range(0, image_shape[1], dtype=tf.float32) - tf.cast(image_shape[1], tf.float32) * 0.5 - 0.5
    label_prop = tf.meshgrid(label_prop_x, label_prop_y, indexing='ij')
    label_prop = [tf.expand_dims(lp, axis=-1) for lp in label_prop]
    label_meshgrid = tf.concat(label_prop, axis=-1)
    all_loss = 0
    for theta in np.linspace(0, 180, angles_count, endpoint=False):
        rotation_matrix = [[tf.cos(theta), -tf.sin(theta)], [tf.sin(theta), -tf.cos(theta)]]
        label_prop_trans = tf.matmul(label_meshgrid, rotation_matrix) + tf.cast(image_shape[0:2], tf.float32) * 0.5 - 0.5
        label_prop_trans = tf.cast(label_prop_trans, tf.int32)
        y_true_trans = tf.gather_nd(y_true, label_prop_trans)
        y_pred_trans = tf.gather_nd(y_pred, label_prop_trans)
        y_true_trans_s = tf.split(y_true_trans, [image_shape[0]//2, image_shape[0]//2], axis=0)
        y_true_trans_s = tf.split(y_true_trans_s[0], [image_shape[1]//2, image_shape[1]//2], axis=1) + tf.split(y_true_trans_s[1], [image_shape[1]//2, image_shape[1]//2], axis=1)
        y_pred_trans_s = tf.split(y_pred_trans, [image_shape[0] // 2, image_shape[0] // 2], axis=0)
        y_pred_trans_s = tf.split(y_pred_trans_s[0], [image_shape[1]//2, image_shape[1]//2], axis=1) + tf.split(y_pred_trans_s[1], [image_shape[1]//2, image_shape[1]//2], axis=1)
        x_d = [mae_loss(tf.reduce_max(y_true_trans_s[idx], 0), tf.reduce_max(y_pred_trans_s[idx], 0)) for idx in [0, 1, 2, 3]]
        y_d = [mae_loss(tf.reduce_max(y_true_trans_s[idx], 1), tf.reduce_max(y_pred_trans_s[idx], 1)) for idx in [0, 1, 2, 3]]
        # x_d = mae_loss(tf.reduce_max(y_true_trans, 0), tf.reduce_max(y_pred_trans[image_shape[0]//2::, :], 0))
        # y_d = mae_loss(tf.reduce_max(y_true_trans, 1), tf.reduce_max(y_pred_trans[:, 0:image_shape[1]//2], 1))
        all_loss = all_loss + tf.reduce_mean(x_d) + tf.reduce_mean(y_d)
    # time_end = time.time()
    # print(time_end - time_start)
    return all_loss/angles_count


def multi_feat_loss(con_feats, fake_feats, weight=None):
    if weight is None:
        multi_weight = [1 for slt in range(len(con_feats))]
    else:
        strips = [np.int32(np.ceil(np.divide(weight.shape, fake_feats[slt].shape))) for slt in range(len(con_feats))]
        multi_weight = [weight[strips[slt][0]//2::strips[slt][0], strips[slt][1]//2::strips[slt][1], strips[slt][2]//2::strips[slt][2], strips[slt][3]//2::strips[slt][3]] for slt in range(len(con_feats))]
    return tf.reduce_sum([mae_loss(con_feats[slt], fake_feats[slt], multi_weight[slt]) for slt in range(len(con_feats))])


def basic_loss_essamble(y_true, y_pred, lossses):
    total_loss = 0
    for loss in lossses:
        if loss == 'maxp':
            total_loss = total_loss + maxproj_loss(y_true, y_pred)
        elif loss == 'sin':
            total_loss = total_loss + sinogram_loss(y_true, y_pred)
        elif loss == 'wll':
            total_loss = total_loss + wavelet_loss(y_true, y_pred)
        elif loss == 'p2p':
            total_loss = total_loss + mae_loss(y_true, y_pred, weight=1.0)
        elif loss == 'cre':
            total_loss = total_loss + cls_loss_with_logits(y_pred=y_pred, y_true=y_true, model='categorical')
        elif loss == 'dice':
            total_loss = total_loss + seg_loss(y_pred=y_pred, y_true=y_true, model='dice')
        elif loss == 'jac':
            total_loss = total_loss + seg_loss(y_pred=y_pred, y_true=y_true, model='jaccard')
        elif loss == 'ct_dice':
            total_loss = total_loss + ct_segmentation_loss(y_pred=y_pred, y_true=y_true, model='dice')
        elif loss == 'ct_jac':
            total_loss = total_loss + ct_segmentation_loss(y_pred=y_pred, y_true=y_true, model='jac')
        elif loss == 'ms_ssim':
            total_loss = total_loss + ms_ssim_loss(y_pred, y_true)
        else:
            pass

    # if 'sin' in lossses:
    #     total_loss = total_loss + sinogram_loss(y_true, y_pred)
    # if 'wll' in lossses:
    #     total_loss = total_loss + wavelet_loss(y_true, y_pred)
    # if 'p2p' in lossses:
    #     total_loss = total_loss + mae_loss(y_true, y_pred, weight=1.0)
    # if 'cre' in lossses:
    #     total_loss = total_loss + cls_loss_with_logits(y_pred=y_pred, y_true=y_true, model='categorical')
    # if 'dice' in lossses:
    #     total_loss = total_loss + seg_loss(y_pred=y_pred, y_true=y_true, model='dice')
    # if 'jac' in lossses:
    #     total_loss = total_loss + seg_loss(y_pred=y_pred, y_true=y_true, model='jaccard')
    # if 'ct_dice' in lossses:
    #     total_loss = total_loss + ct_segmentation_loss(y_pred=y_pred, y_true=y_true, model='dice')
    # if 'ct_jac' in lossses:
    #     total_loss = total_loss + ct_segmentation_loss(y_pred=y_pred, y_true=y_true, model='jac')
    # if 'ms_ssim' in lossses:
    #     total_loss = total_loss + ms_ssim_loss(y_pred, y_true)
    return total_loss



def matrics_ct_segmentation(y_pred, y_true, model='dice'):

    def dice_value(predict, region, model=model):
        ex_axis = [0, 1, 2, 3, 4]
        # ex_axis = tuple(ex_axis[0: np.ndim(region) - 1])
        dv = (2 * np.sum(predict * region) + 1e-3) / (np.sum(predict) + np.sum(region) + 1e-3)
        return dv

    thres = -1100, -900, -150, -10, 150, 20000
    scale, offset = 1000, -1000
    y_true = y_true*scale+offset
    y_pred = y_pred*scale+offset

    dice_values = []

    for idx in range(len(thres)-1):
        region = np.where(np.logical_and(y_true > thres[idx], y_true < thres[idx+1]), 1, 0)
        predict = np.where(np.logical_and(y_pred > thres[idx], y_pred < thres[idx + 1]), 1, 0)
        # region = np.minimum(np.maximum(y_true - thres[idx], 0), 1) - np.minimum(np.maximum(y_true - thres[idx+1], 0), 1)
        # predict = np.minimum(np.maximum(y_pred - thres[idx], 0), 1) - np.minimum(np.maximum(y_pred - thres[idx+1], 0), 1)
        dice_values.append(dice_value(region, predict, model=model))

    return dice_values


def multiple_instensity_metrics(prediction, groundtruth, data_range=1.0):
    ex_axis = [0, 1, 2, 3, 4]
    ex_axis = tuple(ex_axis[0: np.ndim(groundtruth) - 1])
    prediction, groundtruth = prediction / data_range, groundtruth / data_range
    diff_map = prediction - groundtruth
    MAE = np.mean(np.abs(diff_map))
    MSE = np.mean(np.square(diff_map))
    SSIM = np.mean(ssim(groundtruth, prediction, full=False, multichannel=True))
    PSNR = 10 * np.log10((data_range ** 2) / MSE) / 100
    NCC = np.mean(np.multiply(prediction - np.mean(prediction), groundtruth - np.mean(groundtruth)) / (np.std(prediction) * np.std(groundtruth))+1e-6)
    return MAE, MSE, PSNR, NCC, SSIM


def multiple_projection_metrics(prediction, groundtruth, data_range=1):
    ex_axis = [0, 1, 2, 3, 4]

    ex_axis = tuple(ex_axis[0: np.ndim(groundtruth) - 1])
    MAE = np.mean(np.abs(prediction / data_range - groundtruth / data_range))
    MSE = np.mean(np.square(prediction / data_range - groundtruth / data_range))
    SSIM = np.mean(ssim(groundtruth / data_range, prediction / data_range, full=False, multichannel=True))
    PSNR = 10 * np.log10((data_range ** 2) / MSE) / 100
    NCC = np.mean(np.multiply(prediction - np.mean(prediction), groundtruth - np.mean(groundtruth)) / (
                np.std(prediction) * np.std(groundtruth)))
    return [MAE], [MSE], [PSNR], [NCC], [SSIM]


def matrics_synthesis(prediction, groundtruth, data_range=1.0, isinstance=False):
    prediction = prediction
    groundtruth = groundtruth
    ex_axis = [0, 1, 2, 3, 4]
    ex_axis = tuple(ex_axis[0: np.ndim(groundtruth) - 1])

    if isinstance:
        INMT = [multiple_instensity_metrics(prediction, groundtruth, data_range=data_range)]
        DICE = [matrics_ct_segmentation(prediction, groundtruth, model='dice')]

    else:
        INMT = [multiple_instensity_metrics(prediction[idx], groundtruth[idx], data_range=data_range) for idx in range(np.shape(groundtruth)[0])]
        DICE = [matrics_ct_segmentation(prediction[idx], groundtruth[idx], model='dice') for idx in range(np.shape(groundtruth)[0])]

    # return 100*np.mean(INMT, axis=0), 100*np.mean(DICE, axis=0)
    return np.concatenate((100*np.mean(INMT, axis=0), 100*np.mean(DICE, axis=0)), axis=-1)
    # return np.mean(MAE), np.mean(MSE), np.mean(SSIM), np.mean(PSNR), np.std(MAE), np.std(MSE), np.std(SSIM), np.std(PSNR)


def matrics_classification(testvals, labels, thres=None):
    # print(np.mean(testvals, axis=0))

    def softmax(logits):
        return np.exp(logits) / np.sum(np.exp(logits), -1, keepdims=True)

    if thres is not None:
        testvals = testvals - np.array(thres)
    else:
        testvals = testvals - np.mean(testvals, axis=0)
    testvals = softmax(testvals)
    print(np.shape(testvals))
        # losslist = -np.sum(np.multiply(labels, np.log(testvals)), 1)
        # total_loss = np.average(losslist)
    AUC = metrics.roc_auc_score(y_score=testvals, y_true=labels, average='macro')
    ACC = metrics.accuracy_score(y_pred=np.argmax(testvals, axis=-1), y_true=np.argmax(labels, axis=-1))
    BAC = metrics.balanced_accuracy_score(y_pred=np.argmax(testvals, axis=-1), y_true=np.argmax(labels, axis=-1))
    APS = metrics.average_precision_score(y_score=testvals, y_true=labels, average='macro')
    SEN = metrics.recall_score(y_pred=np.argmax(testvals, axis=-1), y_true=np.argmax(labels, axis=-1), pos_label=1, average='macro')
    SPE = metrics.recall_score(y_pred=np.argmax(testvals, axis=-1), y_true=np.argmax(labels, axis=-1), pos_label=0, average='macro')
    COM = metrics.confusion_matrix(y_pred=np.argmax(testvals, axis=-1), y_true=np.argmax(labels, axis=-1))
    F1S = metrics.f1_score(y_pred=np.argmax(testvals, axis=-1), y_true=np.argmax(labels, axis=-1), average='macro')
    MCC = metrics.matthews_corrcoef(y_pred=np.argmax(testvals, axis=-1), y_true=np.argmax(labels, axis=-1))
    # return [AUC*100, ACC*100, BAC*100, REC*100, F1S*100, MCC*100]
    return [AUC*100, ACC*100, SEN*100, SPE*100, F1S*100, MCC*100], metrics.classification_report(y_true=np.argmax(labels, axis=-1), y_pred=np.argmax(testvals, axis=-1))


def matrics_segmentation(prediction, groundtruth, labeltype='category', threshold=0.5):
    if labeltype == 'category' and np.shape(prediction)[-1] > 1:
        prediction_hard = np.argmax(prediction, axis=-1)
        groundtruth_hard = np.argmax(groundtruth, axis=-1)
        prediction_hard = np.concatenate([np.expand_dims(prediction_hard == idx, axis=-1) for idx in range(np.shape(prediction)[-1])], axis=-1)
        groundtruth_hard = np.concatenate([np.expand_dims(groundtruth_hard == idx, axis=-1) for idx in range(np.shape(prediction)[-1])], axis=-1)
    else:
        prediction_hard = np.array(prediction > threshold)
        groundtruth_hard = np.array(groundtruth > threshold)
    # Intersection = [np.array(prediction_hard == idx) & np.array(groundtruth_hard == idx) for idx in range(np.shape(prediction)[-1])]
    # Union = [np.array(prediction_hard == idx) | np.array(groundtruth_hard == idx) for idx in range(np.shape(prediction)[-1])]
        # Intersection = np.array(prediction > threshold) & np.array(groundtruth > threshold)
        # Union = np.array(prediction > threshold) | np.array(groundtruth > threshold)
    # ex_axis = [dd for dd in range(0, np.ndim(Intersection)-1)]
    ex_axis = [0, 1, 2, 3, 4]
    ex_axis = tuple(ex_axis[0: np.ndim(prediction) - 1])
    IoU = (np.sum(prediction_hard & groundtruth_hard, axis=ex_axis)+1e-3)/(np.sum(prediction_hard | groundtruth_hard, axis=ex_axis)+1e-3)
    Jaccard = (np.sum(np.minimum(prediction, groundtruth), axis=ex_axis)+1e-3) / (np.sum(np.maximum(prediction, groundtruth), axis=ex_axis)+1e-3)
    DICE1 = (2*np.sum(prediction_hard*groundtruth_hard, axis=ex_axis)+1e-3)/(np.sum(prediction_hard, axis=ex_axis) + np.sum(groundtruth_hard, axis=ex_axis)+1e-3)
    DICE2 = (2 * np.sum(prediction * groundtruth, axis=ex_axis)+1e-3) / (np.sum(prediction, ex_axis) + np.sum(groundtruth, axis=ex_axis)+1e-3)
    return IoU*100, Jaccard*100, DICE1*100, DICE2*100


def getbondingbox(image, fctr=0, thres=0.5):
        org_shp = np.shape(image)
        locsx, locsy = np.nonzero(np.sum(image > 0.5, axis=0)), np.nonzero(np.sum(image > 0.5, axis=1))
        if len(locsx[0]) == 0 or len(locsy[0]) == 0: return None
        region = np.array([[min(locsy[0]), (max(locsy[0]) + 1 + fctr * org_shp[0])],
                           [min(locsx[0]), (max(locsx[0]) + 1 + fctr * org_shp[1])]]) // (fctr + 1)
        region = region.astype(np.int)
        region[0] = np.minimum(np.maximum(region[0], 0), org_shp[0])
        region[1] = np.minimum(np.maximum(region[1], 0), org_shp[1])
        return region




