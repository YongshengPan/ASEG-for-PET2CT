import numpy as np
from sklearn import metrics
from skimage.metrics import structural_similarity as ssim


def matrics_ct_segmentation(y_pred, y_true, model='dice'):

    def dice_value(predict, region, model=model):
        ex_axis = [0, 1, 2, 3, 4]
        # ex_axis = tuple(ex_axis[0: np.ndim(region) - 1])
        dv = (2 * np.sum(predict * region) + 1e-3) / (np.sum(predict) + np.sum(region) + 1e-3)
        return dv

    thres = -1100, -150, -10, 150, 20000
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




