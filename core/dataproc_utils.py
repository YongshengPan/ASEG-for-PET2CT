import SimpleITK as sitk
import numpy as np
import random
from matplotlib import pyplot as plt


def resize_image_itk_1(itkimage, newSpacing=None, newSize=None, newfactor=None, resamplemethod=sitk.sitkLinear):
    resampler = sitk.ResampleImageFilter()
    originSize = itkimage.GetSize()  # 原来的体素块尺寸
    originSpacing = itkimage.GetSpacing()
    if newSpacing is not None:
        factor = originSpacing / np.array(newSpacing)
    elif newSize is not None:
        factor = [np.nan if newSize[idx] is None else newSize[idx] / originSize[idx] for idx in range(len(originSize))]
        meanfactor = np.nanmean(factor)
        meanspacing = np.nanmean(
            [np.nan if newSize[idx] is None else originSpacing[idx] for idx in range(len(originSize))])
        factor = [factor[idx] if factor[idx] is not np.nan else originSpacing[idx] / meanspacing * meanfactor for idx in
                  range(len(originSize))]
        # factor = np.nan_to_num(factor, nan=nanvalue)
    else:
        factor = np.array(originSpacing) / originSpacing[1] * newfactor
        # print(factor)
    factor = np.nan_to_num(factor, nan=1.0)
    newSize = np.asarray(originSize * factor, np.int32)
    newSpacing = np.array(originSpacing / factor)
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetSize(newSize.tolist())
    resampler.SetOutputSpacing(newSpacing.tolist())
    # print(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
    resampler.SetInterpolator(resamplemethod)
    resampler.SetDefaultPixelValue(0)
    resampler.SetOutputPixelType(itkimage.GetPixelID())
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    return itkimgResampled


def resize_image_itk(itkimage, newSpacing=None, newSize=None, newfactor=None, resamplemethod=sitk.sitkNearestNeighbor):
        resampler = sitk.ResampleImageFilter()
        originSize = itkimage.GetSize()  # 原来的体素块尺寸
        originSpacing = itkimage.GetSpacing()
        if newSpacing is not None:
            newSpacing = [originSpacing[idx] if newSpacing[idx] is None else newSpacing[idx] for idx in range(len(originSpacing))]
            factor = originSpacing/np.array(newSpacing)
        elif newSize is not None:
            factor = [np.nan if newSize[idx] is None else newSize[idx] / originSize[idx] for idx in range(len(originSize))]
            meanfactor = np.nanmean(factor)
            meanspacing = np.nanmean(
                [np.nan if newSize[idx] is None else originSpacing[idx] for idx in range(len(originSize))])
            factor = [factor[idx] if factor[idx] is not np.nan else originSpacing[idx] / meanspacing * meanfactor for
                      idx in range(len(originSize))]
        else:
            factor = np.array(originSpacing)/originSpacing[2]*newfactor
            # print(factor)
        factor = np.nan_to_num(factor, nan=1.0)
        tempSize = np.asarray(originSize*factor, np.int32).tolist()

        if newSize is None: newSize = tempSize
        newSize = [tempSize[idx] if newSize[idx] is None else newSize[idx] for idx in range(3)]
        tempSpacing = np.asarray(originSpacing / factor, np.int32).tolist()
        if newSpacing is None: newSpacing = tempSpacing
        tempSize = np.maximum(tempSize, newSize).tolist()
        for idx in range(3):
            newSpacing[idx] = tempSpacing[idx] if newSpacing[idx] is None else newSpacing[idx]
        resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
        resampler.SetSize(tempSize)
        # outori = resampler.GetOutputOrigin()
        # resampler.SetOutputOrigin((-250.0, 249.0, outori[2]))
        resampler.SetOutputSpacing(newSpacing)
        # resampler.SetTransform(sitk.Transform(3, sitk.sitkIdentity))
        resampler.SetInterpolator(resamplemethod)
        resampler.SetDefaultPixelValue(0)
        resampler.SetOutputPixelType(itkimage.GetPixelID())
        itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
        lowerBoundSz = [(tempSize[idx] - newSize[idx])//2 for idx in range(len(tempSize))]
        upperBoundSz = [(tempSize[idx] - newSize[idx] - lowerBoundSz[idx]) for idx in range(len(tempSize))]
        itkimgResampled = sitk.Crop(itkimgResampled, lowerBoundaryCropSize=lowerBoundSz, upperBoundaryCropSize=upperBoundSz)
        return itkimgResampled


def rotation3d(itkimage, theta, show=False):
    theta_x = np.deg2rad(theta[0])
    theta_y = np.deg2rad(theta[1])
    theta_z = np.deg2rad(theta[2])
    euler_transform = sitk.Euler3DTransform(
        itkimage.TransformContinuousIndexToPhysicalPoint([idx / 2 for idx in itkimage.GetSize()]),
        theta_x, theta_y, theta_z, (0, 0, 0))
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(itkimage)  # 需要重新采样的目标图像
    resampler.SetTransform(euler_transform)
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0)
    # print(itkimage.GetPixelIDValue())
    resampler.SetOutputPixelType(itkimage.GetPixelID())
    itkimgResampled = resampler.Execute(itkimage)  # 得到重新采样后的图像
    if show:
        plt.imshow(sitk.GetArrayFromImage(itkimgResampled)[:, :, 100])
        plt.show()
    return itkimgResampled


def register(fixed, moving, numberOfBins=48, samplingPercentage = 0.10):
    initx = sitk.CenteredTransformInitializer(sitk.Cast(fixed, moving.GetPixelID()), moving, sitk.Euler3DTransform(),
                                              operationMode=sitk.CenteredTransformInitializerFilter.GEOMETRY)
    R = sitk.ImageRegistrationMethod()
    R.SetMetricAsMattesMutualInformation(numberOfBins)
    R.SetMetricSamplingPercentage(samplingPercentage, sitk.sitkWallClock)
    R.SetMetricSamplingStrategy(R.REGULAR)
    R.SetOptimizerAsRegularStepGradientDescent(1.0, .001, 50)
    R.SetInitialTransform(initx)
    R.SetInterpolator(sitk.sitkLinear)
    outTx = R.Execute(sitk.Cast(fixed, sitk.sitkFloat32), sitk.Cast(moving, sitk.sitkFloat32))
    return outTx


def ct_gray2rgb(CT_gray):
    CT1_data = (np.maximum(np.minimum(CT_gray, 1600), 400) - 400) / 600 - 1.0
    CT2_data = (np.maximum(np.minimum(CT_gray, 400), -100) + 100) / 250 - 1.0
    CT3_data = (np.maximum(np.minimum(CT_gray, -100), -1000) + 1000) / 450 - 1.0
    CT_data = np.concatenate((CT1_data, CT2_data, CT3_data), axis=-1)
    return CT_data


def ct_rgb2gray(CT_rgb):
    CT_rgb[:, :, :, 0] = (CT_rgb[:, :, :, 0] + 1.0) * 600 + 400
    CT_rgb[:, :, :, 1] = (CT_rgb[:, :, :, 1] + 1.0) * 250 - 100
    CT_rgb[:, :, :, 2] = (CT_rgb[:, :, :, 2] + 1.0) * 450 - 1000
    CT_data = np.sum(CT_rgb, axis=-1, keepdims=True) + 100 - 400
    return CT_data


def standard_normalization(IMG, remove_tail=True, divide='mean'):
    data = sitk.GetArrayFromImage(IMG)
    data = data - np.min(data)
    IMG = IMG - np.min(data)
    maxvalue, meanvalue, stdvalue = np.max(data), np.mean(data), np.std(data)
    # print(maxvalue, meanvalue, stdvalue)
    data_mask = np.logical_and(data > 0.1*meanvalue, data < meanvalue+stdvalue*3.0)
    mean, std = np.mean(data[data_mask]), np.std(data[data_mask])
    if divide == 'mean':
        nIMG = (IMG - mean) / mean
        min = -1
    else:
        nIMG = (IMG - mean)/std
        min = -mean/std
    if remove_tail:
        nIMG = sitk.Cast(sitk.Minimum(nIMG, sitk.Sqrt(sitk.Maximum(nIMG, 0))) - min, sitk.sitkFloat32)/2.0
    else:
        nIMG = sitk.Cast(nIMG - min, sitk.sitkFloat32) / 2.0
    return nIMG


def histogram_normalization(IMG, thres=0.98, ref_max=1.00, range=None):
    data = sitk.GetArrayFromImage(IMG)
    maxvalue = np.max(data)
    data = data / maxvalue*1000
    hist, edges = np.histogram(data[data > 1], 1000, range=range, density=True)
    hist[0] = 0
    hist = np.cumsum(hist / np.sum(hist))
    hist = hist / hist[-1]
    nIMG = sitk.Cast(IMG, sitk.sitkFloat32) * (ref_max / np.maximum(edges[np.argwhere(hist > thres)[0]], 1e-6))
    return nIMG


def get_aug_crops(center, shift, aug_step, aug_count=1, aug_index=(1,), aug_model='sequency'):
    if aug_model == 'random':
        aug_crops = [
            [center[dim] + min(random.randrange(-shift[dim], shift[dim] + aug_step[dim], aug_step[dim]), shift[dim])
             for dim in range(3)] for idx in range(aug_count)]
        count_of_augs = aug_count
    elif aug_model == 'sequency':
        aug_all_crops = [[center[0] + min(idx, shift[0]),
                          center[1] + min(idy, shift[1]),
                          center[2] + min(idz, shift[2]),
                          ] for idx in np.arange(-shift[0], shift[0] + aug_step[0], aug_step[0])
                         for idy in np.arange(-shift[1], shift[1] + aug_step[1], aug_step[1])
                         for idz in np.arange(-shift[2], shift[2] + aug_step[2], aug_step[2])]
        aug_crops = [aug_all_crops[idx % len(aug_all_crops)] for idx in aug_index]
        count_of_augs = len(aug_all_crops)
    else:
        aug_crops = [center]
        count_of_augs = 1
    return aug_crops, count_of_augs