import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import colorspacious as cs

def plot(img, rescale = False):
    plt.figure(1)
    if(rescale):
        plt.imshow(rescale(img), cmap='Greys_r')
    else:
        plt.imshow(clip(img), cmap='Greys_r')
    
def comparePlot(original_image, img, rescale=False):
    fig = plt.figure(figsize=(16, 8))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2)
    if(rescale):
        ax1.imshow(rescale(original_image), cmap='Greys_r')
        ax2.imshow(rescale(img), cmap='Greys_r')
    else:
        ax1.imshow(clip(original_image), cmap='Greys_r')
        ax2.imshow(clip(img), cmap='Greys_r')

def comparePlotList(original_image, imgList, labels, maxCols = 2, rescale=False):
    fig = plt.figure(figsize=(16, 8))
    imgList = [original_image] + imgList
    labels = ["original"] + labels
    cols = min(maxCols, len(imgList) + 1)
    rows = (len(imgList) + 1) / cols
    for i in range(len(imgList)):
        ax1 = fig.add_subplot(rows, cols, i + 1)
        ax1.set_title(labels[i])
        if(rescale):
            ax1.imshow(rescale(imgList[i]), cmap='Greys_r')
        else:
            ax1.imshow(clip(imgList[i]), cmap='Greys_r')
    
    
def rescale(img, targetMin = 0.0, targetMax = 1.0):
    result = np.copy(img)
    minV, maxV = np.min(img), np.max(img)
    return targetMin + (targetMax - targetMin) / (maxV - minV) * (result - minV)

def clip(img, targetMin = 0.0, targetMax = 1.0):
    result = np.copy(img)
    result[result < targetMin] = targetMin
    result[result > targetMax] = targetMax
    return result

def shiftAndNormalize(img, targetMean):
    mu = np.mean(img)
    result = np.copy(img) - mu
    maxI, minI = np.max(result), np.min(result)
    s1, s2 = (1.0 - targetMean) / maxI, - targetMean / minI
    result = min(s1, s2) * result + targetMean
    return result
    
def w_frac(i, j, gi, gj, alpha_s, alpha_r):
    imj = i - j
    gimgj = gi - gj
    return 1 / ((np.dot(imj, imj) ** (alpha_s / 2) + 0.0001) * (np.dot(gimgj, gimgj) ** (alpha_r / 2) + 0.0001))

def w_exp(i, j, gi, gj, sigma_s, sigma_r):
    imj = i - j
    gimgj = gi - gj
    return np.e**(- np.dot(imj, imj) / (2 * sigma_s**2)) * np.e(- np.dot(gimgj, gimgj) / (2 * sigma_r**2))

def buildA(img, lambda_, r, par1, par2, useFracWeights=True):
    m , n = img.shape[0], img.shape[1] 
    size = m * n

    rowInd = []
    colInd = []
    data = []


    for i in range(m):
        mink = max(i - r, 0)
        for j in range(n):
            minl = max(j - r, 0)
            I = i * n + j
            tot = 0
            for k in range(mink, min(i + r + 1, m)):
                for l in range(minl, min(j + r + 1, n)):
                    J = k * n + l
                    if I != J:
                        if (useFracWeights):
                            val = lambda_ * w_frac(np.array([i, j]), np.array([k, l]), 
                                           img[i, j], img[k, l], par1, par2)
                        else:
                            val = lambda_ * w_exp(np.array([i, j]), np.array([k, l]), 
                                           img[i, j], img[k, l], par1, par2)
                        tot += val
                        rowInd.append(I)
                        colInd.append(J)
                        data.append(-val)
            rowInd.append(I)
            colInd.append(I)
            data.append(tot + 1)
   
    return sp.coo_matrix((data, (rowInd, colInd)), shape=(size, size))

def sigmoid(a,x):
    temp = 1.0 / (1.0 + np.exp(-a * x)) - 0.5
    temp1 = 1.0 / (1.0 + np.exp(-a * 0.5)) - 0.5
    return temp * 0.5/temp1

def multiScaleCorrection(base, details, baseExposure, baseBoost, detailBoosts):
    result  = sigmoid(baseBoost, (baseExposure * base - 56) / 100)*100 + 56
    for D, d in zip(details, detailBoosts):
        result += sigmoid(d, D/100)*100
    return result
    
def changeLightness(img_CIELab, new_lightness):
    img = np.copy(img_CIELab)
    img[:, :, 0] = new_lightness
    return img
    
def rescaleLightness(img_CIELab, targetMin = 0.0, targetMax = 100.0):
    img = np.copy(img_CIELab)
    minV, maxV = np.min(img[:, :, 0]), np.max(img[:, :, 0])
    img[:, :, 0] = targetMin + (targetMax - targetMin) / (maxV - minV) * img[:, :, 0]
    return img
        
    
def saturateColors(img_CIELab, saturation):
    img = np.copy(img_CIELab)
    img[:, :, 1:3] *= saturation
    return img

def loadData(imagePath):
    original_image = mpimg.imread(imagePath)
    if(original_image.dtype == "uint8"):
        original_image = original_image.astype("float")
        original_image /= 255.0
    print(
    "Original image : shape = {s}, data type = {dt}.".format(
        s=original_image.shape, dt=original_image.dtype))
    
    image_CIELab = cs.cspace_convert(original_image, "sRGB1", "CIELab")
    lightness = image_CIELab[:, :, 0]
    return original_image, image_CIELab, lightness
    
def decomposeImage(original_image, image_CIELab, lightness, lambdas, alphas, r, display=True):
    Us = []
    Us_RGB = []
    (m, n) = lightness.shape
    for lambda_, alpha in zip(lambdas, alphas):
        print("WLS with parameters : alpha = {}, lambda = {}".format(alpha, lambda_))
        print("Building system...")
        A = buildA(original_image, lambda_, r, alpha, alpha)
        print("Solving system...")
        U = (sp.linalg.spsolve(A, lightness.flatten())).reshape((m, n))
        Us.append(U)
        if(display):
            U_RGB = cs.cspace_convert(rescaleLightness(changeLightness(image_CIELab, U)), "CIELab", "sRGB1")
            Us_RGB.append(U_RGB)
    
    if(display):
        print("Multi-scale edge-preserving smoothing results :")
        comparePlotList(original_image, Us_RGB, ["lambda = " + str(l) for l in lambdas])
    
    details = []
    details_RGB = []
    prev = lightness
    for U in Us:
        D = prev - U
        details.append(D)
        prev = U
        if(display):
            D_RGB = cs.cspace_convert(rescaleLightness(changeLightness(image_CIELab, D)), "CIELab", "sRGB1")
            details_RGB.append(D_RGB)
        
    base = Us[len(Us) - 1]
    
    if(display):
        print("Multi-scale detail extraction results :")
        comparePlotList(original_image, details_RGB, ["lambda = " + str(l) for l in lambdas])
    
    return base, details
    
def multiScaleEnhancing(original_image, image_CIELab, base, details):
    enhanced_RGB = []
    enhancedFine = multiScaleCorrection(base, details, 1.0, 1.0, [25.0, 1.0])
    enhanced_RGB.append(cs.cspace_convert(saturateColors(changeLightness(image_CIELab, enhancedFine), 1.0), "CIELab", "sRGB1"))
    enhancedMedium = multiScaleCorrection(base, details, 1.0, 1.0, [1.0, 40.0])
    enhanced_RGB.append(cs.cspace_convert(saturateColors(changeLightness(image_CIELab, enhancedMedium), 1.0), "CIELab", "sRGB1"))
    enhancedCoarse = multiScaleCorrection(base, details, 1.1, 15.0, [4.0, 1.0])
    enhanced_RGB.append(cs.cspace_convert(saturateColors(changeLightness(image_CIELab, enhancedCoarse), 1.0), "CIELab", "sRGB1"))
    
    comparePlotList(original_image, enhanced_RGB, ["Fine Scale", "Medium Scale", "Coarse Scale"])
    
def enhancementRoutine(imagePath):
    original_image, image_CIELab, lightness = loadData(imagePath)
    base, details = decomposeImage(original_image, image_CIELab, lightness, lambdas=[0.125, 0.5], alphas=[1.2, 1.2], r=1)
    multiScaleEnhancing(original_image, image_CIELab, base, details)
    