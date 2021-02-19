"""
This code contains a few image processing functions, including a 3x3 kernel convoluter.

17/02/2021
David van Hartevelt
"""

import numpy as np
from localOperator import ProgressBar as pb

def gray_convolute3x3(img, kernel, normalize=True):
    #exception testing
    if len(img.shape) != 2:
        print("Image has to be grayscale.")
        return img

    if len(kernel) != 3:
        print("kernel has to be size 3x3.")
        return img

    for i in range(3):
        if len(kernel[i]) != 3:
            print("kernel has to be size 3x3.")
            return img

    if normalize:
        lowfac, highfac = extractNormalizeFactors(kernel)

    width = img.shape[1]
    height = img.shape[0]

    imgNew = np.zeros((height, width))

    # A sweep over all the pixels in img
    for y in range(1, height - 1):
        for x in range(1, width - 1):
            newValue = 0

            # using the dotproduct of our kernel and a slice of img
            for j in range(-1, 2):
                for i in range(-1, 2):
                    #print(f"[x, y] = {[x,y]}, [i,j] = {[i,j]}.")
                    newValue += img[y + j, x + i] * kernel[1 + j][1 + i]

            if normalize:
                newValue = np.interp(newValue, [lowfac * 255, highfac * 255], [0, 255])

            imgNew[y, x] = newValue

        pb.printProgressBar(y, height - 2, prefix=f'Convoluting...:', length=50)

    return imgNew

def extractNormalizeFactors(kernel):
    lo = 0
    hi = 0
    for j in range(len(kernel)):
        for i in range(len(kernel[0])):
            if kernel[j][i] > 0:
                hi += kernel[j][i]
            else:
                lo += kernel[j][i]

    return lo, hi

"""
Some popular 3x3 filters and operations
"""

kernelBlur = [[1, 1, 1],
              [1, 1, 1],
              [1, 1, 1]]