"""
This contains code for a (3x3) kernal sweeper.

A kernel is an nxn matrix of numbers, where n is an odd number.
By calculating the dotproduct of the kernal and the imagematrix,
  some information or aspect of a picture can be distilled/enhanced.

"""

import numpy as np
from localOperator import ProgressBar as pb

class oddkernel:
    """
    This class contains a kernel of any odd size.

    Arguments:
     self.xDim:      - Required: number of elements in the x direction
     self.yDim:      - Required: number of elements in the y direction
     self.valueMat:  - Required: ValueMatrix of size xDim by yDim
     self.name:      - Optional: name of the kernel

    internally used variables
     self.xMiddle:      middle coordinate for x direction
     self.yMiddle:      middle coordinate for y direction
     self.lowFactor:    normalizing factor, at the lowest end
     self.highFactor:   normalizing factor, at the highest end

    please use getValue() to retreive a value from the matrix.

    The method sweep() to sweep this kernel over a picture
    """
    def __init__(self, size, valueMat, name='unnamed'):
        self.xDim = size[0]
        self.yDim = size[1]
        self.valueMat = valueMat
        self.name = name

        self.xMiddle = int((self.xDim - 1) / 2)
        self.yMiddle = int((self.yDim - 1) / 2)
        self.extract_normalizing_factors()

    def getValue(self, i, j):
        """
        Retrieves value of kernal at position i and j , with respect to the CENTER of the kernel
        :param i: horizontal integer position with respect to the center
        :param y: vertical integer position with respect to the center
        :return: value of matrix
        """
        # exception testing
        if  abs(i) > self.xMiddle or abs(j) > self.yMiddle:
            print("ERROR, coordinate out of scope.")
            return None

        return self.valueMat[self.yMiddle + i][self.xMiddle + j]

    def extract_normalizing_factors(self):
        """
        Extracts normalizing factors from the kernel.
        """
        self.lowFactor = 0
        self.highFactor = 0

        for y in range(-1 * self.yMiddle, self.yMiddle):
            for x in range(-1 * self.xMiddle, self.xMiddle):
                if self.getValue(x, y) > 0:
                    self.highFactor += self.getValue(x, y)
                else:
                    self.lowFactor += self.getValue(x, y)

        pass

    def sweep(self, img):
        """
        Sweeps a kernel over an image

        :param img: (gray) image matrix
        :return: sweeped image matrix
        """

        # exception testing
        if len(img.shape) != 2:
            print("image has to be grayscaled.")
            return img

        width = img.shape[1]
        height = img.shape[0]

        imgNew = np.zeros((height, width), np.uint8)

        # 2D sweep of an odd-sized kernel
        for y in range(self.yMiddle, height - self.yMiddle):
            for x in range(self.xMiddle, width - self.xMiddle):
                # Every pixel of the new picture is a multiplication of the neigbouring
                # pixels multiplied by the kernels relative value.
                newValue = 0

                for j in range(-1 * self.yMiddle, self.yMiddle):
                    for i in range(-1 * self.xMiddle, self.xMiddle):
                        newValue += img[y + j, x + i] * self.getValue(i, j)

                newValue = np.interp(newValue, [self.lowFactor*255, self.highFactor*255], [0, 255])

                imgNew[y,x] = int(newValue)

            pb.printProgressBar(y + self.yMiddle, height - self.yMiddle, prefix=f'Sweeping {self.name} kernel, size {[self.xDim, self.yDim]}:', length=50)

        return imgNew

"""
Now to declare some commonly used kernels.
"""
kernelBlur3 = oddkernel([3, 3], [[1, 1, 1], [1, 1, 1], [1, 1, 1]], name='blurring')
kernelBlur5 = oddkernel([5, 5], [[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]], name='blurring')

kernelGausianLowPass3 = oddkernel([3, 3], [[1, 2, 1], [2, 4, 2], [1, 2, 1]], name='Gausian low-pass')
kernelGausianLowPass5 = oddkernel([5, 5], [[2, 7, 12, 7, 2], [7, 31, 52, 31, 7], [12, 52, 127, 52, 12], [7, 31, 52, 31, 7], [2, 7, 12, 7, 2]], name="Gausian low-pass")

kernelSquareLaplacianHighPass = oddkernel([3, 3], [[0, -1, 0], [-1, 4, -1], [0, -1, 0]], name='square Laplacian high-pass')
kernelFullLaplacianHighPass   = oddkernel([3, 3], [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], name='full Laplacian high-pass')

kernelEdgeDetectionHorizontal = oddkernel([3, 3], [[-1, -1, -1], [2, 2, 2],   [-1, -1, -1]], name='horizontal edge detecting')
kernelEdgeDetectionPlus45     = oddkernel([3, 3], [[2, -1, -1],  [-1, 2, -1], [-1, -1, 2]] , name='+45 deg. edge detecting')
kernelEdgeDetectionVertical   = oddkernel([3, 3], [[-1, 2, -1],  [-1, 2, -1], [-1, 2, -1]] , name='vertical edge detecting')
kernelEdgeDetectionMinus45    = oddkernel([3, 3], [[-1, -1, 2],  [-1, 2, -1], [2, -1, -1]] , name='-45 deg. edge detecting')