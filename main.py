"""
Demonstrating Local operations
"""

import cv2
from localOperator import kernelSweeper as ks

def query_resize(imgOrig, resizeNr):
    smallerPic = input("Would you like to resize the picture to be smaller? [y/n]")
    if (smallerPic.lower() == 'y'):
        print(f"Image resized to one {resizeNr ** 2}th the size.")
        img = cv2.resize(imgOrig, (int(imgOrig.shape[1]/resizeNr), int(imgOrig.shape[0]/resizeNr)))
        return img
    elif (smallerPic.lower() == 'n'):
        print("Image kept the same size.")
        return imgOrig
    else:
        print("Please answer with a [y/n].")
        return query_resize(imgOrig, resizeNr)

def main():
    print("Hello World!")
    imgOrig = cv2.imread("Resources/smollCat.jpeg")

    img = query_resize(imgOrig, 3)

    imgGray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    if (input("Skip blurring? [y/n]").lower() != 'y'):
        imgBlurred   = ks.kernelBlur3.sweep(imgGray)
        imgBigBlur   = ks.kernelBlur5.sweep(imgGray)
        imgSQLapHP   = ks.kernelSquareLaplacianHighPass.sweep(imgGray)
        imgFullLapHP = ks.kernelFullLaplacianHighPass.sweep(imgGray)

        cv2.imshow("Original", imgGray)
        cv2.imshow("Blurred 3", imgBlurred)
        cv2.imshow("Blurred 5", imgBigBlur)
        cv2.imshow("Square Laplacian High-Pass filter", imgSQLapHP)
        cv2.imshow("Full Laplacian High-Pass filter", imgFullLapHP)



        cv2.waitKey(0)


    imgNorth = ks.kernelEdgeDetectionHorizontal.sweep(imgGray)
    imgSouth = imgNorth
    imgEast  = ks.kernelEdgeDetectionVertical.sweep(imgGray)

    cv2.imshow("Original", imgGray)
    cv2.imshow("North and South edge detection", imgNorth)
    cv2.imshow("East edge detection", imgNorth)


    cv2.waitKey(0)

if __name__ == '__main__':
    main()
