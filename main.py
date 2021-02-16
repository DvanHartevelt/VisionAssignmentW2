"""
Demonstrating Local operations
"""

import cv2
from localOperator import kernelSweeper as ks

def query_resize(imgOrig, resizeNr, name='unnamed'):
    smallerPic = input(f"Would you like to resize the {name} image to be one {resizeNr ** 2}th the size? [y/n]")
    if (smallerPic.lower() == 'y'):
        print(f"{name.capitalize()} image resized to one {resizeNr ** 2}th the size.")
        img = cv2.resize(imgOrig, (int(imgOrig.shape[1]/resizeNr), int(imgOrig.shape[0]/resizeNr)))
        return img
    elif (smallerPic.lower() == 'n'):
        print("Image kept the same size.")
        return imgOrig
    else:
        print("Please answer with a [y/n].")
        return query_resize(imgOrig, resizeNr)

def main():
    print("Importing pictures...")
    imgOrig = cv2.imread("Resources/smollCat.jpeg")
    imgMoonO = cv2.imread("Resources/Moon.jpeg")

    imgCat = query_resize(imgOrig, 3, name='cat')
    imgGrayCat = cv2.cvtColor(imgCat, cv2.COLOR_RGB2GRAY)

    imgMoon = query_resize(imgMoonO, 2, name='moon')
    imgGrayMoon = cv2.cvtColor(imgMoon, cv2.COLOR_RGB2GRAY)

    if input("Skip blurring? [y/n]").lower() != 'y':
        """
        Demonstrating Blurring
        """
        imgBlurred   = ks.kernelBlur3.convolute(imgGrayCat)
        imgBigBlur   = ks.kernelBlur5.convolute(imgGray)

        cv2.imshow("Original", imgGrayCat)
        cv2.imshow("Blurred 3", imgBlurred)
        cv2.imshow("Blurred 5", imgBigBlur)

        cv2.waitKey(0)

    if input("Skip Laplacian? [y/n]").lower() != 'y':
        """
        Demonstrating Laplacian, and composite Laplacian counterparts
        """
        imgSquareLap   = ks.kernelSquareLaplacian.convolute(imgGrayMoon)
        imgSquareCLap  = ks.kernelSquareCompositeLaplacian.convolute(imgGrayMoon)
        imgFullLap     = ks.kernelFullLaplacian.convolute(imgGrayMoon)
        imgFullCLap    = ks.kernelFullCompositeLaplacian.convolute(imgGrayMoon)

        cv2.imshow("Original Moon", imgGrayMoon)
        cv2.imshow("Square Lap. Filter", imgSquareLap)
        cv2.imshow("Square Lap. sharpened", imgSquareCLap)
        cv2.imshow("Full Lap. Filter", imgFullLap)
        cv2.imshow("Full Lap. sharpened", imgFullCLap)

    if input("Skip Prewitt? [y,n]").lower() != 'y':
        """
        Demonstrating Prewitt filter
        """
        imgPrewitt = ks.PrewittFilter(imgGrayMoon)

        cv2.imshow("Original Moon", imgGrayMoon)
        cv2.imshow("Prewitt Moon", imgPrewitt)



    # if (input("Skip blurring? [y/n]").lower() != 'y'):
    #     imgBlurred   = ks.kernelBlur3.sweep(imgGrayCat)
    #     #imgBigBlur   = ks.kernelBlur5.sweep(imgGray)
    #     imgSQLapHP   = ks.kernelSquareLaplacianHighPass.sweep(imgGrayCat)
    #     imgFullLapHP = ks.kernelFullLaplacianHighPass.sweep(imgGrayCat)
    #
    #     cv2.imshow("Original", imgGrayCat)
    #     cv2.imshow("Blurred 3", imgBlurred)
    #     #cv2.imshow("Blurred 5", imgBigBlur)
    #
    #
    #
    #     #cv2.waitKey(0)
    #
    #
    # imgNorth = ks.kernelEdgeDetectionHorizontal.sweep(imgGrayCat)
    # imgSouth = imgNorth
    # imgEast  = ks.kernelEdgeDetectionVertical.sweep(imgGrayCat)
    #
    # cv2.imshow("Original", imgGrayCat)
    # cv2.imshow("North and South edge detection", imgNorth)
    # cv2.imshow("East edge detection", imgNorth)


    cv2.waitKey(0)

if __name__ == '__main__':
    main()
