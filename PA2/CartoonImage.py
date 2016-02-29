"""
This program generates a cartoon image from a regular image by generating
canny edge detected image and bilateral filtered image from the same and
bitwise ANDing both.
"""

import cv2,math
import numpy as np
from scipy.spatial import distance
from scipy.spatial.distance import cdist

__author__ = "Aditya Pulekar"

def bilateralFilter(img, kSize, sigmaColor, sigmaSpace, borderType=cv2.BORDER_DEFAULT):
    """
    DESCRIBE YOUR FUNCTION HERE
    (Param descriptions from the OpenCV Documentation)
    :param img: Source 8-bit or floating-point, 1-channel or 3-channel image.
    :param kSize: Diameter of each pixel neighborhood that is used during filtering.
        If it is non-positive, it is computed from sigmaSpace.
    :param sigmaColor (sigma r): Filter sigma in the color space. A larger value of the parameter
        means that farther colors within the pixel neighborhood (see sigmaSpace) will
        be mixed together, resulting in larger areas of semi-equal color.
    :param sigmaSpace (sigma s): Filter sigma in the coordinate space. A larger value of the
        parameter means that farther pixels will influence each other as long as their
        colors are close enough (see sigmaColor ). When d>0, it specifies the neighborhood
        size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace.
    :param borderType: always Reflect_101
    :return: Filtered image of same size and type as img
    """
    #TODO: Write a bilateral filter
    details=img.shape
    img = np.array(img)
    rgb = np.array([0.414,0.589,0.299])
    img_gray = np.sum(img * rgb,axis=-1)/255
    img_gray*=255
    channel=0
    kSize=kSize//3
    I_filtered = img

    #For all three Channels
    while (channel < 3):
        for rows in range(details[0]):
            for columns in range(details[1]):
                if rows+kSize < details[0] and columns + kSize < details[1]:
                    neighborhood = img[rows:rows+kSize,columns:columns+kSize,channel]
                elif rows+kSize < details[0] and columns + kSize >= details[1]:
                    neighborhood = img[rows:rows+kSize,columns:,channel]
                elif rows + kSize >= details[0] and columns + kSize < details[1]:
                    neighborhood = img[rows:,columns:columns + kSize,channel]
                else:
                    neighborhood = img[rows:,columns:,channel]
                neighborhood1D= neighborhood.flatten()
                if len(neighborhood1D) != 0:
                    fr = cv2.getGaussianKernel(min(len(neighborhood),len(neighborhood[0])),sigmaColor)
                    gs = cv2.getGaussianKernel(min(len(neighborhood),len(neighborhood[0])),sigmaSpace)

                    mid = len(neighborhood1D)//2
                    Wp = np.cumsum((fr *(neighborhood1D-neighborhood1D[mid]))
                                                * (gs) * distance.euclidean(neighborhood1D,neighborhood1D[mid])) #* )

                    bilOperation= np.cumsum(neighborhood1D * (fr * (neighborhood1D-neighborhood1D[mid]))
                                                * (gs) * distance.euclidean(neighborhood1D,neighborhood1D[mid])) #
                    WpIndex = Wp.size - 1
                    bilOperationIndex = bilOperation.size - 1
                    x = rows+((neighborhood.shape)[0]//2)
                    y = columns+((neighborhood.shape)[1]//2)
                    # print("X: ", x, "Y: ", y)
                    if Wp[WpIndex] != 0 and x < details[0] and y < details[1]:
                        I_filtered[x,y,channel] = bilOperation[bilOperationIndex]/Wp[WpIndex]
                    elif x >= details[0] and y >= details[1]:
                        I_filtered[x,y,channel] = 0.0    #Since bilOperation[bilOperationIndex] will be zero as well
        channel+=1
    return I_filtered

def neighbors(img,i,j,thresh1,thresh2):
    details=img.shape
    if i < 1:
        if j < 1:
            nb = img[i:i+2,j:j+2].flatten()
        else:
            if j + 1 >= details[1]:
                nb = img[i:i+2,j-1:].flatten()
            else:
                nb = img[i:i+2,j-1:j+2].flatten()
    elif i + 1 >=details[0]:
        if j < 1:
            nb = img[i-1:,j:j+2].flatten()
        else:
            if j+1 >= details[1]:
                nb = img[i-1:,j-1:].flatten()
    else:
        nb = img[i-1:i+2,j-1:j+2].flatten()
    if max(nb) >= thresh2 :
        img[i,j]=255
    else:
        img[i,j]=0


def Canny(img, thresh1, thresh2, L2norm):
    """
    DESCRIBE YOUR FUNCTION HERE
    (Param descriptions from the OpenCV Documentation)
    :param img: 8-bit input image.
    :param thresh1: hysteresis threshold 1.
    :param thresh2: hysteresis threshold 2.
    :param L2norm: boolean to choose between L2norm or L1norm
    :return: a single channel 8-bit with the same size as img
    """
    #TODO: Write a canny edge detector
    filImage = cv2.GaussianBlur(img,(3,3),sigmaX=1.4)
    img_gray = cv2.cvtColor(filImage,cv2.COLOR_BGR2GRAY)
    details = img_gray.shape
    img_gray=np.array(img_gray)
    Gx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    Gy = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])
    kSize = 9
    gradient=[]

    gradientX = np.empty([details[0],details[1]])
    gradientY = np.empty([details[0],details[1]])

    gradientX = cv2.Sobel(img_gray,ddepth=-1,dx=1,dy=0)
    gradientY = cv2.Sobel(img_gray,ddepth=-1,dx=0,dy=1)
    netGradient = cv2.Sobel(img_gray,ddepth=-1,dx=1,dy=1)

    netGradientTheta = np.empty([details[0],details[1]])
    for row in range(gradientX.shape[0]):
        for column in range(gradientX.shape[1]):
            if gradientX[row,column] == 0:
                theta = 90
            else:
                theta = (np.arctan(gradientY[row,column]/gradientX[row,column]))*180/np.pi
            if (theta >=0 and theta <22.5) or (theta >=157.5 and theta <=180):
                    theta = 0
            elif theta >= 22.5 and theta < 67.5:
                    theta = 45
            elif theta >= 67.5 and theta < 112.5:
                    theta = 90
            elif theta >= 112.5 and theta < 157.5:
                    theta = 135
            netGradientTheta[row,column]=theta

    for rows in range(details[0]):
        for column in range(details[1]):
            if netGradientTheta[row,column] == 90:
                if  row < 1:
                    if  netGradient[row,column] != max(netGradient[row,column],netGradient[row+1,column]):
                          netGradient[row,column] = 0
                elif row+1 >=details[0]:
                    if  netGradient[row,column] != max(netGradient[row-1,column],netGradient[row,column]):
                        netGradient[row,column] = 0
                elif  netGradient[row,column] != max(netGradient[row-1,column],netGradient[row,column],netGradient[row+1,column]):
                    netGradient[row,column] = 0

            elif netGradientTheta[row,column] == 45:
                if row < 1 and column+1 >= details[1]:
                    if  netGradient[row,column] != max(netGradient[row,column],netGradient[row+1,column-1]):
                        netGradient[row,column] = 0
                elif row+1 >= details[0] and column < 1:
                    if  netGradient[row,column] != max(netGradient[row-1,column+1],netGradient[row,column]):
                        netGradient[row,column] = 0
                elif  netGradient[row,column] != max(netGradient[row-1,column+1],netGradient[row,column],netGradient[row+1,column-1]):
                    netGradient[row,column] = 0


            elif netGradientTheta[row,column] == 0:
                if column < 1:
                    if  netGradient[row,column] != max(netGradient[row,column],netGradient[row,column+1]):
                        netGradient[row,column] = 0
                elif column+1 >= details[1]:
                    if  netGradient[row,column] != max(netGradient[row,column-1],netGradient[row,column]):
                        netGradient[row,column] = 0
                elif  netGradient[row,column] != max(netGradient[row,column-1],netGradient[row,column],netGradient[row,column+1]):
                    netGradient[row,column] = 0

            else:
                if row < 1 and column < 1:
                    if  netGradient[row,column] != max(netGradient[row,column],netGradient[row+1,column+1]):
                        netGradient[row,column] = 0
                elif row+1 >= details[0] and column+1 >= details[1]:
                    if  netGradient[row,column] != max(netGradient[row-1,column+1],netGradient[row,column]):
                        netGradient[row,column] = 0
                elif  netGradient[row,column] != max(netGradient[row-1,column-1],netGradient[row,column],netGradient[row+1,column+1]):
                    netGradient[row,column] = 0

    #Drawing edges
    for row in range(details[0]):
        for column in range(details[1]):
            if netGradient[row,column] < thresh1:
                netGradient[row,column]=0
            # elif netGradient[row,column] >= thresh1 and netGradient[row,column] < thresh2:
            #     neighbors(netGradient,row,column,thresh1,thresh2)
            else:
                netGradient[row,column] = 255

    netGradient = cv2.adaptiveThreshold(cv2.cvtColor(img,cv2.COLOR_BGR2GRAY), 255,
                                  adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C,thresholdType=cv2.THRESH_BINARY,blockSize=3
                                  ,C=2)
    return netGradient

def cartoonImage(filtered, edges):
    """
    DESCRIBE YOUR FUNCTION HERE
    :param filtered: a bilateral filtered image
    :param edges: a canny edge image
    :return: a cartoon image
    """
    #TODO: Create a cartoon image
    color_edge = cv2.cvtColor(edges,cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(filtered,color_edge)
    return cartoon

def RMSerror(img1, img2):
    """
    A testing function to see how close your images match expectations
    Try to make sure your error is under 1. Some floating point error will occur.
    :param img1: Image 1
    :param img2: Image 2
    :return: The error between the two images
    """
    diff = np.subtract(img1.astype(np.float64), img2.astype(np.float64))
    squaredErr = np.square(diff)
    meanSE = np.divide(np.sum(squaredErr), squaredErr.size)
    RMSE = np.sqrt(meanSE)
    return RMSE

if __name__ == '__main__':
    img = cv2.imread("Castle.jpg")
    bilat = bilateralFilter(img, 9, 50, 100)
    cvbilat = cv2.bilateralFilter(img, 9, 50, 100)
    print "Bilateral Filter RMSE: "+str(RMSerror(bilat, cvbilat))
    edges = Canny(img, 100, 200, True)

    cvedges = cv2.Canny(img, 100, 200, True)
    print "Canny Edge RMSE: "+str(RMSerror(edges, cvedges))
    cartoon = cartoonImage(bilat, edges)
    cv2.imshow("Bilateral", bilat)
    cv2.imwrite("BilateralOutput.jpg", bilat)
    cv2.imshow("Edges", edges)
    cv2.imwrite("CannyOutput.jpg", edges)
    cv2.imshow("Cartoon", cartoon)
    cv2.imwrite("CartoonOutput.jpg", cartoon)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# def findEuclDist(neighborhood,r,c,midY,midX):
#     dist=0
#     for row in range(r,r+len(neighborhood)):
#         for col in range(c,c+len(neighborhood[0])):
#             #np.sqrt(np.sum(np.power(mid-,2),axis=-1))
#             dist+= np.sqrt(np.power(midY-row,2) + np.power(midX-col,2))
#     return dist