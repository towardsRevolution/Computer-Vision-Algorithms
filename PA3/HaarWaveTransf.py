"""
This program gives a 10:1 or more compression ratio for the given image using the Haar Wavelet Transform technique
and also reconstructs the image back from the doctored image (i.e. the compressed stored image)
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt

__author__ = "Aditya Pulekar"

def neighborhoodIntoW(neighborhood,W):
     """
     Performs the dot product of images with a vector
     :return: the resultant matrix after the dot product
     """
     everyRow =[]
     for r in range(8):
        newRow = np.dot(neighborhood[r,:],W)
        everyRow.append(newRow)
     return everyRow

def dataStrEval(I,details,W):
    """
    Performs the row-column transformation evaluation for an entire image
    :param I: Image
    :param details: dimensions of the image
    :param W: vector (could be wavelet or inverse of wavelet)
    :return: the new image matrix
    """
    newImg = np.empty([details[0],details[1]])
    for rows in range(details[0]-8):
        for columns in range(details[1]-8):
            neighborhood = I[rows:rows+8,columns:columns+8]
            new8By8 = neighborhoodIntoW(neighborhood,W)
            newImg[rows:rows+8,columns:columns+8] = new8By8
    return newImg

def main():
    img = cv2.imread("buttress.jpg")
    cv2.imshow("Original image (P)",img)
    cv2.waitKey(0)
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    details_Ori = grayImg.shape
    r=0;c=0
    if details_Ori[0] % 8 != 0:
        r = (details_Ori[0]/8 + 1)*8 - details_Ori[0]
    if details_Ori[1] % 8 != 0:
        c = (details_Ori[1]/8 + 1)*8 - details_Ori[1]
    newImg = np.empty([details_Ori[0]+r,details_Ori[1]+c])
    for row in range(newImg.shape[0]):
        for col in range(newImg.shape[1]):
           if row < details_Ori[0] and col < details_Ori[1]:
               newImg[row,col] = grayImg[row,col]
    details = newImg.shape
    #We will using normalization method.
    A1 = np.matrix('0.707 0 0 0 0.707 0 0 0;0.707 0 0 0 -0.707 0 0 0;\
                  0 0.707 0 0 0 0.707 0 0;0 0.707 0 0 0 -0.707 0 0;\
                  0 0 0.707 0 0 0 0.707 0;0 0 0.707 0 0 0 -0.707 0;\
                  0 0 0 0.707 0 0 0 0.707;0 0 0 0.707 0 0 0 -0.707')
    A2 = np.matrix('0.707 0 0.707 0 0 0 0 0;0.707 0 -0.707 0 0 0 0 0;\
                  0 0.707 0 0.707 0 0 0 0;0 0.707 0 -0.707 0 0 0 0;\
                  0 0 0 0 1 0 0 0;0 0 0 0 0 1 0 0;\
                  0 0 0 0 0 0 1 0;0 0 0 0 0 0 0 1')
    A3 = np.matrix('0.707 0 0 0 0.707 0 0 0;0.707 0 0 0 -0.707 0 0 0;\
                  0 0 1 0 0 0 0 0;0 0 0 1 0 0 0 0;\
                  0 0 0 0 1 0 0 0;0 0 0 0 0 1 0 0;\
                  0 0 0 0 0 0 1 0;0 0 0 0 0 0 0 1')
    init_W = np.dot(A1,A2)
    W = np.dot(init_W,A3)
    newGrayImg = dataStrEval(newImg,details,W)

    #Now take the transpose of the obtained matrix to get the column transformation
    colTrans=newGrayImg.transpose()
    newDetails = colTrans.shape
    TransfImage = dataStrEval(colTrans,newDetails,W)
    finalTransfImg = TransfImage.transpose()
    max=np.amax(finalTransfImg.flatten())
    normFinalImg = finalTransfImg/(max+50)
    cv2.imshow("Wavelet Transformed Image (T)",normFinalImg)
    cv2.waitKey(0)

    # Note: We have to select the threshold for the doctored image in such a way as to
    # balance the conflicting requirements of storage (the more zeros in D, the better)
    # and visual acceptability of the reconstruction R
    DoctoredImg = np.where((finalTransfImg)<(0.81*max),0,finalTransfImg)
    maxForD = np.amax(DoctoredImg.flatten())
    cv2.imshow("Doctored image (D)",DoctoredImg/(maxForD+50))
    cv2.waitKey(0)
    NonZerosForD = len(DoctoredImg[np.nonzero(DoctoredImg)])
    NonZerosForTransf = len(finalTransfImg[np.nonzero(finalTransfImg)])
    print "No. of non-zero pixel values in the Doctored image: ",NonZerosForD
    print "No. of non-zero pixel values in the Wavelet Transformed image: ",NonZerosForTransf
    ratio = float(NonZerosForTransf.__float__()/NonZerosForD)
    print "Compression ratio: ", ratio

    A1_inverse = np.linalg.inv(A1)
    A2_inverse = np.linalg.inv(A2)
    A3_inverse = A3
    init_W_inv = np.dot(A1_inverse,A2_inverse)
    W_inverse = np.dot(init_W_inv,A3)
    transposeD = finalTransfImg.transpose()
    reconstImage1 = dataStrEval(transposeD,details,W_inverse)
    transpReconstImg = reconstImage1.transpose()
    final_R = dataStrEval(transpReconstImg,details,W_inverse)
    maxForFinalR = np.amax(final_R.flatten())
    cv2.imshow("Reconstructed Image from the doctored image (R)",final_R/(maxForFinalR+50))
    cv2.waitKey(0)


if __name__ == "__main__":
    main()