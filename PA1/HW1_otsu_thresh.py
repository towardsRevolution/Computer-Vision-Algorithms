"""
This program takes a color image as an input and performs
Otsu thresholding on it to yield a binary image.
"""

import cv2,math
import numpy as np

__author__ = "Aditya Pulekar"

def main():
    L = int(input("Enter the total number of gray scale levels(256): "))

    #Reading a colored image
    img = cv2.imread("valley.jpg",1)

    #Converting the color image to a gray scale image
    rgbToGray = np.array([0.114,0.589,0.299])
    img_gray = np.sum(img * rgbToGray, axis=-1)/(L-1)
    img_gray*=255

    #Resizing the colored and gray-scale images to fit in the figure window
    img_gray_small = cv2.resize(img_gray,(0,0),fx=0.4, fy=0.4)
    img_color_small = cv2.resize(img,(0,0), fx=0.4, fy=0.4)
    cv2.imshow("Gray-scale valley image",img_color_small)
    cv2.waitKey(0)

    #Procuring the dimensions of the resized gray-scale image
    details =img_gray_small.shape

    #plotting the gray scale histogram
    hist = [0 for itr in range(L)]
    for i in range(details[0]):
        for j in range(details[1]):
            hist[(img_gray_small[i,j]).__int__()]+=1

    #Calculting the probability distribution function (PDF)
    variances = []
    sumOfPixelValues = details[0]*details[1]
    pdf_gray = [(index.__float__()/sumOfPixelValues) for index in hist]
    index = 0

    #Exhaustive search
    while(index < len(pdf_gray)):
        u0=0; u1=0
        W0_1 = pdf_gray[:index]
        W0=sum(W0_1)
        W1_1 = pdf_gray[index:]
        W1=sum(W1_1)
        for m_ptr in range(index):
            u0 += (m_ptr*pdf_gray[m_ptr])
        for m_ptr2 in range(index,len(pdf_gray)):
            u1 += (m_ptr2*pdf_gray[m_ptr2])
        if W0!=0 and W1!=0:
            u0=u0.__float__()/W0; u1=u1.__float__()/W1
        else:
            u0=0; u1=0
        variances.append(W0*W1*math.pow((u0-u1),2))        #These the inter-class variance values
        index+=1

    #Setting the threshold value for the Gray-scale image to convert it into binary image
    thresh = variances.index(max(variances))

    img_bin = np.empty([details[0],details[1]])
    for rows in range(details[0]):
        for cols in range(details[1]):
            if (img_gray_small[rows,cols]) <= thresh:
                img_bin[rows,cols] = 0
            else:
                img_bin[rows,cols] = 1

    #Displaying the Otsu thresholded image
    cv2.imshow('Otsu Thresholded image', img_bin)
    cv2.waitKey(0)

if __name__ == '__main__':
    main()