"""
This program takes a color image as its input, converts it into a gray and binary
images, subplots images for each of the red, green and blue channels, performs arithm-
etic operations amongst the gray and channel images, plots the histogram, PDF, CDF and, also,
equalizes the image (BONUS SECTION).
"""

import cv2
import numpy,math
import matplotlib.pyplot as plt

__author__ = "Aditya Pulekar"

def main():
    #A dictionary maintaining colors with its indices
    COLORS= { 0 : ("BLUE"),
              1 : ("GREEN"),
              2 : ("RED")
    }
    L = int(input("Enter the number of gray-scale levels to be considered for the image (256): "))

    #Reading a colored image
    img = cv2.imread('flower.png',1)
    cv2.imshow('Aditya Pulekar (Colored)',img)
    cv2.waitKey(0)

    #Writing the input color image to a different format
    cv2.imwrite('flower_new.jpg',img)

    #Procuring the dimensions of the input image
    details = img.shape

    #Converting the color image to a gray scale image
    #Faster (Using Numpy)
    rgbToGray = numpy.array([0.114,0.589,0.299])
    img_gray = numpy.sum(img * rgbToGray, axis=-1)/(L-1)
    cv2.imshow('Gray-scale image',img_gray)
    cv2.waitKey(0)

    img_gray= numpy.array(img_gray,dtype='float32')
    img_binary = numpy.empty([details[0],details[1]])

    #Converting gray scale image to binary image using a threshold value of 0.5
    for rows in range(details[0]):
        for cols in range(details[1]):
            if img_gray[rows,cols] <= 0.5:
                img_binary[rows,cols] = 0
            else:
                img_binary[rows,cols] = 1

    #Inbuilt function for gray to binary conversion
    # flag, img_binary = cv2.threshold(img_gray,0.5,1,cv2.THRESH_BINARY)

    cv2.imshow('Binary Image', img_binary)
    cv2.waitKey(0)

    #Displaying images for the RGB Channels in subplots
    img_Blue_Channel = numpy.array(img[:,:,0])
    img_Green_Channel = numpy.array(img[:,:,1])
    img_Red_Channel = numpy.array(img[:,:,2])

    plt.figure(1)
    plt.subplot(221)
    plt.title("Images for RGB Channels")
    plt.ylabel("Blue Channel-->")
    plt.imshow(img_Blue_Channel,'Blues',aspect='equal')


    plt.subplot(222)
    # plt.plot(img_Green_Channel)
    plt.ylabel("Green Channel-->")
    plt.imshow(img_Green_Channel,'Greens',aspect='equal')

    plt.subplot(223)
    # plt.plot(img_Red_Channel)
    plt.ylabel("Red Channel-->")
    plt.imshow(img_Red_Channel,'Reds',aspect='equal')

    plt.show()

    #Some Arithmetic Operations on the images

    #Treatment of negative pixel values differ as per the implementation of the operator. Negative pixel values may be
    #set to zero if the image format does not support it, else it may be wrapped to certain values (For example, a pixel
    #value of -20 may be wrapped to 236


    blue_Red = img_Blue_Channel + img_Red_Channel
    red_green = img_Red_Channel + img_Green_Channel
    blue_green = img_Blue_Channel + img_Green_Channel
    img_gray2 = img_gray *  L-1
    gray_blue = img_Blue_Channel - img_gray2

    plt.figure(2)
    plt.subplot(321)
    plt.title("Arithmetic Operations on the image")
    plt.imshow(blue_Red,aspect='equal',cmap=plt.cm.gray)
    plt.ylabel("B + R")

    plt.subplot(322)
    plt.imshow(red_green,aspect='equal',cmap=plt.cm.gray)
    plt.ylabel("R + G")

    plt.subplot(323)
    plt.imshow(blue_green,aspect='equal',cmap=plt.cm.gray)
    plt.ylabel("B + G")

    plt.subplot(324)
    plt.imshow(gray_blue,aspect='equal',cmap=plt.cm.gray)
    plt.ylabel("Gray - B")

    plt.show()

    # (NOTE)Below two operations are invalid
    # gray_green_Div = img_gray / img_Green_Channel
    # red_gray_Div = img_Red_Channel / img_gray

    # plt.subplot(325)
    # plt.imshow(gray_green_Div,aspect='equal')
    # plt.ylabel("Gray / G")

    # plt.subplot(326)
    # plt.imshow(red_gray_Div,aspect='equal')
    # plt.ylabel("R / Gray")


    #Plotting the histogram
    cumuSum = 0

    #Plotting the gray scale histogram
    hist = [0 for itr in range(L)]
    for i in range(details[0]):
        for j in range(details[1]):
            hist[(img_gray[i,j]*(L-1)).__int__()]+=1
    plt.figure(3)
    plt.subplot(311)
    plt.plot(hist, 'ro-')
    plt.title("Histogram,PDF and CDF of the gray-scale image")
    plt.xlabel("Pixel Intensities--->")
    plt.ylabel("Frequency of Pixels--->")

    #Plotting the probability distribution function(PDF) and cumulative
    #distribution function(CDF)

    sumOfPixelValues = details[0]*details[1]
    pdf_gray = [(index.__float__()/sumOfPixelValues) for index in hist]
    cdf_gray = []
    for i in range(len(pdf_gray)):
       cumuSum += pdf_gray[i]
       cdf_gray.append(cumuSum)

    plt.subplot(312)
    plt.plot(pdf_gray, 'go-')
    plt.xlabel("Pixel Intensities--->")
    plt.ylabel("Probability -->")

    plt.subplot(313)
    plt.plot(cdf_gray, 'bo-')
    plt.xlabel("Pixel Intensities -->")
    plt.ylabel("Cumu Probabilty -->")

    plt.show()


    #You may try it on "flower.png" as well as "hazecity.png"
    #Histogram Equalization for the given image (BONUS)
    cdf_new = []

    #CDF Calculation
    for i in range(len(hist)):
       cumuSum += hist[i]
       cdf_new.append(cumuSum)
    hist_Eq = []
    for index in range(L):
        hist_Eq.append(math.floor(((cdf_new[index] - min(cdf_new))/((details[0]*details[1]) - min(cdf_new)))*(L-1)))

    plt.figure(4)

    #Equalized Image
    newImage = numpy.empty([details[0],details[1]])
    for rows in range(details[0]):
        for cols in range(details[1]):
            newImage[rows,cols] = hist_Eq[(img_gray[rows,cols]*(L-1)).__int__()]

    print "Image Dimensions: ",newImage.shape
    plt.title("Image after Histogram Equalization")
    plt.imshow(newImage,cmap=plt.cm.gray)
    plt.show()

    #OPTIONAL SECTION
    #channels = 0
    #
    #Finding the color histogram and PDF,CDF for individual channels
    # while channels < 3:
    #     scale = [0 for i in range(L)]
    #     for row in range(details[0]):
    #         for col in range(details[1]):
    #             scale[img[row,col,channels]]+=1
    #     # fig = plt.figure()
    #     plt.figure(5)
    #     plt.subplot(311)
    #     plt.plot(scale, 'ro-')
    #     # fig.suptitle("Aditya Pulekar( " + COLORS[channels] + " Channel's Histogram" + " )")
    #     plt.title("Aditya Pulekar( " + COLORS[channels] + " Channel's Histogram, PDF and CDF" + " )")
    #     plt.xlabel("Pixel Intensities--->")
    #     plt.ylabel("Frequency of Pixels--->")
    #
    #     #PDF
    #     sumOfPixelValues = details[0]*details[1]
    #     pdf = [(index.__float__()/sumOfPixelValues) for index in scale]
    #     # print pdf
    #     # plt.figure(2)
    #     plt.subplot(312)
    #     plt.plot(pdf, 'go-')
    #     plt.xlabel("Pixel Intensities--->")
    #     plt.ylabel("Probability -->")
    #
    #     #CDF
    #     cdf = []
    #     cumuSum = 0
    #     for i in range(len(pdf)):
    #         cumuSum += pdf[i]
    #         cdf.append(cumuSum)
    #     plt.subplot(313)
    #     plt.plot(cdf, 'bo-')
    #     plt.xlabel("Pixel Intensities -->")
    #     plt.ylabel("Cumu Probabilty -->")
    #     plt.show()
    #     channels+=1
    # plt.show()

if __name__ == '__main__':
    main()