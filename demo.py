import cv2
import numpy,math
import matplotlib.pyplot as plt

def main():
    COLORS= { 0 : ("BLUE"),
              1 : ("GREEN"),
              2 : ("RED")
    }
    img = cv2.imread('flower.png',1)
    cv2.imshow('Aditya Pulekar (Colored)',img)
    cv2.waitKey(0)
    cv2.imwrite('flower_new.jpg',img)
    details = img.shape
    #Faster (Using Numpy)
    rgbToGray = numpy.array([0.114,0.589,0.299])
    img_gray = numpy.sum(img * rgbToGray, axis=-1)/255

    cv2.imshow('Aditya Pulekar (Gray)',img_gray)
    cv2.waitKey(0)

    img_gray= numpy.array(img_gray,dtype='float32')
    img_binary = numpy.empty([details[0],details[1]])

    #Learn to iterate through a 2d matrix in a single statement (try numpy)
    for rows in range(details[0]):
        for cols in range(details[1]):
            if img_gray[rows,cols] <= 0.5:
                img_binary[rows,cols] = 0
            else:
                img_binary[rows,cols] = 1
    # flag, img_binary = cv2.threshold(img_gray,0.5,1,cv2.THRESH_BINARY)
    cv2.imshow('Aditya Pulekar (Binary)', img_binary)
    cv2.waitKey(0)

    img_Blue_Channel = numpy.array(img[:,:,0])
    img_Green_Channel = numpy.array(img[:,:,1])
    img_Red_Channel = numpy.array(img[:,:,2])

    plt.figure(1)
    plt.subplot(311)
    plt.ylabel("Blue Channel-->")
    plt.imshow(img_Blue_Channel,'Blues',aspect='equal')        #plt.imshow(img_Blue_Channel)

    # cv2.title()

    plt.subplot(312)
    # plt.plot(img_Green_Channel)
    plt.ylabel("Green Channel-->")
    plt.imshow(img_Green_Channel,'Greens',aspect='equal')       #"Aditya Pulekar (Green Channel)",

    plt.subplot(313)
    # plt.plot(img_Red_Channel)
    plt.ylabel("Red Channel-->")
    plt.imshow(img_Red_Channel,'Reds',aspect='equal')         #"Aditya Pulekar (Red Channel)",

    plt.show()

    #Some Arithmetic Operations on the images

    #(COPIED)
    # Implementations of the operator vary as to what they do if the output pixel values are negative. Some work with
    # image formats that support negatively-valued pixels, in which case the negative values are fine (and the way in which
    # they are displayed will be determined by the display colormap). If the image format does not support negative numbers
    # then often such pixels are just set to zero (i.e. black typically). Alternatively, the operator may
    # `wrap' negative values, so that for instance -30 appears in the output as 226 (assuming 8-bit pixel values).


    blue_Red = img_Blue_Channel + img_Red_Channel
    red_green = img_Red_Channel + img_Green_Channel
    blue_green = img_Blue_Channel + img_Green_Channel
    img_gray2 = img_gray *  255
    gray_blue = img_Blue_Channel - img_gray2

    plt.figure(2)
    plt.title("Arithmetic Operations on the image")
    plt.subplot(321)
    plt.imshow(blue_Red,aspect='equal')
    plt.ylabel("B + R")

    plt.subplot(322)
    plt.imshow(red_green,aspect='equal')
    plt.ylabel("R + G")

    plt.subplot(323)
    plt.imshow(blue_green,aspect='equal')
    plt.ylabel("B + G")

    plt.subplot(324)
    plt.imshow(gray_blue,aspect='equal')
    plt.ylabel("Gray - B")

    plt.show()

    # Below two operations are invalid (Reason?)
    # gray_green_Div = img_gray / img_Green_Channel
    # red_gray_Div = img_Red_Channel / img_gray

    # plt.subplot(325)
    # plt.imshow(gray_green_Div,aspect='equal')
    # plt.ylabel("Gray / G")

    # plt.subplot(326)
    # plt.imshow(red_gray_Div,aspect='equal')
    # plt.ylabel("R / Gray")



    #Slower Way (Colored to Grayscale)
    # img_gray = numpy.empty([details[0],details[1]])
    # for i in range(details[0]):
    #     for j in range(details[1]):
    #         value = 0.589*img[i,j,2] + 0.299*img[i,j,1] + 0.114*img[i,j,0]
    #         img_gray[i,j] = value/255
    # cv2.imshow('Aditya Pulekar',img_gray)
    # cv2.waitKey(0)


    #Plotting the histogram
    channels = 0
    cumuSum = 0

    #Taking the gray scale histogram
    hist = [0 for itr in range(256)]
    for i in range(details[0]):
        for j in range(details[1]):
            hist[(img_gray[i,j]*255).__int__()]+=1
    plt.figure(3)
    plt.subplot(311)
    plt.plot(hist, 'ro-')
    plt.title("Histogram,PDF and CDF of the gray-scale image")
    plt.xlabel("Pixel Intensities--->")
    plt.ylabel("Frequency of Pixels--->")

    sumOfPixelValues = details[0]*details[1]

    #PDF
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

    #Histogram Equalization
    cdf_new = []
    for i in range(len(hist)):
       cumuSum += hist[i]
       cdf_new.append(cumuSum)
    print(len(cdf_new))
    hist_Eq = []
    for index in range(256):
        hist_Eq.append(math.floor(((cdf_new[index] - min(cdf_new))/((details[0]*details[1]) - min(cdf_new)))*255))
    print(len(hist_Eq))


    plt.figure(4)
    #Equalized Image
    newImage = numpy.empty([details[0],details[1]])
    for rows in range(details[0]):
        for cols in range(details[1]):
            newImage[rows,cols] = hist_Eq[(img_gray[rows,cols]*255).__int__()]

    print(newImage.shape)
    plt.title("Image after Histogram Equalization")
    plt.imshow(newImage,cmap=plt.cm.gray)
    plt.show()

    #Finding the color histogram and PDF,CDF for individual channels
    while channels < 3:
        scale = [0 for i in range(256)]
        for row in range(details[0]):
            for col in range(details[1]):
                scale[img[row,col,channels]]+=1
        # fig = plt.figure()
        plt.figure(5)
        plt.subplot(311)
        plt.plot(scale, 'ro-')
        # fig.suptitle("Aditya Pulekar( " + COLORS[channels] + " Channel's Histogram" + " )")
        plt.title("Aditya Pulekar( " + COLORS[channels] + " Channel's Histogram, PDF and CDF" + " )")
        plt.xlabel("Pixel Intensities--->")
        plt.ylabel("Frequency of Pixels--->")

        sumOfPixelValues = details[0]*details[1]

        #PDF
        pdf = [(index.__float__()/sumOfPixelValues) for index in scale]
        # print pdf
        # plt.figure(2)
        plt.subplot(312)
        plt.plot(pdf, 'go-')
        plt.xlabel("Pixel Intensities--->")
        plt.ylabel("Probability -->")

        #CDF
        cdf = []
        for i in range(len(pdf)):
            cumuSum += pdf[i]
            cdf.append(cumuSum)
        plt.subplot(313)
        plt.plot(cdf, 'bo-')
        plt.xlabel("Pixel Intensities -->")
        plt.ylabel("Cumu Probabilty -->")
        plt.show()
        channels+=1
    # plt.show()

main()