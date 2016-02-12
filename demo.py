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
    print img_gray
    img_binary = numpy.empty([details[0],details[1]])

    #Learn to iterate through a 2d matrix in a single statement (try numpy)
    for rows in range(details[0]):
        for cols in range(details[1]):
            if img_gray[rows,cols] <= 0.5:
                img_binary[rows,cols] = 0
            else:
                img_binary[rows,cols] = 1
    flag, img_binary = cv2.threshold(img_gray,0.5,1,cv2.THRESH_BINARY)
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
    #cv2.waitKey(0)
    # cv2.imshow('Aditya Pulekar (Blue Channel)', img_Blue_Channel)
    # cv2.waitKey(0)
    # cv2.imshow('Aditya Pulekar (Green Channel)', img_Green_Channel)
    # cv2.waitKey(0)
    # cv2.imshow('Aditya Pulekar (Red Channel)', img_Red_Channel)
    # cv2.waitKey(0)



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
    plt.figure(2)
    plt.plot(hist, 'ro-')
    plt.title("Histogram of the gray-scale image")
    plt.xlabel("Pixel Intensities--->")
    plt.ylabel("Frequency of Pixels--->")
    plt.show()


    #Taking the color histogram
    while channels < 3:
        scale = [0 for i in range(256)]
        for row in range(details[0]):
            for col in range(details[1]):
                scale[img[row,col,channels]]+=1
        # fig = plt.figure()
        plt.figure(3)
        plt.subplot(311)
        plt.plot(scale, 'ro-')
        # fig.suptitle("Aditya Pulekar( " + COLORS[channels] + " Channel's Histogram" + " )")
        plt.title("Aditya Pulekar( " + COLORS[channels] + " Channel's Histogram" + " )")
        plt.xlabel("Pixel Intensities--->")
        plt.ylabel("Frequency of Pixels--->")

        sumOfPixelValues = details[0]*details[1]

        #PDF
        pdf = [(index.__float__()/sumOfPixelValues) for index in scale]
        # print pdf
        # plt.figure(2)
        plt.subplot(312)
        plt.plot(pdf, 'go-')
        # fig.suptitle("Probability distribution function for " + COLORS[channels] + " Channel")
        # plt.title("Probability distribution function for " + COLORS[channels] + " Channel")
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