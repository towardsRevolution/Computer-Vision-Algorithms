import cv2
import numpy,math
import matplotlib.pyplot as plt

def main():
    COLORS= { 0 : ("BLUE"),
              1 : ("GREEN"),
              2 : ("RED")
    }
    img = cv2.imread('hazecity.png',1)
    cv2.imshow('Aditya Pulekar (Colored)',img)
    cv2.waitKey(0)
    # cv2.imwrite('flower_new.jpg',img)
    details = img.shape
    #Faster (Using Numpy)
    rgbToGray = numpy.array([0.114,0.589,0.299])
    img_gray = numpy.sum(img * rgbToGray, axis=-1)/255

    cv2.imshow('Aditya Pulekar (Gray)',img_gray)
    cv2.waitKey(0)


    #Plotting the histogram
    channels = 0
    cumuSum = 0

    #Taking the gray scale histogram
    hist = [0 for itr in range(256)]
    for i in range(details[0]):
        for j in range(details[1]):
            hist[(img_gray[i,j]*255).__int__()]+=1
    plt.figure(1)
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


    plt.figure(2)
    #Equalized Image
    newImage = numpy.empty([details[0],details[1]])
    for rows in range(details[0]):
        for cols in range(details[1]):
            newImage[rows,cols] = hist_Eq[(img_gray[rows,cols]*255).__int__()]

    plt.title("Image after Histogram Equalization")
    plt.imshow(newImage,cmap=plt.cm.gray)
    plt.show()


main()