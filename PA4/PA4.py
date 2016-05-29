"""
The program calculates the Histogram of Oriented Gradient feature descriptor for the
given interest points, matches the features based on euclidean distances between them
and compares HoG with Speed-Up Robust Features (SURF)

@author: Aditya Pulekar

(Referred "http://stackoverflow.com/questions/20259025/module-object-has-no-attribute-drawmatches-opencv-python"
to concatenate two images and draw matches for SURF).
"""
import cv2
import math,pdb
import numpy as np
import matplotlib.pyplot as plt

kp1 = [[402, 372],
       [371, 230],
       [156, 381],
       [419, 231],
       [323, 322]]

kp2 = [[325, 232],
       [300, 90],
       [81, 230],
       [348, 94],
       [249, 182]]

def HOG(img, x, y):
    """
    Generate a descriptor around a given point in an image
    :param img: the image to describe
    :param x: the x value of the point to be described
    :param y: the y value of the point to be described
    :return: the [1, 128] descriptor of the point
    """
    #TODO: write a HOG descriptor here
    des=[]
    row=0
    sub_image = img[x-8:x+8,y-8:y+8]
    while row < len(sub_image):
        col=0
        while col < len(sub_image[0]):
            temp_vector = [0 for i in range(8)]
            new_subimage = sub_image[row:row+4,col:col+4]
            x_gradient = cv2.Sobel(new_subimage,ddepth=-1,dx=1,dy=0)
            y_gradient = cv2.Sobel(new_subimage,ddepth=-1,dx=0,dy=1)
            theta = np.empty([x_gradient.shape[0],x_gradient.shape[1]])
            for i in range(len(x_gradient)):
                for j in range(len(x_gradient[0])):
                    if x_gradient[i,j] == 0:
                        theta[i,j] = 90
                    else:
                        theta[i,j] = np.arctan(y_gradient[i,j]/x_gradient[i,j])*(180/np.pi)
            theta_iter = theta.flatten()        #To avoid nested for loops for 4x4 theta
            for i in range(len(theta_iter)):
                if theta_iter[i] < 45:
                    temp_vector[0]=temp_vector[0]+1
                elif theta_iter[i] >= 45 and theta_iter[i] < 90:
                    temp_vector[1]=temp_vector[1]+1
                elif theta_iter[i] >= 90 and theta_iter[i] < 135:
                    temp_vector[2]=temp_vector[2]+1
                elif theta_iter[i] >= 135 and theta_iter[i] < 180:
                    temp_vector[3]=temp_vector[3]+1
                elif theta_iter[i] >= 180 and theta_iter[i] < 225:
                    temp_vector[4]=temp_vector[4]+1
                elif theta_iter[i] >= 225 and theta_iter[i] < 270:
                    temp_vector[5]=temp_vector[5]+1
                elif theta_iter[i] >= 270 and theta_iter[i] < 315:
                    temp_vector[6]=temp_vector[6]+1
                elif theta_iter[i] >= 315 and theta_iter[i] < 360:
                    temp_vector[7]=temp_vector[7]+1
            des.extend(temp_vector)
            col=col+4
        row=row+4
    return des

def sumOfSquares(f1,f2):
    sum1=0;sum2=0
    for i in f1:
        sum1=sum1+math.pow(i,2)
    for i in f2:
        sum2=sum2+math.pow(i,2)
    rootOfSquare1 = math.sqrt(sum1)
    rootOfSquare2 = math.sqrt(sum2)
    return [rootOfSquare1,rootOfSquare2]

def matcher(features1, features2):
    """
    Matches the descriptors from one image to the
    descriptors from another image
    :param features1: the first array of features [n, 128]
    :param features2: the second array of features [n, 128]
    :return: matching point pairs [[index1, index2], ... ]
    """
    #TODO: write a matching function
    #Performing the L2-Norm
    new_features1=[]
    new_features2=[]
    for itr in range(5):
        [rootOfSquare1,rootOfSquare2] = sumOfSquares(features1[itr],features2[itr])
        new_features1.append(np.array(features1[itr])/rootOfSquare1)
        new_features2.append(np.array(features2[itr])/rootOfSquare2)
    indices = []
    for itr in range(len(new_features1)):
        findMinDist=[]
        #findMaxCosineVal=[]
        for itr2 in range(len(new_features2)):
            f1 = new_features1[itr]
            f2 = new_features2[itr2]

            #For evaluating the cosine similarity
            # [rootOfSquare1,rootOfSquare2] = sumOfSquares(f1,f2)
            # numerator = np.array(f1)*np.array(f2)
            # numeratorSum = sum(numerator)
            # denominator = rootOfSquare1*rootOfSquare2
            # cosine = np.divide(numeratorSum,denominator)
            # findMaxCosineVal.append(cosine)

            #For evaluating the similarity based on euclidean distance
            Dist = np.array(f1) - np.array(f2)
            sum=0
            for i in Dist:
                sum=sum+math.pow(i,2)
            rootOfSum = math.sqrt(sum)
            findMinDist.append(rootOfSum)
        # print "itr: ", itr, " Matching scores: ", findMaxCosineVal
        # bestMatch = findMaxCosineVal.index(max(findMaxCosineVal))
        bestMatch = findMinDist.index(min(findMinDist))
        indices.append([itr,bestMatch])
    return indices

def surfMatcher(img1_color,img2_color):
    s = cv2.SURF(450)
    keyPt1,descp1 = s.detectAndCompute(img1_color,None)
    new_Img = cv2.drawKeypoints(img1_color,keyPt1,None,(0,255,0))
    plt.title("SURF Features for aerial1")
    cv2.imwrite("SURF_Features_Aerial1.jpg",new_Img)
    plt.imshow(new_Img)
    plt.show()
    keyPt2,descp2 = s.detectAndCompute(img2_color,None)  #"None" has been given for the mask
    new_Img = cv2.drawKeypoints(img2_color,keyPt2,None,(0,255,0))
    plt.title("SURF Features for aerial2")
    cv2.imwrite("SURF_Features_Aerial2.jpg",new_Img)
    plt.imshow(new_Img)
    plt.show()

    #Concatenate two images
    out_image = concatenateTwoImages(img1_color.shape[0],img1_color.shape[1],
                                     img2_color.shape[0],img2_color.shape[1])
    c1 = img1_color.shape[1]
    m = np.amax(out_image.flatten())
    cv2.imshow("Concatenated Image for SURF",out_image/m)
    cv2.waitKey(0)

    bf_match = cv2.BFMatcher(cv2.NORM_L2,crossCheck=False)
    # match_pts= bf_match.knnMatch(np.asarray(descp1,np.float32),np.asarray(descp2,np.float32),1)
    match_pts= bf_match.match(np.asarray(descp1,np.float32),np.asarray(descp2,np.float32))
    match_pts_sorted = sorted(match_pts,reverse=True)                 #Why do we get better on taking the higher values?
    #Peformance of surf changes every time we run the program
    #Drawing matches for SURF
    for itr in match_pts_sorted[:5]:
        indexForI2 =itr.trainIdx
        indexForI1 = itr.queryIdx
        (ptX2,ptY2)=tuple(keyPt2[indexForI2].pt)   #(X--> Columns, Y--> Rows)
        (ptX1,ptY1)=tuple(keyPt1[indexForI1].pt)
        cv2.circle(out_image,(int(ptX2)+c1,int(ptY2)),10,(0,255,0),thickness=3)
        cv2.circle(out_image,(int(ptX1),int(ptY1)),10,(0,255,0),thickness=3)
        cv2.line(out_image,(int(ptX1),int(ptY1)),(int(ptX2)+c1,int(ptY2)),(0,255,0),thickness=2)
    cv2.imshow("Concatenated Image for SURF (With final key points)",out_image/m)
    cv2.imwrite("SURF_Features.jpg",out_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    pdb.set_trace()

def concatenateTwoImages(r1,c1,r2,c2):
    r2=img2_color.shape[0];c2=img2_color.shape[1]
    r1=img1_color.shape[0];c1=img2_color.shape[1]
    out_image=np.empty([max(r1,r2),c2+c1,3],dtype=np.float32)
    out_image[:r1,:c1]=np.dstack([img1_color])
    out_image[:r2,c1:]=np.dstack([img2_color])
    return out_image

if __name__ == '__main__':
    img1_color = cv2.imread("aerial1.jpg", 1)
    img2_color = cv2.imread("aerial2.jpg", 1)
    img1 = cv2.cvtColor(img1_color,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2_color,cv2.COLOR_BGR2GRAY)
    des1 = []
    des2 = []
    print "img1 details: ", img1.shape,"  img2 details: ", img2.shape
    for i in range(5):
        des1.append(HOG(img1,kp1[i][1],kp1[i][0]))
        des2.append(HOG(img2,kp2[i][1],kp2[i][0]))
    indices = matcher(des1, des2)

    print "indices: ",indices
    #TODO: display the output in a meaningful way
    #Concatenating two images
    coloredFeatureImage = concatenateTwoImages(img1_color.shape[0],img1_color.shape[1],
                                     img2_color.shape[0],img2_color.shape[1])
    color_m = np.amax(coloredFeatureImage.flatten())
    coloredFeatureImage/=color_m
    dictForPoints={}
    for r in range(coloredFeatureImage.shape[0]):
        for c in range(coloredFeatureImage.shape[1]):
            if [c,r] in kp1:
                cv2.circle(coloredFeatureImage,(c,r),10,(0,255,0),thickness=3)
                dictForPoints[(c,r)] = (c,r)
            if [c-img2.shape[1],r] in kp2:
                cv2.circle(coloredFeatureImage,(c,r),10,(0,255,0),thickness=3)
                dictForPoints[(c-img2.shape[1],r)] = (c,r)
    cv2.imshow("Feature Matching Image with circles (HoG)",coloredFeatureImage)
    cv2.imwrite("HoG_Features_highlighted.jpg",coloredFeatureImage*color_m)
    cv2.waitKey(0)

    for itr in indices:
        if itr[0] == itr[1]:
            cv2.line(coloredFeatureImage,dictForPoints[tuple(kp1[itr[0]])],dictForPoints[tuple(kp2[itr[1]])],(0,255,0),thickness=2)
        else:
            cv2.line(coloredFeatureImage,dictForPoints[tuple(kp1[itr[0]])],dictForPoints[tuple(kp2[itr[1]])],(0,0,255),thickness=2)
    cv2.imshow("Feature Matching Image with lines (HoG)",coloredFeatureImage)
    cv2.imwrite("HoG_Features_matched.jpg",coloredFeatureImage*color_m)
    cv2.waitKey(0)

    #Comparison of HoG with SURF
    surfMatcher(img1_color,img2_color)