"""
This program gives solution to all the problems mentioned in our final exam.

Referred:
For Object tracking: http://www.pyimagesearch.com/2015/09/21/opencv-track-object-movement/
For Image composting: http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_pyramids/py_pyramids.html
"""
import cv2,math
import numpy as np,matplotlib.pyplot as plt

__author__="Aditya Pulekar"

#DONE
def problem_1(gray_img):
    equalizedImg=cv2.equalizeHist(gray_img)
    concatenatedImage=np.hstack([gray_img,equalizedImg])
    cv2.imshow("Result after contrast enhancement",concatenatedImage)
    cv2.imwrite("P1.jpg",concatenatedImage)
    cv2.waitKey(0)

#DONE
def problem_4(img,gray_img):
    #Cropping the given gray-scale image
    i_img=img[33:54,27:34]
    res=cv2.matchTemplate(img,i_img,method=cv2.cv.CV_TM_CCOEFF_NORMED)
    pts=np.where(res>=0.86)      #0.835
    for itr in range(len(pts[0])):
        cv2.rectangle(img,(pts[1][itr],pts[0][itr]),(pts[1][itr]+10,pts[0][itr]+20),(0,0,255),thickness=2)
    cv2.imshow("Final Image",img)
    cv2.imwrite("P4.jpg",img)
    cv2.waitKey(0)

#DONE
def problem_5():
    vid=cv2.VideoCapture("trackball.avi");count=0;
    while(True):
        status,frame=vid.read()
        if status==True:
            frame_hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
            m=cv2.inRange(frame_hsv,np.array((150,95,95)),np.array((190,255,255)))  #This gives the range for the BGR values..Lower and upper bound
            m=cv2.erode(m,kernel=None,iterations=3);m=cv2.dilate(m,kernel=None,iterations=3);
            contour=cv2.findContours(m,mode=cv2.RETR_EXTERNAL,method=cv2.CHAIN_APPROX_TC89_KCOS)[-2]
            if len(contour)>0:
                circle=max(contour,key=cv2.contourArea)
                # Moment=cv2.moments(circle);
                # cen=((int(Moment["m10"])/int(Moment["m00"])),(int(Moment["m01"]),int(Moment["m00"])))
                ((col,row),rad)=cv2.minEnclosingCircle(circle)
                cv2.circle(frame,center=(int(col),int(row)),radius=int(rad),color=(0,0,255),thickness=2)
            cv2.imshow("Ball Tracked",frame)
            k=cv2.waitKey(1)&0xFF;count+=1
        else:
            break
    vid.release()
    cv2.destroyAllWindows()

#PROBLEM 5 using MeanShift Algorithm
# def problem_5():
#     vid=cv2.VideoCapture("trackball.avi")
#     status,frame=vid.read()
#     init_r,init_c,init_h,init_w=55,139,20,20
#     trackWindow=(init_c,init_r,init_w,init_h)
#     frame_hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
#     roi=frame[init_r:init_r+init_h,init_c:init_c+init_w]
#
#     #Problem lies here..i.e. we may be choosing the wrong channel for tracking
#     # (Rectify the below two lines and we should get the answer)
#     m=cv2.inRange(frame_hsv,np.array((160.,100.,100.)),np.array((200.,255.,255.)))  #This gives the range for the BGR values..Lower and upper bound
#
#     frame_hist=cv2.calcHist([frame_hsv],channels=[0],mask=None,histSize=[180],ranges=[0,180])  #Note: We have not put a mask
#
#     cv2.normalize(frame_hist,frame_hist,0,255, cv2.NORM_MINMAX)
#     termination=(cv2.TERM_CRITERIA_COUNT|cv2.TERM_CRITERIA_EPS,1,10)
#     while(True):
#         status,frame=vid.read()
#         if status==True:
#             frame_hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
#             destn=cv2.calcBackProject([frame_hsv],[0],frame_hist,[0,180],1)
#             r,trackWindow=cv2.meanShift(destn,trackWindow,termination)
#             # r,trackWindow=cv2.CamShift(destn,trackWindow,termination)
#
#             # #For camShift()
#             # p=cv2.cv.BoxPoints(r)
#             # p=np.int0(p)
#             # print p
#             # cv2.polylines(frame,[p],True,255,2)
#             # cv2.imshow("Image with BB",frame)
#             # cv2.waitKey(0)
#             #For meanShift()
#             (c,r,w,h)=trackWindow
#             cv2.rectangle(frame,(c,r),(c+w,r+h),(0,0,255),thickness=2)
#             cv2.imshow("Image with bounding box",frame)
#             cv2.waitKey(0)
#         else:
#             break
#     vid.release()

#DONE
def problem_3(gray_img,img):
    gray_img=cv2.GaussianBlur(gray_img,(11,11),0)
    c=cv2.HoughCircles(gray_img,method=cv2.cv.CV_HOUGH_GRADIENT,dp=1,minDist=17,param1=117,param2=16,minRadius=6,maxRadius=27)
    #c=Coordinates
    for itr in c:
        for p in range(c.shape[1]):
            cv2.circle(img,(itr[p][0],itr[p][1]),itr[p][2],(0,0,255),thickness=3)
    cv2.imshow("Circles Detected",img)
    cv2.imwrite("P3.jpg",img)
    cv2.waitKey(0)

#DONE
def problem_6(img):
    cv2.imshow("Original Image",img)
    cv2.waitKey(0)
    m_1=cv2.inRange(img,np.array((10,15,100)),np.array((45,65,255))) # Red
    m_2=cv2.inRange(img,np.array((19,140,180)),np.array((55,195,255))) # Yellow
    m_3=cv2.inRange(img,np.array((70,40,40)),np.array((165,110,110))) # Grey
    img1=cv2.bitwise_and(img,img,mask=m_1)
    img2=cv2.bitwise_and(img,img,mask=m_2)
    img3=cv2.bitwise_and(img,img,mask=m_3)
    cv2.imshow("RED",img1)
    cv2.imwrite("P6_1.jpg",img1)
    cv2.waitKey(0)
    cv2.imshow("YELLOW",img2)
    cv2.imwrite("P6_2.jpg",img2)
    cv2.waitKey(0)
    cv2.imshow("GREY",img3)
    cv2.imwrite("P6_3.jpg",img3)
    cv2.waitKey(0)

#DONE
def problem_7(imgA,imgB):
    #Another solution using image pyramids
    #For Gaussain Pyramids
    diffLevels_A=imgA.copy()
    GaussPyr_A=[]
    GaussPyr_A.append(diffLevels_A)         # Gaussian Pyramids for imgA
    for itr in range(4):
        diffLevels_A=cv2.pyrDown(diffLevels_A)
        GaussPyr_A.append(diffLevels_A)
    diffLevels_B=imgB.copy()                # Gaussian Pyramids for imgB
    GaussPyr_B=[];GaussPyr_B.append(diffLevels_B)
    for itr in range(4):
        diffLevels_B=cv2.pyrDown(diffLevels_B)
        GaussPyr_B.append(diffLevels_B)

    #For Laplacian Pyramids   (Laplacian pyramids will be appended from lowest resolution to highest resolution)
    LaplacePyr_A=[GaussPyr_A[3]]   #Since we start building the Laplacian pyramids from the bottom
    for itr in range(3,0,-1):
        temp_A=cv2.pyrUp(GaussPyr_A[itr])
        d=(GaussPyr_A[itr-1].shape[0],GaussPyr_A[itr-1].shape[1],3)
        temp_A=np.resize(temp_A,d)
        LDiff=cv2.subtract(GaussPyr_A[itr-1],temp_A)      #Bcoz "GaussPyr_A[itr-1]" has a higher resolution than "GaussPyr_A[itr]"
        LaplacePyr_A.append(LDiff)

    LaplacePyr_B=[GaussPyr_B[3]]
    for itr in range(3,0,-1):
        temp_B=cv2.pyrUp(GaussPyr_B[itr])
        d=(GaussPyr_B[itr-1].shape[0],GaussPyr_B[itr-1].shape[1],3)
        temp_B=np.resize(temp_B,d)
        LDiff=cv2.subtract(GaussPyr_B[itr-1],temp_B)
        LaplacePyr_B.append(LDiff)

    #Blending the two Laplacian Pyramids (all resolution levels)
    Blend=[]
    #Note: Blend will have pyramids blended from lower to higher resolution
    for LapA,LapB in zip(LaplacePyr_A,LaplacePyr_B):
        Lr,Lc,dimension=LapA.shape
        temp=np.hstack((LapA[:,0:Lc/2],LapB[:,Lc/2:]))
        # Laplacian pyramid at each level is blended. This will help reconstruction of image
        Blend.append(temp)

    #Reconstructing the Image from the pyramids (Laplcian to Gaussian)
    final_temp=Blend[0]
    for itr in range(1,4):
        final_temp=cv2.pyrUp(final_temp)
        d=(Blend[itr].shape[0],Blend[itr].shape[1],3)
        final_temp=np.resize(final_temp,d)
        final_temp=cv2.add(final_temp,Blend[itr])       #L[i]=G[i]-G[i-1]..diff of gaussian..So, G[i]=L[i]+G[i-1]

    final_img=np.hstack((imgA[:,0:Lc/2],imgB[:,Lc/2:]))
    cv2.imshow("Final Blended Image",final_temp)
    cv2.imwrite("P_7.jpg",final_temp)
    cv2.waitKey(0)


def main():
    flag=1
    while(flag==1):
        probNum=int(input("Enter the problem number:"))
        if probNum == 1:
            #Solution to problem 1
            img=cv2.imread("highway.png",1)
            gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            problem_1(gray_img)
        elif probNum == 2:
            print "PROBLEM 2 WAS DONE IN MATLAB."
        elif probNum == 3:
            #Solution to problem 3
            img=cv2.imread("cropcirlces.png",1)
            gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            problem_3(gray_img,img)
        elif probNum == 4:
            #Solution to problem 4
            img=cv2.imread("scantext.png",1)
            gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            #(286, 525, 3)-->Image dimensions
            problem_4(img,gray_img)
        elif probNum == 5:
            #Solution to problem 5
            problem_5()
        elif probNum == 6:
            #Solution to problem 6
            img=cv2.imread("berries.png",1)
            problem_6(img)
        elif probNum == 7:
            #Solution to problem 7
            imgA=cv2.imread("apple.jpg",1)
            imgB=cv2.imread("orange.jpg",1)

            problem_7(imgA,imgB)
            #One solution (without using image pyramids)
            # blendedImg=cv2.addWeighted(imgA,1,imgB,1,0)
            # cv2.imshow("Blended Image",blendedImg)
            # cv2.waitKey(0)
        else:
            print "You entered the wrong problem number! Please enter again..."
        ch=int(input("Would you like to continue?(Yes-->Press 1 or No--> Press 2) "))
        if ch==1:
            pass
        else:
            flag=0
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()