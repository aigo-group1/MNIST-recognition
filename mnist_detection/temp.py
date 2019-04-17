import cv2
import numpy as np

img = cv2.imread("D:/MNIST-recognition/server/upload/image 3.jpg")
height, width, channels = img.shape

print(width)
print(height)

#resize anh 1/2
resized_image = cv2.resize(img, (int(1024), int(768)))
#resized_image = img
#chuyen anh mau ve anh xam
img_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

#chuyen anh ve binary
imgBinary = cv2.threshold(img_gray, 135, 255, cv2.THRESH_BINARY)[1]

#bounding image,lay ra cac doi tuong
contours,hierachy=cv2.findContours(imgBinary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
print(len(contours))
widthBound = 0
heightBound =0
dem = 0

def sortSecond(val): 
    return val[1]  

def sortFirst(val):
        return val[0]
def checkContainRect(x1,y1,w1,h1,countours):
        for countour in countours:
                (x2,y2,w2,h2) = cv2.boundingRect(countour)
                if w2<100 and x2<x1 and x2+w2>x1+w1 and y2<y1 and y2+h2>y1+h1 :
                        return True
        return False        

lstFilter = []
lstRect = []
#contours.sort(key = sortSecond)
# lay tong trong so width va trong height
for contour in contours:
    (x,y,w,h) = cv2.boundingRect(contour)
    if w<100 and h <100 and w>10 and h>10:
        rs = checkContainRect(x,y,w,h,contours)
        if rs == False:
                rect = []
                rect.append(x)
                rect.append(y)
                rect.append(w)
                rect.append(h)    
                lstRect.append(rect)
                cv2.rectangle(resized_image, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (255, 0, 0), 2)
                widthBound = widthBound + w
                heightBound = heightBound + h
                dem = dem + 1

#cv2.imshow("test-anh",resized_image)
#cv2.waitKey(0)

lstRect.sort(key = sortFirst)
lstSame = []
for i in range(0,len(lstRect)):
        if i!=len(lstRect)-1 and abs(lstRect[i][0] - lstRect[i+1][0])<5:
                lstSame.append(lstRect[i])
        else:
                lstSame.append(lstRect[i])
                lstFilter.append(lstSame) 
                lstSame = []

#lay cot co so o nhieu nhat de loc label
#remove doi tuong bat thuong ra khoi cot

for filter in lstFilter:
        filter.sort(key=sortSecond)
        for i in range(len(filter)-1):
                rect = filter[i]
                #cv2.rectangle(resized_image, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (255, 0, 0), 2)
                #cv2.imshow("test-anh",resized_image)
                #cv2.waitKey(0)
                if filter[i]!=None and filter[i][0]-filter[i+1][0] <-5: 
                      filter[i] = None  
                      #i=i+1
 
dataRs =[]
maxlen = 10
passlen = 0
for filter in lstFilter:
        if len(filter) > maxlen or passlen>3:
                passlen = passlen +1
                for rect in filter:
                        if rect !=None:
                                #cv2.rectangle(resized_image, (rect[0],rect[1]), (rect[0]+rect[2],rect[1]+rect[3]), (255, 0, 0), 2)
                                dataRs.append(rect)
                
#cv2.imshow("test-anh",resized_image)
#cv2.waitKey(0)

dataRs.sort(key = sortSecond)
lstFilter=[]
lstSame=[]
for i in range(0,len(dataRs)):
        if i!=len(dataRs)-1 and abs(dataRs[i][1] - dataRs[i+1][1])<5:
                lstSame.append(dataRs[i])
        else:
                lstSame.append(dataRs[i])
                lstFilter.append(lstSame) 
                lstSame = []

dataIndentity = []
dataBirthday= []
for i in range(0,len(lstFilter)):
        if len(lstFilter[i]) == 12:
                dataImageID = []
                for j in range (0,len(lstFilter[i])):
                        crop_img = imgBinary[lstFilter[i][j][1]:lstFilter[i][j][1]+lstFilter[i][j][3], lstFilter[i][j][0]:lstFilter[i][j][0]+lstFilter[i][j][2]]
                        demPixel = 0
                        for m in range(5,lstFilter[i][j][3]-5):
                                for n in range(5,lstFilter[i][j][2]-5):
                                        pixel = crop_img[m][n]
                                        if pixel == 0:
                                                demPixel = demPixel +1                    

                        if demPixel >5:
                                crop_img = imgBinary[lstFilter[i][j][1]+3:lstFilter[i][j][1]+lstFilter[i][j][3]-3, lstFilter[i][j][0]+3:lstFilter[i][j][0]+lstFilter[i][j][2]-3]
                                crop_img = cv2.resize(crop_img,(28,28))
                                dataImageID.append(crop_img)
                                cv2.rectangle(resized_image, (lstFilter[i][j][0],lstFilter[i][j][1]), (lstFilter[i][j][0]+lstFilter[i][j][2],lstFilter[i][j][1]+lstFilter[i][j][3]), (255, 0, 0), 2)
                        #cv2.imshow("test-anh",crop_img)
                        #cv2.waitKey(0)
                if len(dataImageID) >0:
                        dataIndentity.append(dataImageID)               

        if len(lstFilter[i]) == 8:
                dataImageBirthday = []
                for j in range (0,len(lstFilter[i])):
                        crop_img = imgBinary[lstFilter[i][j][1]:lstFilter[i][j][1]+lstFilter[i][j][3], lstFilter[i][j][0]:lstFilter[i][j][0]+lstFilter[i][j][2]]
                        demPixel = 0
                        for m in range(5,lstFilter[i][j][3]-5):
                                for n in range(5,lstFilter[i][j][2]-5):
                                        pixel = crop_img[m][n]
                                        if pixel == 0:
                                                demPixel = demPixel +1                    

                        if demPixel >5:
                                crop_img = imgBinary[lstFilter[i][j][1]+3:lstFilter[i][j][1]+lstFilter[i][j][3]-3, lstFilter[i][j][0]+3:lstFilter[i][j][0]+lstFilter[i][j][2]-3]
                                crop_img = cv2.resize(crop_img,(28,28))
                                dataImageBirthday.append(crop_img)
                                cv2.rectangle(resized_image, (lstFilter[i][j][0],lstFilter[i][j][1]), (lstFilter[i][j][0]+lstFilter[i][j][2],lstFilter[i][j][1]+lstFilter[i][j][3]), (255, 0, 0), 2)
                                
                if len(dataImageBirthday) >0:
                        dataBirthday.append(dataImageBirthday) 

#cv2.imshow("test-anh",resized_image)
#cv2.waitKey(0)

dataFinal = []
dataFinal.append(dataIndentity)
dataFinal.append(dataBirthday)    
x = 0
print(len(dataFinal[0]))
print(len(dataFinal[1]))





