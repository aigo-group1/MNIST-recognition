import cv2
import numpy as np

img = cv2.imread("test.jpg")
height, width, channels = img.shape

print(width)
print(height)

#resize anh 1/2
resized_image = cv2.resize(img, (int(width/2), int(height/2)))
#resized_image = img
#chuyen anh mau ve anh xam
img_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

#chuyen anh ve binary
imgBinary = cv2.threshold(img_gray, 130, 255, cv2.THRESH_BINARY)[1]

#bounding image,lay ra cac doi tuong
contours,hierachy=cv2.findContours(imgBinary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
#
print(len(contours))
widthBound = 0
heightBound =0
dem = 0

def sortSecond(val): 
    return val[1]  

def checkContainRect(x1,y1,w1,h1,countours):
        for countour in countours:
                (x2,y2,w2,h2) = cv2.boundingRect(countour)
                if w2<100 and x2<x1 and x2+w2>x1+w1 and y2<y1 and y2+h2>y1+h1 :
                        return True
        return False        

# lay tong trong so width va trong height
for contour in contours:
    (x,y,w,h) = cv2.boundingRect(contour)
    if w<100 and h <100:
        widthBound = widthBound + w
        heightBound = heightBound + h
        dem = dem + 1
#lay trung binh tong cua withd height
dataRs=[]
for contour in contours:
    (x,y,w,h) = cv2.boundingRect(contour)
    #nhung o vuong w > width trung binh and
    if w<100 and h <100 and w>widthBound/dem and h > heightBound/dem:
        rs = checkContainRect(x,y,w,h,contours)
        if rs == False:
                #cv2.rectangle(resized_image, (x,y), (x+w,y+h), (255, 0, 0), 2)
                data = []
                data.append(x)
                data.append(y)
                data.append(w)
                data.append(h)
                dataRs.append(data)

#crop_img = imgBinary[y:y+h, x:x+w]
                #demPixel = 0
                #for i in range(5,h-5):
                        #for j in range(5,w-5):
                                #pixel = crop_img[i][j]
                                #if pixel == 0:
                                        #demPixel = demPixel +1


print(widthBound/dem)

dataRs.sort(key = sortSecond)

dataFirst=[]
dataSecond= []
dattaLast=[]
dataTotal =[]

currentData = "First"
demTake = 0

for i in range(0,len(dataRs)):
        if currentData == "First":
                if(demTake<20):
                        dataFirst.append(dataRs[i])
                        demTake = demTake+1
                else:
                        dataTotal.append(dataFirst)
                        dataFirst=[]
                        currentData = "Second"
                        demTake = 0

        if currentData == "Second":
                if(demTake<12):
                        dataSecond.append(dataRs[i])
                        demTake = demTake+1
                else:
                        dataTotal.append(dataSecond)
                        dataSecond=[]
                        currentData = "Last"
                        demTake = 0
                
        if currentData == "Last":
                if(demTake<8):
                        dattaLast.append(dataRs[i])
                        demTake = demTake+1
                else:
                        dataTotal.append(dattaLast)
                        dattaLast=[]
                        currentData = "First"
                        demTake = 0

dataFinal = []
dataIndentity = []
dataBirthday = []
for i in range(0,len(dataTotal)):
        if len(dataTotal[i]) == 12:
                dataImageID = []
                for j in range (0,len(dataTotal[i])):
                        crop_img = imgBinary[dataTotal[i][j][1]:dataTotal[i][j][1]+dataTotal[i][j][3], dataTotal[i][j][0]:dataTotal[i][j][0]+dataTotal[i][j][2]]
                        demPixel = 0
                        for m in range(5,dataTotal[i][j][3]-5):
                                for n in range(5,dataTotal[i][j][2]-5):
                                        pixel = crop_img[m][n]
                                        if pixel == 0:
                                                demPixel = demPixel +1                    

                        if demPixel >5:
                                crop_img = imgBinary[dataTotal[i][j][1]+3:dataTotal[i][j][1]+dataTotal[i][j][3]-3, dataTotal[i][j][0]+3:dataTotal[i][j][0]+dataTotal[i][j][2]-3]
                                crop_img = cv2.resize(crop_img,(28,28))
                                dataImageID.append(crop_img)
                                #cv2.imshow("test-anh",crop_img)
                                #cv2.waitKey(0)
                if len(dataImageID) >0:
                        dataIndentity.append(dataImageID)               

        if len(dataTotal[i]) == 8:
                dataImageBirthday = []
                for j in range (0,len(dataTotal[i])):
                        crop_img = imgBinary[dataTotal[i][j][1]:dataTotal[i][j][1]+dataTotal[i][j][3], dataTotal[i][j][0]:dataTotal[i][j][0]+dataTotal[i][j][2]]
                        demPixel = 0
                        for m in range(5,dataTotal[i][j][3]-5):
                                for n in range(5,dataTotal[i][j][2]-5):
                                        pixel = crop_img[m][n]
                                        if pixel == 0:
                                                demPixel = demPixel +1                    

                        if demPixel >5:
                                crop_img = imgBinary[dataTotal[i][j][1]+3:dataTotal[i][j][1]+dataTotal[i][j][3]-3, dataTotal[i][j][0]+3:dataTotal[i][j][0]+dataTotal[i][j][2]-3]
                                crop_img = cv2.resize(crop_img,(28,28))
                                dataImageBirthday.append(crop_img)
                                #cv2.imshow("test-anh",crop_img)
                                #cv2.waitKey(0)
                if len(dataImageBirthday) >0:
                        dataBirthday.append(dataImageBirthday) 

dataFinal.append(dataIndentity)
dataFinal.append(dataBirthday)    

cv2.imshow("test-anh",resized_image)
cv2.waitKey(0)



