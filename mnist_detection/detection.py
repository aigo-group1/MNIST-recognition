import cv2
import numpy as np

def save_for_debug(imagePath, debug_img, debug_name):
    save_path = imagePath+"-"+debug_name+".jpg"
    cv2.imwrite(save_path, debug_img)


def save_contour_for_debug(imagePath, debug_img, contours, debug_name):
    save_path = imagePath+"-"+debug_name+".jpg"
    output = debug_img.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(save_path, output)

def sortSecond(val):
    return val[1]


def sortFirst(val):
    return val[0]


def checkContainRect(x1, y1, w1, h1, countours):
    for countour in countours:
        (x2, y2, w2, h2) = cv2.boundingRect(countour)
        if w2 < 100 and x2 < x1 and x2+w2 > x1+w1 and y2 < y1 and y2+h2 > y1+h1:
            return True
    return False

def detect_image(path):
    img = cv2.imread(path)
    height, width, channels = img.shape

    #print(width)
    #print(height)
    # resize anh 1/2
    resized_image = cv2.resize(img, (int(1024), int(768)))
    #resized_image = img
    # chuyen anh mau ve anh xam
    img_gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    # chuyen anh ve binary
    imgBinary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY)[1]

    # bounding image,lay ra cac doi tuong
    contours, hierachy = cv2.findContours(
        imgBinary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    #save_contour_for_debug("D:\\Python\\img_debug\\img",
    #                    resized_image, contours, "step_contours")


    # print(len(contours))
    widthBound = 0
    heightBound = 0
    dem = 0

    lstFilter = []
    lstRect = []
    #contours.sort(key = sortSecond)
    # lay tong trong so width va trong height
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w < 100 and h < 100 and w > 10 and h > 10:
            rs = checkContainRect(x, y, w, h, contours)
            if rs == False:
                rect = []
                rect.append(x)
                rect.append(y)
                rect.append(w)
                rect.append(h)
                lstRect.append(rect)
                widthBound = widthBound + w
                heightBound = heightBound + h
                dem = dem + 1

    averageWidtd = widthBound/dem
    averageHeight = heightBound/dem

    # loai bo cac o có chieu cao nho hon chieu cao trung binh
    for rect in lstRect:
        if(rect[3] < averageHeight-5):
            lstRect.remove(rect)

    #for rect in lstRect:
    #    cv2.rectangle(resized_image, (rect[0], rect[1]),
    #                (rect[0]+rect[2], rect[1]+rect[3]), (255, 0, 0), 2)


    # cv2.imshow("test-anh",resized_image)
    # cv2.waitKey(0)
    lstRect.sort(key=sortSecond)
    # print(lstRect)

    lstRows = []
    row = []
    for i in range(0, len(lstRect)):
        if i != len(lstRect)-1 and lstRect[i+1][1] - lstRect[i][1] < 10:
            row.append(lstRect[i])
        else:
            row.append(lstRect[i])
            lstRows.append(row)
            row = []


    # remove hang ho ten
    for row in lstRows:
        if len(row) == 20:
            lstRows.remove(row)
    lstDataFinal = []
    # cat image va tinh pixel
    for row in lstRows:
        dataFinal = []
        for rect in row:
            crop_img = imgBinary[rect[1]:rect[1] +
                                rect[3], rect[0]:rect[0]+rect[2]]
            #crop_img = cv2.resize(crop_img,(28,28))

            # cv2.imshow("test-anh",crop_img)
            # cv2.waitKey(0)
            # dem pixel de remove
            demPixel = 0
            for m in range(5, rect[3]-5):
                for n in range(5, rect[2]-5):
                    pixel = crop_img[m][n]
                    if pixel < 130:
                        demPixel = demPixel + 1

            if demPixel > 5:
                crop_img = cv2.resize(crop_img, (28, 28))
                dataFinal.append(crop_img)
            else:
                # cv2.imshow("test-anh",crop_img)
                # cv2.waitKey(0)
                dataFinal.append(None)
        lstDataFinal.append(dataFinal)
    # print(lstRows)
    # cv2.imshow("test-anh",resized_image)
    # cv2.waitKey(0)
    return lstDataFinal

#list_image = detect_image("mnist_detection/test3.jpg") 
#for i in range(len(list_image)):
#    for j in range(len(list_image[i])):
#        cv2.imwrite("piece"+str(i)+"-"+str(j)+".jpg",list_image[i][j])
#print("")
