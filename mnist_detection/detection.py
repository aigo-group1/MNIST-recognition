import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def sortFirst(val):
    return val[0]


def sortSecond(val):
    return val[1]


def save_for_debug(imagePath, debug_img, debug_name, image_shape):
    str1 = '_'.join(str(e) for e in image_shape)
    save_path = imagePath+"-"+debug_name+"-"+str1+".jpg"
    cv2.imwrite(save_path, debug_img)


def save_contour_for_debug(imagePath, debug_img, contours, debug_name, image_shape):
    str1 = '_'.join(str(e) for e in image_shape)
    save_path = imagePath+"-"+debug_name+"-"+str1+".jpg"
    output = debug_img.copy()
    for i in contours:
        (x, y, w, h) = cv2.boundingRect(i)
        cv2.rectangle(output, (x, y), (x+w, y+h), (0, 0, 255), 2)
    cv2.imwrite(save_path, output)
    #print("sum of contours = ",len(contours))


def editting_image(image_path):
    image = cv2.imread(image_path)
    #save_for_debug(image_path, image, "00_original", image.shape)

    im_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #save_for_debug(image_path, im_gray, "01_gray", im_gray.shape)

    thre = cv2.threshold(im_gray, 127, 255, cv2.THRESH_BINARY)[1]
    #save_for_debug(image_path, thre, "02_thresh_binary", thre.shape)

    contours = cv2.findContours(
        thre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    #save_contour_for_debug(image_path, image, contours, "03_contours", thre.shape)

    height, width, channels = image.shape
    #print("height = ", height)
    #print("width = ",width)
    #print("channels = ",channels)

    for i in contours:
        (x, y, w, h) = cv2.boundingRect(i)
        if(w > width/2 and h > height/2):
            image = image[y:y+h, x:x+w, :]
            #save_for_debug(image_path, image, "04_image_after_cutting", image.shape)

            im_gray = im_gray[y:y+h, x:x+w]
            #save_for_debug(image_path, im_gray, "04_gray_after_cutting", im_gray.shape)

    height1, width1, channels1 = image.shape

    thre = cv2.threshold(im_gray, 127, 255, cv2.THRESH_BINARY_INV)[1]
    #save_for_debug(image_path, thre, "05_thresh_binary_inv", thre.shape)

    contours = cv2.findContours(
        thre, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    #save_contour_for_debug(image_path, image, contours, "06_contours_inv", thre.shape)

    h_avg = 0
    index = 0
    A1 = np.array([height1/2, width1/2])
    B1 = np.array([height1/2, width1/2])
    A2 = np.array([height1/2, width1/2])
    B2 = np.array([height1/2, width1/2])
    for i in contours:
        (x, y, w, h) = cv2.boundingRect(i)
        if(w < 100 and h < 100):
            h_avg += h
            index += 1
    #print("index = ",index)
    h_avg /= index
    #sum_contours_ivalid = 0
    for i in contours:
        (x, y, w, h) = cv2.boundingRect(i)
        if(h > h_avg and w < 100 and h < 100):
            #sum_contours_ivalid += 1
            if(math.sqrt(x*x + y*y) < math.sqrt(A1[0]*A1[0] + A1[1]*A1[1])):
                A1[0] = x
                A1[1] = y
            if(math.sqrt(x*x + (y-height)*(y-height)) < math.sqrt(A2[0]*A2[0] + (A2[1]-height)*(A2[1]-height))):
                A2[0] = x
                A2[1] = y
            if(math.sqrt((x-width)*(x-width) + y*y) < math.sqrt((B1[0]-width)*(B1[0]-width) + B1[1]*B1[1])):
                B1[0] = x
                B1[1] = y
            if(math.sqrt((x-width)*(x-width) + (y-height)*(y-height)) < math.sqrt((B2[0]-width)*(B2[0]-width) + (B2[1]-height)*(B2[1]-height))):
                B2[0] = x
                B2[1] = y
        # cv2.rectangle(im_cut,(x,y),(x+w,y+h),(0,255,0),3)
    #print("sum_contours_ivalid = ",sum_contours_ivalid)
    A1[1] -= height//100
    B1[1] -= height//100
    A2[1] = A2[1] + height//30
    B2[1] = A2[1]
    B1[0] += width//40
    B2[0] += width//40
    #print("A1 = ", A1)
    #print("A2 = ", A2)
    #print("B1 = ", B1)
    #print("B2 = ", B2)

    pts1 = np.float32([A1, B1, A2, B2])
    pts2 = np.float32([[0, 0], [1500, 0], [0, 1000], [1500, 1000]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst1 = cv2.warpPerspective(image, M, (1500, 1000))
    #save_for_debug(image_path, dst1, "07_cutting_word", dst1.shape)

    dst2 = cv2.warpPerspective(thre, M, (1500, 1000))
    #save_for_debug(image_path, dst2, "08_cutting_word_binary_inv", dst2.shape)

    contours = cv2.findContours(
        dst2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    #save_contour_for_debug(image_path, dst1, contours, "09_contours_inv", dst1.shape)
    return dst2

#img = editting_image("mnist_detection/test.jpg")


def detect_image(path_image):
    image = editting_image(path_image)
    contours = cv2.findContours(
        image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    # print(len(contours))
    listRect = []
    for i in contours:
        (x, y, w, h) = cv2.boundingRect(i)
        if(w > 10 and h > 10 and w < 100 and h < 100):
            listRect.append([x, y, w, h])
    # print(len(listRect))
    listRect.sort(key=sortSecond)
    index = 0
    list_image_detected = []
    while index < len(listRect):
        index += 20
        list_iden = listRect[index:index+12]
        # print(list_iden)
        list_iden.sort(key=sortFirst)
        # print(list_iden)
        iden = []
        for i in range(len(list_iden)):
            (x, y, w, h) = list_iden[i]
            piece_iden = image[y:y+h, x:x+w]
            #cv2.imwrite("piece"+str(i)+".jpg", piece_image)
            piece_iden = cv2.resize(
                piece_iden, (28, 28), interpolation=cv2.INTER_AREA)
            iden.append(piece_iden)
        list_image_detected.append(iden)
        index += 12
        list_date = listRect[index:index+8]
        list_date.sort(key=sortFirst)
        # print(list_date)
        date = []
        for i in range(len(list_date)):
            (x, y, w, h) = list_date[i]
            piece_date = image[y:y+h, x:x+w]
            #cv2.imwrite("piece"+str(i)+".jpg", piece_date)
            piece_date = cv2.resize(
                piece_date, (28, 28), interpolation=cv2.INTER_AREA)
            date.append(piece_date)
        list_image_detected.append(date)
        index += 8
    return list_image_detected

"""
image = detect_image("mnist_detection/test3.jpg")
for i in range(len(image)):
    for j in range(len(image[i])):
        cv2.imwrite("piece"+str(i)+"-"+str(j)+".jpg",image[i][j])
"""
