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
        if w > 50 and w < 100 and h > 50 and h < 100:
            cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 3)
    cv2.imwrite(save_path, output)
    #print("sum of contours = ",len(contours))


def editting_image(image_path):
    image = cv2.imread(image_path)
    #save_for_debug(image_path, image, '00_origin', image.shape)

    im_resized = cv2.resize(image, (2000, 1500),interpolation=cv2.INTER_AREA)
    #save_for_debug(image_path, im_resized,'01_resized', im_resized.shape)

    lower_white = np.array([0, 0, 50], dtype=np.uint8)
    upper_white = np.array([117, 135, 255], dtype=np.uint8)

    mask = cv2.inRange(im_resized, lower_white, upper_white)
    #save_for_debug(image_path, mask, "02_mask_red", mask.shape)

    mask = cv2.GaussianBlur(mask, (9, 9), 0)
    #save_for_debug(image_path, mask, "03_mask_gauss", mask.shape)

    contours = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
    #save_contour_for_debug(image_path, im_resized, contours,
    #                       "04_contours", im_resized.shape)
    return contours
#img = editting_image("mnist_detection/image4.jpg")


def detect_image(path_image):
    contours = editting_image(path_image)
    #print(len(contours))
    img = cv2.resize(cv2.imread(path_image), (2000, 1500),
                     interpolation=cv2.INTER_AREA)
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #save_for_debug(path_image, im_gray, "05_gray", im_gray.shape)

    thre = cv2.threshold(im_gray, 127, 255, cv2.THRESH_BINARY_INV)[1]
    #save_for_debug(path_image, thre, "06_thresh_binary_inv", thre.shape)

    listRect = []
    for i in contours:
        (x, y, w, h) = cv2.boundingRect(i)
        if(w > 50 and h > 50 and w < 100 and h < 100):
            listRect.append([x, y, w, h])
    print(len(listRect))
    listRect.sort(key=sortSecond)
    index = 0
    list_image_detected = np.zeros(shape=(1, 28, 28))

    while index < len(listRect):
        index += 16
        list_iden = listRect[index:index+12]
        list_iden.sort(key=sortFirst)
        for i in range(len(list_iden)):
            (x, y, w, h) = list_iden[i]
            piece_iden = thre[y+7:y+h-7, x+7:x+w-7]
            piece_iden = cv2.resize(
                piece_iden, (28, 28), interpolation=cv2.INTER_AREA)
            piece_iden = np.array(piece_iden).reshape(1, 28, 28)
            list_image_detected = np.append(
                list_image_detected, piece_iden, axis=0)
        index += 12
        list_date = listRect[index:index+8]
        list_date.sort(key=sortFirst)
        for i in range(len(list_date)):
            (x, y, w, h) = list_date[i]
            piece_date = thre[y+7:y+h-7, x+7:x+w-7]
            piece_date = cv2.resize(
                piece_date, (28, 28), interpolation=cv2.INTER_AREA)
            piece_date = np.array(piece_date).reshape(1, 28, 28)
            list_image_detected = np.append(
                list_image_detected, piece_date, axis=0)
        index += 8

    list_image_detected = np.delete(list_image_detected, 0, 0)
    list_image_detected = np.expand_dims(
        list_image_detected, axis=-1).astype(np.float32)/255.0
    return list_image_detected


#image = detect_image("mnist_detection/image2.jpg")
#print(image.shape)
#for i in range(len(image)):
#    cv2.imwrite("piece"+str(i)+"-"+".jpg",image[i])
