import cv2
import numpy as np

<<<<<<< HEAD
def sortFirst(val):
    return val[0]
=======

def save_for_debug(imagePath, debug_img, debug_name):
    save_path = imagePath+"-"+debug_name+".jpg"
    cv2.imwrite(save_path, debug_img)


def save_contour_for_debug(imagePath, debug_img, contours, debug_name):
    save_path = imagePath+"-"+debug_name+".jpg"
    output = debug_img.copy()
    cv2.drawContours(output, contours, -1, (0, 255, 0), 2)
    cv2.imwrite(save_path, output)
>>>>>>> ff109ed5b5207efd335b2ec8ad9f1dc69f26bd37

def sortSecond(val):
    return val[1]

<<<<<<< HEAD
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
=======

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
    imgBinary = cv2.threshold(img_gray, 135, 255, cv2.THRESH_BINARY)[1]

    # bounding image,lay ra cac doi tuong
    contours, hierachy = cv2.findContours(
        imgBinary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    save_contour_for_debug("D:\\Python\\img_debug\\img",
                        resized_image, contours, "step_contours")


>>>>>>> ff109ed5b5207efd335b2ec8ad9f1dc69f26bd37
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

    for rect in lstRect:
        cv2.rectangle(resized_image, (rect[0], rect[1]),
                    (rect[0]+rect[2], rect[1]+rect[3]), (255, 0, 0), 2)


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
            #crop_img = cv2.resize(crop_img,(32,32))

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
                crop_img = cv2.resize(crop_img, (32, 32))
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

list_image = detect_image("mnist_detection/test3.jpg") 
#print("")