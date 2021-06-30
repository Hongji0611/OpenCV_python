#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ipywidgets as widgets 
from IPython.display import display
import cv2
import numpy as np
from ipywidgets import Box
from PIL import Image
from math import acos, pi, sqrt
import io
import time
from matplotlib import pyplot as plt


# In[2]:


def histogramEqualization(name):
    src = cv2.imread("./example/hw1/%s" %name, cv2.IMREAD_GRAYSCALE)
    
    if src is None:
        src = cv2.imread("./example/hw1/3. Negative test.tif", cv2.IMREAD_GRAYSCALE)
    
    height, width = src.shape[0], src.shape[1]

    histogram = np.zeros(256)
    lookUpTable = np.zeros(256)

    for i in range(height):
        for j in range(width):
            histogram[src[i][j]] += 1

    sum = 0.0
    scale_factor = 255.0/(height * width)

    for i in range(256):
        sum += histogram[i]
        lookUpTable[i] = round( sum * scale_factor )

    for i in range(height):
        for j in range(width):
            src[i][j] = lookUpTable[src[i][j]]

    cv2.imwrite("./result/histogramEqualization.png", src) 
    
    show_img = widgets.Image(
        value = cv2.imencode(".jpeg", src)[1].tobytes(),
        format = 'jpeg'
    )
    display(show_img)
    time.sleep(2)
    show_img.close()


# In[3]:


def negative(name):
    src = cv2.imread("./example/hw1/%s" %name, cv2.IMREAD_GRAYSCALE)
    
    if src is None:
        src = cv2.imread("./example/hw1/3. Negative test.tif", cv2.IMREAD_GRAYSCALE)
    
    height, width = src.shape[0], src.shape[1]
    
    for i in range(height):
        for j in range(width):
            src[i][j] = 255. - src[i][j]

    cv2.imwrite("./result/negative.png", src) 
    
    show_img = widgets.Image(
        value = cv2.imencode(".jpeg", src)[1].tobytes(),
        format = 'jpeg'
    )
    display(show_img)
    time.sleep(2)
    show_img.close()


# In[4]:


def powerLawTransformations(name):
    src = cv2.imread("./example/hw1/%s" %name, cv2.IMREAD_GRAYSCALE)
    
    if src is None:
        src = cv2.imread("./example/hw1/3. Negative test.tif", cv2.IMREAD_GRAYSCALE)
    
    height, width = src.shape[0], src.shape[1]
    
    r = 1.2
    for i in range(height):
        for j in range(width):
            src[i][j] = 255*(src[i][j]/255.)**r

    cv2.imwrite("./result/powerLawTransformations.png", src) 
    
    show_img = widgets.Image(
        value = cv2.imencode(".jpeg", src)[1].tobytes(),
        format = 'jpeg'
    )
    display(show_img)
    time.sleep(2)
    show_img.close()


# In[5]:


def gaussian1D(sigma):
    a = sigma
    
    if a % 2 == 0: #mask size가 짝수일 경우 1을 키운다
        a+=1
        
    a2 = a/2
    filter_size = np.array(range(-int(a/2),int(a/2)+1))
    result = np.zeros(len(filter_size))
    
    for i in range(len(filter_size)):
        x = i-a2
        result[i] =  float(np.exp(-(x**2)/(2*sigma**2)) / (2*3.14*sigma**2)) #가우시안 공식
    
    total = np.sum(result)
    result/=total
    
    return result


# In[6]:


def gaussian2D(sigma):
    # 두 벡터를 외적하여 x,y 2차원 mask를 생성
    result = np.outer(gaussian1D(sigma),gaussian1D(sigma))
    total = np.sum(result)
    result/=total
    
    return result


# In[7]:


def convolution(array, f):
    a = (len(f)-1)//2
    # boundery를 0으로 채운다
    img_pad = np.pad(array, ((a,a),(a,a)),'constant')
    
    #convolution을 위해 2번 뒤집는다
    mask = np.rot90(f)
    mask = np.rot90(mask)
    
    result_img = np.zeros((len(array), len(array[0])))
    result_img = result_img.astype('float32')
    
    for i in range(len(result_img)):
        for j in range(len(result_img[0])):
            result_img[i][j] = np.sum(img_pad[i:i+len(f),j:j+len(f)]*mask)
    
    return result_img


# In[8]:


def convolution2(img, size, mask):
    
    if len(img.shape) == 3: # 컬러 이미지
        H, W, C = img.shape
    else:  # 흑백 이미지
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape

    pad = size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)

    tmp = out.copy()
    
    for i in range(H): 
        for j in range(W): 
            for k in range(C): 
                out[pad + i, pad + j, k] = np.sum(mask * tmp[i: i + size, j: j + size, k])
                
    # 0~255사이의 값으로 변환
    out = np.clip(out, 0, 255) 
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8) 
    
    return out


# In[9]:


def convolution_color(array, f):
    a = (len(f)-1)//2
    # boundery를 0으로 채운다
    img_pad = np.pad(array, ((a,a),(a,a),(0,0)),'constant', constant_values=0)
    
    #convolution을 위해 2번 뒤집는다
    mask = np.rot90(f)
    mask = np.rot90(mask)
    
    #color img 이므로 3차원으로 배열 설정
    result_img = np.zeros((len(array),len(array[0]),3))
    result_img = result_img.astype('float32')
    
    for i in range(len(result_img)):
        for j in range(len(result_img[0])):
            result_img[i][j][0] = np.sum(img_pad[i:i+len(f), j:j+len(f),0]*mask)
            result_img[i][j][1] = np.sum(img_pad[i:i+len(f), j:j+len(f),1]*mask)
            result_img[i][j][2] = np.sum(img_pad[i:i+len(f), j:j+len(f),2]*mask)
    return result_img


# In[10]:


def gaussianFiltering_color(name):
    try:
        img = Image.open("./example/hw2/%s" %name)
    except IOError:
        img = Image.open("./example/hw2/Salt&pepper noise.png")

    img_arr = np.asarray(img)
    img_arr = img_arr.astype('float32')

    img_result = convolution_color(img_arr, gaussian2D(10))
    img_result = img_result.astype('uint8')
    img_result = Image.fromarray(img_result)
    
    save_img = np.array(img_result)
    cv2.imwrite("./result/gaussianFilterColor.png", save_img) 
   
    img_byte_arr = io.BytesIO()
    img_result.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    show_img = widgets.Image(
        value = img_byte_arr,
        format='png',
    )
    display(show_img)
    time.sleep(2)
    show_img.close()


# In[11]:


def gaussianFiltering_gray(name):
    try:
        img = Image.open("./example/hw2/%s" %name)
    except IOError:
        img = Image.open("./example/hw2/Fig0504(i)(salt-pepper-noise).jpg")

    img_arr = np.asarray(img)
    img_arr = img_arr.astype('float32')

    img_result = convolution(img_arr, gaussian2D(3))
    img_result = img_result.astype('uint8')
    img_result = Image.fromarray(img_result)
    
    save_img = np.array(img_result)
    cv2.imwrite("./result/gaussianFilterGray.png", save_img) 
   
    img_byte_arr = io.BytesIO()
    img_result.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    show_img = widgets.Image(
        value=img_byte_arr,
        format='png',
        width=300,
        height=400,
    )
    display(show_img)
    time.sleep(2)
    show_img.close()


# In[12]:


def median_gray(name):
    img = cv2.imread("./example/hw2/%s" %name)
    
    if img is None:
        img = cv2.imread("./example/hw2/Fig0504(i)(salt-pepper-noise).jpg")
    
    height, width, channel = img.shape

    # 결과 배열 생성
    out1 = np.zeros((height + 2, width + 2, channel), dtype=np.float) 
    out1[1: 1 + height, 1: 1 + width] = img.copy().astype(np.float)
    temp1 = out1.copy()

    mask = 3

    for i in range(height):
        for j in range(width): 
            for k in range(channel): # mask size만큼 중간값 계산
                out1[1 + i, 1 + j, k] = np.median(temp1[i:i + mask, j:j + mask, k])

    out1 = out1[1:1 + height, 1:1 + width].astype(np.uint8) 
    cv2.imwrite("./result/medianFilterGray.png", out1) 
    
    show_img = widgets.Image(
        value = cv2.imencode(".jpeg", out1)[1].tobytes(),
        format = 'jpeg'
    )
    display(show_img)
    time.sleep(2)
    show_img.close()


# In[13]:


def median_color(name):
    img = cv2.imread("./example/hw2/%s" %name)
    
    if img is None:
        img = cv2.imread("./example/hw2/Salt&pepper noise.png")
    
    height, width, channel = img.shape

    # 결과 배열 생성
    out1 = np.zeros((height + 2, width + 2, channel), dtype=np.float) 
    out1[1: 1 + height, 1: 1 + width] = img.copy().astype(np.float)
    temp1 = out1.copy()

    mask = 3

    for i in range(height):
        for j in range(width): 
            for k in range(channel): 
                out1[1 + i, 1 + j, k] = np.median(temp1[i:i + 15, j:j + 15, k])


    out1 = out1[1:1 + height, 1:1 + width].astype(np.uint8) 
    cv2.imwrite("./result/medianFilterColor.png", out1) 
    
    show_img = widgets.Image(
        value = cv2.imencode(".jpeg", out1)[1].tobytes(),
        format = 'jpeg'
    )
    display(show_img)
    time.sleep(2)
    show_img.close()


# In[14]:


def average_filter(name):
    img = cv2.imread("./example/hw2/%s" %name)
    
    if img is None:
        img = cv2.imread("./example/hw2/Fig0504(i)(salt-pepper-noise).jpg")
    
    if len(img.shape) == 3: # 컬러 이미지
        H, W, C = img.shape
    else:  # 흑백 이미지
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape

    pad = 3 // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)

    # mask 생성
    mask = np.ones((3, 3))/3**2

    tmp = out.copy()
    
    for i in range(H): 
        for j in range(W): 
            for k in range(C): 
                out[pad + i, pad + j, k] = np.sum(mask * tmp[i: i + 3, j: j + 3, k])
                
    # 0~255사이의 값으로 변환
    out = np.clip(out, 0, 255) 
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8) 
    cv2.imwrite("./result/averageFilterGray.png", out) 
    
    show_img = widgets.Image(
        value = cv2.imencode(".jpeg", out)[1].tobytes(),
        format = 'jpeg'
    )
    display(show_img)
    time.sleep(2)
    show_img.close()


# In[31]:


def highBoostFilter(name):
    img = cv2.imread("./example/hw2/%s" %name)
    
    if img is None:
        img = cv2.imread("./example/hw2/Fig0327(a)(tungsten_original).jpg")

    A = 1.2

    mask2 = np.array([[-1,-1,-1],
                      [-1,A+8,-1],
                      [-1,-1,-1]])

    result2 = convolution2(img, 3, mask2)
    cv2.imwrite("./result/highBoostFilterGray.png", result2) 
    
    show_img = widgets.Image(
        value = cv2.imencode(".jpeg", result2)[1].tobytes(),
        format = 'jpeg'
    )
    display(show_img)
    time.sleep(2)
    show_img.close()


# In[32]:


def sobel_X():
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    return kernel


# In[33]:


def sobel_Y():
    kernel = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])
    return kernel


# In[34]:


def prewitt_X():
    kernel = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    return kernel


# In[35]:


def prewitt_Y():
    kernel = np.array([[1, 1, 1], [0, 0, 0], [-1, -1, -1]])
    return kernel


# In[36]:


def gradient(img, size, mask):
    if len(img.shape) == 3: 
        H, W, C = img.shape
    else: 
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape

    pad = size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)

    tmp = out.copy()
    
    for i in range(H): 
        for j in range(W): 
            for k in range(C): 
                out[pad + i, pad + j, k] = np.sum(mask * tmp[i: i + size, j: j + size, k]) 
    out = np.clip(out, 0, 255) 
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8) 
    
    return out


# In[37]:


def prewittEdgeDetection(name):
    img = cv2.imread("./example/hw2/%s" %name)
    
    if img is None:
        img = cv2.imread("./example/hw2/Fig0327(a)(tungsten_original).jpg")
    
    out_x = gradient(img, 3, prewitt_X())
    out_y = gradient(img, 3, prewitt_Y())

    merged = out_x+out_y
    
    cv2.imwrite("./result/prewittEdgeDetection.png", merged) 
    
    show_img = widgets.Image(
        value = cv2.imencode(".jpeg", merged)[1].tobytes(),
        format = 'jpeg'
    )
    display(show_img)
    time.sleep(2)
    show_img.close()
    


# In[38]:


def sobelEdgeDetection(name):
    img = cv2.imread("./example/hw2/%s" %name)
    
    if img is None:
        img = cv2.imread("./example/hw2/Fig0327(a)(tungsten_original).jpg")
    
    out_x = gradient(img, 3, sobel_X())
    out_y = gradient(img, 3, sobel_Y())

    merged = out_x+out_y
    
    cv2.imwrite("./result/sobelEdgeDetection.png", merged) 
    
    show_img = widgets.Image(
        value = cv2.imencode(".jpeg", merged)[1].tobytes(),
        format = 'jpeg'
    )
    display(show_img)
    time.sleep(2)
    show_img.close()
    


# In[39]:


def log_5x5():
    kernel = np.array([[0,0,1,0,0], [0,1,2,1,0], [1,2,-16,2,1], [0,1,2,1,0],[0,0,1,0,0]])
    return kernel


# In[40]:


def LoG_filter(img, size, mask):
    if len(img.shape) == 3: 
        H, W, C = img.shape
    else: 
        img = np.expand_dims(img, axis=-1)
        H, W, C = img.shape

    pad = size // 2
    out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=np.float)
    out[pad: pad + H, pad: pad + W] = img.copy().astype(np.float)

    tmp = out.copy()
    
    for i in range(H): 
        for j in range(W): 
            for k in range(C): 
                out[pad + i, pad + j, k] = np.sum(mask * tmp[i: i + size, j: j + size, k]) 
    out = np.clip(out, 0, 255) 
    out = out[pad: pad + H, pad: pad + W].astype(np.uint8) 
    
    return out


# In[41]:


def logEdgeDetection(name):
    img = cv2.imread("./example/hw2/%s" %name)
    
    if img is None:
        img = cv2.imread("./example/hw2/Fig0327(a)(tungsten_original).jpg")
        
    out2 = LoG_filter(img, 5, log_5x5())
    
    cv2.imwrite("./result/LoGEdgeDetection.png", out2) 
    
    show_img = widgets.Image(
        value = cv2.imencode(".jpeg", out2)[1].tobytes(),
        format = 'jpeg'
    )
    display(show_img)
    time.sleep(2)
    show_img.close()
    


# In[42]:


def cannyEdgeDetection(name):
    img = cv2.imread("./example/hw2/%s" %name)
    
    if img is None:
        img = cv2.imread("./example/hw2/Fig0327(a)(tungsten_original).jpg")
    
    out2 = cv2.Canny(img, 50, 100)
    
    cv2.imwrite("./result/cannyEdgeDetection.png", out2) 
    
    show_img = widgets.Image(
        value = cv2.imencode(".jpeg", out2)[1].tobytes(),
        format = 'jpeg'
    )
    display(show_img)
    time.sleep(2)
    show_img.close()


# In[43]:


def handDetection(name):
    cap = cv2.VideoCapture('./example/hw3/%s' %name)
    k = cap.isOpened()
    
    if k==False:
        cap.open("./example/hw3/Hand Video2.mov")

    width = int(cap.get(3)) # 가로 길이
    height = int(cap.get(4)) # 세로 길이
    fps = cap.get(cv2.CAP_PROP_FPS)

    count = 0

    # 동영상을 output으로 저장할 때
    fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    out = cv2.VideoWriter('./result/output.avi', fcc, fps, (width, height))

    widget1 = widgets.Image(layout = widgets.Layout(border="solid"), width="50%")
    widget2 = widgets.Image(layout = widgets.Layout(border="solid"), width="50%")

    output = [widget1, widget2]
    box = Box(children=output)
    display(box)

    while True :
        try:
            ret, image = cap.read() 
            if not ret:
                break
            #1. 전처리
            image = cv2.GaussianBlur(image, (5,5),0)

            #2.피부 검출
            YCrCb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            mask_hand = cv2.inRange(YCrCb,np.array([0,138,76]),np.array([255,175,127]))
            mask_color = cv2.bitwise_and(image,image, mask=mask_hand)

            #3. 후처리
            mask_color = cv2.erode(mask_color,None,1)

            count += 1 
            out.write(mask_color)
            widget1.value = cv2.imencode(".jpeg", image)[1].tobytes()
            widget2.value = cv2.imencode(".jpeg", mask_color)[1].tobytes()

        except KeyboardInterrupt: break
    if cap.isOpened():
        cap.release()

    if out.isOpened():
        out.release()
    widget1.close()
    widget2.close()


# In[44]:


def get_background(name):
    cap = cv2.VideoCapture('./example/hw3/%s' %name)
    k = cap.isOpened()
    
    if k==False:
        cap.open("./example/hw3/Project_outdoor video1.mov")

    width = int(cap.get(3)) # 가로 길이
    height = int(cap.get(4)) # 세로 길이
    
    back_img = np.zeros(shape=(height,width,3),dtype=np.float32)

    widget1 = widgets.Image(layout = widgets.Layout(border="solid"), width="50%") 
    widget2 = widgets.Image(layout = widgets.Layout(border="solid"), width="50%") 
    items = [widget1, widget2]
    box = Box(children=items)
    display(box)

    count = 0

    while (cap.isOpened()) :
        try:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            cv2.accumulate(frame,back_img)
            average_back = back_img/count
            result_img = cv2.convertScaleAbs(average_back)

            widget1.value = cv2.imencode(".jpeg", frame)[1].tobytes()
            widget2.value = cv2.imencode(".jpeg", result_img)[1].tobytes()

        except KeyboardInterrupt: break

    if cap.isOpened():
        cap.release()
    widget1.close()
    widget2.close()
    cv2.imwrite('./background/back_img.png',result_img)


# In[45]:


def vehicleDetection(name):
    cap = cv2.VideoCapture('./example/hw3/%s' %name)
    k = cap.isOpened()
    
    if k==False:
        cap.open("./example/hw3/Project_outdoor video1.mov")

    width = int(cap.get(3)) # 가로 길이
    height = int(cap.get(4)) # 세로 길이
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    count = 0

    back_img = cv2.imread('./background/back_img.png')
    back_img = cv2.resize(back_img, dsize=(width,height), interpolation=cv2.INTER_AREA)

    widget1 = widgets.Image(layout = widgets.Layout(border="solid"), width="50%") 
    widget2 = widgets.Image(layout = widgets.Layout(border="solid"), width="50%")

    items = [widget1, widget2]
    box = Box(children=items)
    display(box)
    
    # 동영상을 output으로 저장할 때
    fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
    out = cv2.VideoWriter('./result/output.avi', fcc, fps, (width, height))

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                break
            sub_img = cv2.absdiff(frame,back_img)

            B,G,R = cv2.split(sub_img)
            ret,B = cv2.threshold(B,35,255,cv2.THRESH_BINARY)
            ret,G = cv2.threshold(G,35,255,cv2.THRESH_BINARY)
            ret,R = cv2.threshold(R,35,255,cv2.THRESH_BINARY)

            thres_img = cv2.bitwise_or(B,G)
            thres_img = cv2.bitwise_or(R,thres_img)

            thres_img = cv2.dilate(thres_img,None,1)
            thres_img = cv2.erode(thres_img,None, 3)

            box_round,temp = cv2.findContours(thres_img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            for i, now in enumerate(box_round):
                area = cv2.contourArea(now)
                if area>110:
                    x,y,width,height = cv2.boundingRect(now)
                    cv2.rectangle(frame,(x,y),(x+width,y+height),(0,255, 0),2)

            widget1.value = cv2.imencode(".png", frame)[1].tobytes()
            widget2.value = cv2.imencode(".png", thres_img)[1].tobytes()
            count +=1
            out.write(frame)

        except KeyboardInterrupt:
            break

    if cap.isOpened():
        cap.release()
    widget1.close()
    widget2.close()


# In[46]:


def main():
    fileName = widgets.Textarea(
        placeholder='Type file name',
        description='String:',
    )

    btn_hand_detect = widgets.Button(description="hand detection")
    btn_get_background = widgets.Button(description="get background")
    btn_vehicle_detect = widgets.Button(description="vehicle detection")

    btn_gaussian_gray = widgets.Button(description="gaussian filter-gray")
    btn_median_gray = widgets.Button(description="median filter-gray")
    btn_average_gray = widgets.Button(description="average filter")
    btn_gaussian_color = widgets.Button(description="gaussian filter-color")
    btn_median_color = widgets.Button(description="median filter-color")

    btn_highBoost_gray = widgets.Button(description="highBoost filter-gray")
    btn_sobel_edge = widgets.Button(description="sobel edge detection")
    btn_prewitt_edge = widgets.Button(description="prewitt edge detection")
    btn_log_edge = widgets.Button(description="LoG edge detection")
    btn_canny_edge = widgets.Button(description="canny edge detection")

    btn_histogramEqualization = widgets.Button(description="histogram equalization")
    btn_negative = widgets.Button(description="negative")
    btn_powerLawTransformations = widgets.Button(description="power law transformations")


    output = widgets.Output()

    btn_items = [btn_hand_detect, btn_get_background, btn_vehicle_detect]
    btn_box = Box(children=btn_items)

    btn_items1 = [btn_gaussian_gray, btn_median_gray, btn_average_gray, btn_gaussian_color, btn_median_color]
    btn_box1 = Box(children=btn_items1)

    btn_items2 = [btn_highBoost_gray, btn_sobel_edge, btn_prewitt_edge, btn_log_edge, btn_canny_edge]
    btn_box2 = Box(children=btn_items2)

    btn_items3 = [btn_histogramEqualization, btn_negative,btn_powerLawTransformations]
    btn_box3 = Box(children=btn_items3)

    def btn_hand_detect_clicked(b):
        with output:
            handDetection(fileName.value)
    btn_hand_detect.on_click(btn_hand_detect_clicked)

    def btn_get_background_clicked(b):
        with output:
            get_background(fileName.value)
    btn_get_background.on_click(btn_get_background_clicked)

    def btn_vehicle_detect_clicked(b):
        with output:
            vehicleDetection(fileName.value)
    btn_vehicle_detect.on_click(btn_vehicle_detect_clicked)

    def btn_gaussian_gray_clicked(b):
        with output:
            gaussianFiltering_gray(fileName.value)
    btn_gaussian_gray.on_click(btn_gaussian_gray_clicked)

    def btn_median_gray_clicked(b):
        with output:
            median_gray(fileName.value)
    btn_median_gray.on_click(btn_median_gray_clicked)

    def btn_median_color_clicked(b):
        with output:
            median_color(fileName.value)
    btn_median_color.on_click(btn_median_color_clicked)

    def btn_average_gray_clicked(b):
        with output:
            average_filter(fileName.value)
    btn_average_gray.on_click(btn_average_gray_clicked)

    def btn_highBoost_gray_clicked(b):
        with output:
            highBoostFilter(fileName.value)
    btn_highBoost_gray.on_click(btn_highBoost_gray_clicked)

    def btn_sobel_edge_clicked(b):
        with output:
            sobelEdgeDetection(fileName.value)
    btn_sobel_edge.on_click(btn_sobel_edge_clicked)

    def btn_prewitt_edge_clicked(b):
        with output:
            prewittEdgeDetection(fileName.value)
    btn_prewitt_edge.on_click(btn_prewitt_edge_clicked)

    def btn_log_edge_clicked(b):
        with output:
            logEdgeDetection(fileName.value)
    btn_log_edge.on_click(btn_log_edge_clicked)

    def btn_canny_edge_clicked(b):
        with output:
            cannyEdgeDetection(fileName.value)
    btn_canny_edge.on_click(btn_canny_edge_clicked)

    def btn_gaussian_color_clicked(b):
        with output:
            gaussianFiltering_color(fileName.value)
    btn_gaussian_color.on_click(btn_gaussian_color_clicked)

    def btn_histogramEqualization_clicked(b):
        with output:
            histogramEqualization(fileName.value)
    btn_histogramEqualization.on_click(btn_histogramEqualization_clicked)

    def btn_negative_clicked(b):
        with output:
            negative(fileName.value)
    btn_negative.on_click(btn_negative_clicked)

    def btn_powerLawTransformations_clicked(b):
        with output:
            powerLawTransformations(fileName.value)
    btn_powerLawTransformations.on_click(btn_powerLawTransformations_clicked)

    display(fileName)
    display(btn_box)
    display(btn_box1)
    display(btn_box2)
    display(btn_box3)
    display(output)


# In[47]:


if __name__ == '__main__':
    main()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




