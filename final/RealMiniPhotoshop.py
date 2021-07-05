#!/usr/bin/env python
# coding: utf-8

# In[1]:


import cv2
import sys
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QApplication, QWidget, QTabWidget, QVBoxLayout

import numpy as np
from PIL import Image
from math import acos, pi, sqrt
import io
import time

class Startvidieo(QtCore.QObject):

    VideoSignal1 = QtCore.pyqtSignal(QtGui.QImage)

    def __init__(self, parent=None):
        super(Startvidieo, self).__init__(parent)
    
    def convolution_color(self, array, f):
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
    
    def convolution(self, array, f):
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
    
    def gaussian2D(self, sigma):
        # 두 벡터를 외적하여 x,y 2차원 mask를 생성
        result = np.outer(self.gaussian1D(sigma),self.gaussian1D(sigma))
        total = np.sum(result)
        result/=total

        return result
    
    def gaussian1D(self, sigma):
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
    
    @QtCore.pyqtSlot()
    def gaussianFiltering_color(self, name):
        img = cv2.imread("./example/hw2/%s" %name)

        if img is None:
            img = cv2.imread("./example/hw2/lover.jpg")
        
        img_arr = np.asarray(img)
        img_arr = img_arr.astype('float32')
        
        if len(img.shape) == 3: # 컬러 이미지
            H, W, C = img.shape
            img_arr = self.convolution_color(img_arr, self.gaussian2D(10))
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
            
        else:  # 흑백 이미지
            img = np.expand_dims(img, axis=-1)
            H, W, C = img.shape
            img_arr = self.convolution(img_arr, self.gaussian2D(10))

        img_result = img_arr.astype('uint8')
        img_result = cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB)
        cv2.imwrite("./result/gaussianFilter.png", img_result) 

        out = cv2.resize(img_result, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        out = QImage(out.data, 640, 480, out.strides[0], QImage.Format_RGB888)
        self.VideoSignal1.emit(out)
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(25, loop.quit) #25 ms
        loop.exec_()
#         cv2.imshow("gaussianFilter",img_result)
#         cv2.waitKey()
    
    @QtCore.pyqtSlot()
    def median_color(self, name):
        img = cv2.imread("./example/hw2/%s" %name)

        if img is None:
            img = cv2.imread("./example/hw2/lover.jpg")

        height, width, channel = img.shape

        # 결과 배열 생성
        out1 = np.zeros((height + 2, width + 2, channel), dtype=float) 
        out1[1: 1 + height, 1: 1 + width] = img.copy().astype(float)
        temp1 = out1.copy()

        mask = 3

        for i in range(height):
            for j in range(width): 
                for k in range(channel): 
                    out1[1 + i, 1 + j, k] = np.median(temp1[i:i + 15, j:j + 15, k])


        out1 = out1[1:1 + height, 1:1 + width].astype(np.uint8) 
        cv2.imwrite("./result/medianFilterColor.png", out1) 

        out = cv2.resize(out1, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        out = QImage(out.data, 640, 480, out.strides[0], QImage.Format_RGB888)
        self.VideoSignal1.emit(out)
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(25, loop.quit) #25 ms
        loop.exec_()
    
    @QtCore.pyqtSlot()
    def averageFilter(self, name):
        name = str(name)
        print(name)
        img = cv2.imread("./example/hw2/%s" %name)

        if img is None:
            img = cv2.imread("./example/hw2/Fig0504(i)(salt-pepper-noise).jpg")

        if len(img.shape) == 3: # 컬러 이미지
            H, W, C = img.shape
        else:  # 흑백 이미지
            img = np.expand_dims(img, axis=-1)
            H, W, C = img.shape

        pad = 3 // 2
        out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=float)
        out[pad: pad + H, pad: pad + W] = img.copy().astype(float)

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
        
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        out = cv2.resize(out, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
        out = QImage(out.data, 640, 480, out.strides[0], QImage.Format_RGB888)
        self.VideoSignal1.emit(out)
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(25, loop.quit) #25 ms
        loop.exec_()
    
    @QtCore.pyqtSlot()
    def handDetection(self,name):
        cap = cv2.VideoCapture('./example/hw3/%s' %name)
        k = cap.isOpened()

        if k==False:
            cap.open("./example/hw3/Hand Video2.mov")

        width = int(cap.get(3)) # 가로 길이
        height = int(cap.get(4)) # 세로 길이
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # 동영상을 output으로 저장할 때
        fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        out = cv2.VideoWriter('./result/output.avi', fcc, fps, (width, height))
        
        while True:
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
                out.write(mask_color)
                
                mask_color = cv2.cvtColor(mask_color, cv2.COLOR_BGR2RGB)
                mask_color = cv2.resize(mask_color, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
                mask_color = QImage(mask_color.data, 640, 480, mask_color.strides[0], QImage.Format_RGB888)
                self.VideoSignal1.emit(mask_color)

                loop = QtCore.QEventLoop()
                QtCore.QTimer.singleShot(25, loop.quit) #25 ms
                loop.exec_()
            except KeyboardInterrupt: break
        if cap.isOpened():
            cap.release()
        if out.isOpened():
            out.release()
        
    @QtCore.pyqtSlot()
    def get_background(self, name):
        cap = cv2.VideoCapture('./example/hw3/%s' %name)
        k = cap.isOpened()

        if k==False:
            cap.open("./example/hw3/Project_outdoor video1.mov")

        width = int(cap.get(3)) # 가로 길이
        height = int(cap.get(4)) # 세로 길이
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        back_img = np.zeros(shape=(height,width,3),dtype=np.float32)
        count = 0
        
        while True:
            try:
                ret, image = cap.read()

                if not ret:
                    break

                count += 1
                cv2.accumulate(image, back_img)
                average_back = back_img/count
                result_img = cv2.convertScaleAbs(average_back)
                
                temp_img = cv2.resize(result_img, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
                temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
                temp_img = QImage(temp_img.data, 640, 480, temp_img.strides[0], QImage.Format_RGB888)
                self.VideoSignal1.emit(temp_img)

                loop = QtCore.QEventLoop()
                QtCore.QTimer.singleShot(25, loop.quit) #25 ms
                loop.exec_()
            except KeyboardInterrupt: break
        if cap.isOpened():
            cap.release()
        cv2.imwrite('./background/back_img.png',result_img)
    
    @QtCore.pyqtSlot()
    def vehicleDetection(self, name):
        cap = cv2.VideoCapture('./example/hw3/%s' %name)
        k = cap.isOpened()

        if k==False:
            cap.open("./example/hw3/Project_outdoor video1.mov")

        width = int(cap.get(3)) # 가로 길이
        height = int(cap.get(4)) # 세로 길이
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        back_img = cv2.imread('./background/back_img.png')
        back_img = cv2.resize(back_img, dsize=(width,height), interpolation=cv2.INTER_AREA)
        
        # 동영상을 output으로 저장할 때
        fcc = cv2.VideoWriter_fourcc('D', 'I', 'V', 'X')
        out = cv2.VideoWriter('./result/output.avi', fcc, fps, (width, height))
        
        while True:
            try:
                ret, image = cap.read()
                if not ret:
                    break
                sub_img = cv2.absdiff(image, back_img)

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
                        cv2.rectangle(image,(x,y),(x+width,y+height),(0,255, 0),2)
                out.write(image)
                
                temp_img = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
                temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB)
                temp_img = QImage(temp_img.data, 640, 480, temp_img.strides[0], QImage.Format_RGB888)
                self.VideoSignal1.emit(temp_img)

                loop = QtCore.QEventLoop()
                QtCore.QTimer.singleShot(25, loop.quit) #25 ms
                loop.exec_()
            except KeyboardInterrupt: break
        if cap.isOpened():
            cap.release()
        if out.isOpened():
            out.release()
    
    
    def gradient(self, img, size, mask):
        if len(img.shape) == 3: 
            H, W, C = img.shape
        else: 
            img = np.expand_dims(img, axis=-1)
            H, W, C = img.shape

        pad = size // 2
        out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=float)
        out[pad: pad + H, pad: pad + W] = img.copy().astype(float)

        tmp = out.copy()

        for i in range(H): 
            for j in range(W): 
                for k in range(C): 
                    out[pad + i, pad + j, k] = np.sum(mask * tmp[i: i + size, j: j + size, k]) 
        out = np.clip(out, 0, 255) 
        out = out[pad: pad + H, pad: pad + W].astype(np.uint8) 

        return out

    @QtCore.pyqtSlot()
    def sobelEdgeDetection(self, name):
        img = cv2.imread("./example/hw2/%s" %name)

        if img is None:
            img = cv2.imread("./example/hw2/Fig0327(a)(tungsten_original).jpg")
            
        kernel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        kernel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        out_x = self.gradient(img, 3, kernel_x)
        out_y = self.gradient(img, 3, kernel_y)

        merged = out_x+out_y

        cv2.imwrite("./result/sobelEdgeDetection.png", merged) 

        out = cv2.resize(merged, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
        out = QImage(out.data, 640, 480, out.strides[0], QImage.Format_RGB888)
        self.VideoSignal1.emit(out)
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(25, loop.quit) #25 ms
        loop.exec_()
    
    @QtCore.pyqtSlot()
    def highBoostFilter(self, name):
        img = cv2.imread("./example/hw2/%s" %name)

        if img is None:
            img = cv2.imread("./example/hw2/Fig0327(a)(tungsten_original).jpg")

        A = 1.2

        mask2 = np.array([[-1,-1,-1],
                          [-1,A+8,-1],
                          [-1,-1,-1]])

        result2 = self.convolution2(img, 3, mask2)
        cv2.imwrite("./result/highBoostFilterGray.png", result2) 

        result2 = cv2.resize(result2, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
        result2 = QImage(result2.data, 640, 480, result2.strides[0], QImage.Format_RGB888)
        self.VideoSignal1.emit(result2)
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(25, loop.quit) #25 ms
        loop.exec_()
    
    @QtCore.pyqtSlot()
    def cannyEdgeDetection(self, name):
        img = cv2.imread("./example/hw2/%s" %name)

        if img is None:
            img = cv2.imread("./example/hw2/Fig0327(a)(tungsten_original).jpg")

        out2 = cv2.Canny(img, 50, 100)

        cv2.imwrite("./result/cannyEdgeDetection.png", out2) 

        out = cv2.resize(out2, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        out = QImage(out.data, 640, 480, out.strides[0], QImage.Format_RGB888)
        self.VideoSignal1.emit(out)
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(25, loop.quit) #25 ms
        loop.exec_()
#         cv2.imshow("canny",out2)
#         cv2.waitKey()
    
    def LoG_filter(self, img, size, mask):
        if len(img.shape) == 3: 
            H, W, C = img.shape
        else: 
            img = np.expand_dims(img, axis=-1)
            H, W, C = img.shape

        pad = size // 2
        out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=float)
        out[pad: pad + H, pad: pad + W] = img.copy().astype(float)

        tmp = out.copy()

        for i in range(H): 
            for j in range(W): 
                for k in range(C): 
                    out[pad + i, pad + j, k] = np.sum(mask * tmp[i: i + size, j: j + size, k]) 
        out = np.clip(out, 0, 255) 
        out = out[pad: pad + H, pad: pad + W].astype(np.uint8) 

        return out
    
    
    @QtCore.pyqtSlot()
    def logEdgeDetection(self, name):
        img = cv2.imread("./example/hw2/%s" %name)

        if img is None:
            img = cv2.imread("./example/hw2/Fig0327(a)(tungsten_original).jpg")
        
        kernel = np.array([[0,0,1,0,0], [0,1,2,1,0], [1,2,-16,2,1], [0,1,2,1,0],[0,0,1,0,0]])
        out2 = self.LoG_filter(img, 5, kernel)

        cv2.imwrite("./result/LoGEdgeDetection.png", out2) 

        result2 = cv2.resize(out2, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
        result2 = QImage(result2.data, 640, 480, result2.strides[0], QImage.Format_RGB888)
        self.VideoSignal1.emit(result2)
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(25, loop.quit) #25 ms
        loop.exec_()
    
    @QtCore.pyqtSlot()
    def negative(self, name):
        src = cv2.imread("./example/hw1/%s" %name)

        if src is None:
            src = cv2.imread("./example/hw1/3. Negative test.tif", cv2.IMREAD_GRAYSCALE)

        height, width = src.shape[0], src.shape[1]

        for i in range(height):
            for j in range(width):
                src[i][j] = 255. - src[i][j]

        cv2.imwrite("./result/negative.png", src) 

        out = cv2.resize(src, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        out = QImage(out.data, 640, 480, out.strides[0], QImage.Format_RGB888)
        self.VideoSignal1.emit(out)
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(25, loop.quit) #25 ms
        loop.exec_()
#         cv2.imshow("negative",src)
#         cv2.waitKey()
    
    @QtCore.pyqtSlot()
    def histogramEqualization(self, name):
        src = cv2.imread("./example/hw1/%s" %name)

        if src is None:
            src = cv2.imread("./example/hw1/3. Negative test.tif", cv2.IMREAD_GRAYSCALE)

        height, width = src.shape[0], src.shape[1]

        histogram = np.zeros(256)
        lookUpTable = np.zeros(256)

        for i in range(height):
            for j in range(width):
                histogram[src[i][j]] += 1

        sumv = 0.0
        scale_factor = 255.0/(height * width)

        for i in range(256):
            sumv += histogram[i]
            lookUpTable[i] = round( sumv * scale_factor )

        for i in range(height):
            for j in range(width):
                src[i][j] = lookUpTable[src[i][j]]

        cv2.imwrite("./result/histogramEqualization.png", src) 

        out = cv2.resize(src, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        out = QImage(out.data, 640, 480, out.strides[0], QImage.Format_RGB888)
        self.VideoSignal1.emit(out)
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(25, loop.quit) #25 ms
        loop.exec_()
#         cv2.imshow("histogramEqualization",src)
#         cv2.waitKey()
    
    @QtCore.pyqtSlot()
    def powerLawTransformations(self, name):
        src = cv2.imread("./example/hw1/%s" %name)

        if src is None:
            src = cv2.imread("./example/hw1/3. Negative test.tif", cv2.IMREAD_GRAYSCALE)

        height, width = src.shape[0], src.shape[1]

        r = 1.2
        for i in range(height):
            for j in range(width):
                src[i][j] = 255*(src[i][j]/255.)**r

        cv2.imwrite("./result/powerLawTransformations.png", src) 
        
        out = cv2.resize(src, dsize=(640, 480), interpolation=cv2.INTER_LINEAR)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        out = QImage(out.data, 640, 480, out.strides[0], QImage.Format_RGB888)
        self.VideoSignal1.emit(out)
        loop = QtCore.QEventLoop()
        QtCore.QTimer.singleShot(25, loop.quit) #25 ms
        loop.exec_()
    
    def convolution2(self, img, size, mask):
        if len(img.shape) == 3: # 컬러 이미지
            H, W, C = img.shape
        else:  # 흑백 이미지
            img = np.expand_dims(img, axis=-1)
            H, W, C = img.shape

        pad = size // 2
        out = np.zeros((H + pad * 2, W + pad * 2, C), dtype=float)
        out[pad: pad + H, pad: pad + W] = img.copy().astype(float)

        tmp = out.copy()

        for i in range(H): 
            for j in range(W): 
                for k in range(C): 
                    out[pad + i, pad + j, k] = np.sum(mask * tmp[i: i + size, j: j + size, k])

        # 0~255사이의 값으로 변환
        out = np.clip(out, 0, 255) 
        out = out[pad: pad + H, pad: pad + W].astype(np.uint8) 
        return out
    

class ImageViewer(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.image = QtGui.QImage()
        self.setAttribute(QtCore.Qt.WA_OpaquePaintEvent)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QtGui.QImage()

    def initUI(self):
        self.setWindowTitle('Test')

    @QtCore.pyqtSlot(QtGui.QImage)
    def setImage(self, image):
        if image.isNull():
            print("Viewer Dropped frame!")

        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    
    #thread
    thread = QtCore.QThread()
    thread.start()
    vid = Startvidieo()
    vid.moveToThread(thread)
    image_viewer1 = ImageViewer()
    vid.VideoSignal1.connect(image_viewer1.setImage)
    
    #버튼 및 텍스트 생성
    textEdit = QtWidgets.QTextEdit()
    
    push_button1 = QtWidgets.QPushButton('hand detection')
    push_button1.clicked.connect(lambda: vid.handDetection(textEdit.toPlainText()))
    push_button2 = QtWidgets.QPushButton('get background')
    push_button2.clicked.connect(lambda: vid.get_background(textEdit.toPlainText()))
    push_button3 = QtWidgets.QPushButton('vehicle detection')
    push_button3.clicked.connect(lambda: vid.vehicleDetection(textEdit.toPlainText()))
    
    push_button4 = QtWidgets.QPushButton('average filter')
    push_button4.clicked.connect(lambda: vid.averageFilter(textEdit.toPlainText()))
    push_button5 = QtWidgets.QPushButton('gaussian filter')
    push_button5.clicked.connect(lambda: vid.gaussianFiltering_color(textEdit.toPlainText()))
    push_button6 = QtWidgets.QPushButton('median filter')
    push_button6.clicked.connect(lambda: vid.median_color(textEdit.toPlainText()))
    
    push_button7 = QtWidgets.QPushButton('highBoost filter')
    push_button7.clicked.connect(lambda: vid.highBoostFilter(textEdit.toPlainText()))
    push_button8 = QtWidgets.QPushButton('gradiant edge detection')
    push_button8.clicked.connect(lambda: vid.sobelEdgeDetection(textEdit.toPlainText()))
    push_button9 = QtWidgets.QPushButton('LoG edge detection')
    push_button9.clicked.connect(lambda: vid.logEdgeDetection(textEdit.toPlainText()))
    push_button10 = QtWidgets.QPushButton('canny edge detection')
    push_button10.clicked.connect(lambda: vid.cannyEdgeDetection(textEdit.toPlainText()))
    
    push_button11 = QtWidgets.QPushButton('histogram equalization')
    push_button11.clicked.connect(lambda: vid.histogramEqualization(textEdit.toPlainText()))
    push_button12 = QtWidgets.QPushButton('negative')
    push_button12.clicked.connect(lambda: vid.negative(textEdit.toPlainText()))
    push_button13 = QtWidgets.QPushButton('power law transformation')
    push_button13.clicked.connect(lambda: vid.powerLawTransformations(textEdit.toPlainText()))
    
    #버튼 레이아웃
    horizontal_layout0 = QtWidgets.QHBoxLayout()
    horizontal_layout0.addWidget(textEdit)
    
    horizontal_layout = QtWidgets.QHBoxLayout()
    horizontal_layout.addWidget(push_button1)
    horizontal_layout.addWidget(push_button2)
    horizontal_layout.addWidget(push_button3)
    
    horizontal_layout2 = QtWidgets.QHBoxLayout()
    horizontal_layout2.addWidget(push_button4)
    horizontal_layout2.addWidget(push_button5)
    horizontal_layout2.addWidget(push_button6)
    
    horizontal_layout3 = QtWidgets.QHBoxLayout()
    horizontal_layout3.addWidget(push_button7)
    horizontal_layout3.addWidget(push_button8)
    horizontal_layout3.addWidget(push_button9)
    horizontal_layout3.addWidget(push_button10)
    
    horizontal_layout4 = QtWidgets.QHBoxLayout()
    horizontal_layout4.addWidget(push_button11)
    horizontal_layout4.addWidget(push_button12)
    horizontal_layout4.addWidget(push_button13)
    
    #전체 레이아웃
    vertical_layout = QtWidgets.QVBoxLayout()
    vertical_layout.addWidget(image_viewer1)
    vertical_layout.addLayout(horizontal_layout0)
    vertical_layout.addLayout(horizontal_layout)
    vertical_layout.addLayout(horizontal_layout2)
    vertical_layout.addLayout(horizontal_layout3)
    vertical_layout.addLayout(horizontal_layout4)
    
    layout_widget = QtWidgets.QWidget()
    layout_widget.setLayout(vertical_layout)
    
    #show window
    main_window = QtWidgets.QMainWindow()
    main_window.setCentralWidget(layout_widget)
    main_window.setWindowTitle('Mini Photoshop by Jiwoo Hong')
    main_window.show()
    sys.exit(app.exec_())
    


# In[ ]:





# In[ ]:




