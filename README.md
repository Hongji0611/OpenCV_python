## **소개**
hw1, hw2, hw3를 통해 영상 처리를 python opencv으로 구현.
final에서 미니 포토샵을 통해 이미지 및 영상을 변환하는 작은 포토샵을 GUI를 사용하여 구현.

## 개발 환경

- 사용 언어: Python
- 라이브러리: opencv
- 패키지 관리 시스템: anaconda
- GUI : PyQt5

## HW1 Point Processing
![image](https://user-images.githubusercontent.com/63103070/124430498-32b78980-ddaa-11eb-8932-50a812794d40.png)
1. Histogram Equalization을 이용한 대비개선
 - opencv 라이브러리를 사용하여 이미지를 가져온다.
 - 픽셀값을 읽으면서 누적 histogram을 계산한다. (2D -> 1D)
 - Look up table을 생성한다.
 - table을 읽으면서 이미지의 값을 변경해 대비를 향상시킨다.
 - 결과 이미지를 저장하고, 화면에 출력한다.

![image](https://user-images.githubusercontent.com/63103070/124430531-3f3be200-ddaa-11eb-83a7-e9ddd570795a.png)
2. negative
 - gray scale 이미지의 픽셀 값을 하나씩 읽으면서 255에서 값을 빼 반전시킨다.
 - 결과 이미지를 저장하고, 화면에 출력한다.

![image](https://user-images.githubusercontent.com/63103070/124430575-4e229480-ddaa-11eb-8924-8f451fd16f31.png)
3. power law transformation
 - 이미지를 가져오고 𝑠 = 𝑐𝑟^𝛾 에서 c = 1 로 고정한 뒤 다음을 각 픽셀에 대해 반복한다.
 - r은 [0, 1] 사이의 값 이므로, 255로 나누어 준 뒤 이를 감마 제곱한다. 
 - 원래의 범위인 [0, 255]으로 되돌리기 위해 255.를 다시 곱한다.
 - 결과 이미지를 저장하고, 화면에 출력한다.
 
 코드 분석 결과, Histogram Equalization > Power law transformation > Negative 순으로 비용이
많이 발생함을 볼 수 있다. 
 Histogram-Equalization을 사용한 경우, 대부분의 이미지는 잘 개선되었다. 그러나 가우시안 분포 형태이거나, 어둠과 밝음의 값 차이가 명확한 경우에는 Power-law-transformation 또는 Negative를 사용한 경우에서 더 대비가 향상된 이미지를 얻을 수 있었다. 따라서 목적이 명확하거나 정해진 히스토그램 형태가 존재한다면, HE보다 성능 비용이 저렴한 방식을 선택하는 것이 좋다. 그 이외에 히스토그램이 매번 달라지는 이미지를 처리해야 한다면, 대체적으로 잘 개선시키는 HE를 사용하는 것이 낫다는 결론을 도출했다.

cf.자세한 결과 분석은 첨부된 pdf 참조

## HW2 Area processing (Smoothing, Sharpening, Edge detection)
![image](https://user-images.githubusercontent.com/63103070/124430649-6397be80-ddaa-11eb-8538-ea6a29bd5403.png)
[Smoothing]
1. Gaussian filtering
  - sigma값을 10으로 하여 1D에서의 gaussian값을 행, 열 방향으로 구하여 mask를 얻는다.
  - 컬러 이미지일 경우, RGB 3개의 영역에서 convolution을 진행한다.
  - 결과 이미지를 저장하고, 화면에 출력한다.
  
2. median filtering
  - mask의 크기를 15로 설정한다.
  - 15*15 크기만큼의 픽셀읠 나열해서 media값을 구한다.
  - 결과 이미지를 저장하고, 화면에 출력한다.

3. average filtering
  - average를 위한 mask를 생성한다.
  - 이미지와 mask를 convolution한다.
  - 각 픽셀의 값이 0~255가 넘지 않도록 수정한다.
  - 결과 이미지를 저장하고, 화면에 출력한다

실행 시간: average < gaussian < median
노이즈 제거 효과: average < gaussian < median
결론, 실행시간만을 고려한다면 average filter를 사용
노이즈 제거 성능만을 고려한다면 median filter를 사용
빠른 속도와 노이즈 제거 성능을 모두 고려한다면 Gaussian filter를 사용

![image](https://user-images.githubusercontent.com/63103070/124430715-790ce880-ddaa-11eb-9740-1223f44528d0.png)
[Sharpening]
1. Highboost filtering
  - high boost filter mask를 생성한다. 
      mask = [-1 -1 -1; -1 9.2 -1; -1 -1 -1;]
  - mask와 이미지를 convolution한다.
  - 결과 이미지를 저장하고, 화면에 출력한다

![image](https://user-images.githubusercontent.com/63103070/124430747-8924c800-ddaa-11eb-87f5-49739397f4ee.png)
[Edge Detection]
1. Gradient (Sobel, Prewitt)
  - x축 및 y축으로 sobel mask를 생성한다.
     kernel_x = [-1 0 1; -2 0 2; -1 0 1;]
     kernel_y = [1 2 1; 0 0 0; -1 -2 -1;]
  - mask와 이미지를 convolution한다.
  - 결과 이미지를 저장하고, 화면에 출력한다.
  
2. LoG
  - LoG mask를 생성한다.
      mask = [0 0 1 0 0; 0 1 2 1 0; 1 2 -16 2 1; 0 1 2 1 0; 0 0 1 0 0;]
  - mask와 이미지를 convolution한다.
  - 결과 이미지를 저장하고, 화면에 출력한다.
  
3. Canny edge operator
  - opencv 라이브러리 함수 cv2.Canny 사용

cf.자세한 결과 분석은 첨부된 pdf 참조

## HW3 Object recognition in videos
![image](https://user-images.githubusercontent.com/63103070/124430929-bb362a00-ddaa-11eb-81d0-079dfa00bcb5.png)
[ 피부색으로 객체 추출 – color 사용 ]
- 영상을 cv2.VidieCapture를 통해 가져온다.
- 프레임마다 아래 과정을 수행한다.
    1) Gaussian filter를 통해 이미지를 bluring한다.
    2) 해당 프레임을 YCrCb로 색을 추출한다.
    3) 피부색 범위를 포함하는 mask를 생성한다.
    4) 프레임과 mask를 and연산하여 배경과 object를 구분한다.
    5) 결과를 프레임별로 png파일 및 비디오로 저장하고, anaconda에 원본과 결과 이미지를 출력한다.

1. Hand Video2.mov – 손 추출
2. Project –hand gesture.AVI – 얼굴 및 손 추출

![image](https://user-images.githubusercontent.com/63103070/124430974-c8ebaf80-ddaa-11eb-85e8-f2503e2019ff.png)
![image](https://user-images.githubusercontent.com/63103070/124431066-e456ba80-ddaa-11eb-9831-7a7dae757ef6.png)
[ 배경 제거로 움직이는 객체 추출 ]
- 영상을 cv2.VidieCapture를 통해 가져온다.
- 프레임마다 아래 과정을 수행한다.
    1) Gaussian filter를 통해 이미지를 bluring한다.
    2) 해당 프레임을 YCrCb로 색을 추출한다.
    3) 피부색 범위를 포함하는 mask를 생성한다. 
        (대비가 명확하지 않은 이미지이기 때문에 앞의 범위보다 좀더 타이트하게 범위를 수정)
    4) 프레임과 mask를 and연산하여 배경과 object를 구분한다.
    5) 결과를 프레임별로 png파일 및 비디오로 저장하고, anaconda에 원본과 결과 이미지를 출력한다.
1. Project_outdoor video1.mov – vehicles 추출
2. Car video2.mp4 – vehicles 추출

cf.자세한 결과 분석은 첨부된 pdf 참조

## Final Project. 미니 포토샵
[프로그램 실행 및 준비사항]
1. RealMiniPhotoshop.py 파일을 다운로드합니다.
2. 실행파일과 background, example, result 폴더는 동일한 폴더에 존재해야 합니다.
3. cv2, PyQt5 등 필요한 라이브러리를 준비합니다.
4. 변환하고 싶은 파일을 example 폴더에 저장합니다.
5. RealMiniPhotoshop.py를 통해 프로그램을 실행합니다.
(*사용자의 실행 환경에 따라 작동이 보장되지 않을 수 있습니다.)

[파일 준비사항]
- hw3 폴더에는 handDetection, get background, vehicle detection을 위한 영상 파일을 저장합니다.
- hw2 폴더에는 gaussian filter, median filter, average filter, gradient detection, high boost filter, canny edge 
detection, LoG edge detection을 위한 이미지 파일을 저장합니다.
- hw1 폴더에는 negative, histogram equalization, power law transformation을 위한 이미지 파일을 저장합니다.
(*지정되지 않은 폴더에 영상 및 이미지를 준비할 경우 정상 작동이 보장되지 않습니다.)


![image](https://user-images.githubusercontent.com/63103070/124430052-a4430800-dda9-11eb-9ef1-2e5c85a67ac2.png)
[화면 구성]
1. 가장 위쪽 레이어에서 영상 및 이미지가 변환된 모습을 확인할 수 있습니다. 또
한 변환된 영상 및 이미지는 result 폴더 및 background 폴더에 저장됩니다.
2. 변환하려는 파일이 폴더에 저장되어 있을 때, 해당 파일명을 text창에 적고 하단
버튼을 통해 변환할 수 있습니다.
3. 하단 버튼을 통해 포토샵 기능을 사용하실 수 있습니다.
- 1행: 영상 처리
(hand detection, get background, vehicle detection)
- 2행: 이미지 처리 [color, gray 모두 가능]
(average filter, gaussian filter, median filter)
- 3행: 이미지 edge 처리
(high-boost filter, gradient, LoG, canny edge detection)
- 4행: 이미지 대비 향상
(histogram equalization, negative, power law transformation)

cf.자세한 결과 분석은 첨부된 pdf 참조
