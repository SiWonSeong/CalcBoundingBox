import numpy as np
import cv2
import os

def calc_threshold(hist, ratio):
    th = 0
    total_ratio = 0.0
    total_pixel = np.sum(hist)
    for idx in range(0, 255):
        total_ratio += float(hist[idx]/total_pixel) # 밝기 값 i에 대한 비율을 누적합
        if total_ratio >= ratio: # ratio를 기준으로 threshold 결정
            th = idx
            break
    return int(th) # 밝기 값 return

def calc_bounding_box(bin_img):
    x_min = len(bin_img[0]) - 1
    y_min = len(bin_img) - 1
    x_max, y_max = 0, 0
    for y in range(len(bin_img) - 1):
        for x in range(len(bin_img[0]) - 1):
            if bin_img[y, x]==255:
                if not(bin_img[y-1, x] == 255 & bin_img[y, x-1] == 255 &
                       bin_img[y+1, x] == 255 & bin_img[y, x+1] == 255): # 외접
                    if x < x_min:
                        x_min = x
                    elif x > x_max:
                        x_max = x
                    if y < y_min:
                        y_min = y
                    elif y > y_max:
                        y_max = y
    return x_min, y_min, x_max, y_max

def full_pipeline(path, save, ratio):
    img = cv2.imread(path, cv2.COLOR_RGB2BGR)
    print("image loaded: " + path)

    # 1. 흑백 이미지로 변환
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 2. 밝기값 누적 히스토그램
    hist = cv2.calcHist([gray_img], [0], None, [256], [0, 256])

    # 3. ratio에 해당하는 밝기값을 기준으로 threshold
    th = calc_threshold(hist, ratio)

    # 4. threshold를 기준으로 이진 영상으로 변환
    _, bin_img = cv2.threshold(gray_img, th, 255, cv2.THRESH_BINARY)
    # cv2.imshow('bin_img', bin_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    print("threshold:", th)

    # 5. white에 해당하는 픽셀에서 최대값/최솟값 계산
    x_min, y_min, x_max, y_max = calc_bounding_box(bin_img)

    # 6. 바운딩 박스 생성
    img_with_box = cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)

    # 7. 결과 이미지 저장
    cv2.imwrite(save, img_with_box)
    print("image saved: " + save + "\n")

    return x_min, y_min, x_max, y_max

if __name__ == '__main__':
    data_path = "./sample_data"
    result_path = "./result"
    data_list = os.listdir(data_path)

    for i in range(len(data_list)):
        class_path = "/" + data_list[i]
        class_list = os.listdir(data_path + class_path)

        result = []  # x_min, y_min, x_max, y_max
        bounding_box_info = ""
        text_path = result_path + class_path + "/bounding_box_list.txt"
        for j in range(len(class_list)):
            img_path = data_path + class_path + "/" + class_list[j]
            save_path = result_path + class_path + "/" + class_list[j][:-4] + "_result" + class_list[j][-4:]
            os.makedirs(result_path, exist_ok=True)
            os.makedirs(result_path + class_path, exist_ok=True)
            result = full_pipeline(img_path, save_path, ratio=0.99)
            bounding_box_info += str(result[0])+", "+str(result[1])+", "+str(result[2])+", "+str(result[3])+"\n"

        # 8. 바운딩 박스 정보 저장
        txt_file = open(text_path, 'w')
        txt_file.write(bounding_box_info)
        txt_file.close()
