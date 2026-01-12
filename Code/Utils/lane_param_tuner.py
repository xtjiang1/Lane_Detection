import cv2
import numpy as np


def adjust_parameters(image):
    # 创建窗口和滑动条
    cv2.namedWindow('Adjust Parameters')
    cv2.createTrackbar('L_min', 'Adjust Parameters', 120, 255, lambda x: None)
    cv2.createTrackbar('L_max', 'Adjust Parameters', 255, 255, lambda x: None)
    cv2.createTrackbar('S_min', 'Adjust Parameters', 20, 255, lambda x: None)
    cv2.createTrackbar('S_max', 'Adjust Parameters', 255, 255, lambda x: None)

    while True:
        # 获取滑动条值
        l_min = cv2.getTrackbarPos('L_min', 'Adjust Parameters')
        l_max = cv2.getTrackbarPos('L_max', 'Adjust Parameters')
        s_min = cv2.getTrackbarPos('S_min', 'Adjust Parameters')
        s_max = cv2.getTrackbarPos('S_max', 'Adjust Parameters')

        # 转换到HLS并应用阈值
        hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        l_mask = cv2.inRange(l_channel, l_min, l_max)
        s_mask = cv2.inRange(s_channel, s_min, s_max)

        # 组合掩码
        combined_mask = cv2.bitwise_and(l_mask, s_mask)

        # 显示结果
        cv2.imshow('Original', image)
        cv2.imshow('Result', combined_mask)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    return l_min, l_max, s_min, s_max


# 使用方法
image = cv2.imread('your_image.png')
l_min, l_max, s_min, s_max = adjust_parameters(image)
print(f"Use these values: l_min={l_min}, l_max={l_max}, s_min={s_min}, s_max={s_max}")