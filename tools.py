import numpy as np
import cv2

from scipy.interpolate import make_lsq_spline
import matplotlib.pyplot as plt
from matplotlib import animation

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog, QMainWindow


class GuiTools():
    ACTION = {27: 'esc', 67: 'clear', 99: 'clear', 82: 'reset', 114: 'reset',
              26: 'last atcion', 66: 'back', 98: 'back', 84: 'test', 116: 'test',
              76: 'line', 108: 'line', 83: 'speckle', 115: 'speckle',
              32: 'space', 77: 'median filter', 109: 'median filter'}

    # 將 Dcm 檔影像加上圖片編號
    def add_page(self, imgs: np, start: int = 0) -> np:
        for i in range(len(imgs)):
            s = f"{i + start}/{len(imgs) - 1 + start}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(imgs[i], s, (0, 15), font, .5, (255, 255, 255), 2)
        return imgs

    # 將 Dcm 檔影像加上圖片編號
    def add_page_single(self, img: np, num: int, total: int) -> np:
        s = f"{num}/{total}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, s, (0, 15), font, .5, (255, 255, 255), 2)
        return img

    # 查詢按鍵指令
    def find_action(self, key: int) -> str:
        # if key != -1:print(key)
        try:
            return self.ACTION[key]
        except:
            return None

    # 將cv2轉為 QImg
    def convert2qtimg(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, bytesPerComponent = img.shape
        bytesPerLine = 3 * width
        QImg = QtGui.QImage(img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        return QImg

    def lsq_spline_medain(self, y: list, k: int = 3) -> list:
        '''
        只取中間開始變化處的 lsq spline
        :param y: 原始資料
        :param k: 近似為 P(k) 多項式， k = 3 即為 cube spline
        :return: 近似資料
        '''
        x = np.asarray([i for i in range(len(y))])
        y = np.asarray(y)
        max = np.max(y)
        min = np.min(y)
        stage = (max - min) / 10

        try:
            start = np.where(np.abs(y - y[0]) > stage)[0][0]
            end = np.where(np.abs(y - y[-1]) > stage)[0][-1]

            start_split = y[:start]
            end_split = y[end + 1:]
            median_split = y[start:end + 1]

            l = len(median_split)
            x_split = np.asarray([i for i in range(l)])
            t = np.linspace(0, l, num=5, dtype='int').reshape(-1)
            t = np.r_[(0,) * (k + 1), t[1:-1], (l - 1,) * (k + 1)]

            spl = make_lsq_spline(x_split, median_split, t, k)

            output = np.hstack((start_split, spl(x_split), end_split))
        except:
            output = y

        return output

    def moving_average(self, y: list, r: int):
        w = len(y)
        result = np.zeros(w)

        y = y[1:1 + r][::-1] + y + y[-2 - r:-2][::-1]
        y = np.array(y).reshape(-1)

        kernel = np.ones(2 * r + 1) / (2 * r + 1)

        for i in range(w):
            result[i] = np.sum(y[i:i + 2 * r + 1] * kernel)

        return result

    def diff(self, y: list):

        result = np.zeros(len(y))
        y = y[0] + y

        for i in range(1, len(y)):
            result[i - 1] = y[i] - y[i - 1]

        return result

    # 找最長水平線的中間值
    def find_best_frame(self, temps, window=5, thre=0.005):

        size = len(temps)
        temp = np.array(self.diff(self.moving_average(temps, window)))
        temp = np.argwhere(np.abs(temp) < thre).reshape(-1)
        temp = temp[temp > size * 0.25]  # 頭 20% 不計算
        temp = temp[temp < size * 0.75]  # 尾 20% 不計算

        g = 0
        long_max = 0
        tar_frame = 0
        while g < len(temp) - 1:
            long = 1
            g_end = g_start = temp[g]
            while g < len(temp) - 1 and temp[g + 1] - temp[g] == 1:
                g_end = temp[g]
                long += 1
                g += 1

            if long > long_max:
                long_max = long
                tar_frame = (g_start + g_end) // 2

            g += 1

        return tar_frame


class Cv2Tools():

    # 影像切換
    def photo_switch(self, switch: str, page: int, max: int) -> int:
        if switch == 'last':
            page = 0 if page == 0 else page - 1

        elif switch == 'next':
            page = max - 1 if max == max - 1 else page + 1

        return page

    # 換算距離
    def count_distance(self, point1: tuple, point2: tuple, delta: np) -> (tuple, tuple):
        '''
        計算兩點距離、標示點的位置
        :param point1: 任一一點
        :param point2: 任一一點
        :return: textPoint（點的位置）, d（距離）
        '''
        p1 = np.array(point1)
        p2 = np.array(point2)
        textPoint = tuple(((p1 + p2) / 2).astype('int'))
        d = np.sqrt(np.sum(np.power((p1 - p2) * delta, 2)))
        return textPoint, d

    # 轉換 rectangle 的點為左上右下
    def point_converter(self, point1: tuple, point2: tuple) -> (tuple, tuple):
        '''
        將矩陣點轉換為 左上、右下
        :param point1:
        :param point2:
        :return: p1（小點）, p2（大點）
        '''
        p1 = (min(point1[0], point2[0]), min(point1[1], point2[1]))
        p2 = (max(point1[0], point2[0]), max(point1[1], point2[1]))
        return p1, p2

    # 輸入中心點、右下角點，回傳校正後的左上、右下點
    def get_search_window(self, center: tuple, corner: tuple, template: int) -> (tuple, tuple, tuple, tuple):
        '''
        算出 searchwindow 範圍
        :param center:
        :param corner:
        :return:
        '''
        # ROI part
        startPoint = tuple(2 * np.asarray(center) - np.asarray(corner))
        s1, s2 = self.point_converter(startPoint, corner)

        c1 = (s1[0] - template // 2, s1[1] - template // 2)
        c2 = (s2[0] + template // 2, s2[1] + template // 2)

        return s1, s2, c1, c2

    def color_iterater(self, x, saturation, lightness):
        colors = np.linspace(0, 180, x + 1)
        img = np.zeros((1, 1, 3), dtype='uint8')
        result = []

        for i in range(x):
            img[:, :, ] = [colors[i], saturation, lightness]
            img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

            result.append(tuple([int(i) for i in img[0, 0,]]))

        return result

    def local_histogram_equalization(self, img: np, r: int) -> np:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = img.shape
        output = np.zeros((h, w), dtype='uint8')
        img = np.pad(img, ((r, r), (r, r)), 'edge')  # add padding

        for i in range(h):
            for j in range(w):
                temp = img[i: i + 2 * r + 1, j: j + 2 * r + 1]
                output[i, j] = cv2.equalizeHist(temp)[r, r]

        return output.astype('uint8')
