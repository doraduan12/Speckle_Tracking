import numpy as np
import cv2

from scipy.interpolate import make_lsq_spline, BSpline
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import animation


class Tools():


    ACTION = {27: 'esc', 67: 'clear', 99: 'clear', 82: 'reset', 114: 'reset',
              26: 'last atcion', 66: 'back', 98: 'back', 84: 'test', 116: 'test',
              76: 'line', 108: 'line', 83: 'speckle', 115: 'speckle',
              32: 'space', 77: 'median filter', 109: 'median filter'}


    # 查詢按鍵指令
    def find_ACTION(self, key: int) -> str:
        # if key != -1:print(key)
        try:
            return self.ACTION[key]
        except:
            return None

    # 將 Dcm 檔影像加上圖片編號
    def addPage(self, imgs: np) -> np:
        for i in range(len(imgs)):
            s = str(i) + '/' + str(len(imgs) - 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(imgs[i], s, (0, 15), font, .5, (255, 255, 255), 2)
        return imgs

    # 影像切換
    def photo_switch(self, switch:str, page:int, max:int) -> int:
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

        c1 = (s1[0] - template//2, s1[1] - template//2)
        c2 = (s2[0] + template//2, s2[1] + template//2)

        return s1, s2, c1, c2


