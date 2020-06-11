import pydicom
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

import sys
import os
import datetime
import time

from cv2_gui import *
from tools import GuiTools
gui_tool = GuiTools()



if __name__ == '__main__':

    path = '../dicom/US00001L.DCM'
    dicom = pydicom.read_file(path)
    IMGS = gui_tool.add_page(dicom.pixel_array)


    # 呼叫 cv2 GUI class 的參數
    kwargs = {
        'imgs': IMGS,
        'window_name': 'US00001L',
        'delta_x': 0,
        'delta_y': 0,
        'temp_size': 32,
        'default_search': 10,
        'cost': 'SAD',
        'draw_delay': 20
    }
    cv2_gui = Cv2Line(**kwargs)

    while True:
        cv2.setMouseCallback(cv2_gui.window_name, cv2_gui.click_event)  # 設定滑鼠回饋事件

        action = gui_tool.find_action(cv2.waitKey(1))  # 設定鍵盤回饋事件

        # 「esc」 跳出迴圈
        if action == 'esc':
            break

        # 「r」 重置
        if action == 'reset':
            # 清除 strain curve 圖片
            cv2_gui.reset()

        # 「s」 執行 speckle tracking
        if action == 'speckle':
            t1 = time.time()
            cv2_gui.tracking(show=False)
            t2 = time.time()
            print('Speckle Tracking costs {} seconds.\n'.format(t2 - t1))


        # 「t」 增加預設點數（測試時用）
        if action == 'test':
            print('add point')
            if mode == 'line':
                cv2_gui.addPoint((224, 217), (243, 114))
                # cv2_gui.addPoint((313, 122), (374, 292))
                # cv2_gui.addPoint((318, 131), (310, 174))
            elif mode == 'point':
                cv2_gui.addPoint((224, 217))
                cv2_gui.addPoint((243, 114))
                cv2_gui.addPoint((224, 217))
                cv2_gui.addPoint((374, 292))

        # 按空白鍵查看點數狀況
        if action == 'space':
            labeled_points = ''
            for k in range(len(cv2_gui.target_point)):
                if k % 2 == 1:
                    labeled_points += f"{cv2_gui.target_point[k - 1]}, {cv2_gui.target_point[k]}\n"

            textBrowser_labeled_points.setText(labeled_points)

            print('target_point : ', cv2_gui.target_point)
            print('track_done : ', cv2_gui.track_done)
            print('search_point : ', cv2_gui.search_point)  # 目前沒用
            print('search_shift : ', cv2_gui.search_shift)
            print('result_points: ', cv2_gui.result_point)
            print()

    cv2.destroyWindow(cv2_gui.window_name)  # （按 esc 跳出迴圈後）關閉視窗






