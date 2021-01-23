import pydicom
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from matplotlib import animation
import json

import sys
import os
import datetime
import time

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QMessageBox, QInputDialog
from main_window import Ui_MainWindow

from cv2_gui import *
from tools import GuiTools

gui_tool = GuiTools()


def get_N_sided(N, p1, p2):
	x1, y1 = p1
	x2, y2 = p2

	dx = x2 - x1
	dy = y1 - y2
	r = (dx ** 2 + dy ** 2) ** 0.5
	theta = np.arctan2(dy, dx)
	d_theta = 2*np.pi/N

	result = []
	for i in range(N):
		new_theta = theta + d_theta * i
		x = x1 + np.cos(new_theta) * r
		y = y1 - np.sin(new_theta) * r
		result.append([int(x), int(y)])

	return np.array(result)

def my_draw_contour(img, points, colors, colors_max, bold):
    new_point = points + [points[0]]
    for i in range(len(points)):
        cv2.line(img, new_point[i], new_point[i+1], colors[i%colors_max], bold)

    return img

def run_cv2(mw):
    if not mw.filename:
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle('Warning')
        msg.setText('Please select the dicom or the image file.\n')
        msg.setIcon(QtWidgets.QMessageBox.Warning)

        msg.exec_()
        return

    cv2.destroyAllWindows()
    mw.btn_add_line.show()

    # 清除 strain curve 圖片
    mw.label_show_curve.setPixmap(QtGui.QPixmap(""))

    # 紀錄設定的開始與結束頁
    mw.start = mw.spinBox_start.value()
    mw.end = mw.spinBox_end.value() + 1

    # 多邊形邊樹
    edge = 5

    # 呼叫 cv2 GUI class 的參數
    kwargs = {
        'main_window': mw,
        'imgs': mw.IMGS[mw.start:mw.end:, :, :],
        'window_name': mw.filename,
        'delta_x': float(mw.doubleSpinBox_delta_x.value()) / 1000,
        'delta_y': float(mw.doubleSpinBox_delta_y.value()) / 1000,
        'temp_size': int(mw.spinBox_temp_size.value()),
        'default_search': int(mw.spinBox_search_range.value()),
        'method': mw.json_para['method'],
        'draw_delay': int(mw.spinBox_drawing_delay.value()),
        'json_para': mw.json_para,
        'edge': edge,
    }

    # 設定模式
    if mw.radioButton_line.isChecked():
        mw.mode = 'ishan'
        mw.cv2_gui = Cv2Line_ishan(**kwargs)

        #################### 額外控制功能的動作 ####################
        # ADD point 格式：
        # (288, 114), (266, 194)
        # (326, 123), (329, 184)
        # (342, 105), (368, 179)
        add_points = mw.textBrowser_auto_add_point.toPlainText()
        mw.textBrowser_labeled_points.setText(add_points)
        if add_points != '':
            try:
                target_point = []
                add_points = add_points.replace('(', '').replace(')', '').replace(' ', '').replace('\n', '').split(
                    ',')
                for i in range(0, len(add_points), 4):
                    # for point in add_points:
                    x1, y1, x2, y2 = add_points[i], add_points[i + 1], add_points[i + 2], add_points[i + 3]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    if mw.scaling != 100:
                        x1 = (x1 * mw.scaling) // 100
                        y1 = (y1 * mw.scaling) // 100
                        x2 = (x2 * mw.scaling) // 100
                        y2 = (y2 * mw.scaling) // 100

                    target_point.append((x1, x2))

                    # 如果下次的四個點無法算完
                    if i + 8 > len(add_points): break

                mw.cv2_gui.addPoint_ishan(target_point)

            except Exception as e:
                print(e)
                print("###########################\n"
                      "# 輸入點的格式錯誤，應為：\n"
                      "# (288, 114), (266, 194),\n"
                      "# (326, 123), (329, 184),\n"
                      "# (342, 105), (368, 179),\n"
                      "###########################")

    mw.cv2_gui = Cv2Line_ishan(**kwargs)
    mw.use_json('write')

    ###################### 主程式運行 ######################
    cv2.setMouseCallback(mw.cv2_gui.window_name, mw.cv2_gui.click_event)  # 設定滑鼠回饋事件
    while True:
        action = gui_tool.find_action(cv2.waitKey(1))  # 設定鍵盤回饋事件

        # 「esc」 跳出迴圈
        if action == 'esc':
            break

        # 「r」 重置
        if action == 'reset':
            # 清除 strain curve 圖片
            mw.textBrowser_labeled_points.clear()
            mw.textBrowser_auto_add_point.clear()
            mw.textBrowser_target_frame.clear()
            mw.label_show_curve.setPixmap(QtGui.QPixmap(""))
            mw.cv2_gui.reset()

        # 「s」 執行 speckle tracking
        if action == 'speckle':
            t1 = time.time()
            mw.cv2_gui.tracking(show=True if mw.checkBox_Animation.isChecked() else False)
            t2 = time.time()
            print('Speckle Tracking costs {} seconds.\n'.format(t2 - t1))
            mw.plot_strain_curve(0)

            # 自動存檔？
            if mw.checkBox_auto_save.isChecked(): mw.auto_save_files()


        # 「t」 增加預設點數（測試時用）
        if action == 'test':
            cv2.imshow('LHE', cv2_tool.local_histogram_equalization(mw.IMGS[0], 35))
            cv2.setMouseCallback('LHE', mw.cv2_gui.click_event)  # 設定滑鼠回饋事件
            cv2.waitKey(1)

        # print(f"mw.cv2_gui.result_distance :\n{mw.cv2_gui.result_distance}")

        # 按空白鍵查看點數狀況
        if action == 'space':
            labeled_points = ''

            print('mw.target_point :\n', mw.cv2_gui.target_point)
            print('\nself.track_done :\n', mw.cv2_gui.track_done)
            print('\nself.search_shift :\n', mw.cv2_gui.search_shift)
            print('\nself.result_points:\n', mw.cv2_gui.result_point)
            print('\nself.result_strain:\n', mw.cv2_gui.result_strain)
            print('\nself.result_distance:\n', mw.cv2_gui.result_distance)
            print()

    cv2.destroyAllWindows()  # （按 esc 跳出迴圈後）關閉視窗




class Cv2Line_ishan():

    def __init__(self, main_window, imgs: np, delta_x: float, delta_y: float, window_name: str,
                 temp_size: int, default_search: int, method: str, draw_delay: int, json_para: dict, edge: int):

        self.mw = main_window
        self.json_para = json_para

        self.IMGS = imgs
        self.window_name = window_name

        self.current_page = 0
        self.default_search = default_search
        self.temp_size = temp_size

        self.delta_x = delta_x
        self.delta_y = delta_y
        if self.delta_x == 0 or self.delta_y == 0:
            self.delta = np.array([1, 1])
        else:
            self.delta = np.array([self.delta_x, self.delta_y])

        print("The shape of dicom is :", self.IMGS.shape)

        self.IMGS_GRAY = np.asarray([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in self.IMGS])

        self.img_label = np.copy(self.IMGS)
        self.num_of_img, self.h, self.w = self.IMGS.shape[:3]
        self.num_of_line = 0

        # 畫圖顏色
        self.color_index = 0
        self.num_of_color = self.json_para['line']['color']['amount']
        self.colors = cv2_tool.color_iterater(x=self.num_of_color,
                                              saturation=self.json_para['line']['color']['saturation'],
                                              lightness=self.json_para['line']['color']['lightness'])
        self.current_color = self.colors[self.color_index % self.num_of_color]
        self.edge = edge

        # 讀取 json 中 font 與 line 的參數
        self.font_show = self.json_para['font']['show']
        self.font_size = self.json_para['font']['size']
        self.font_bold = self.json_para['font']['bold']
        self.line_bold = self.json_para['line']['bold']

        # 點相關參數
        self.target_point = []  # -> tuple
        self.track_done = []
        self.search_point = []  # -> list -> tuple
        self.search_shift = []
        self.result_point = {}
        self.result_distance = {}
        self.result_dx = {}
        self.result_dy = {}
        self.result_strain = {}

        # 顯示
        # cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.mw.w, self.mw.h)
        cv2.createTrackbar('No', self.window_name, 0, self.num_of_img - 1, self.track_change)
        cv2.imshow(self.window_name, self.img_label[self.current_page])
        cv2.waitKey(1)

        self.speckle_tracking = SpeckleTracking(method=method)

    # 重置所有動作
    def reset(self):
        self.img_label = np.copy(self.IMGS)
        cv2.imshow(self.window_name, self.img_label[self.current_page])

        self.num_of_line = 0
        self.color_index = 0
        self.current_color = self.colors[self.color_index % self.num_of_color]
        self.target_point = []
        self.track_done = []
        self.search_point = []
        self.search_shift = []
        self.result_point = {}
        self.result_distance = {}
        self.result_dx = {}
        self.result_dy = {}
        self.result_strain = {}

        print('Reseting complete.')

    # track bar 更動
    def track_change(self, x: int):
        '''
        Track bar 變動時的呼叫函數
        :param x: 變動後的值
        :return: None
        '''
        self.current_page = x
        cv2.imshow(self.window_name, self.img_label[self.current_page])

    # 滑鼠事件
    def click_event(self, event, x, y, flags, param):

        # 滾輪選擇照片
        if event == cv2.EVENT_MOUSEWHEEL:
            if flags < 0:
                self.current_page = cv2_tool.photo_switch('next', self.current_page, self.num_of_img)
            elif flags > 0:
                self.current_page = cv2_tool.photo_switch('last', self.current_page, self.num_of_img)

            # 更新 Trackbar，__track_change會更新圖片
            cv2.setTrackbarPos('No', self.window_name, self.current_page)

        # 劃出線段（左鍵點擊時）
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_drag = False
            point = (x, y)  # 記錄起點

            self.target_point.append(point)
            self.track_done.append(False)
            self.search_shift.append((self.default_search // 2, self.default_search // 2))

            img = my_draw_contour(np.copy(self.IMGS[self.current_page]), self.target_point, self.colors, self.num_of_color, self.line_bold)
            self.img_label[self.current_page] = img
            cv2.imshow(self.window_name, self.img_label[self.current_page])
            cv2.waitKey(1)

            points_show = ""
            for i in range(len(self.target_point)):
                j = i+1 if i+1 != len(self.target_point) else 0
                points_show += f"{self.target_point[i]}, {self.target_point[j]},\n"

            self.mw.textBrowser_labeled_points.setText(points_show)
            self.num_of_line += 1



        # 預覽線段（左鍵拖曳時）
        elif flags == 1 & cv2.EVENT_FLAG_LBUTTON:
            self.mouse_drag = True
            pass

        # 確定線段（左鍵放開時）
        elif event == cv2.EVENT_LBUTTONUP:
            if self.mouse_drag:
                self.mouse_drag = False  # 拖曳重置
                pass

        # 設定 Search Window（右鍵點擊時）
        if event == cv2.EVENT_RBUTTONDOWN:
            self.mouse_drag = False
            pass

        # 畫 Search Window 範圍（右鍵拖曳時）
        elif flags == 2 & cv2.EVENT_FLAG_RBUTTON:
            self.mouse_drag = True
            pass

        # 確定 Search Window 範圍（右鍵放開時）
        elif event == cv2.EVENT_RBUTTONUP:
            if self.mouse_drag:
                self.mouse_drag = False  # 拖曳重置
                pass

    # TODO: 測試時方便建立線段
    def addPoint_ishan(self, target_points):
        self.target_point = target_points
        self.track_done = [False] * len(target_points)
        self.search_shift = [(self.default_search // 2, self.default_search // 2)] * len(target_points)

        img = my_draw_contour(np.copy(self.IMGS[self.current_page]), self.target_point, self.colors, self.num_of_color,
                              self.line_bold)
        self.img_label[self.current_page] = img
        cv2.imshow(self.window_name, self.img_label[self.current_page])

        points_show = ""
        for i in range(len(self.target_point)):
            j = i + 1 if i + 1 != len(self.target_point) else 0
            points_show += f"{self.target_point[i]}, {self.target_point[j]},\n"

        self.mw.textBrowser_labeled_points.setText(points_show)
        self.num_of_line += 1


    # 畫線的 Speckle Tracking
    def tracking(self, show=False):
        finish_already = True

        for i in range(self.num_of_line):
        #     point1 = self.target_point[i]
        #     point2 = self.target_point[i+1] if i+1 != self.num_of_line else self.target_point[0]
        #     text_point, d, dx, dy = cv2_tool.count_distance(point1, point2, self.delta)
            self.result_distance[i] = []

        progress_denominator = (len(self.track_done) - np.sum(np.asarray(self.track_done))) * (len(self.IMGS) - 1)
        progress_fraction = 0
        for j, (tp, s_shift, done) in enumerate(zip(self.target_point, self.search_shift, self.track_done)):

            # 如果該點完成，跳過該點
            if done: continue

            finish_already = False
            self.track_done[j] = True
            self.result_point[j] = [tp]

            # color = self.colors[(j // 2) % self.num_of_color]

            print('Now is tracking point{}/{}.'.format(j + 1, len(self.target_point)))

            result = tp

            # 從圖1開始抓出，當作 Candidate
            for i in range(1, self.num_of_img):
                progress_fraction += 1
                # target, img1, img2, search_shift, temp_size
                result = self.speckle_tracking.method(result, self.IMGS_GRAY[i - 1], self.IMGS_GRAY[i], s_shift,
                                                      self.temp_size)

                self.result_point[j].append(result)

                cv2.circle(self.img_label[i], result, 2, (0, 0, 255), thickness=-1)

                if show:
                    self.show_progress_bar(self.img_label[i], progress_fraction, progress_denominator)

            self.show_progress_bar(np.copy(self.img_label[0]), progress_fraction, progress_denominator, pos="top")

        for i in range(self.num_of_img):
            points = []
            for j in range(self.num_of_line):
                points.append(self.result_point[j][i])

                point1 = self.result_point[j][i]
                point2 = self.result_point[j+1][i] if j+1 != self.num_of_line else self.result_point[0][i]
                text_point, d, dx, dy = cv2_tool.count_distance(point1, point2, self.delta)
                self.result_distance[j].append(d)

            self.img_label[i] = my_draw_contour(np.copy(self.IMGS[i]), points, self.colors, self.num_of_color, self.line_bold)

        cv2.imshow(self.window_name, self.img_label[0])
        cv2.waitKey(1)

        for i in self.result_distance.keys():
            d_list = np.asarray(self.result_distance[i])
            self.result_strain[i] = list((d_list - d_list[0]) / d_list[0])

    def show_progress_bar(self, img, fraction, denominator, pos='down'):
        if pos == 'down':
            temp_img = cv2.line(np.copy(img), (0, self.h - 1), (((self.w - 1) * fraction) // denominator, self.h - 1),
                            (216, 202, 28), 5)
        elif pos == 'top':
            temp_img = cv2.line(np.copy(img), (0, 0), (((self.w - 1) * fraction) // denominator, 0),
                                (216, 202, 28), 5)
        cv2.imshow(self.window_name, temp_img)
        cv2.waitKey(1)
