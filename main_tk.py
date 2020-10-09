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


# TODO
def load_file(mw):
    files, filetype = QFileDialog.getOpenFileNames(mw, "選擇文件", mw.default_path,  # 起始路径
                                                       "All Files (*);;Dicom Files (*.dcm);;Png Files (*.png);;JPEG Files (*.jpeg)")
    # 如果沒讀到檔案
    if len(files) == 0: return

    files = list(filter(lambda x: 'png' in x or 'jpg' in x or 'jpeg' in x, files))
    mw.files = files

    # 更新預設路徑
    mw.default_path = os.path.split(files[0])[0]
    mw.json_para['path'] = mw.default_path
    mw.use_json('write')

    # 副檔名
    mw.extension = os.path.splitext(files[0])[-1].lower()

    # 如果讀到圖檔
    if mw.extension == '.png' or mw.extension == '.jpg' or mw.extension == '.jpeg':

        browse_path = os.path.split(files[0])[0]
        mw.filename = os.path.split(browse_path)[-1]

        # 輸出影向預設的路徑與檔案名稱
        mw.default_path = os.path.split(browse_path)[0]
        mw.default_filename = mw.filename

        # 排序圖檔
        files = np.asarray(files)
        temp = np.asarray([int(file.split('.')[0].split('/')[-1]) for file in files])
        temp = np.argsort(temp)
        mw.files = files[temp]

        mw.IMGS = gui_tool.add_page(np.array([cv2.imdecode(np.fromfile(file, dtype=np.uint8), -1) for file in files[:50]]))    # 預覽 50 張

        mw.img_preview = mw.IMGS[0]
        mw.num_of_img, mw.h, mw.w = mw.IMGS.shape[:3]

        filetype = '.' + files[0].split('.')[-1]

        mw.date, mw.time, system_name = '', '', ''

        deltax, deltay = mw.json_para['delta_x'], mw.json_para['delta_y']


    # 如果讀入檔案不是 圖檔
    else:
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle('Warning')
        msg.setText('Please select the dicom or the image file.\n')
        msg.setIcon(QtWidgets.QMessageBox.Warning)

        msg.exec_()
        return

    # 寫入 檔案路徑
    mw.textBrowser_browse.setText(browse_path)

    # 顯示超音波廠商、字體白色、根據字數調整 label size
    mw.label_manufacturer.setText(system_name)
    mw.label_manufacturer.setStyleSheet("color:white")
    mw.label_manufacturer.adjustSize()

    # 寫入 file detail 內容
    mw.label_filetype_show.setText(filetype)
    mw.label_image_size_show.setText(str(mw.w) + ' x ' + str(mw.h))
    mw.label_date_show.setText(str(mw.date))
    mw.label_time_show.setText(str(mw.time))
    mw.label_frame_show.setText(str(len(mw.files)))

    # 更新 josn 參數
    mw.json_para['delta_x'] = deltax
    mw.json_para['delta_y'] = deltay
    mw.doubleSpinBox_delta_x.setValue(deltax)
    mw.doubleSpinBox_delta_y.setValue(deltay)

    # horizontalSlider_preview 設定最大值、歸零
    mw.horizontalSlider_preview.setMaximum(mw.num_of_img-1)
    mw.horizontalSlider_preview.setValue(0)

    # 預設的 template block 與 search window
    mw.spinBox_temp_size.setValue(mw.json_para['template_size'])
    mw.spinBox_temp_size.setRange(1, mw.h // 2)
    mw.spinBox_search_range.setValue(mw.json_para['search_size'])
    mw.spinBox_search_range.setRange(1, mw.h // 2)

    # 預設的 draw delay
    mw.default_draw_delay = mw.json_para['draw_delay']
    mw.spinBox_drawing_delay.setValue(mw.default_draw_delay)
    mw.spinBox_drawing_delay.setRange(1, 100)

    # 建立預覽圖片、自適化調整
    mw.show_preview_img(np.copy(mw.img_preview), mw.json_para['template_size'],
                        mw.json_para['search_size'])

    # 預設上下限與初始頁數
    mw.spinBox_start.setRange(0, mw.num_of_img - 1)
    mw.spinBox_start.setValue(0)
    mw.spinBox_end.setRange(0, mw.num_of_img - 1)
    mw.spinBox_end.setValue(mw.num_of_img - 1)

    #################### 額外控制功能的動作 ####################
    mw.spinBox_target_frame.setRange(0, mw.num_of_img - 1)

    # 清空 points
    mw.textBrowser_labeled_points.setText('')
    mw.textBrowser_auto_add_point.setText('')

    with open('saved_points.json', 'r') as f:
        saved_points = json.loads(f.read())
        name = os.path.split(mw.json_para['path'])[-1] + '_' + str(
            mw.date) + '_' + mw.filename + mw.extension
        if name in saved_points.keys():
            mw.textBrowser_auto_add_point.setText(saved_points[name])


def run_cv2(mw, multi_mode=False):
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

    # 呼叫 cv2 GUI class 的參數
    kwargs = {
        'mw': mw,
        'imgs': mw.IMGS[mw.start:mw.end:, :, :],
        'window_name': mw.filename,
        'delta_x': float(mw.doubleSpinBox_delta_x.value()) / 1000,
        'delta_y': float(mw.doubleSpinBox_delta_y.value()) / 1000,
        'temp_size': int(mw.spinBox_temp_size.value()),
        'default_search': int(mw.spinBox_search_range.value()),
        'method': mw.json_para['method'],
        'draw_delay': int(mw.spinBox_drawing_delay.value()),
        'json_para': mw.json_para
    }

    # 設定模式
    # if mw.radioButton_line.isChecked():
    if True:
        mw.mode = 'line'
        mw.cv2_gui = Cv2Line_tk(**kwargs)

        #################### 額外控制功能的動作 ####################
        # ADD point 格式：
        # (288, 114), (266, 194)
        # (326, 123), (329, 184)
        # (342, 105), (368, 179)
        add_points = mw.textBrowser_auto_add_point.toPlainText()
        mw.textBrowser_labeled_points.setText(add_points)
        if add_points != '':
            try:
                add_points = add_points.replace('(', '').replace(')', '').replace(' ', '').replace('\n', '').split(
                    ',')
                for i in range(0, len(add_points), 4):
                    # for point in add_points:
                    x1, y1, x2, y2 = add_points[i], add_points[i + 1], add_points[i + 2], add_points[i + 3]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    mw.cv2_gui.addPoint((x1, y1), (x2, y2))
                    # 如果下次的四個點無法算完
                    if i + 8 > len(add_points): break

            except Exception as e:
                print("###########################\n"
                      "# 輸入點的格式錯誤，應為：\n"
                      "# (288, 114), (266, 194),\n"
                      "# (326, 123), (329, 184),\n"
                      "# (342, 105), (368, 179),\n"
                      "###########################")

    # elif mw.radioButton_draw.isChecked():
    #     mw.mode = 'point'
    #     mw.cv2_gui = Cv2Point(**kwargs)

    mw.use_json('write')

    ###################### 主程式運行 ######################
    while True:
        cv2.setMouseCallback(mw.cv2_gui.window_name, mw.cv2_gui.click_event)  # 設定滑鼠回饋事件

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
        if action == 'speckle' or multi_mode:
            t1 = time.time()
            mw.cv2_gui.tracking(show=True if mw.checkBox_Animation.isChecked() else False)
            t2 = time.time()
            print('Speckle Tracking costs {} seconds.\n'.format(t2 - t1))

            # Line 模式畫圖
            if mw.mode == 'line':

                # 計算位置
                target_frame = gui_tool.find_best_frame(
                    mw.cv2_gui.result_distance[1] if len(mw.cv2_gui.result_distance) > 1 else
                    mw.cv2_gui.result_distance[0])

                # 畫 curve，如果是郁文用的就自動偵測
                if mw.action_user_yuwen.isChecked():
                    mw.spinBox_target_frame.setValue(target_frame)
                    mw.plot_strain_curve(target_frame)
                else:
                    mw.plot_strain_curve(0)

                # 自動存檔？
                if mw.checkBox_auto_save.isChecked(): mw.auto_save_files()

                mw.console_nb += "\t".join(['{:.3f}'.format(mw.cv2_gui.result_distance[k][0]) for k in
                                              mw.cv2_gui.result_distance.keys()][::-1]) + '\n'
                mw.console_after += "\t".join(
                    ['{:.3f}'.format(mw.cv2_gui.result_distance[k][target_frame]) for k in
                     mw.cv2_gui.result_distance.keys()][::-1]) + '\n'
                mw.console_text = 'NB:\n' + mw.console_nb + '\nAfter:\n' + mw.console_after + '\n'

                mw.console_text += 'Result Points:\n'
                for k in range(len(mw.cv2_gui.result_point)):
                    if k % 2 == 1:
                        mw.console_text += f"{mw.cv2_gui.result_point[k - 1][target_frame]}, {mw.cv2_gui.result_point[k][target_frame]}\n"

                mw.textBrowser_target_frame.setText(mw.console_text)

                if multi_mode: break

            # Point 模式不動作
            # elif mw.mode == 'point':
            #     pass

        # 「t」 增加預設點數（測試時用）
        # if action == 'test':
        #     cv2.imshow('LHE', cv2_tool.local_histogram_equalization(mw.IMGS[0], 35))
        #     cv2.setMouseCallback('LHE', mw.cv2_gui.click_event)  # 設定滑鼠回饋事件
        #     cv2.waitKey(1)

        # print(f"mw.cv2_gui.result_distance :\n{mw.cv2_gui.result_distance}")

        # 按空白鍵查看點數狀況
        if action == 'space':
            labeled_points = ''

            print('mw.target_point :\n', mw.cv2_gui.target_point)
            print('\nself.track_done :\n', mw.cv2_gui.track_done)
            print('\nself.search_shift :\n', mw.cv2_gui.search_shift)
            print('\nself.result_points:\n', mw.cv2_gui.result_point)
            print('\nself.result_strain:\n', mw.cv2_gui.result_strain)
            print()

    cv2.destroyAllWindows()  # （按 esc 跳出迴圈後）關閉視窗




class Cv2Line_tk():
    def __init__(self, mw, imgs: np, delta_x: float, delta_y: float, window_name: str,
                 temp_size: int, default_search: int, method: str, draw_delay: int, json_para: dict):

        self.mw = mw
        self.json_para = json_para

        self.IMGS = imgs
        self.img_label = np.copy(self.IMGS)
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

        self.num_of_img, self.h, self.w = self.IMGS.shape[:3]

        # 畫圖顏色
        self.color_index = 0
        self.num_of_color = self.json_para['line']['color']['amount']
        self.colors = cv2_tool.color_iterater(x=self.num_of_color,
                                              saturation=self.json_para['line']['color']['saturation'],
                                              lightness=self.json_para['line']['color']['lightness'])
        self.current_color = self.colors[self.color_index % self.num_of_color]

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
        self.result_strain = {}

        # 顯示
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar('No', self.window_name, 0, self.num_of_img - 1, self.track_change)
        cv2.imshow(self.window_name, self.img_label[self.current_page])
        cv2.waitKey(1)

        self.speckle_tracking = SpeckleTracking(method=method)

    # 重置所有動作
    def reset(self):
        self.img_label = np.copy(self.IMGS)
        cv2.imshow(self.window_name, self.img_label[self.current_page])

        self.color_index = 0
        self.current_color = self.colors[self.color_index % self.num_of_color]
        self.target_point = []
        self.track_done = []
        self.search_point = []
        self.search_shift = []
        self.result_point = {}
        self.result_distance = {}
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
            self.point1 = (x, y)  # 記錄起點

        # 預覽線段（左鍵拖曳時）
        elif flags == 1 & cv2.EVENT_FLAG_LBUTTON:
            self.mouse_drag = True

            # 複製目前畫面，在放開滑鼠之前都在複製畫面上作圖，否則會有許多線段互相覆蓋
            temp_img = np.copy(self.img_label[self.current_page])
            # print(self.current_color)
            cv2.line(temp_img, self.point1, (x, y), self.current_color, thickness=self.line_bold)

            # 計算距離、顯示距離的座標
            text_point, d = cv2_tool.count_distance(self.point1, (x, y), self.delta)
            font = cv2.FONT_HERSHEY_SIMPLEX
            if self.font_show:
                cv2.putText(temp_img, '{:4.3f}{}'.format(d, '(p)' if self.delta_x == 0 else ''), text_point, font,
                            self.font_size, (255, 255, 255), self.font_bold)

            # 刷新畫面
            cv2.imshow(self.window_name, temp_img)

        # 確定線段（左鍵放開時）
        elif event == cv2.EVENT_LBUTTONUP:
            if self.mouse_drag:
                self.mouse_drag = False  # 拖曳重置

                # 紀錄 point2 的點
                self.point2 = (x, y)

                # 作圖
                cv2.line(self.img_label[self.current_page], self.point1, self.point2, self.current_color,
                         thickness=self.line_bold)
                cv2.circle(self.img_label[self.current_page], self.point1, 0, self.current_color, thickness=2)
                cv2.circle(self.img_label[self.current_page], self.point2, 0, self.current_color, thickness=2)

                # 計算距離 -> 尚未加入 List
                text_point, d = cv2_tool.count_distance(self.point1, self.point2, self.delta)
                font = cv2.FONT_HERSHEY_SIMPLEX
                if self.font_show:
                    cv2.putText(self.img_label[self.current_page],
                                '{:4.3f}{}'.format(d, '(p)' if self.delta_x == 0 else ''),
                                text_point, font, self.font_size, (255, 255, 255), self.font_bold)

                # 新增點參數
                self.target_point.extend([self.point1, self.point2])
                self.track_done.extend([False, False])

                # 計算預設的 search window
                x, y = self.point1
                s11, s12, _, _ = cv2_tool.get_search_window((x, y), (
                x + self.default_search // 2, y + self.default_search // 2), self.temp_size)
                x, y = self.point2
                s21, s22, _, _ = cv2_tool.get_search_window((x, y), (
                x + self.default_search // 2, y + self.default_search // 2), self.temp_size)

                self.search_point.extend([[s11, s12], [s21, s22]])
                self.search_shift.extend([(self.default_search // 2, self.default_search // 2),
                                          (self.default_search // 2, self.default_search // 2)])

                print(f"{self.point1}, {self.point2}")

                cv2.imshow(self.window_name, self.img_label[self.current_page])

                # 先將第一點的距離輸入結果
                self.result_distance[self.color_index] = [d]

                # 更新顏色
                self.color_index += 1
                self.current_color = self.colors[self.color_index % self.num_of_color]

                # 更新座標
                points_show = self.mw.textBrowser_labeled_points.toPlainText()
                points_show += f"{self.point1}, {self.point2},\n"
                self.mw.textBrowser_labeled_points.setText(points_show)

        # 設定 Search Window（右鍵點擊時）
        if event == cv2.EVENT_RBUTTONDOWN:
            self.mouse_drag = False
            # 計算點擊位置與各點之間的距離
            target = np.asarray(self.target_point)
            diff = target - np.asarray([x, y])
            # 距離最近者為此次框 ROI 的點
            self.t_point_index = np.argmin(np.sum(np.square(diff), axis=1))
            self.t_point = self.target_point[self.t_point_index]


        # 畫 Search Window 範圍（右鍵拖曳時）
        elif flags == 2 & cv2.EVENT_FLAG_RBUTTON:
            self.mouse_drag = True

            # 複製框圖模板
            temp_img = np.copy(self.img_label[self.current_page])

            # 計算 Search Winodw, Calculate Range
            s1, s2, c1, c2 = cv2_tool.get_search_window(self.t_point, (x, y), self.temp_size)

            cv2.rectangle(temp_img, s1, s2, (255, 0, 0), thickness=1)
            cv2.rectangle(temp_img, c1, c2, (255, 255, 0), thickness=1)

            # 更新圖片
            cv2.imshow(self.window_name, temp_img)


        # 確定 Search Window 範圍（右鍵放開時）
        elif event == cv2.EVENT_RBUTTONUP:
            if self.mouse_drag:
                self.mouse_drag = False  # 拖曳重置

                tx, ty = self.t_point

                # 計算 Search Winodw, Calculate Range
                s1, s2, c1, c2 = cv2_tool.get_search_window((tx, ty), (x, y), self.temp_size)

                # 紀錄範圍
                self.search_point[self.t_point_index] = [s1, s2]
                self.search_shift[self.t_point_index] = (abs(x - tx), abs(y - ty))

                # 畫圖
                cv2.rectangle(self.img_label[self.current_page], s1, s2, (0, 0, 255), thickness=1)
                cv2.rectangle(self.img_label[self.current_page], c1, c2, (255, 255, 0), thickness=1)

                # 更新圖片
                cv2.imshow(self.window_name, self.img_label[self.current_page])

    # 測試時方便建立線段
    def addPoint(self, point1, point2):
        # 作圖
        cv2.line(self.img_label[self.current_page], point1, point2, self.current_color, thickness=self.line_bold)
        cv2.circle(self.img_label[self.current_page], point1, 2, self.current_color, thickness=-1)
        cv2.circle(self.img_label[self.current_page], point2, 2, self.current_color, thickness=-1)

        # 計算距離 -> 尚未加入 List TODO
        text_point, d = cv2_tool.count_distance(point1, point2, self.delta)
        font = cv2.FONT_HERSHEY_SIMPLEX
        if self.font_show:
            cv2.putText(self.img_label[self.current_page], '{:4.3f}{}'.format(d, '(p)' if self.delta_x == 0 else ''),
                        text_point, font, self.font_size, (255, 255, 255), self.font_bold)

        # 新增點參數
        self.target_point.extend([point1, point2])
        self.track_done.extend([False, False])

        x, y = point1
        s11, s12, _, _ = cv2_tool.get_search_window((x, y),
                                                    (x + self.default_search // 2, y + self.default_search // 2),
                                                    self.temp_size)
        x, y = point2
        s21, s22, _, _ = cv2_tool.get_search_window((x, y),
                                                    (x + self.default_search // 2, y + self.default_search // 2),
                                                    self.temp_size)

        self.search_point.extend([[s11, s12], [s21, s22]])
        self.search_shift.extend([(self.default_search // 2, self.default_search // 2),
                                  (self.default_search // 2, self.default_search // 2)])

        cv2.imshow(self.window_name, self.img_label[self.current_page])

        # 先將第一點的距離輸入結果
        self.result_distance[self.color_index] = [d]

        self.color_index += 1
        self.current_color = self.colors[self.color_index]

    # 畫線的 Speckle Tracking
    def tracking(self, show=False):
        progress_denominator = len(self.mw.files) - 1
        progress_fraction = 0

        # 如果沒有新的點，回傳不做
        for done in self.track_done:
            if not done: break
        else:
            return

        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        path = self.mw.default_path + '/' + self.mw.default_filename + '/_' + self.mw.default_filename + '_' + self.mw.json_para['method'] + '.mp4'
        videowriter = cv2.VideoWriter(path, fourcc, self.mw.json_para['video_fps'], (self.w, self.h))

        for i in range(1, len(self.mw.files)):

            progress_fraction += 1
            img_tem = cv2.cvtColor(cv2.imdecode(np.fromfile(self.mw.files[i - 1], dtype=np.uint8), -1), cv2.COLOR_BGR2GRAY)
            img_can = cv2.cvtColor(cv2.imdecode(np.fromfile(self.mw.files[i], dtype=np.uint8), -1), cv2.COLOR_BGR2GRAY)
            img_label = cv2.imdecode(np.fromfile(self.mw.files[i], dtype=np.uint8), -1)

            for j, (s_shift, done) in enumerate(zip(self.search_shift, self.track_done)):

                if i == 1: self.result_point[j] = [self.target_point[j]]
                if i == len(self.mw.files) - 1: self.track_done[j] = True

                result = self.speckle_tracking.method(self.result_point[j][-1], img_tem, img_can, s_shift, self.temp_size)
                self.result_point[j].append(result)

                if j % 2 == 1:
                    # 抓出前次結果的點
                    result_tem = self.result_point[j - 1][-1]

                    color = self.colors[(j // 2) % self.num_of_color]

                    # 畫線、計算（顯示）距離
                    cv2.line(img_label, result_tem, result, color, thickness=self.line_bold)
                    text_point, d = cv2_tool.count_distance(result_tem, result, self.delta)
                    if self.font_show:
                        cv2.putText(img_label, '{:4.3f}{}'.format(d, '(p)' if self.delta_x == 0 else ''),
                                    text_point,
                                    cv2.FONT_HERSHEY_SIMPLEX, self.font_size, (255, 255, 255), self.font_bold)
                    self.result_distance[j // 2].append(d)
            
            # cv2.imshow('test', img_label)
            # cv2.waitKey(0)
            videowriter.write(gui_tool.add_page_single(img_label, i, len(self.mw.files)))
            if i < len(self.img_label): self.img_label[i] = np.copy(img_label)

            self.show_progress_bar(np.copy(self.img_label[0]), progress_fraction, progress_denominator, pos='top')

        videowriter.release()
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
