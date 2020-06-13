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

import cgitb

cgitb.enable(format='text')

'''
20200316 meeting

1. 把線段的 dx, dy 分量顯示出來（除了直接運算 distance，還要還原原本的 dx, dy）
2. 星期五高教計畫將操作流程、簡介報告出來

3. 加上選擇頁數，在下星期一報告
'''

import img.iconQrc


# TODO 顏色轉換
# TODO 重製 temp 與 search 按鈕

# TODO 新增 target point 顯示視窗??
# TODO 新贓錨點??

class My_MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()

        self.setupUi(self)
        # add icon https://www.jianshu.com/p/c1e75244b6f3,
        # https://typecoder.blogspot.com/2018/04/pyqt-pyinstallerwinicon.html
        self.setWindowIcon(QtGui.QIcon(':/icon.png'))

        # 先將顏色功能藏起來
        self.btn_color.hide()

        # add line 隱藏
        self.btn_add_line.hide()

        # 還沒做好的小畫板
        self.mplwidget_show_curve.hide()

        # 讀取記錄檔
        self.json_para = self.use_json('read')

        self.setup()

    def setup(self):

        # 設定 json 內容
        self.default_path = self.json_para['path']

        if self.json_para['method'] == 'SAD':
            self.radioButton_SAD.setChecked(True)
        elif self.json_para['method'] == 'PPMCC':
            self.radioButton_PPMCC.setChecked(True)
        elif self.json_para['method'] == 'NCC':
            self.radioButton_CC.setChecked(True)
        elif self.json_para['method'] == 'OF':
            self.radioButton_Optical.setChecked(True)

        self.checkBox_Animation.setChecked(self.json_para['animation'])

        self.btn_save_video_checkable.setChecked(self.json_para['auto_save_video'])
        self.btn_save_csv_checkable.setChecked(self.json_para['auto_save_csv'])
        self.btn_save_curve_checkable.setChecked(self.json_para['auto_save_curve'])
        self.btn_save_ani_curve_checkable.setChecked(self.json_para['auto_save_ani_curve'])


        # 初始化防止特定錯誤
        self.filename = ''
        self.cv2_gui = ''
        self.mode = ''
        self.result_curve_temp = ''

        # 初始化 Console 內容
        self.console_text = ''

        # 按下 選路徑(btn_path) 按鈕
        self.btn_browse.clicked.connect(self.clicked_btn_path)

        # 設定更新 template, search window spinbox 的動作
        self.spinBox_temp_size.valueChanged.connect(self.spinBox_temp_changed)
        self.spinBox_search_range.valueChanged.connect(self.spinBox_search_changed)

        # 更新更新 draw delay 的大小
        self.spinBox_drawing_delay.valueChanged.connect(self.spinBox_drawing_delay_changed)

        # 按下執行時的動作
        self.btn_run.clicked.connect(self.clicked_btn_run)

        # 按下 COLOR 轉換色彩
        self.btn_color.clicked.connect(self.clicked_btn_color)

        # 按下 delta 設定 delta
        self.btn_set_delta.clicked.connect(self.clicked_btn_set_delta)

        # 按下 open folder
        self.btn_open_folder.clicked.connect(self.clicked_btn_open_folder)

        # 新增點功能
        self.btn_add_line.clicked.connect(self.clicked_btn_add_line)

        # 滑動 horizontal slide 的動作
        self.horizontalSlider_preview.valueChanged.connect(self.slide_change)

        # SAVE 相關
        # 按下儲存結果
        self.btn_save_video.clicked.connect(self.clicked_btn_save_video)
        self.btn_save_csv.clicked.connect(self.clicked_btn_save_csv)
        self.btn_save_curve.clicked.connect(self.clicked_btn_save_curve)
        self.btn_save_ani_curve.clicked.connect(self.clicked_btn_save_ani_curve)

        # mode 切換
        self.radioButton_line.toggled.connect(self.radio_btn_line_change)

        # mode == line 中的 方法切換
        self.radioButton_spline.toggled.connect(self.radio_btn_curve_change)
        self.radioButton_strain.toggled.connect(self.radio_btn_curve_change)

        # 按下軟體資訊
        self.action_version.triggered.connect(self.action_soft_information)

        # 按下 Reset Setting
        self.action_reset_setting.triggered.connect(self.action_reset_setting_triggered)

        # 按下 open setting file
        self.action_open_setting_file.triggered.connect(self.action_open_setting_triggered)

        # 按下 open saved points
        self.action_open_saved_points.triggered.connect(self.action_open_saved_points_triggered)

        # 設定更新頁數的 spinbox 動作
        self.spinBox_start.valueChanged.connect(self.spinBox_start_change)
        self.spinBox_end.valueChanged.connect(self.spinBox_end_change)

        # 按下 animation -> 更新 json 設定檔
        self.checkBox_Animation.clicked.connect(self.checkBox_Animation_change)

        # 更改方法
        self.radioButton_PPMCC.clicked.connect(self.method_changed)
        self.radioButton_SAD.clicked.connect(self.method_changed)
        self.radioButton_CC.clicked.connect(self.method_changed)
        self.radioButton_Optical.clicked.connect(self.method_changed)

        # auto save 相關 -> 更新 json 設定檔
        self.checkBox_auto_save.clicked.connect(self.checkBox_auto_save_change)
        self.btn_save_video_checkable.clicked.connect(self.checkBox_auto_save_btn)
        self.btn_save_csv_checkable.clicked.connect(self.checkBox_auto_save_btn)
        self.btn_save_curve_checkable.clicked.connect(self.checkBox_auto_save_btn)
        self.btn_save_ani_curve_checkable.clicked.connect(self.checkBox_auto_save_btn)



        #################### 額外控制功能的動作 ####################

        # 按下 auto integral，更改 target frame 的屬性（可調 / 不可調）
        self.checkBox_auto_integral.clicked.connect(self.checkBox_auto_integral_change)

        # 更新 integral ratio 與 integral line 的動作
        self.spinBox_integral_line.valueChanged.connect(self.spinBox_integral_change)
        self.spinBox_integral_ratio.valueChanged.connect(self.spinBox_integral_change)

        # 變更 spin target frame 時，更新參考點
        self.spinBox_target_frame.valueChanged.connect(self.spinBox_target_frame_chane)

        # 記錄點
        self.btn_save_points.clicked.connect(self.clicked_btn_save_points)

    # 按下 選路徑(btn_path) 按鈕的動作
    @pyqtSlot()
    def clicked_btn_path(self):
        files, filetype = QFileDialog.getOpenFileNames(self, "選擇文件", self.default_path,  # 起始路径
                                                       "All Files (*);;Dicom Files (*.dcm);;Png Files (*.png);;JPEG Files (*.jpeg)")

        # 如果有讀到檔案
        if len(files) > 0:
            # 更新預設路徑
            self.default_path = os.path.split(files[0])[0]
            self.json_para['path'] = self.default_path
            self.use_json('write')

            # 副檔名
            self.extension = os.path.splitext(files[0])[-1].lower()

            # 如果讀取到 Dicom 檔
            if self.extension == '.dcm':
                file = files[0]
                browse_path = file
                self.default_path, self.default_filename = os.path.split(browse_path)
                self.default_filename = self.default_filename.split('.')[0]
                self.filename = os.path.splitext(os.path.split(file)[-1])[0]

                dicom = pydicom.read_file(file)
                self.IMGS = gui_tool.add_page(dicom.pixel_array)
                self.img_preview = self.IMGS[0]
                self.num_of_img, self.h, self.w = self.IMGS.shape[:3]

                filetype = '.dcm'

                self.date, self.time = '', ''

                # 讀取 dicom 的日期
                if 'InstanceCreationDate' in dir(dicom):
                    self.date = dicom.InstanceCreationDate
                    self.date = datetime.date(int(self.date[:4]), int(self.date[4:6]), int(self.date[6:]))
                elif 'StudyDate' in dir(dicom):
                    self.date = dicom.StudyDate
                    self.date = datetime.date(int(self.date[:4]), int(self.date[4:6]), int(self.date[6:]))

                # 讀取 dicom 的時間
                if 'InstanceCreationTime' in dir(dicom):
                    self.time = dicom.InstanceCreationTime
                    self.time = datetime.time(int(self.time[:2]), int(self.time[2:4]), int(self.time[4:]))
                elif 'StudyTime' in dir(dicom):
                    self.time = dicom.StudyTime
                    self.time = datetime.time(int(self.time[:2]), int(self.time[2:4]), int(self.time[4:]))

                # 讀取 dicom 中 儀器廠商
                if 'ManufacturerModelName' in dir(dicom) and 'Manufacturer' in dir(dicom):
                    system_name = dicom.Manufacturer + ' - ' + dicom.ManufacturerModelName
                elif 'Manufacturer' in dir(dicom):
                    system_name = dicom.Manufacturer
                elif 'ManufacturerModelName' in dir(dicom):
                    system_name = dicom.ManufacturerModelName
                else:
                    system_name = ''

                # 讀取 dicom 的 delta x, delta y
                if 'SequenceOfUltrasoundRegions' in dir(dicom):
                    deltax = dicom.SequenceOfUltrasoundRegions[0].PhysicalDeltaX
                    deltay = dicom.SequenceOfUltrasoundRegions[0].PhysicalDeltaY
                    deltax = deltax // 0.000001 / 1000
                    deltay = deltay // 0.000001 / 1000

                elif 'PixelSpacing' in dir(dicom):
                    deltax, deltay = dicom.PixelSpacing
                    deltax = deltax // 0.000001 / 1000
                    deltay = deltay // 0.000001 / 1000

                else:
                    deltax, deltay = self.json_para['delta_x'], self.json_para['delta_y']

                # 讀取 dicom 的 fps
                if 'RecommendedDisplayFrameRate' in dir(dicom):
                    self.json_para['video_fps'] = dicom.RecommendedDisplayFrameRate
                else:
                    pass


            # 如果讀到圖檔
            elif self.extension == '.png' or self.extension == '.jpg' or self.extension == '.jpeg' or self.extension == '.mp4':

                browse_path = os.path.split(files[0])[0]
                self.filename = os.path.split(browse_path)[-1]

                # 輸出影向預設的路徑與檔案名稱
                self.default_path = os.path.split(browse_path)[0]
                self.default_filename = self.filename

                if self.extension == '.mp4':
                    self.filename = os.path.splitext(os.path.split(files[0])[-1])[0]
                    capture = cv2.VideoCapture(files[0])
                    ret, frame = capture.read()
                    IMGS = []
                    while ret:
                        IMGS.append(frame)
                        ret, frame = capture.read()

                    capture.release()
                    self.IMGS = np.asarray(IMGS)

                else:
                    # 排序圖檔
                    files = np.asarray(files)
                    temp = np.asarray([int(file.split('.')[0].split('/')[-1]) for file in files])
                    temp = np.argsort(temp)
                    files = files[temp]

                    self.IMGS = gui_tool.add_page(
                        np.asarray([cv2.imdecode(np.fromfile(file, dtype=np.uint8), -1) for file in files]))
                    if np.ndim(self.IMGS) == 3:
                        self.IMGS = cv2.merge([self.IMGS, self.IMGS, self.IMGS])

                self.img_preview = self.IMGS[0]
                self.num_of_img, self.h, self.w = self.IMGS.shape[:3]

                filetype = '.' + files[0].split('.')[-1]

                self.date, self.time, system_name = '', '', ''

                deltax, deltay = self.json_para['delta_x'], self.json_para['delta_y']


            # 如果讀入檔案不是 dicom 或是 圖檔
            else:
                msg = QtWidgets.QMessageBox()
                msg.setWindowTitle('Warning')
                msg.setText('Please select the dicom or the image file.\n')
                msg.setIcon(QtWidgets.QMessageBox.Warning)

                msg.exec_()
                return

            # 寫入 檔案路徑
            self.textBrowser_browse.setText(browse_path)

            # 顯示超音波廠商、字體白色、根據字數調整 label size
            self.label_manufacturer.setText(system_name)
            self.label_manufacturer.setStyleSheet("color:white")
            self.label_manufacturer.adjustSize()

            # 寫入 file detail 內容
            self.label_filetype_show.setText(filetype)
            self.label_image_size_show.setText(str(self.w) + ' x ' + str(self.h))
            self.label_date_show.setText(str(self.date))
            self.label_time_show.setText(str(self.time))
            self.label_frame_show.setText(str(self.num_of_img))

            # 更新 josn 參數
            self.json_para['delta_x'] = deltax
            self.json_para['delta_y'] = deltay
            self.doubleSpinBox_delta_x.setValue(deltax)
            self.doubleSpinBox_delta_y.setValue(deltay)

            # horizontalSlider_preview 設定最大值、歸零
            self.horizontalSlider_preview.setMaximum(len(self.IMGS) - 1)
            self.horizontalSlider_preview.setValue(0)

            # 預設的 template block 與 search window
            self.spinBox_temp_size.setValue(self.json_para['template_size'])
            self.spinBox_temp_size.setRange(1, self.h // 2)
            self.spinBox_search_range.setValue(self.json_para['search_size'])
            self.spinBox_search_range.setRange(1, self.h // 2)

            # 預設的 draw delay
            self.default_draw_delay = self.json_para['draw_delay']
            self.spinBox_drawing_delay.setValue(self.default_draw_delay)
            self.spinBox_drawing_delay.setRange(1, 100)

            # 建立預覽圖片、自適化調整
            self.show_preview_img(np.copy(self.img_preview), self.json_para['template_size'],
                                  self.json_para['search_size'])

            # 預設上下限與初始頁數
            self.spinBox_start.setRange(0, self.num_of_img - 1)
            self.spinBox_end.setRange(0, self.num_of_img - 1)
            self.spinBox_start.setValue(0)
            self.spinBox_end.setValue(self.num_of_img - 1)

            #################### 額外控制功能的動作 ####################
            self.spinBox_target_frame.setRange(0, self.num_of_img - 1)

            # 清空 points
            self.textBrowser_labeled_points.setText('')
            self.textBrowser_auto_add_point.setText('')

            with open('saved_points.json', 'r') as f:
                saved_points = json.loads(f.read())
                name = os.path.split(self.json_para['path'])[-1] + '_' + str(self.date) + '_' +self.filename + self.extension
                if name in saved_points.keys():
                    self.textBrowser_auto_add_point.setText(saved_points[name])




        # 如果沒有選擇檔案的話
        else:
            pass

    @pyqtSlot()
    def clicked_btn_run(self):

        if not self.filename:
            msg = QtWidgets.QMessageBox()
            msg.setWindowTitle('Warning')
            msg.setText('Please select the dicom or the image file.\n')
            msg.setIcon(QtWidgets.QMessageBox.Warning)

            msg.exec_()
            return

        cv2.destroyAllWindows()
        self.btn_add_line.show()

        # 清除 strain curve 圖片
        self.label_show_curve.setPixmap(QtGui.QPixmap(""))

        # 紀錄設定的開始與結束頁
        self.start = self.spinBox_start.value()
        self.end = self.spinBox_end.value() + 1



        # 呼叫 cv2 GUI class 的參數
        kwargs = {
            'main_window': self,
            'imgs': self.IMGS[self.start:self.end:, :, :],
            'window_name': self.filename,
            'delta_x': float(self.doubleSpinBox_delta_x.value()) / 1000,
            'delta_y': float(self.doubleSpinBox_delta_y.value()) / 1000,
            'temp_size': int(self.spinBox_temp_size.value()),
            'default_search': int(self.spinBox_search_range.value()),
            'method': self.json_para['method'],
            'draw_delay': int(self.spinBox_drawing_delay.value()),
            'json_para': self.json_para
        }

        # 設定模式
        if self.radioButton_line.isChecked():
            self.mode = 'line'
            self.cv2_gui = Cv2Line(**kwargs)

            #################### 額外控制功能的動作 ####################
            # ADD point 格式：
            # (288, 114), (266, 194)
            # (326, 123), (329, 184)
            # (342, 105), (368, 179)
            add_points = self.textBrowser_auto_add_point.toPlainText()
            self.textBrowser_labeled_points.setText(add_points)
            if add_points != '':
                try:
                    add_points = add_points.replace('(', '').replace(')', '').replace(' ', '').replace('\n', '').split(
                        ',')
                    for i in range(0, len(add_points), 4):
                        # for point in add_points:
                        x1, y1, x2, y2 = add_points[i], add_points[i + 1], add_points[i + 2], add_points[i + 3]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        self.cv2_gui.addPoint((x1, y1), (x2, y2))
                        # 如果下次的四個點無法算完
                        if i + 8 > len(add_points): break

                except Exception as e:
                    print("###########################\n"
                          "# 輸入點的格式錯誤，應為：\n"
                          "# (288, 114), (266, 194),\n"
                          "# (326, 123), (329, 184),\n"
                          "# (342, 105), (368, 179),\n"
                          "###########################")


        elif self.radioButton_draw.isChecked():
            self.mode = 'point'
            self.cv2_gui = Cv2Point(**kwargs)

        self.use_json('write')

        ###################### 主程式運行 ######################
        while True:
            cv2.setMouseCallback(self.cv2_gui.window_name, self.cv2_gui.click_event)  # 設定滑鼠回饋事件

            action = gui_tool.find_action(cv2.waitKey(1))  # 設定鍵盤回饋事件

            # 「esc」 跳出迴圈
            if action == 'esc':
                break

            # 「r」 重置
            if action == 'reset':
                # 清除 strain curve 圖片
                self.textBrowser_labeled_points.clear()
                self.textBrowser_auto_add_point.clear()
                self.textBrowser_target_frame.clear()
                self.label_show_curve.setPixmap(QtGui.QPixmap(""))
                self.cv2_gui.reset()

            # 「s」 執行 speckle tracking
            if action == 'speckle':
                t1 = time.time()
                self.cv2_gui.tracking(show=True if self.checkBox_Animation.isChecked() else False)
                t2 = time.time()
                print('Speckle Tracking costs {} seconds.\n'.format(t2 - t1))

                if self.mode == 'line':
                    # 畫 curve
                    self.plot_strain_curve()

                    # 自動存檔？
                    if self.checkBox_auto_save.isChecked():
                        self.auto_save_files()

                    # 顯示資料在 console
                    target_frame = int(self.spinBox_target_frame.text())
                    self.console_text = 'Length:\n'
                    for k in self.cv2_gui.result_distance.keys():
                        self.console_text += '{:.3f}\n'.format(self.cv2_gui.result_distance[k][target_frame])
                    self.console_text += '\n'

                    self.console_text += 'Result Points:\n'
                    for k in range(len(self.cv2_gui.result_point)):
                        if k % 2 == 1:
                            self.console_text += f"{self.cv2_gui.result_point[k - 1][target_frame]}, {self.cv2_gui.result_point[k][target_frame]}\n"

                    self.textBrowser_target_frame.setText(self.console_text)

                elif self.mode == 'point':
                    pass

            # 「t」 增加預設點數（測試時用）
            if action == 'test':
                pass

                # print(f"self.cv2_gui.result_distance :\n{self.cv2_gui.result_distance}")

            # 按空白鍵查看點數狀況
            if action == 'space':
                labeled_points = ''

                print('self.target_point : ', self.cv2_gui.target_point)
                print('self.track_done : ', self.cv2_gui.track_done)
                print('self.search_point : ', self.cv2_gui.search_point)  # 目前沒用
                print('self.search_shift : ', self.cv2_gui.search_shift)
                print('self.result_points: ', self.cv2_gui.result_point)
                print()

        cv2.destroyWindow(self.cv2_gui.window_name)  # （按 esc 跳出迴圈後）關閉視窗

    def plot_strain_curve(self):
        # 開始繪圖
        self.fig, self.ax = plt.subplots()
        # figure = self.mplwidget_show_curve.canvas.figure.add_subplot(111)

        plt.xlabel('frame')
        # figure.set_xlabel('frame')
        if self.json_para['curve_bound'] != 0:
            plt.ylim(-self.json_para['curve_bound'], self.json_para['curve_bound'])

        for i in self.cv2_gui.result_distance.keys():

            # 抓出對應的顏色，並轉呈 matplotlib 的 RGB 0-1 格式
            color = tuple([self.cv2_gui.colors[i % self.cv2_gui.num_of_color][-j] / 255 for j in range(1, 4)])

            if self.radioButton_strain.isChecked():
                plt.axhline(0, color='k', alpha=0.2)
                # figure.axhline(0, color='k', alpha=0.2)
                if self.radioButton_spline.isChecked():
                    plt.plot([i for i in range(self.start, self.end)],
                             gui_tool.lsq_spline_medain(self.cv2_gui.result_strain[i]), color=color)
                    # figure.plot([i for i in range(self.start, self.end)], gui_tool.lsq_spline_medain(self.cv2_gui.result_strain[i]), color=color)
                elif self.radioButton_original.isChecked():
                    plt.plot([i for i in range(self.start, self.end)], self.cv2_gui.result_strain[i], color=color)
                    # figure.plot([i for i in range(self.start, self.end)], self.cv2_gui.result_strain[i], color=color)

                plt.ylabel('Strain')
                # figure.set_ylabel('Strain')
                plt.title('Strain curve')
                # figure.set_title('Strain curve')

            elif self.radioButton_distance.isChecked():
                if self.radioButton_spline.isChecked():
                    plt.plot([i for i in range(self.start, self.end)],
                             gui_tool.lsq_spline_medain(self.cv2_gui.result_distance[i]), color=color)
                    # figure.plot([i for i in range(self.start, self.end)], gui_tool.lsq_spline_medain(self.cv2_gui.result_distance[i]), color=color)
                elif self.radioButton_original.isChecked():
                    plt.plot([i for i in range(self.start, self.end)], self.cv2_gui.result_distance[i], color=color)
                    # figure.plot([i for i in range(self.start, self.end)], self.cv2_gui.result_distance[i], color=color)

                plt.ylabel('Distance')
                # figure.set_ylabel('Distance')
                plt.title('Distance curve')
                # figure.set_title('Distance curve')

        # self.mplwidget_show_curve.canvas.draw()
        plt.savefig(self.default_path + '/output.png')
        plt.close()

        self.result_curve_temp = cv2.imdecode(np.fromfile(self.default_path + '/output.png', dtype=np.uint8),
                                              -1)  # 解決中文路徑問題
        # self.result_curve_temp = cv2.imread(self.default_path + '/output.png') # 中文路徑會報錯
        os.remove(self.default_path + '/output.png')

        self.label_show_curve.setPixmap(QtGui.QPixmap(gui_tool.convert2qtimg(self.result_curve_temp)))
        self.label_show_curve.setScaledContents(True)

    def clicked_btn_set_delta(self):
        if not self.filename:
            return

        input_dy, okPressed = QInputDialog.getInt(self, "Set Delta x/y", "Line length (mm):", 5, 1, 100, 1)
        if okPressed:
            set_delta = SetDelta(self.IMGS[0])

            while set_delta.undo:
                cv2.setMouseCallback(set_delta.window_name, set_delta.click_event)  # 設定滑鼠回饋事件
                cv2.waitKey(1)

            cv2.destroyWindow(set_delta.window_name)
            dy = abs(set_delta.point2[1] - set_delta.point1[1])
            self.json_para['delta_x'] = 1000 * input_dy / dy
            self.json_para['delta_y'] = 1000 * input_dy / dy
            self.doubleSpinBox_delta_x.setValue(self.json_para['delta_y'])
            self.doubleSpinBox_delta_y.setValue(self.json_para['delta_y'])

    def clicked_btn_add_line(self):
        if not self.filename:
            return

        x1, okPressed = QInputDialog.getInt(self, "Add line", "x1:", 0, 0, self.w - 1, 1)
        if okPressed: y1, okPressed = QInputDialog.getInt(self, "Add line", "y1:", 0, 0, self.h - 1, 1)
        if okPressed: x2, okPressed = QInputDialog.getInt(self, "Add line", "x2:", 0, 0, self.w - 1, 1)
        if okPressed: y2, okPressed = QInputDialog.getInt(self, "Add line", "y2:", 0, 0, self.h - 1, 1)
        if not okPressed: return

        self.cv2_gui.addPoint((x1, y1), (x2, y2))

    # TODO 待修正
    @pyqtSlot()
    def clicked_btn_color(self):
        if not self.filename:
            return

        # 將圖片從 YBR 轉成 BGR 通道
        self.IMGS = np.asarray([cv2.cvtColor(
            pydicom.pixel_data_handlers.util.convert_color_space(img, 'YBR_FULL', 'RGB'), cv2.COLOR_RGB2BGR) for img in
                                self.IMGS])
        self.img_preview = self.IMGS[0]

        # 建立預覽圖片、自適化調整
        self.show_preview_img(np.copy(self.img_preview), self.json_para['template_size'], self.json_para['search_size'])

    # 存檔的按鈕
    def clicked_btn_save_video(self):
        # 如果尚未選擇影像，或是尚未運行 cv2，不運行按鈕
        if not self.filename or not self.cv2_gui: return

        fps, okPressed = QInputDialog.getInt(self, "Set output video FPS", "FPS:", self.json_para['video_fps'], 1,
                                             10000, 1)
        if okPressed == False: return

        path, filetype = QFileDialog.getSaveFileName(self, "文件保存",
                                                     self.default_path + '/' + self.default_filename + '.mp4',
                                                     "All Files (*);;MP4 Files (*.mp4)")

        # 如果沒有選擇存檔路徑，結束 function
        if not path: return

        self.json_para['video_fps'] = fps
        self.use_json('write')

        # 強制轉換副檔名為 mp4
        filename = os.path.split(path)[-1]
        if filename.split('.')[-1] != 'mp4':
            path = path.split('.')[0] + '.mp4'

        # 開始寫入 mp4
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        videowriter = cv2.VideoWriter(path, fourcc, fps, (self.w, self.h))

        for img in self.cv2_gui.img_label:
            videowriter.write(img)
        videowriter.release()

        # 通知視窗
        msg = QMessageBox()
        msg.setWindowTitle('Save completed.')
        msg.setIcon(QMessageBox.Information)
        msg.setText('Result saved finish.\n')

        Open = msg.addButton('Show in explorer', QMessageBox.AcceptRole)
        Ok = msg.addButton('OK', QMessageBox.DestructiveRole)
        msg.setDefaultButton(Ok)
        reply = msg.exec()

        # 如果選擇開啟資料夾，則運行
        if reply == 0:
            os.startfile(os.path.split(path)[0])

    # 儲存成 CSV 檔案
    def clicked_btn_save_csv(self):
        # 如果尚未選擇影像，或是尚未運行 cv2，不運行按鈕
        if not self.filename or not self.cv2_gui:
            return
        path, filetype = QFileDialog.getSaveFileName(self, "文件保存",
                                                     self.default_path + '/' + self.default_filename + '.csv',
                                                     "All Files (*);;CSV Files (*.csv)")

        # 如果沒有選擇存檔路徑，結束 function
        if not path:
            return

        # 將結果點轉成 dataframe、更改 colums 文字
        select_df = pd.DataFrame(self.cv2_gui.result_point)
        select_df.columns = ['Point {}'.format(i) for i in select_df.columns]

        if self.mode == 'line':
            for i, (d, s) in enumerate(zip(self.cv2_gui.result_distance.values(), self.cv2_gui.result_strain.values())):
                select_df.insert(i * 4 + 2, 'Distance {} -> {}'.format(i * 2, i * 2 + 1), d)
                select_df.insert(i * 4 + 3, 'Strain {} -> {}'.format(i * 2, i * 2 + 1), s)

        select_df.to_csv(path, index=True, sep=',')

        # 通知視窗
        msg = QMessageBox()
        msg.setWindowTitle('Save completed.')
        msg.setIcon(QMessageBox.Information)
        msg.setText('Result saved finish.\n')

        Open = msg.addButton('Show in explorer', QMessageBox.AcceptRole)
        Ok = msg.addButton('OK', QMessageBox.DestructiveRole)
        msg.setDefaultButton(Ok)
        reply = msg.exec()

        # 如果選擇開啟資料夾，則運行
        if reply == 0:
            os.startfile(os.path.split(path)[0])

    # 如果要儲存 Curve
    def clicked_btn_save_curve(self):
        # 如果尚未選擇影像，或是尚未運行 cv2，不運行按鈕，或是不是畫線模式（沒有 curve）
        if not self.filename or not self.cv2_gui:
            return

        path, filetype = QFileDialog.getSaveFileName(self, "文件保存",
                                                     self.default_path + '/' + self.default_filename + '.png',
                                                     "All Files (*);;PNG Files (*.png)")

        # 如果沒有選擇存檔路徑，結束 function
        if not path:
            return

        # 解決中文路徑問題
        cv2.imencode('.png', self.result_curve_temp)[1].tofile(path)
        # cv2.imwrite(path, self.result_curve_temp)

        # 通知視窗
        msg = QMessageBox()
        msg.setWindowTitle('Save completed.')
        msg.setIcon(QMessageBox.Information)
        msg.setText('Result saved finish.\n')

        Open = msg.addButton('Show in explorer', QMessageBox.AcceptRole)
        Ok = msg.addButton('OK', QMessageBox.DestructiveRole)
        msg.setDefaultButton(Ok)
        reply = msg.exec()

        # 如果選擇開啟資料夾，則運行
        if reply == 0:
            os.startfile(os.path.split(path)[0])


    def clicked_btn_save_ani_curve(self, not_auto=''):
        # 如果不是從自動存檔呼叫的，需要選擇檔案路徑
        if not_auto != 'auto':
            # 如果尚未選擇影像，或是尚未運行 cv2，不運行按鈕
            if not self.filename or not self.cv2_gui: return

            fps, okPressed = QInputDialog.getInt(self, "Set output video FPS", "FPS:", self.json_para['video_fps'], 1,
                                                 10000, 1)

            if okPressed == False: return

            path, filetype = QFileDialog.getSaveFileName(self, "文件保存",
                                                         self.default_path + '/' + self.default_filename + '_'+ self.json_para['method'] + '_ani_curve.mp4',
                                                         "All Files (*);;MP4 Files (*.mp4)")

            if not path: return # 如果沒有選擇存檔路徑，結束 function

            self.json_para['video_fps'] = fps

        # 如果是自動存檔呼叫的，指定路徑
        else:
            path = self.default_path + '/' + self.default_filename + '_' + self.json_para['method'] + '_ani_curve.mp4'



        # plt.xlabel('frame')
        if self.json_para['curve_bound'] != 0:
            plt.ylim(-self.json_para['curve_bound'], self.json_para['curve_bound'])
            line, = self.ax.plot([0, 0], [-self.json_para['curve_bound'], self.json_para['curve_bound']], color='k', alpha=0.2)
        else:
            line, = self.ax.plot([0, 0], [0.4, -0.4],color='k', alpha=0.2)

        # 設定動態更新
        def animate(i):
            line.set_xdata([i, i])
            return line,

        # 設定初始狀態
        def init():
            line.set_xdata([0, 0])
            return line,

        # 建立 animation 物件，frames = 影像數量， interval = 更新時間（ms）
        ani = animation.FuncAnimation(fig=self.fig, func=animate, frames=self.num_of_img, init_func=init, interval=1000, blit=False)

        ani.save(f"temp.gif", writer='imagemagick', fps=20)
        gif = cv2.VideoCapture(f"temp.gif")
        ret, frame = gif.read()
        h, w, ch = frame.shape
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        videowriter = cv2.VideoWriter(path, fourcc, self.json_para['video_fps'], (w, h))
        while ret:
            videowriter.write(frame)
            ret, frame = gif.read()

        videowriter.release()


    # 儲存所有結果
    def auto_save_files(self):
        if not self.filename or not self.cv2_gui:
            return

        path = self.default_path + '/' + self.default_filename + '_' + self.json_para['method'] + '.all'
        if path.split('.')[-1] == 'all':
            path = path.split('.')[0]

        if self.json_para['auto_save_video']:
            # 儲存 mp4
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            videowriter = cv2.VideoWriter(path + '.mp4', fourcc, self.json_para['video_fps'], (self.w, self.h))

            for img in self.cv2_gui.img_label:
                videowriter.write(img)
            videowriter.release()


        # 儲存 csv
        select_df = pd.DataFrame(self.cv2_gui.result_point)
        select_df.columns = ['Point {}'.format(i) for i in select_df.columns]

        if self.mode == 'line':
            for i, (d, s) in enumerate(
                    zip(self.cv2_gui.result_distance.values(), self.cv2_gui.result_strain.values())):
                select_df.insert(i * 4 + 2, 'Distance {} -> {}'.format(i * 2, i * 2 + 1), d)
                select_df.insert(i * 4 + 3, 'Strain {} -> {}'.format(i * 2, i * 2 + 1), s)

            if self.json_para['auto_save_curve']:
            # 儲存 curve
                cv2.imencode('.png', self.result_curve_temp)[1].tofile(path + '.png')

        if self.json_para['auto_save_csv']:
            select_df.to_csv(path + '.csv', index=True, sep=',')


        if self.json_para['auto_save_ani_curve']:
            self.clicked_btn_save_ani_curve('auto')


    def action_soft_information(self):
        msg = QMessageBox()
        msg.setWindowTitle('Software Information')
        msg.setWindowIcon(QtGui.QIcon(':/icon.png'))
        msg.setText(
            'Author: Yuwen Huang\n\n' +
            'Latest Update: 20200613\n\n' +
            'Website: https://github.com/Yuwen0810/Speckle_Tracking\n'
        )
        reply = msg.exec()

    # Reset Setting
    def action_reset_setting_triggered(self):
        with open('setting_default.json', 'r') as f_default, open('setting.json', 'w') as f:
            f_default_content = f_default.read()
            self.json_para = json.loads(f_default_content)

            print(f_default_content, file=f)

        # 重製設定
        self.spinBox_temp_size.setValue(self.json_para['template_size'])
        self.spinBox_search_range.setValue(self.json_para['search_size'])
        self.default_draw_delay = self.json_para['draw_delay']
        self.radioButton_PPMCC.setChecked(True)

    # Open Setting file
    def action_open_setting_triggered(self):
        os.startfile(os.path.join(os.getcwd(), 'setting.json'))

    # Open Saved points file
    def action_open_saved_points_triggered(self):
        os.startfile(os.path.join(os.getcwd(), 'saved_points.json'))


    def radio_btn_line_change(self):
        if self.radioButton_line.isChecked():
            self.stackedWidget_mode.setCurrentIndex(0)
        else:
            self.stackedWidget_mode.setCurrentIndex(1)

    def radio_btn_curve_change(self):
        if not self.filename or not self.cv2_gui:
            return

        if self.mode == 'line':
            self.plot_strain_curve()

    # 更改
    def slide_change(self):
        if not self.filename:
            return

        # 建立預覽圖片、自適化調整
        self.img_preview = self.IMGS[self.horizontalSlider_preview.value()]
        self.show_preview_img(np.copy(self.img_preview), self.json_para['template_size'], self.json_para['search_size'])

    # 顯示預覽影像、自適化調整
    def show_preview_img(self, img, temp, search):
        x, y = self.w // 2, self.h // 2

        t_shift = temp // 2
        s_shift = search // 2

        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

        cv2.rectangle(img, (x - s_shift, y - s_shift), (x + s_shift, y + s_shift), (0, 0, 255), 1)
        cv2.rectangle(img, (x - s_shift - t_shift, y - s_shift - t_shift),
                      (x - s_shift + t_shift, y - s_shift + t_shift), (255, 255, 0), 1)

        self.label_preview.setPixmap(QtGui.QPixmap(gui_tool.convert2qtimg(img)))
        self.label_preview.setScaledContents(True)

    def spinBox_drawing_delay_changed(self, x):
        self.json_para['draw_delay'] = x

    def spinBox_temp_changed(self, x):
        self.json_para['template_size'] = x
        self.show_preview_img(np.copy(self.img_preview), self.json_para['template_size'], self.json_para['search_size'])

    def spinBox_search_changed(self, x):
        self.json_para['search_size'] = x
        self.show_preview_img(np.copy(self.img_preview), self.json_para['template_size'], self.json_para['search_size'])

    def spinBox_start_change(self, x):
        self.show_preview_img(np.copy(self.IMGS[x]), self.json_para['template_size'], self.json_para['search_size'])

    def spinBox_end_change(self, x):
        self.show_preview_img(np.copy(self.IMGS[x]), self.json_para['template_size'], self.json_para['search_size'])

    def checkBox_Animation_change(self):
        self.json_para['animation'] = self.checkBox_Animation.isChecked()
        self.use_json('write')

    def checkBox_auto_save_change(self):
        if self.checkBox_auto_save.isChecked():
            self.stackedWidget_auto_save.setCurrentIndex(1)
        else:
            self.stackedWidget_auto_save.setCurrentIndex(0)

        self.json_para['auto_save'] = self.checkBox_auto_save.isChecked()
        self.use_json('write')


    def method_changed(self):
        # 判斷 COST 方法
        if self.radioButton_SAD.isChecked():
            self.json_para['method'] = 'SAD'
        elif self.radioButton_PPMCC.isChecked():
            self.json_para['method'] = 'PPMCC'
        elif self.radioButton_CC.isChecked():
            self.json_para['method'] = 'NCC'
        elif self.radioButton_Optical.isChecked():
            self.json_para['method'] = 'OF'
        self.use_json('write')

    def checkBox_auto_save_btn(self):
        self.json_para['auto_save_video'] = True if self.btn_save_video_checkable.isChecked() else False
        self.json_para['auto_save_csv'] = True if self.btn_save_csv_checkable.isChecked() else False
        self.json_para['auto_save_curve'] = True if self.btn_save_curve_checkable.isChecked() else False
        self.json_para['auto_save_ani_curve'] = True if self.btn_save_ani_curve_checkable.isChecked() else False

        self.use_json('write')

    def checkBox_auto_integral_change(self):
        if self.checkBox_auto_integral.isChecked():
            self.spinBox_target_frame.setReadOnly(True)
        else:
            self.spinBox_target_frame.setReadOnly(False)

    def spinBox_integral_change(self):
        if self.checkBox_auto_integral.isChecked() == False:
            return

        target_line = self.spinBox_integral_line.value()
        ratio = self.spinBox_integral_ratio.value() / 100

        cumsum = np.cumsum(np.abs(self.cv2_gui.result_distance[target_line - 1]))
        thre = cumsum[-1] * ratio
        closer = np.argmin(np.abs(cumsum - thre))

        self.spinBox_target_frame.setValue(closer)

    def clicked_btn_save_points(self):
        with open('./saved_points.json', 'r') as f:
            saved_points = json.loads(f.read())

        name = os.path.split(self.json_para['path'])[-1] + '_' + str(self.date) + '_' +self.filename + self.extension
        saved_points[name] = self.textBrowser_labeled_points.toPlainText()

        with open('./saved_points.json', 'w') as f:
            print(json.dumps(saved_points, indent=4), file=f)

    def clicked_btn_open_folder(self):
        os.startfile(self.default_path)

    def spinBox_target_frame_chane(self, x):
        if not self.cv2_gui:
            return

        target_frame = x
        self.console_text = 'Length:\n'
        for k in self.cv2_gui.result_distance.keys():
            self.console_text += '{:.3f}\n'.format(self.cv2_gui.result_distance[k][target_frame])
        self.console_text += '\n'

        self.console_text += 'Result Points:\n'
        for k in range(len(self.cv2_gui.result_point)):
            if k % 2 == 1:
                self.console_text += f"{self.cv2_gui.result_point[k - 1][x]}, {self.cv2_gui.result_point[k][x]}\n"

        self.textBrowser_target_frame.setText(self.console_text)

    def use_json(self, mode='write'):
        if mode == 'write':
            with open('setting.json', 'w') as f:
                print(json.dumps(self.json_para, indent=4), file=f)
                return None
        elif mode == 'read':
            try:
                with open('setting.json', 'r') as f:
                    return json.loads(f.read())
            except:
                with open('setting_default.json', 'r') as f:
                    return json.loads(f.read())


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = My_MainWindow()
    window.show()
    sys.exit(app.exec_())
