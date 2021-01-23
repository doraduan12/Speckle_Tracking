import pydicom
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
from matplotlib import animation
import json
from io import BytesIO
import PIL

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
import main_general
import main_tk
import main_ishan

gui_tool = GuiTools()

import cgitb

cgitb.enable(format='text')

'''
20200316 meeting

1. 把線段的 dx, dy 分量顯示出來（除了直接運算 distance，還要還原原本的 dx, dy）
2. 星期五高教計畫將操作流程、簡介報告出來

3. 加上選擇頁數，在下星期一報告


未來改進方向
https://blog.csdn.net/jacke121/article/details/54718563
CV2 選框套件

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

        # 讀取記錄檔
        self.json_para = self.use_json('read')

        # console 初始
        self.console_nb = ""
        self.console_after = ""

        # fig 初始化
        self.fig = ""


        # 把專屬工具欄關掉，在特定使用者時才開啟
        self.menuYuwen.menuAction().setVisible(False)


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

        if self.json_para['user'] == 'yuwen':
            self.action_user_yuwen.setChecked(True)
            self.menuYuwen.menuAction().setVisible(True)    # 開啟專屬工具欄
        elif self.json_para['user'] == 'tk':
            self.action_user_tk.setChecked(True)
        elif self.json_para['user'] == 'ishan':
            self.action_user_ishan.setChecked(True)

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
        self.files = []

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
        # self.btn_color.clicked.connect(self.clicked_btn_color)

        # 按下 delta 設定 delta
        self.btn_set_delta.clicked.connect(self.clicked_btn_set_delta)

        # 放大 plt 功能
        self.btn_show_curve.clicked.connect(self.clicked_btn_show_curve)

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

        ''' Action  '''

        # 按下軟體資訊
        self.action_version.triggered.connect(self.action_soft_information)

        # 按下 Change auto save path
        self.action_auto_save_path.triggered.connect(self.action_auto_save_path_triggered)

        # 按下 Reset Setting
        self.action_reset_setting.triggered.connect(self.action_reset_setting_triggered)

        # 按下 open setting file
        self.action_open_setting_file.triggered.connect(self.action_open_setting_triggered)

        # 按下 open saved points
        self.action_open_saved_points.triggered.connect(self.action_open_saved_points_triggered)

        # 按下 multi files
        self.action_multi_files.triggered.connect(self.action_multi_files_triggered)

        # 連結改變模式
        self.action_user_tk.triggered.connect(self.action_user_tk_change)
        self.action_user_yuwen.triggered.connect(self.action_user_yuwen_change)
        self.action_user_ishan.triggered.connect(self.action_user_ishan_change)

        # 按下
        self.action_resize_input.triggered.connect(self.action_resize_input_triggered)


        ''' Spin box '''

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
        # self.checkBox_yuwen.clicked.connect(self.checkBox_auto_integral_change)

        # 更新 integral ratio 與 integral line 的動作

        # 變更 spin target frame 時，更新參考點
        self.spinBox_target_frame.valueChanged.connect(self.spinBox_target_frame_chane)

        # 記錄點
        self.btn_save_points.clicked.connect(self.clicked_btn_save_points)

        # 清除 console 紀錄的文字
        self.btn_console_clear.clicked.connect(self.clicked_btn_console_clear)



    # 按下 選路徑(btn_path) 按鈕的動作
    @pyqtSlot()
    def clicked_btn_path(self, files=None):

        if self.action_user_tk.isChecked():
            main_tk.load_file(self)
        else:
            main_general.load_file(self, files)


    @pyqtSlot()
    def clicked_btn_run(self, multi_mode=False):

        if self.action_user_tk.isChecked():
            main_tk.run_cv2(self)
        elif self.action_user_ishan.isChecked():
            main_ishan.run_cv2(self)
        else:
            main_general.run_cv2(self, multi_mode)


    def plot_strain_curve(self, axv=0):
        # 開始繪圖
        self.fig, self.ax = plt.subplots()
        # plt.figure()
        plt.xlabel('frame')
        if self.json_para['curve_bound'] != 0:
            plt.ylim(-self.json_para['curve_bound'], self.json_para['curve_bound'])

        for i in self.cv2_gui.result_distance.keys():

            # 抓出對應的顏色，並轉呈 matplotlib 的 RGB 0-1 格式
            color = tuple([self.cv2_gui.colors[i % self.cv2_gui.num_of_color][-j] / 255 for j in range(1, 4)])

            if self.radioButton_strain.isChecked():
                plt.axhline(0, color='k', alpha=0.2)
                if self.radioButton_spline.isChecked():
                    plt.plot(gui_tool.lsq_spline_medain(self.cv2_gui.result_strain[i]), color=color)
                elif self.radioButton_original.isChecked():
                    plt.plot(self.cv2_gui.result_strain[i], color=color)

                plt.ylabel('Strain')
                plt.title('Strain curve')

            elif self.radioButton_distance.isChecked():
                if self.radioButton_spline.isChecked():
                    plt.plot(gui_tool.lsq_spline_medain(self.cv2_gui.result_distance[i]), color=color)
                elif self.radioButton_original.isChecked():
                    plt.plot(self.cv2_gui.result_distance[i], color=color)

                plt.ylabel('Distance')
                plt.title('Distance curve')

        if axv != 0: plt.axvline(axv, color='k', alpha=0.2)

        # 申請緩衝地址
        buffer_ = BytesIO()

        # 儲存在記憶體中，而不是在本地磁碟，注意這個預設認為你要儲存的就是plt中的內容
        plt.savefig(buffer_, format='png')
        plt.close()
        buffer_.seek(0)

        # 用PIL或CV2從記憶體中讀取
        dataPIL = PIL.Image.open(buffer_)

        # 轉換為nparrary，PIL轉換就非常快了，data即為所需
        self.result_curve_temp = cv2.cvtColor(np.asarray(dataPIL), cv2.COLOR_BGR2RGB)

        # 顯示
        self.label_show_curve.setPixmap(QtGui.QPixmap(gui_tool.convert2qtimg(self.result_curve_temp)))
        self.label_show_curve.setScaledContents(True)

        # 釋放快取
        buffer_.close()

    def clicked_btn_set_delta(self):
        if not self.filename:
            return

        input_dy, okPressed = QInputDialog.getInt(self, "Set Delta x/y", "Line length (unit):", 5, 1, 100, 1)
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


    # 按下隱藏按鈕的 show
    def clicked_btn_show_curve(self):
        plt.close()
        try:
            plt.figure()
            plt.xlabel('frame')
            if self.json_para['curve_bound'] != 0:
                plt.ylim(-self.json_para['curve_bound'], self.json_para['curve_bound'])

            for i in self.cv2_gui.result_distance.keys():

                # 抓出對應的顏色，並轉呈 matplotlib 的 RGB 0-1 格式
                color = tuple([self.cv2_gui.colors[i % self.cv2_gui.num_of_color][-j] / 255 for j in range(1, 4)])

                if self.radioButton_strain.isChecked():
                    plt.axhline(0, color='k', alpha=0.2)
                    if self.radioButton_spline.isChecked():
                        plt.plot(gui_tool.lsq_spline_medain(self.cv2_gui.result_strain[i]), color=color)
                    elif self.radioButton_original.isChecked():
                        plt.plot(self.cv2_gui.result_strain[i], color=color)

                    plt.ylabel('Strain')
                    plt.title('Strain curve')

                elif self.radioButton_distance.isChecked():
                    if self.radioButton_spline.isChecked():
                        plt.plot(gui_tool.lsq_spline_medain(self.cv2_gui.result_distance[i]), color=color)
                    elif self.radioButton_original.isChecked():
                        plt.plot(self.cv2_gui.result_distance[i], color=color)

                    plt.ylabel('Distance')
                    plt.title('Distance curve')

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(e)


    def clicked_btn_add_line(self):
        if not self.filename:
            return

        x1, okPressed = QInputDialog.getInt(self, "Add line", "x1:", 0, 0, self.w - 1, 1)
        if okPressed: y1, okPressed = QInputDialog.getInt(self, "Add line", "y1:", 0, 0, self.h - 1, 1)
        if okPressed: x2, okPressed = QInputDialog.getInt(self, "Add line", "x2:", 0, 0, self.w - 1, 1)
        if okPressed: y2, okPressed = QInputDialog.getInt(self, "Add line", "y2:", 0, 0, self.h - 1, 1)
        if not okPressed: return

        self.cv2_gui.addPoint((x1, y1), (x2, y2))

    # # TODO 待修正
    # @pyqtSlot()
    # def clicked_btn_color(self):
    #     if not self.filename:
    #         return
    #
    #     # 將圖片從 YBR 轉成 BGR 通道
    #     self.IMGS = np.asarray([cv2.cvtColor(
    #         pydicom.pixel_data_handlers.util.convert_color_space(img, 'YBR_FULL', 'RGB'), cv2.COLOR_RGB2BGR) for img in
    #         self.IMGS])
    #     self.img_preview = self.IMGS[0]
    #
    #     # 建立預覽圖片、自適化調整
    #     self.show_preview_img(np.copy(self.img_preview), self.json_para['template_size'], self.json_para['search_size'])

    # 存檔的按鈕
    def clicked_btn_save_video(self):
        # 如果尚未選擇影像，或是尚未運行 cv2，不運行按鈕
        if not self.filename or not self.cv2_gui: return

        fps, okPressed = QInputDialog.getInt(self, "Set output video FPS", "FPS:", self.json_para['video_fps'], 1,
                                             10000, 1)
        if okPressed == False: return

        path, filetype = QFileDialog.getSaveFileName(self, "文件保存",
                                                     self.json_para['save_path'] + '/' + self.default_filename + '.mp4',
                                                     "All Files (*);;MP4 Files (*.mp4)")

        # 如果沒有選擇存檔路徑，結束 function
        if not path: return

        self.json_para['video_fps'] = fps
        self.json_para['save_path'] = os.path.split(path)[0]
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
                                                     self.json_para['save_path'] + '/' + self.default_filename + '.csv',
                                                     "All Files (*);;CSV Files (*.csv)")

        # 如果沒有選擇存檔路徑，結束 function
        if not path:
            return

        self.json_para['save_path'] = os.path.split(path)[0]
        self.use_json('write')

        # 將結果點轉成 dataframe、更改 colums 文字
        select_df = pd.DataFrame(self.cv2_gui.result_point)
        select_df.columns = [f'Point {i}' for i in select_df.columns]

        if self.mode == 'line':
            for i, (dx, dy, d, s) in enumerate(zip(self.cv2_gui.result_dx.values(), self.cv2_gui.result_dy.values(), self.cv2_gui.result_distance.values(), self.cv2_gui.result_strain.values())):
                select_df.insert(i * 6 + 2, 'dx {} -> {}'.format(i * 2, i * 2 + 1), dx)
                select_df.insert(i * 6 + 3, 'dy {} -> {}'.format(i * 2, i * 2 + 1), dy)
                select_df.insert(i * 6 + 4, 'Distance {} -> {}'.format(i * 2, i * 2 + 1), d)
                select_df.insert(i * 6 + 5, 'Strain {} -> {}'.format(i * 2, i * 2 + 1), s)

        elif self.mode == 'ishan':
            line_num = len(self.cv2_gui.result_distance)
            for i, (d, s) in enumerate(zip(self.cv2_gui.result_distance.values(), self.cv2_gui.result_strain.values())):
                select_df.insert(line_num + 2 * i, f"Distance p{i} -> p{i+1 if i+1 != line_num else 0}", d)
                select_df.insert(line_num + 2 * i + 1, f"Strain p{i} -> p{i+1 if i+1 != line_num else 0}", s)



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
                                                     self.json_para['save_path'] + '/' + self.default_filename + '.png',
                                                     "All Files (*);;PNG Files (*.png)")

        # 如果沒有選擇存檔路徑，結束 function
        if not path:
            return

        self.json_para['save_path'] = os.path.split(path)[0]
        self.use_json('write')

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

    def clicked_btn_save_ani_curve(self, is_auto=''):
        # 如果不是從自動存檔呼叫的，需要選擇檔案路徑
        if is_auto != 'auto':
            # 如果尚未選擇影像，或是尚未運行 cv2，不運行按鈕
            if not self.filename or not self.cv2_gui: return

            fps, okPressed = QInputDialog.getInt(self, "Set output video FPS", "FPS:", self.json_para['video_fps'], 1,
                                                 10000, 1)

            if okPressed == False: return

            path, filetype = QFileDialog.getSaveFileName(self, "文件保存",
                                                         self.json_para[
                                                             'save_path'] + '/' + self.default_filename + '_' +
                                                         self.json_para['method'] + '_ani_curve.mp4',
                                                         "All Files (*);;MP4 Files (*.mp4)")

            if not path: return  # 如果沒有選擇存檔路徑，結束 function

            self.json_para['video_fps'] = fps
            self.json_para['save_path'] = os.path.split(path)[0]
            self.use_json('write')

        # 如果是自動存檔呼叫的，指定路徑
        else:
            path = self.json_para['auto_save_path'] + '/' + self.default_filename + '_' + self.json_para[
                'method'] + '_ani_curve.mp4'

        # fig, ax = plt.subplots()
        # plt.xlabel('frame')
        if self.json_para['curve_bound'] != 0:
            plt.ylim(-self.json_para['curve_bound'], self.json_para['curve_bound'])
            line, = self.ax.plot([0, 0], [-self.json_para['curve_bound'], self.json_para['curve_bound']], color='k',
                                 alpha=0.2)
        else:
            line, = self.ax.plot([0, 0], [0.4, -0.4], color='k', alpha=0.2)

        # 設定動態更新
        def animate(i):
            line.set_xdata([i, i])
            return line,

        # 設定初始狀態
        def init():
            line.set_xdata([0, 0])
            return line,

        # 建立 animation 物件，frames = 影像數量， interval = 更新時間（ms）
        ani = animation.FuncAnimation(fig=self.fig, func=animate, frames=self.num_of_img, init_func=init, interval=1000,
                                      blit=False)

        ani.save("temp.gif", writer='imagemagick', fps=20)
        gif = cv2.VideoCapture("temp.gif")
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

        path = self.json_para['auto_save_path'] + '/' + self.default_filename + '_' + self.json_para['method'] + '.all'
        if path.split('.')[-1] == 'all':
            path = path.split('.')[0]

        if self.json_para['auto_save_video']:
            # 儲存 mp4
            fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
            videowriter = cv2.VideoWriter(path + '.mp4', fourcc, self.json_para['video_fps'], (self.w, self.h))

            for img in self.cv2_gui.img_label:
                videowriter.write(img)
            videowriter.release()

        target_frame = int(self.spinBox_target_frame.value())
        cv2.imencode('.png', self.cv2_gui.img_label[target_frame])[1].tofile(path + f'_{target_frame}.png')

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

    # Auto save path
    def action_auto_save_path_triggered(self):
        dir_choose = QFileDialog.getExistingDirectory(self,
                                                      "選取資料夾",
                                                      self.json_para['auto_save_path'])  # 起始路径

        if dir_choose != "":
            self.json_para['auto_save_path'] = dir_choose
            self.use_json('write')

    # Open Setting file
    def action_open_setting_triggered(self):
        os.startfile(os.path.join(os.getcwd(), 'setting.json'))

    # Open Saved points file
    def action_open_saved_points_triggered(self):
        os.startfile(os.path.join(os.getcwd(), 'saved_points.json'))

    def action_multi_files_triggered(self):
        files, filetype = QFileDialog.getOpenFileNames(self, "選擇文件", self.default_path,  # 起始路径
                                                       "All Files (*);;Dicom Files (*.dcm)")

        if len(files) == 0: return

        self.textBrowser_target_frame.clear()

        # 依序解析檔案
        for file in files:
            self.clicked_btn_path(files=[file])
            self.clicked_btn_run(multi_mode=True)

    # 切換使用者的功能
    def action_user_tk_change(self):
        if self.action_user_tk.isChecked():
            self.action_user_yuwen.setChecked(False)
            self.action_user_ishan.setChecked(False)
            self.json_para['user'] = "tk"

            # 暫時借放
            self.menuYuwen.menuAction().setVisible(False)
        else:
            self.json_para['user'] = ""
        self.use_json(mode='write')

    def action_user_yuwen_change(self):
        if self.action_user_yuwen.isChecked():
            self.action_user_tk.setChecked(False)
            self.action_user_ishan.setChecked(False)
            self.json_para['user'] = "yuwen"

            self.menuYuwen.menuAction().setVisible(True)
        else:
            self.json_para['user'] = ""
            self.menuYuwen.menuAction().setVisible(False)
        self.use_json(mode='write')

    def action_user_ishan_change(self):
        if self.action_user_ishan.isChecked():
            self.action_user_tk.setChecked(False)
            self.action_user_yuwen.setChecked(False)
            self.json_para['user'] = "ishan"

            # 暫時借放
            self.menuYuwen.menuAction().setVisible(False)
        else:
            self.json_para['user'] = ""
        self.use_json(mode='write')


    def action_resize_input_triggered(self):
        if not self.filename:
            return

        scaling, okPressed = QInputDialog.getInt(self, "Scaling", "ratio (%):", 100, 10, 200, 1)
        new_h = (self.h * scaling)//100
        new_w = (self.w * scaling)//100

        self.IMGS = np.array([cv2.resize(img, (new_w, new_h)) for img in self.IMGS])
        self.img_preview = self.IMGS[0]
        self.h = new_h
        self.w = new_w
        self.scaling = scaling

    def radio_btn_line_change(self):
        if self.radioButton_line.isChecked():
            self.stackedWidget_mode.setCurrentIndex(0)
        else:
            self.stackedWidget_mode.setCurrentIndex(1)

    # 只要跟換圖片有關的動作
    def radio_btn_curve_change(self):
        if not self.filename or not self.cv2_gui:
            return

        if self.mode == 'line' or self.mode == 'ishan':
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

    def clicked_btn_save_points(self):
        with open('./saved_points.json', 'r') as f:
            saved_points = json.loads(f.read())

        name = os.path.split(self.json_para['path'])[-1] + '_' + str(self.date) + '_' + self.filename + self.extension
        name += "" if self.mode != "ishan" else "_ishan"
        saved_points[name] = self.textBrowser_labeled_points.toPlainText()

        with open('./saved_points.json', 'w') as f:
            print(json.dumps(saved_points, indent=4), file=f)

    def clicked_btn_console_clear(self):
        self.spinBox_target_frame.setValue(0)
        self.console_nb = ""
        self.console_after = ""
        self.textBrowser_target_frame.setText("")

    def clicked_btn_open_folder(self):
        os.startfile(self.default_path)

    def spinBox_target_frame_chane(self, x):
        if not self.cv2_gui:
            return
        self.plot_strain_curve(axv=x)

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
