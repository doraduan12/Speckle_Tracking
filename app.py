import pydicom
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
matplotlib.use("Qt5Agg")  # 声明使用QT5
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

import sys
import os
import datetime
import time

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog, QMainWindow, QMessageBox
from main_window import Ui_MainWindow


from cv2_gui import *
from tools import GuiTools
gui_tool = GuiTools()

import cgitb
cgitb.enable( format = 'text')


# TODO 顏色轉換
# TODO 重製 temp 與 search 按鈕

# TODO 新增 target point 顯示視窗??
# TODO 新贓錨點??

class My_MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        # 先將顏色功能藏起來
        self.btn_color.hide()


        self.setup()


    def setup(self):

        # 初始化防止特定錯誤
        self.filename = ''
        self.cv2_gui = ''
        self.mode = ''

        # 按下 選路徑(btn_path) 按鈕
        self.btn_browse.clicked.connect(self.clicked_btn_path)

        # 設定更新 spinbox 的動作
        self.spinBox_temp_size.valueChanged.connect(self.spinBox_temp_changed)
        self.spinBox_search_range.valueChanged.connect(self.spinBox_search_changed)

        # 按下執行時的動作
        self.btn_run.clicked.connect(self.clicked_btn_run)

        # 按下 COLOR 轉換色彩
        self.btn_color.clicked.connect(self.clicked_btn_color)

        # 滑動 horizontal slide 的動作
        self.horizontalSlider_preview.valueChanged.connect(self.slide_change)

        # SAVE 相關
        # 按下儲存結果
        self.btn_save_result.clicked.connect(self.clicked_btn_save_result)
        self.btn_save_csv.clicked.connect(self.clicked_btn_ave_csv)

        #
        self.radioButton_line.toggled.connect(self.radio_btn_line_change)

        #
        self.radioButton_spline.toggled.connect(self.radio_btn_curve_change)
        self.radioButton_strain.toggled.connect(self.radio_btn_curve_change)


    # 按下 選路徑(btn_path) 按鈕的動作
    @pyqtSlot()
    def clicked_btn_path(self):
        # files, filetype = QFileDialog.getOpenFileNames(self, "選取資料夾").replace('/', '\\')     # 開啟選取檔案的視窗
        files, filetype = QFileDialog.getOpenFileNames(self,  "選擇文件", '../dicom/', # 起始路径
                                                       "All Files (*);;Dicom Files (*.dcm);;Png Files (*.png);;JPEG Files (*.jpeg)")

        if len(files) > 0:
            # 副檔名
            extension = os.path.splitext(files[0])[-1].lower()

            # 如果讀取到 Dicom 檔
            if extension == '.dcm':
                file = files[0]
                browse_path = file
                self.default_path, self.default_filename = os.path.split(browse_path)
                self.default_filename = self.default_filename.split('.')[0]
                self.filename = os.path.splitext(os.path.split(file)[-1])[0]

                dicom = pydicom.read_file(file)
                self.IMGS = dicom.pixel_array
                self.img_preview = self.IMGS[0]
                self.num_of_img, self.h, self.w = self.IMGS.shape[:3]

                filetype = '.dcm'

                date, time = '', ''

                # 讀取 dicom 的日期
                if 'InstanceCreationDate' in dir(dicom):
                    date = dicom.InstanceCreationDate
                    date = datetime.date(int(date[:4]), int(date[4:6]), int(date[6:]))
                elif 'StudyDate' in dir(dicom):
                    date = dicom.StudyDate
                    date = datetime.date(int(date[:4]), int(date[4:6]), int(date[6:]))

                # 讀取 dicom 的時間
                if 'InstanceCreationTime' in dir(dicom):
                    time = dicom.InstanceCreationTime
                    time = datetime.time(int(time[:2]), int(time[2:4]), int(time[4:]))
                elif 'StudyTime' in dir(dicom):
                    time = dicom.StudyTime
                    time = datetime.time(int(time[:2]), int(time[2:4]), int(time[4:]))


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

                elif 'PixelSpacing' in dir(dicom):
                    deltax, deltay = dicom.PixelSpacing

                else:
                    deltax, deltay = 0, 0


                # 讀取 dicom 的 fps
                if 'RecommendedDisplayFrameRate' in dir(dicom):
                    self.FPS = dicom.RecommendedDisplayFrameRate
                else:
                    self.FPS = 20



            # 如果讀到圖檔
            elif extension == '.png' or extension =='.jpg' or extension == '.jpeg':
                browse_path = os.path.split(files[0])[0]
                self.filename = os.path.split(browse_path)[-1]

                # 輸出影向預設的路徑與檔案名稱
                self.default_path = os.path.split(browse_path)[0]
                self.default_filename = self.filename

                # 排序圖檔
                files = np.asarray(files)
                temp = np.asarray([int(file.split('.')[0].split('/')[-1]) for file in files])
                temp = np.argsort(temp)
                files = files[temp]
                self.IMGS = np.asarray([cv2.imread(file) for file in files])
                self.img_preview = self.IMGS[0]
                self.num_of_img, self.h, self.w = self.IMGS.shape[:3]

                filetype = '.' + files[0].split('.')[-1]

                date, time, system_name = '', '', ''

                # TODO deltax, y = 0 的時候顯示 pixel 值
                # TODO 或是讓使用者畫線，換算 dx, dy
                deltax, deltay = 0, 0

                # 設定 fps
                self.FPS = 20


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
            self.label_date_show.setText(str(date))
            self.label_time_show.setText(str(time))
            self.label_frame_show.setText(str(self.num_of_img))

            self.doubleSpinBox_delta_x.setValue(deltax // 0.000001 / 1000)
            self.doubleSpinBox_delta_y.setValue(deltay // 0.000001 / 1000)

            # horizontalSlider_preview 設定最大值、歸零
            self.horizontalSlider_preview.setMaximum(len(self.IMGS) - 1)
            self.horizontalSlider_preview.setValue(0)

            # 預設的 template block 與 search window
            self.default_template = 32
            self.default_search = 10
            self.spinBox_temp_size.setValue(self.default_template)
            self.spinBox_temp_size.setRange(1, self.h//2)
            self.spinBox_search_range.setValue(self.default_search)
            self.spinBox_search_range.setRange(1, self.h//2)

            # 預設的 draw delay
            self.default_draw_delay = 20
            self.spinBox_drawing_delay.setValue(self.default_draw_delay)
            self.spinBox_drawing_delay.setRange(1, 100)

            # 建立預覽圖片、自適化調整
            self.show_preview_img(np.copy(self.img_preview), self.default_template, self.default_search)



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

        # 清除 strain curve 圖片
        self.label_show_curve.setPixmap(QtGui.QPixmap(""))


        # 判斷模式
        if self.radioButton_line.isChecked():
            mode = 'line'
        elif self.radioButton_draw.isChecked():
            mode = 'point'

        # 判斷 COST 方法
        if self.radioButton_SAD.isChecked():
            cost = 'sad'
        elif self.radioButton_SSD.isChecked():
            cost = 'ssd'


        kwargs = {
            'imgs': self.IMGS,
            'window_name': self.filename,
            'delta_x': float(self.doubleSpinBox_delta_x.value())/1000,
            'delta_y': float(self.doubleSpinBox_delta_y.value())/1000,
            'temp_size': int(self.spinBox_temp_size.value()),
            'default_search': int(self.spinBox_search_range.value()),
            'cost': cost,
            'draw_delay': int(self.spinBox_drawing_delay.value())
        }

        # 設定模式
        if self.radioButton_line.isChecked():
            self.mode = 'line'
            self.cv2_gui = Cv2Line(**kwargs)
        elif self.radioButton_draw.isChecked():
            self.mode = 'point'
            self.cv2_gui = Cv2Point(**kwargs)


        while True:
            cv2.setMouseCallback(self.cv2_gui.window_name, self.cv2_gui.click_event)  # 設定滑鼠回饋事件

            action = gui_tool.find_action(cv2.waitKey(1))  # 設定鍵盤回饋事件

            # 「esc」 跳出迴圈
            if action == 'esc':
                break

            # 「r」 重置
            if action == 'reset':
                # 清除 strain curve 圖片
                self.label_show_curve.setPixmap(QtGui.QPixmap(""))
                self.cv2_gui.reset()

            # 「s」 執行 speckle tracking
            if action == 'speckle':
                t1 = time.time()
                self.cv2_gui.tracking(show=True if self.checkBox_show_process.isChecked() else False)
                t2 = time.time()
                print('Speckle Tracking costs {} seconds.\n'.format(t2 - t1))

                if self.mode == 'line':
                    self.plot_strain_curve()
                elif self.mode == 'point':
                    pass

            # 「t」 增加預設點數（測試時用）
            if action == 'test':
                print('add point')
                if self.mode == 'line':
                    self.cv2_gui.addPoint((224, 217), (243, 114))
                    self.cv2_gui.addPoint((313, 122), (374, 292))
                elif self.mode == 'point':
                    self.cv2_gui.addPoint((224, 217))
                    self.cv2_gui.addPoint((243, 114))
                    self.cv2_gui.addPoint((224, 217))
                    self.cv2_gui.addPoint((374, 292))


            # 按空白鍵查看點數狀況
            if action == 'space':
                print('self.target_point : ', self.cv2_gui.target_point)
                print('self.track_done : ', self.cv2_gui.track_done)
                print('self.search_point : ', self.cv2_gui.search_point) # 目前沒用
                print('self.search_shift : ', self.cv2_gui.search_shift)
                print()

        cv2.destroyWindow(self.cv2_gui.window_name)  # （按 esc 跳出迴圈後）關閉視窗


    def plot_strain_curve(self):
        # 開始繪圖
        plt.figure()
        plt.axhline(0, color='k', alpha=0.2)
        plt.xlabel('frame')

        for i in self.cv2_gui.result_distance.keys():

            # 抓出對應的顏色，並轉呈 matplotlib 的 RGB 0-1 格式
            color = tuple([self.cv2_gui.colors[i][-j] / 255 for j in range(1, 4)])
            if self.radioButton_strain.isChecked():
                if self.radioButton_spline.isChecked():
                    plt.plot([i for i in range(self.num_of_img)], gui_tool.lsq_spline_medain(self.cv2_gui.result_strain[i]), color=color)
                elif self.radioButton_original.isChecked():
                    plt.plot([i for i in range(self.num_of_img)], self.cv2_gui.result_strain[i], color=color)

                plt.ylabel('Strain')
                plt.title('Strain curve')

            elif self.radioButton_distance.isChecked():
                if self.radioButton_spline.isChecked():
                    plt.plot([i for i in range(self.num_of_img)], gui_tool.lsq_spline_medain(self.cv2_gui.result_distance[i]), color=color)
                elif self.radioButton_original.isChecked():
                    plt.plot([i for i in range(self.num_of_img)], self.cv2_gui.result_distance[i], color=color)

                plt.ylabel('Distance')
                plt.title('Distance curve')

        # TODO 改善顯示流程

        plt.savefig(self.default_path + '/output.png')
        plt.close()

        # TODO 解決讀取中文路徑會出錯的問題
        self.result_curve_temp = cv2.imread(self.default_path + '/output.png')
        os.remove(self.default_path + '/output.png')

        self.label_show_curve.setPixmap(QtGui.QPixmap(gui_tool.convert2qtimg(self.result_curve_temp)))
        self.label_show_curve.setScaledContents(True)



    # TODO 待修正
    @pyqtSlot()
    def clicked_btn_color(self):
        if not self.filename:
            return

        # 將圖片從 YBR 轉成 BGR 通道
        self.IMGS = np.asarray([cv2.cvtColor(pydicom.pixel_data_handlers.util.convert_color_space(img, 'YBR_FULL', 'RGB'), cv2.COLOR_RGB2BGR) for img in self.IMGS])
        self.img_preview = self.IMGS[0]

        # 建立預覽圖片、自適化調整
        self.show_preview_img(np.copy(self.img_preview), self.default_search, self.default_search)

    # 存檔的按鈕
    def clicked_btn_save_result(self):
        # 如果尚未選擇影像，或是尚未運行 cv2，不運行按鈕
        if not self.filename or not self.cv2_gui:
            return

        path, filetype = QFileDialog.getSaveFileName(self, "文件保存", self.default_path + '/' + self.default_filename + '.mp4', "All Files (*);;MP4 Files (*.mp4)")

        # 如果沒有選擇存檔路徑，結束 function
        if not path:
            return

        # 強制轉換副檔名為 mp4
        filename = os.path.split(path)[-1]
        if filename.split('.')[-1] != 'mp4':
            path = path.split('.')[0] + '.mp4'

        # 開始寫入 mp4
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        videowriter = cv2.VideoWriter(path, fourcc, self.FPS, (self.w, self.h))

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
    def clicked_btn_ave_csv(self):
        # 如果尚未選擇影像，或是尚未運行 cv2，不運行按鈕
        if not self.filename or not self.cv2_gui:
            return
        print(self.cv2_gui.result_point)
        path, filetype = QFileDialog.getSaveFileName(self, "文件保存", self.default_path + '/' + self.default_filename + '.csv', "All Files (*);;CSV Files (*.csv)")

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
        self.show_preview_img(np.copy(self.img_preview), self.default_template, self.default_search)


    # 顯示預覽影像、自適化調整
    def show_preview_img(self, img, temp, search):
        x, y = self.w//2, self.h//2

        t_shift = temp//2
        s_shift = search//2

        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

        cv2.rectangle(img, (x-s_shift, y-s_shift), (x+s_shift, y+s_shift), (0, 0, 255), 1)
        cv2.rectangle(img, (x-s_shift-t_shift, y-s_shift-t_shift), (x-s_shift+t_shift, y-s_shift+t_shift), (255, 255, 0), 1)

        self.label_preview.setPixmap(QtGui.QPixmap(gui_tool.convert2qtimg(img)))
        self.label_preview.setScaledContents(True)


    def spinBox_temp_changed(self, x):
        self.default_template = x
        self.show_preview_img(np.copy(self.img_preview), self.default_template, self.default_search)

    def spinBox_search_changed(self, x):
        self.default_search = x
        self.show_preview_img(np.copy(self.img_preview), self.default_template, self.default_search)








if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = My_MainWindow()
    window.show()
    sys.exit(app.exec_())
