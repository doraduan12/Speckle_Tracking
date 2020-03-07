import pydicom
import cv2
import numpy as np

import sys
import os
import datetime
import time

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog, QMainWindow
from main_window import Ui_MainWindow

from cv2_gui import Cv2Gui
from tools import Tools
tool = Tools()

import cgitb
cgitb.enable( format = 'text')

# TODO Cost 方法 選擇器
# TODO 新增 target point 顯示視窗
# TODO 新增 點模式／線條模式

# TODO 新贓錨點??

class My_MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.setup()



    def setup(self):

        self.filename = ''

        # 按下 選路徑(btn_path) 按鈕
        self.btn_browse.clicked.connect(self.clicked_btn_path)

        self.btn_run.clicked.connect(self.clicked_btn_run)

        # 設定更新 spinbox 的動作
        self.spinBox_temp_size.valueChanged.connect(lambda x: self.show_preview_img(np.copy(self.img_preview), x, self.spinBox_search_range.value()))
        self.spinBox_search_range.valueChanged.connect(lambda x: self.show_preview_img(np.copy(self.img_preview), self.spinBox_temp_size.value(), x))





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
                self.filename = os.path.splitext(os.path.split(file)[-1])[0]

                dicom = pydicom.read_file(file)
                self.IMGS = dicom.pixel_array
                self.img_preview = self.IMGS[0]
                num_of_img, h, w = self.IMGS.shape[:3]

                filetype = '.dcm'

                # 讀取 dicom 的時間
                if 'InstanceCreationDate' in dir(dicom):
                    date = dicom.InstanceCreationDate
                    date = datetime.date(int(date[:4]), int(date[4:6]), int(date[6:]))
                if 'InstanceCreationTime' in dir(dicom):
                    time = dicom.InstanceCreationTime
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

            # 如果讀到圖檔
            elif extension == '.png' or extension =='.jpg' or extension == '.jpeg':
                browse_path = os.path.split(files[0])[0]
                self.filename = os.path.split(browse_path)[-1]

                # 排序圖檔
                files = np.asarray(files)
                temp = np.asarray([int(file.split('.')[0].split('/')[-1]) for file in files])
                temp = np.argsort(temp)
                files = files[temp]
                self.IMGS = np.asarray([cv2.imread(file) for file in files])
                self.img_preview = self.IMGS[0]
                num_of_img, h, w = self.IMGS.shape[:3]

                filetype = '.' + files[0].split('.')[-1]

                date, time, system_name = '', '', ''

                # TODO deltax, y = 0 的時候顯示 pixel 值
                # TODO 或是讓使用者畫線，換算 dx, dy
                deltax, deltay = 0, 0


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

            # 顯示超音波廠商、根據字數調整 label size
            self.label_manufacturer.setText(system_name)
            self.label_manufacturer.adjustSize()

            # 寫入 file detail 內容
            self.label_filetype_show.setText(filetype)
            self.label_image_size_show.setText(str(w) + ' x ' + str(h))
            self.label_date_show.setText(str(date))
            self.label_time_show.setText(str(time))
            self.label_frame_show.setText(str(num_of_img))

            self.doubleSpinBox_delta_x.setValue(deltax // 0.000001 / 1000)
            self.doubleSpinBox_delta_y.setValue(deltay // 0.000001 / 1000)


            # 預設的 template block 與 search window
            default_template = 32
            default_search = 10
            self.spinBox_temp_size.setValue(default_template)
            self.spinBox_temp_size.setRange(1, h//2)
            self.spinBox_search_range.setValue(default_search)
            self.spinBox_search_range.setRange(1, h//2)

            # 建立預覽圖片、自適化調整
            self.show_preview_img(np.copy(self.img_preview), default_template, default_search)

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

        kwargs = {
            'imgs': self.IMGS,
            'window_name': self.filename,
            'delta_x': float(self.doubleSpinBox_delta_x.value())/1000,
            'delta_y': float(self.doubleSpinBox_delta_y.value())/1000,
            'temp_size': int(self.spinBox_temp_size.value()),
            'default_search': int(self.spinBox_search_range.value()),
            'mode': 'sad'
        }

        cv2_gui = Cv2Gui(**kwargs)

        while True:
            cv2.setMouseCallback(cv2_gui.window_name, cv2_gui.click_event)  # 設定滑鼠回饋事件

            action = tool.find_action(cv2.waitKey(1))  # 設定鍵盤回饋事件

            # 「esc」 跳出迴圈
            if action == 'esc':
                break

            # 「r」 重置
            if action == 'reset':
                cv2_gui.reset()

            # 「s」 執行 speckle tracking
            if action == 'speckle':
                t1 = time.time()
                cv2_gui.speckle_tracking(show=False)
                t2 = time.time()
                print('Speckle Tracking costs {} seconds.\n'.format(t2 - t1))

            # 「t」 增加預設點數（測試時用）
            # if action == 'test':
            #     print('add point')
            #     # dcm.addPoint((224, 217), (243, 114))
            #     dcm.addPoint((313, 122), (374, 292))

            # 按空白鍵查看點數狀況
            if action == 'space':
                print('self.target_point : ', cv2_gui.target_point)
                print('self.track_done : ', cv2_gui.track_done)
                print('self.search_point : ', cv2_gui.search_point) # 目前沒用
                print('self.search_shift : ', cv2_gui.search_shift)
                print()

        cv2.destroyWindow(cv2_gui.window_name)  # （按 esc 跳出迴圈後）關閉視窗



    def show_preview_img(self, img, temp, search):
        h, w, _ = img.shape
        x, y = w//2, h//2

        t_shift = temp//2
        s_shift = search//2

        cv2.circle(img, (x, y), 1, (0, 0, 255), -1)

        cv2.rectangle(img, (x-s_shift, y-s_shift), (x+s_shift, y+s_shift), (0, 0, 255), 1)
        cv2.rectangle(img, (x-s_shift-t_shift, y-s_shift-t_shift), (x-s_shift+t_shift, y-s_shift+t_shift), (255, 255, 0), 1)

        self.label_preview.setPixmap(QtGui.QPixmap(self.convert2qtimg(img)))
        self.label_preview.setScaledContents(True)


    # 將cv2轉為 QImg
    def convert2qtimg(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, bytesPerComponent = img.shape
        bytesPerLine = 3 * width
        QImg = QtGui.QImage(img.data, width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
        return QImg




if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = My_MainWindow()
    window.show()
    sys.exit(app.exec_())
