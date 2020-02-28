import pydicom
import cv2
import numpy as np

import sys
import os
import datetime

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMainWindow
from main_window import Ui_MainWindow



class My_MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.setup()



    def setup(self):
        # 按下 選路徑(btn_path) 按鈕
        self.btn_browse.pressed.connect(self.pressed_btn_path)



    # 按下 選路徑(btn_path) 按鈕的動作
    def pressed_btn_path(self):
        # files, filetype = QFileDialog.getOpenFileNames(self, "選取資料夾").replace('/', '\\')     # 開啟選取檔案的視窗
        files, filetype = QFileDialog.getOpenFileNames(self,  "多文件选择", './', # 起始路径
                                    "All Files (*);;Dicom Files (*.dcm);;Png Files (*.png);;JPEG Files (*.jpeg)")


        # 如果讀取到 Dicom 檔
        if files[0].split('.')[-1].lower() == 'dcm':
            self.textBrowser_browse.setText(files[0])

            dicom = pydicom.read_file(files[0])
            img = dicom.pixel_array[0]
            num_of_img, h, w, _ = dicom.pixel_array.shape

            dt = dicom.AcquisitionDateTime
            dt = datetime.datetime(int(dt[0:4]), int(dt[4:6]), int(dt[6:8]), int(dt[8:10]), int(dt[10:12]), int(dt[12:]))

            # 讀取 delta x, delta y
            if 'SequenceOfUltrasoundRegions' in dir(dicom):
                self.DELTAX = dicom.SequenceOfUltrasoundRegions[0].PhysicalDeltaX
                self.DELTAY = dicom.SequenceOfUltrasoundRegions[0].PhysicalDeltaY

            elif 'PixelSpacing' in dir(dicom):
                self.DELTAX, self.DELTAY = dicom.PixelSpacing

            else:
                self.DELTAX, self.DELTAY = 0, 0


        # 如果讀到圖檔
        elif len(files) > 0:
            files = np.asarray(files)
            temp = np.asarray([int(file.split('.')[0].split('/')[-1]) for file in files])
            temp = np.argsort(temp)
            files = files[temp]

            img = cv2.imread(files[0])
            dt = ''
            h, w = img.shape[:2]
            num_of_img = len(files)
            self.DELTAX = 0
            self.DELTAY = 0

        self.textBrowser_image_size.setText(str(w) + ' x ' + str(h))
        self.textBrowser_datetime.setText(str(dt))
        self.textBrowser_frame.setText(str(num_of_img))
        self.textBrowser_delta_x.setText(str(self.DELTAX // 0.001) + ' mm')
        self.textBrowser_delta_y.setText(str(self.DELTAY // 0.001) + ' mm')


        # 建立預覽圖片、自適化調整
        self.label_preview.setPixmap(QtGui.QPixmap(self.convert2qtimg(img)))
        self.label_preview.setScaledContents(True)

    # TODO 完善讀圖的時間與可修改 dx, dy



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
