import pydicom
import cv2
import numpy as np

import sys
import os
import datetime

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QFileDialog, QMainWindow
from main_window import Ui_MainWindow

from cv2_gui import Cv2Gui

import cgitb
cgitb.enable( format = 'text')


class My_MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.setup()



    def setup(self):
        # 按下 選路徑(btn_path) 按鈕
        self.btn_browse.pressed.connect(self.pressed_btn_path)

        self.btn_run.pressed.connect(self.pressed_btn_run)




    # 按下 選路徑(btn_path) 按鈕的動作
    def pressed_btn_path(self):
        # files, filetype = QFileDialog.getOpenFileNames(self, "選取資料夾").replace('/', '\\')     # 開啟選取檔案的視窗
        files, filetype = QFileDialog.getOpenFileNames(self,  "選擇文件", './', # 起始路径
                                                       "All Files (*);;Dicom Files (*.dcm);;Png Files (*.png);;JPEG Files (*.jpeg)")


        if len(files) > 0:
            # 如果讀取到 Dicom 檔

            if os.path.splitext(files[0])[-1].lower() == '.dcm':
                file = files[0]
                browse_path = file
                self.filename = os.path.splitext(os.path.split(file)[-1])[0]

                dicom = pydicom.read_file(file)
                self.IMGS = dicom.pixel_array
                img_preview = self.IMGS[0]
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
            else:
                browse_path = os.path.split(files[0])[0]
                self.filename = os.path.split(browse_path)[-1]

                # 排序圖檔
                files = np.asarray(files)
                temp = np.asarray([int(file.split('.')[0].split('/')[-1]) for file in files])
                temp = np.argsort(temp)
                files = files[temp]
                self.IMGS = np.asarray([cv2.imread(file) for file in files])
                img_preview = self.IMGS[0]
                num_of_img, h, w = self.IMGS.shape[:3]

                filetype = '.' + files[0].split('.')[-1]

                date = ''
                time = ''

                system_name = ''

                deltax = 0
                deltay = 0


            self.textBrowser_browse.setText(browse_path)

            # 顯示超音波廠商、根據字數調整 label size
            self.label_manufacturer.setText(system_name)
            self.label_manufacturer.adjustSize()

            self.label_filetype_show.setText(filetype)
            self.label_image_size_show.setText(str(w) + ' x ' + str(h))
            self.label_date_show.setText(str(date))
            self.label_time_show.setText(str(time))
            self.label_frame_show.setText(str(num_of_img))
            self.textBrowser_delta_x.setText(str(deltax // 0.001) + ' mm')
            self.textBrowser_delta_y.setText(str(deltay // 0.001) + ' mm')

            # 建立預覽圖片、自適化調整
            self.label_preview.setPixmap(QtGui.QPixmap(self.convert2qtimg(img_preview)))
            self.label_preview.setScaledContents(True)

        else:
            # TODO 輸入格式錯的視窗
            pass



    def pressed_btn_run(self):
        # TODO 按下 run

        kwargs = {
            'imgs': self.IMGS,
            'window_name': self.filename,
            'delta_x': 0,
            'delta_y': 0,
            'temp_size': 32,
            'default_search': 10
        }

        print(0.1)
        cv2_gui = Cv2Gui(**kwargs)
        print(0.2)
        cv2_gui.main()




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
