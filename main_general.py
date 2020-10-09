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


def load_file(mw, files=None):   # mw = main window
    if files == None:
        files, filetype = QFileDialog.getOpenFileNames(mw, "選擇文件", mw.default_path,  # 起始路径
                                                       "All Files (*);;Dicom Files (*.dcm);;Png Files (*.png);;JPEG Files (*.jpeg)")

    # 如果有讀到檔案
    if len(files) == 0: return


    # 更新預設路徑
    mw.default_path = os.path.split(files[0])[0]
    mw.json_para['path'] = mw.default_path
    mw.use_json('write')

    # 副檔名
    mw.extension = os.path.splitext(files[0])[-1].lower()

    # 如果讀取到 Dicom 檔
    if mw.extension == '.dcm':
        file = files[0]
        browse_path = file
        mw.default_path, mw.default_filename = os.path.split(browse_path)
        mw.default_filename = mw.default_filename.split('.')[0]
        mw.filename = os.path.splitext(os.path.split(file)[-1])[0]

        dicom = pydicom.read_file(file)
        mw.IMGS = gui_tool.add_page(dicom.pixel_array)
        mw.img_preview = mw.IMGS[0]
        mw.num_of_img, mw.h, mw.w = mw.IMGS.shape[:3]

        filetype = '.dcm'

        mw.date, mw.time = '', ''

        # 讀取 dicom 的日期
        if 'InstanceCreationDate' in dir(dicom):
            mw.date = dicom.InstanceCreationDate
            mw.date = datetime.date(int(mw.date[:4]), int(mw.date[4:6]), int(mw.date[6:]))
        elif 'StudyDate' in dir(dicom):
            mw.date = dicom.StudyDate
            mw.date = datetime.date(int(mw.date[:4]), int(mw.date[4:6]), int(mw.date[6:]))

        # 讀取 dicom 的時間
        try:
            if 'InstanceCreationTime' in dir(dicom):
                mw.time = dicom.InstanceCreationTime
                mw.time = datetime.time(int(mw.time[:2]), int(mw.time[2:4]), int(mw.time[4:]))
            elif 'StudyTime' in dir(dicom):
                mw.time = dicom.StudyTime
                mw.time = datetime.time(int(mw.time[:2]), int(mw.time[2:4]), int(mw.time[4:]))
        except:
            mw.time = ''

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
            deltax, deltay = mw.json_para['delta_x'], mw.json_para['delta_y']

        # 讀取 dicom 的 fps
        if 'RecommendedDisplayFrameRate' in dir(dicom):
            mw.json_para['video_fps'] = dicom.RecommendedDisplayFrameRate
        else:
            pass


    # 如果讀到圖檔
    elif mw.extension == '.png' or mw.extension == '.jpg' or mw.extension == '.jpeg' or mw.extension == '.mp4' or mw.extension == '.avi':

        browse_path = os.path.split(files[0])[0]
        mw.filename = os.path.split(browse_path)[-1]

        # 輸出影向預設的路徑與檔案名稱
        mw.default_path = os.path.split(browse_path)[0]
        mw.default_filename = mw.filename

        if mw.extension == '.mp4' or mw.extension == '.avi':
            mw.filename = os.path.splitext(os.path.split(files[0])[-1])[0]
            capture = cv2.VideoCapture(files[0])
            ret, frame = capture.read()
            IMGS = []
            while ret:
                IMGS.append(frame)
                ret, frame = capture.read()

            capture.release()
            mw.IMGS = np.asarray(IMGS)

        else:
            # 排序圖檔
            files = np.asarray(files)
            temp = np.asarray([int(file.split('.')[0].split('/')[-1]) for file in files])
            temp = np.argsort(temp)
            files = files[temp]

            mw.IMGS = gui_tool.add_page(
                np.asarray([cv2.imdecode(np.fromfile(file, dtype=np.uint8), -1) for file in files]))
            if np.ndim(mw.IMGS) == 3:
                mw.IMGS = cv2.merge([mw.IMGS, mw.IMGS, mw.IMGS])

        mw.img_preview = mw.IMGS[0]
        mw.num_of_img, mw.h, mw.w = mw.IMGS.shape[:3]

        filetype = '.' + files[0].split('.')[-1]

        mw.date, mw.time, system_name = '', '', ''

        deltax, deltay = mw.json_para['delta_x'], mw.json_para['delta_y']


    # 如果讀入檔案不是 dicom 或是 圖檔
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
    mw.label_frame_show.setText(str(mw.num_of_img))

    # 更新 josn 參數
    mw.json_para['delta_x'] = deltax
    mw.json_para['delta_y'] = deltay
    mw.doubleSpinBox_delta_x.setValue(deltax)
    mw.doubleSpinBox_delta_y.setValue(deltay)

    # horizontalSlider_preview 設定最大值、歸零
    mw.horizontalSlider_preview.setMaximum(len(mw.IMGS) - 1)
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
    mw.spinBox_end.setRange(0, mw.num_of_img - 1)
    mw.spinBox_start.setValue(0)
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
        'main_window': mw,
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
    if mw.radioButton_line.isChecked():
        mw.mode = 'line'
        mw.cv2_gui = Cv2Line(**kwargs)

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

    elif mw.radioButton_draw.isChecked():
        mw.mode = 'point'
        mw.cv2_gui = Cv2Point(**kwargs)

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

                # 畫 curve，如果是郁文用的就自動偵測
                if mw.action_user_yuwen.isChecked():
                    # 計算位置
                    target_frame = gui_tool.find_best_frame(mw.cv2_gui.result_distance[1] if len(mw.cv2_gui.result_distance) > 1 else mw.cv2_gui.result_distance[0])

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
            elif mw.mode == 'point':
                pass

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






