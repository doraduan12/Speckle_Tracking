import matplotlib.pyplot as plt
import pydicom

import cv2
import numpy as np
import time
# from speckle_tracking import SpeckleTracking
from tools import Tools
import os


tool = Tools()

class Cv2Gui(Tools):

    def __init__(self, imgs:np, delta_x: float, delta_y: float, window_name: str,
                 temp_size: int=32, default_search: int=10):

        self.IMGS = imgs
        self.window_name = window_name

        self.current_page = 0
        self.default_search = default_search
        self.temp_size = temp_size

        self.delta_x = delta_x
        self.delta_y = delta_y
        self.delta = np.array([self.delta_x, self.delta_y])

        print("The shape of dicom is :", self.IMGS.shape)

        self.IMGS = tool.addPage(self.IMGS)
        self.img_label = np.copy(self.IMGS)
        self.num_of_img, self.h, self.w, _ = self.IMGS.shape

        # 點相關參數
        self.target_point = []  # -> tuple
        self.track_done = []
        self.roi_point = []  # -> list -> tuple
        self.roi_shift = []
        self.result_point = {}

        # 顯示
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar('No', self.window_name, 0, self.num_of_img - 1, self.track_change)
        cv2.imshow(self.window_name, self.img_label[self.current_page])
        cv2.waitKey(1)





    # 重置所有動作
    def reset(self):
        self.img_label = np.copy(self.IMGS)
        cv2.imshow(self.window_name, self.img_label[self.current_page])
        self.target_point = []
        self.track_done = []
        self.roi_point = []
        self.roi_shift = []
        self.result_point = {}

        print('Reseting complete.')


    # track bar 更動
    def track_change(self, x:int):
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
                self.current_page = self.photo_switch('next', self.current_page, self.num_of_img)
            elif flags > 0:
                self.current_page = self.photo_switch('last', self.current_page, self.num_of_img)

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
            cv2.line(temp_img, self.point1, (x, y), (0, 0, 255), thickness=1)

            # 計算距離、顯示距離的座標
            text_point, d = tool.count_distance(self.point1, (x, y), self.delta)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(temp_img, '{:4.3f}'.format(d), text_point, font, .5, (255, 255, 255), 1)

            # 刷新畫面
            cv2.imshow(self.window_name, temp_img)

        # 確定線段（左鍵放開時）
        elif event == cv2.EVENT_LBUTTONUP:
            if self.mouse_drag:
                self.mouse_drag = False  # 拖曳重置

                # 紀錄 point2 的點
                self.point2 = (x, y)

                # 作圖
                cv2.line(self.img_label[self.current_page], self.point1, self.point2, (0, 0, 255), thickness=1)
                cv2.circle(self.img_label[self.current_page], self.point1, 0, (0, 0, 255), thickness=2)
                cv2.circle(self.img_label[self.current_page], self.point2, 0, (0, 0, 255), thickness=2)

                # 計算距離 -> 尚未加入 List TODO
                text_point, d = tool.count_distance(self.point1, self.point2, self.delta)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(self.img_label[self.current_page], '{:4.3f}'.format(d), text_point, font, .5,
                            (255, 255, 255), 1)

                # 新增點參數
                self.target_point.extend([self.point1, self.point2])
                self.track_done.extend([False, False])

                # 計算預設的 search window
                x, y = self.point1
                s11, s12, _, _ = tool.get_search_window((x, y), (x + self.default_search//2, y+self.default_search//2), self.temp_size)
                x, y = self.point2
                s21, s22, _, _ = tool.get_search_window((x, y), (x + self.default_search//2, y+self.default_search//2), self.temp_size)

                self.roi_point.extend([[s11, s12], [s21, s22]])
                self.roi_shift.extend([(self.default_search // 2, self.default_search // 2), (self.default_search // 2, self.default_search // 2)])

                print(self.point1, self.point2)

                cv2.imshow(self.window_name, self.img_label[self.current_page])


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
            s1, s2, c1, c2 = tool.get_search_window(self.t_point, (x, y), self.temp_size)


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
                s1, s2, c1, c2 = tool.get_search_window((tx, ty), (x, y), self.temp_size)

                # 紀錄範圍
                self.roi_point[self.t_point_index] = [s1, s2]
                self.roi_shift[self.t_point_index] = (abs(x-tx), abs(y-ty))

                # 畫圖
                cv2.rectangle(self.img_label[self.current_page], s1, s2, (0, 0, 255), thickness=1)
                cv2.rectangle(self.img_label[self.current_page], c1, c2, (255, 255, 0), thickness=1)


                # 更新圖片
                cv2.imshow(self.window_name, self.img_label[self.current_page])




    def main(self):
        while True:
            cv2.setMouseCallback(self.window_name, self.click_event)  # 設定滑鼠回饋事件

            action = self.find_ACTION(cv2.waitKey(1))    # 設定鍵盤回饋事件

            # 「esc」 跳出迴圈
            if action == 'esc':
                break

            # 「r」 重置
            if action == 'reset':
                self.reset()

            # 「s」 執行 speckle tracking
            # if action == 'speckle':
            #     t1 = time.time()
            #     self.speckleTracking(record=True, show=True)
            #     t2 = time.time()
            #     print('Speckle Tracking costs {} seconds.\n'.format(t2 - t1))
            #     plt.show()


            # 「t」 增加預設點數（測試時用）
            # if action == 'test':
            #     print('add point')
            #     # dcm.addPoint((224, 217), (243, 114))
            #     dcm.addPoint((313, 122), (374, 292))

            # 按空白鍵查看點數狀況
            if action == 'space':
                print('dcm.target_point : ', dcm.target_point)
                print('dcm.track_done : ', dcm.track_done)
                print('dcm.roi_point : ', dcm.roi_point)
                print('dcm.roi_shift : ', dcm.roi_shift)
                print()


        cv2.destroyWindow(self.window_name) # （按 esc 跳出迴圈後）關閉視窗

if __name__ == '__main__':
    pass












