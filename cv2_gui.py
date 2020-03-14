import pydicom
import cv2
import numpy as np
import matplotlib.pyplot as plt

import time
import os

from speckle_tracking import SpeckleTracking
from tools import Cv2Tools
# from mouse_event import

cv2_tool = Cv2Tools()


class Cv2Gui():

    def __init__(self, imgs:np, delta_x: float, delta_y: float, window_name: str,
                 temp_size: int=32, default_search: int=10, cost: str='sad'):

        self.IMGS = imgs
        self.window_name = window_name

        self.current_page = 0
        self.default_search = default_search
        self.temp_size = temp_size

        self.delta_x = delta_x
        self.delta_y = delta_y
        self.delta = np.array([self.delta_x, self.delta_y])

        print("The shape of dicom is :", self.IMGS.shape)

        self.IMGS = cv2_tool.add_page(self.IMGS)
        self.IMGS_GRAY = np.asarray([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in self.IMGS])

        self.img_label = np.copy(self.IMGS)
        self.num_of_img, self.h, self.w, _ = self.IMGS.shape

        # 畫圖顏色
        self.color_index = 0
        self.num_of_color = 10
        self.colors = cv2_tool.color_iterater(x=self.num_of_color)
        self.current_color = self.colors[self.color_index % self.num_of_color]

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

        self.speckle_tracking = SpeckleTracking(cost_method=cost)



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
            cv2.line(temp_img, self.point1, (x, y), self.current_color, thickness=1)

            # 計算距離、顯示距離的座標
            text_point, d = cv2_tool.count_distance(self.point1, (x, y), self.delta)
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
                cv2.line(self.img_label[self.current_page], self.point1, self.point2, self.current_color, thickness=1)
                cv2.circle(self.img_label[self.current_page], self.point1, 0, self.current_color, thickness=2)
                cv2.circle(self.img_label[self.current_page], self.point2, 0, self.current_color, thickness=2)

                # 計算距離 -> 尚未加入 List
                text_point, d = cv2_tool.count_distance(self.point1, self.point2, self.delta)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(self.img_label[self.current_page], '{:4.3f}'.format(d), text_point, font, .5,
                            (255, 255, 255), 1)

                # 新增點參數
                self.target_point.extend([self.point1, self.point2])
                self.track_done.extend([False, False])

                # 計算預設的 search window
                x, y = self.point1
                s11, s12, _, _ = cv2_tool.get_search_window((x, y), (x + self.default_search//2, y+self.default_search//2), self.temp_size)
                x, y = self.point2
                s21, s22, _, _ = cv2_tool.get_search_window((x, y), (x + self.default_search//2, y+self.default_search//2), self.temp_size)

                self.search_point.extend([[s11, s12], [s21, s22]])
                self.search_shift.extend([(self.default_search // 2, self.default_search // 2), (self.default_search // 2, self.default_search // 2)])

                print(self.point1, self.point2)

                cv2.imshow(self.window_name, self.img_label[self.current_page])

                # 先將第一點的距離輸入結果
                self.result_distance[self.color_index] = [d]

                # 更新顏色
                self.color_index += 1
                self.current_color = self.colors[self.color_index % self.num_of_color]


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
                self.search_shift[self.t_point_index] = (abs(x-tx), abs(y-ty))

                # 畫圖
                cv2.rectangle(self.img_label[self.current_page], s1, s2, (0, 0, 255), thickness=1)
                cv2.rectangle(self.img_label[self.current_page], c1, c2, (255, 255, 0), thickness=1)


                # 更新圖片
                cv2.imshow(self.window_name, self.img_label[self.current_page])


    # 測試時方便建立線段
    def addPoint(self, point1, point2):
        # 作圖
        cv2.line(self.img_label[self.current_page], point1, point2, self.current_color, thickness=1)
        cv2.circle(self.img_label[self.current_page], point1, 2, self.current_color, thickness=-1)
        cv2.circle(self.img_label[self.current_page], point2, 2, self.current_color, thickness=-1)

        # 計算距離 -> 尚未加入 List TODO
        text_point, d = cv2_tool.count_distance(point1, point2, self.delta)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.img_label[self.current_page], '{:4.3f}'.format(d), text_point, font, .5,
                    (255, 255, 255), 1)

        # 新增點參數
        self.target_point.extend([point1, point2])
        self.track_done.extend([False, False])

        x, y = point1
        s11, s12, _, _ = cv2_tool.get_search_window((x, y), (x + self.default_search // 2, y + self.default_search // 2), self.temp_size)
        x, y = point2
        s21, s22, _, _ = cv2_tool.get_search_window((x, y), (x + self.default_search // 2, y + self.default_search // 2), self.temp_size)

        self.search_point.extend([[s11, s12], [s21, s22]])
        self.search_shift.extend([(self.default_search // 2, self.default_search // 2), (self.default_search // 2, self.default_search // 2)])

        cv2.imshow(self.window_name, self.img_label[self.current_page])

        # 先將第一點的距離輸入結果
        self.result_distance[self.color_index] = [d]

        self.color_index += 1
        self.current_color = self.colors[self.color_index]


    # 畫線的 Speckle Tracking
    def tracking(self, show=False):
        finish_already = True
        for j, (tp, s_shift, done) in enumerate(zip(self.target_point, self.search_shift, self.track_done)):

            # 如果該點完成，跳過該點
            if done: continue

            finish_already = False
            self.track_done[j] = True
            self.result_point[j] = [tp]


            color = self.colors[(j//2) % self.num_of_color]

            print('Now is tracking point{}.'.format(j + 1))

            result = tp

            # 從圖1開始抓出，當作 Candidate
            for i in range(1, self.num_of_img):
                # target, img1, img2, search_shift, temp_size
                result = self.speckle_tracking.full(result, self.IMGS_GRAY[i-1], self.IMGS_GRAY[i], s_shift, self.temp_size)
                self.result_point[j].append(result)

                cv2.circle(self.img_label[i], result, 2, color, thickness=-1)

                # 若運算的點為直線的第二端，開始畫線
                if j % 2 == 1:
                    # 抓出前次結果的點
                    p_last = self.result_point[j - 1][i]

                    # 畫線、計算（顯示）距離
                    cv2.line(self.img_label[i], p_last, result, color, thickness=1)
                    text_point, d = cv2_tool.count_distance(p_last, result, self.delta)
                    cv2.putText(self.img_label[i], '{:4.3f}'.format(d), text_point, cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                    self.result_distance[j//2].append(d)

                if show:
                    cv2.imshow(self.window_name, self.img_label[i])
                    cv2.waitKey(1)

        cv2.imshow(self.window_name, self.img_label[0])
        cv2.waitKey(1)


if __name__ == '__main__':
    pass


