import matplotlib.pyplot as plt
import pydicom
import cv2
import numpy as np
import time
# from speckle_tracking import SpeckleTracking
from tools import Tools
import os



class Cv2Gui():

    def __init__(self, imgs:np, delta_x: float, delta_y: float, window_name: str,
                 temp_size: int=32, default_search: int=10):
        self.IMGS = imgs
        self.window_name = window_name

        self.current_page = 0
        self.search_range = default_search
        self.temp_size = temp_size

        self.delta_x = delta_x
        self.delta_y = delta_y
        self.delta = np.array([self.delta_x, self.delta_y])

        print("The shape of dicom is :", self.IMGS.shape)

        self.IMGS = self.addPage(self.IMGS)
        self.img_label = np.copy(self.IMGS)
        self.num_of_img, self.h, self.w, _ = self.IMGS.shape

        # 點相關參數
        self.target_point = []  # -> tuple
        self.track_done = []
        self.roi_point = []  # -> list -> tuple
        self.roi_shift = []
        self.search_point = []  # -> list -> tuple 目前沒用
        self.result_point = {}

        # 顯示
        cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
        cv2.createTrackbar('No', self.window_name, 0, self.num_of_img - 1, self.track_change)
        cv2.imshow(self.window_name, self.img_label[self.current_page])
        cv2.waitKey(1)




    # 將 Dcm 檔影像加上圖片編號
    def addPage(self, imgs: np) -> np:
        for i in range(len(imgs)):
            s = str(i) + '/' + str(len(imgs) - 1)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(imgs[i], s, (0, 15), font, .5, (255, 255, 255), 2)
        return imgs

    # 重置所有動作
    def reset(self):
        self.img_label = np.copy(self.IMGS)
        cv2.imshow(self.window_name, self.img_label[self.current_page])
        self.target_point = []
        self.track_done = []
        self.roi_point = []
        self.roi_shift = []
        self.search_point = []
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

    def click_event(self):
        pass

    def main(self):

        while True:
            print(1)
            cv2.setMouseCallback(self.window_name, self.click_event)  # 設定滑鼠回饋事件
            print(1.2)
            # TODO 疑似 pyqt5 無法進行外部無窮迴圈
            action = t.find_ACTION(cv2.waitKey(1))    # 設定鍵盤回饋事件

            print(2)
            # 「esc」 跳出迴圈
            if action == 'esc':
                break

            print(3)
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
                print('dcm.search_point : ', dcm.search_point)
                print()


        cv2.destroyAllWindows() # （按 esc 跳出迴圈後）關閉視窗

if __name__ == '__main__':
    hello = Cv2_gui('')













