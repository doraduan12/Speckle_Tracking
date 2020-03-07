import cv2
import numpy as np
import time


class SpeckleTracking():

    def __init__(self, cost_method='sad'):

        cost_method = cost_method.lower()
        if cost_method == 'sad': self.cost_method = self.SAD
        elif cost_method == 'ssd': self.cost_method = self.SSD
        elif cost_method == 'census': self.cost_method = self.census()


    ############################################
    #               COST 方法
    ############################################
    SAD = lambda self, img1, img2: np.sum(np.abs(img1.astype('int') - img2.astype('int')))
    SSD = lambda self, img1, img2: np.sum(np.square(img1.astype('int') - img2.astype('int')))

    def census(self, img1, img2):
        h, w = img1.shape
        census1 = img1 < img1[h // 2, w // 2]
        census2 = img2 < img2[h // 2, w // 2]

        hamming = np.sum(np.logical_xor(census1, census2))

        return hamming
    ############################################
    ############################################


    # target, img1, img2, search_shift, temp_size
    def full(self, target: tuple, img1: np, img2: np, search_shift: tuple, temp_size: int) -> tuple:

        # 將原本位置紀錄為答案（以免找不到點）
        output = target
        tx, ty = target

        t_shift = temp_size//2

        now_window = img1[ty - t_shift: ty + t_shift, tx - t_shift: tx + t_shift]
        next_window = img2[ty - t_shift: ty + t_shift, tx - t_shift: tx + t_shift]

        cost_min = self.cost_method(now_window, next_window)

        search_x, search_y = search_shift

        for i in range(-search_x, search_x):
            for j in range(-search_y, search_y):

                if i == 0 and j == 0: continue

                next_window = img2[ty + i - t_shift: ty + i + t_shift, tx + j - t_shift: tx + j + t_shift]

                cost = self.cost_method(now_window, next_window)
                if cost < cost_min:
                    cost_min = cost
                    output = (tx + j, ty + i)

        return output





