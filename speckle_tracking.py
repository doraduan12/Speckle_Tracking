import cv2
import numpy as np
import time


class SpeckleTracking():

    def __init__(self, method='PPMCC'):
        if method == 'PPMCC':
            self.method = self.full_Correlation_coefficient
        elif method == 'SAD':
            self.method = self.full_SAD
        elif method == 'NCC':
            self.method = self.full_Cross_Correlation
        elif method == 'OF':
            self.method = self.optical_flow
            self.lk_params = dict(winSize=(17, 17),
                             maxLevel=2,
                             criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))



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

    def optical_flow(self, target: tuple, img1: np, img2: np, search_shift: tuple, temp_size: int) -> tuple:
        # Parameters for lucas kanade optical flow

        target = np.asarray(target, dtype='float32').reshape(-1, 1, 2)

        result, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, target, None, **self.lk_params)

        result_x, result_y = result.reshape(-1)

        return (result_x, result_y)


    def full_Correlation_coefficient(self, target: tuple, img1: np, img2: np, search_shift: tuple, temp_size: int) -> tuple:
        # cv2.matchTemplate 是從左上角點來匹配，並且不會新增 padding，因此要對search window 的角落多開半個 temp_size
        temp_bias = temp_size//2
        search_x, search_y = search_shift
        tx, ty = target

        template = img2[ty - search_y - temp_bias: ty + search_y + temp_bias,
                          tx - search_x - temp_bias: tx + search_x + temp_bias]

        target_img = img1[ty - temp_bias: ty + temp_bias, tx - temp_bias: tx + temp_bias]


        result = cv2.matchTemplate(template, target_img, cv2.TM_CCOEFF_NORMED)
        # cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        print(max_val)

        result_x, result_y = max_loc
        
        return (tx - search_x + result_x, ty - search_y + result_y)




    def full_Cross_Correlation(self, target: tuple, img1: np, img2: np, search_shift: tuple, temp_size: int) -> tuple:
        # cv2.matchTemplate 是從左上角點來匹配，並且不會新增 padding，因此要對search window 的角落多開半個 temp_size
        temp_bias = temp_size // 2
        search_x, search_y = search_shift
        tx, ty = target

        template = img2[ty - search_y - temp_bias: ty + search_y + temp_bias,
                   tx - search_x - temp_bias: tx + search_x + temp_bias]

        target_img = img1[ty - temp_bias: ty + temp_bias, tx - temp_bias: tx + temp_bias]

        result = cv2.matchTemplate(template, target_img, cv2.TM_CCORR_NORMED)
        # cv2.normalize(result, result, 0, 1, cv2.NORM_MINMAX, -1)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        result_x, result_y = max_loc

        return (tx - search_x + result_x, ty - search_y + result_y)




    # target, img1, img2, search_shift, temp_size
    def full_SAD(self, target: tuple, img1: np, img2: np, search_shift: tuple, temp_size: int) -> tuple:
        # 將原本位置紀錄為答案（以免找不到點）
        output = target
        tx, ty = target

        t_shift = temp_size//2

        now_window = img1[ty - t_shift: ty + t_shift, tx - t_shift: tx + t_shift]
        next_window = img2[ty - t_shift: ty + t_shift, tx - t_shift: tx + t_shift]

        cost_min = self.SSD(now_window, next_window)

        search_x, search_y = search_shift

        for i in range(-search_x, search_x):
            for j in range(-search_y, search_y):

                if i == 0 and j == 0: continue

                next_window = img2[ty + i - t_shift: ty + i + t_shift, tx + j - t_shift: tx + j + t_shift]

                cost = self.SSD(now_window, next_window)
                if cost < cost_min:
                    cost_min = cost
                    output = (tx + j, ty + i)

        return output





