import numpy as np
from scipy.interpolate import make_lsq_spline, BSpline
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib import animation

class Tools():


    ACTION = {27: 'esc', 67: 'clear', 99: 'clear', 82: 'reset', 114: 'reset',
              26: 'last atcion', 66: 'back', 98: 'back', 84: 'test', 116: 'test',
              76: 'line', 108: 'line', 83: 'speckle', 115: 'speckle',
              32: 'space', 77: 'median filter', 109: 'median filter'}

    # 查詢按鍵指令
    def find_ACTION(self, key: int) -> str:
        # if key != -1:print(key)
        try:
            return self.ACTION[key]
        except:
            return None


    # 轉換 rectangle 的點為左上右下
    def pointConverter(self, point1: tuple, point2: tuple) -> (tuple, tuple):
        '''
        將矩陣點轉換為 左上、右下
        :param point1:
        :param point2:
        :return: p1（小點）, p2（大點）
        '''
        p1 = (min(point1[0], point2[0]), min(point1[1], point2[1]))
        p2 = (max(point1[0], point2[0]), max(point1[1], point2[1]))
        return p1, p2

    def countDist(self, point1: tuple, point2: tuple, delta: 'numpy') -> (tuple, tuple):
        '''
        計算兩點距離、標示點的位置
        :param point1: 任一一點
        :param point2: 任一一點
        :return: textPoint（點的位置）, d（距離）
        '''
        p1 = np.array(point1)
        p2 = np.array(point2)
        textPoint = tuple(((p1 + p2) / 2).astype('int'))
        d = np.sqrt(np.sum(np.power((p1 - p2) * delta, 2)))
        return textPoint, d

    def getROI(self, center: tuple, mouse: tuple, pLevel: int) -> (tuple, tuple, tuple, tuple):
        '''
        算出 ROI 範圍
        :param center:
        :param mouse:
        :param pLevel:
        :return:
        '''
        # ROI part
        startPoint = tuple(2 * np.asarray(center) - np.asarray(mouse))
        r1, r2 = self.pointConverter(startPoint, (mouse))

        # Search Window part
        pParameter = np.asarray([2 ** (pLevel - 1), 2 ** (pLevel - 1)])
        s1 = tuple(np.asarray(r1) - pParameter)
        s2 = tuple(np.asarray(r2) + pParameter)

        return r1, r2, s1, s2


    # 平滑曲線
    def lsq_spline(self, y:list, k:int = 3) -> list :
        '''
        任意階的 lsq spline
        :param y: 原始資料
        :param k: 近似為 P(k) 多項式， k = 3 即為 cube spline
        :return: 近似資料
        '''
        l = len(y)
        x = np.asarray([i for i in range(l)])
        t = np.linspace(0, l, num=5, dtype='int').reshape(-1)
        k = 25
        t = np.r_[(0,) * (k + 1), t[1:-1], (l-1,) * (k + 1)]

        spl = make_lsq_spline(x, y, t, k)

        return spl(x)



    # 平滑曲線
    def lsq_spline_medain(self, y: list, k: int = 3) -> list:
        '''
        只取中間開始變化處的 lsq spline
        :param y: 原始資料
        :param k: 近似為 P(k) 多項式， k = 3 即為 cube spline
        :return: 近似資料
        '''
        x = np.asarray([i for i in range(len(y))])
        y = np.asarray(y)
        max = np.max(y)
        min = np.min(y)
        stage = (max-min)/10

        start = np.where(np.abs(y-y[0]) > stage)[0][0]
        end = np.where(np.abs(y-y[-1]) > stage)[0][-1]

        start_split = y[:start]
        end_split = y[end+1:]
        median_split = y[start:end+1]

        l = len(median_split)
        x_split = np.asarray([i for i in range(l)])
        t = np.linspace(0, l, num=5, dtype='int').reshape(-1)
        t = np.r_[(0,) * (k + 1), t[1:-1], (l - 1,) * (k + 1)]

        spl = make_lsq_spline(x_split, median_split, t, k)

        output = np.hstack((start_split, spl(x_split), end_split))

        return output



    def spline2(self, y) -> (list, list):
        y = [i for num, i in enumerate(y) if num%5 == 0]
        l = len(y)
        x = np.asarray([i for i in range(l)])
        f = interp1d(x, y, kind='cubic')

        xnew = np.linspace(0, l - 1, num=l//2, endpoint=True)
        return xnew, f(xnew)



    # 顯示動態曲線
    def show_dynamic_curve(self, y, filename=None, fps=25) -> None:
        '''
        :param y: list，內容為 displacement 或 strain
        :param filename: 儲存的檔案名稱（.gif），如果沒填表布儲存
        :param fps: 儲存 gif 的 fps
        :return: None
        '''

        length = len(y) # 抓取資料長度
        x = np.asarray([i for i in range(length)])  # 還原 x 資料

        # 整理 y 資料
        y = np.asarray(y)
        max = np.max(y)
        min = np.min(y)

        # 創建子畫板
        fig, ax = plt.subplots()

        ax.plot(x, y)   # 畫資料曲線
        plt.axhline(0, color='k', alpha=0.2)    # 畫 y = 0 （ x 軸）

        # 設定 strain 的上下限
        bound = 0.3
        plt.ylim((-bound, bound))

        # 曲線的座標
        plt.xlabel('frame')
        plt.ylabel('strain')
        plt.title('Strain curve')

        # 設置追蹤線
        line, = ax.plot([0, 0], [-bound, bound], color='r', alpha=0.4)

        # 設定動態更新
        def animate(i):
            line.set_xdata([i, i])
            return line,

        # 設定初始狀態
        def init():
            line.set_xdata([0, 0])
            return line,

        # 建立 animation 物件，frames = 影像數量， interval = 更新時間（ms）
        ani = animation.FuncAnimation(fig=fig,
                                      func=animate,
                                      frames=length,
                                      init_func=init,
                                      interval=1000,
                                      blit=False)

        # 如果有設定檔名，則開始存檔
        if filename:
            ani.save(filename, writer='imagemagick', fps=fps)



if __name__ == '__main__':
    pass
