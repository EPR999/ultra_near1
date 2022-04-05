import os
import cv2
import numpy as np
import scipy.ndimage as ndimage
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.rcParams["font.size"] = 20
import pandas as pd

#import cv2
import sys
import os.path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import sympy as sym
#from sympy import Array
#from numba import jit,autojit
from mpmath import *
import math
#import Jacobian_henon
import copy


a_orb = 0.3
b_orb = 0.30
dimension = 2
k = 3.0
xf = 1.2
xb = 1.0
T = 0.05
a_orb = 0.22#0.22
b_orb = 0.06#0.06
t = 0#60/180 * np.pi
stdrad = 0.08
e2 = k * xf * (np.exp(-8 * xb * (2 * xf - xb)) / (1 - np.exp(-32 * xf * xb)))


def dotV(x):
    return k * x * np.exp(-8 * x**2) - e2 * (np.exp(-8 * pow(x - xb, 2)) -  np.exp(-8 * pow(x + xb, 2)))
def U(qp):
     return np.array([ qp[0] + qp[1] - dotV(qp[0]) , qp[1] -  dotV(qp[0])  ])

#正の時間方向の写像
def U(qp):
    return np.array([qp[0] + T * (qp[1] - T *   dotV(qp[0]) ), qp[1]  - T  * dotV( qp[0] )])

#def U(qp):
#    return np.array([qp[0] + T * qp[1], qp[1]  - T  * dotV( qp[0] + T * qp[1] )])

#OVERLIDE
#def U(qp):
#    return qp[0] + qp[1] -  dotV(qp[0]), qp[1] - dotV(qp[0]) 

lamb = 1.2
def dotV(q):  #kicked potenal の時間微分
        return q + 2/lamb * np.sin(q/lamb)


class ScatteringMap: #made by R.Kogawa
    def __init__(self,xb,k,e2):
        self.xb = xb
        self.k = k
        self.e2 = e2
    def U(self,qp): #正の時間方向の写像
        return [qp[0] + qp[1] - 0.5 * dotV(qp[0]), qp[1] - 0.5 * dotV(qp[0]) - 0.5 * dotV(qp[0] + qp[1] - 0.5 * dotV(qp[0]))]
    def Ui(self,qp): #inverse of
        return [qp[0] - qp[1] - 0.5 * dotV(qp[0]), qp[1] + 0.5 * dotV(qp[0]) + 0.5 * dotV(qp[0] - qp[1] - 0.5 * dotV(qp[0]))]

def quantumnumber(scale):
    file_path = 'test{}.png'.format(str(round(scale,2)).replace(".",""))
    basename = os.path.splitext(os.path.basename(file_path))[0]
    ### 画像読み込み
    img = cv2.imread(file_path, 1)
    img = cv2.bitwise_not(img)
    # 画像の高さと幅を取得
    h, w, c = img.shape
    # 拡大(拡大することで輪郭がぼやける。このぼやけにより境界を識別しやすくする)
    scale = 3
    img_resize = cv2.resize(img, (w * scale, h * scale))

    ### 画像処理
    # グレースケールに変換
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    # ガウシアンによるスムージング処理（ぼかし）
    img_blur = cv2.GaussianBlur(img_gray, (5,5), 0)

    # 二値化と大津処理
    r, dst = cv2.threshold(img_blur,120, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # モルフォロジー膨張処理
    kernel = np.ones((3,3), np.uint8)
    dst = cv2.dilate(dst, kernel, iterations = 1)

    # 画像ファイルに保存
    cv2.imwrite(basename + '_thresholds.jpg', dst)


    # In[3]:


    # もし画像欠けがあった場合に塗りつぶす処理
    dst_fill = ndimage.binary_fill_holes(dst).astype(int) * 255
    cv2.imwrite(basename + '_thresholds_fill.jpg', dst_fill)


    # In[4]:


    # 境界を検出して描画する
    contours, _ = cv2.findContours(dst_fill.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)


    img_contour = cv2.drawContours(img_resize, contours, -1, (0,0,255), 1)
    cv2.imwrite(basename + '_counter.jpg', img_contour)


    # In[5]:


    # 面積、重心、輪郭長さを抽出する
    Areas = []
    with open(basename + '_data.csv', 'w') as f:
        if len(contours) > 3 : 
            return None
        for i, contour in enumerate(contours):
            # 面積
            area = cv2.contourArea(contour)
            area = area / 1000
            Areas.append(area)

            print(area)
            # 輪郭の重心
            M = cv2.moments(contour)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])  

            #輪郭（境界）の長さ
            perimeter = cv2.arcLength(contours[i],True)        
            
            # 画像に出力する
            #img2 = img_resize.copy()
            img2 = cv2.drawContours(img_resize, contours, i, (0, 0, 255), 3)
            cv2.putText(img2, str('{:.1f}'.format(area)), (cx, cy),
                        cv2.FONT_HERSHEY_PLAIN, # フォントタイプ
                        3, # 文字サイズ
                        (0, 0, 0), # 文字色：白(255, 255, 255)　黒(0, 0, 0)
                        2, # 文字太さ
                        cv2.LINE_AA)

            if i == (len(contours)-1):
                img2_resize = cv2.resize(img2, (w, h))
                cv2.imwrite(basename + '_' + str(i) + '.jpg', img2_resize)

            # csvファイルに保存
            if i == 0:
                my_columns_list = ['ID', 'Area', 'x_center_of_gravity', 'y_center_of_gravity', 'Perimeter']
                my_columns_str = ','.join(my_columns_list)
                f.write(my_columns_str + '\n')
            else:
                my_data_list = [str(i), str(area), str(cx), str(cy), str(perimeter)]
                my_data_str = ','.join(my_data_list)
                f.write(my_data_str + '\n')

        Area_sum = sum(Areas)
        hbar = 1
        print("n = ",1/hbar * ((Area_sum - 25.2965)/17614.807) * 3600 /(2 * np.pi )) 
        n = 1/hbar * ((Area_sum-25.2965)/17614.807) * 3600 /(2 * np.pi )
        return n
        print('Area_sum', Area_sum)


# In[6]:


    # ヒストグラム表示
    fig = plt.figure(figsize=(8,6))
    plt.title("histogram")
    plt.xlabel('Area')
    plt.ylabel('frequency')
    plt.tick_params()
    plt.grid()
    plt.hist(Areas, bins=10, rwidth=0.9) # binsは区分数
    plt.savefig(basename + '_histogram.jpg')

    plt.close()


    # In[7]:


    # 面積を割合で出力する
    df = pd.read_csv(basename + '_data.csv')
    df[r'Area[%]'] = df['Area'] / Area_sum * 100
    df.to_csv(basename + '_data_2.csv')
    df


    # In[8]:


    ### 累積分布図を表示する
    # ヒストグラムデータを抽出
    values, base = np.histogram(df['Area'], bins=40) # binsは区分数

    # 要素を足し合わせて、numpyアレイで出力する
    y = np.cumsum(values)

    # グラフをプロット
    fig = plt.figure(figsize=(8,6))
    plt.plot(base[:-1], y, color='black', linewidth = 4)

    # 以下、グラフオプション
    plt.xlabel('Area')
    plt.ylabel('p')
    #plt.ylim(0, 1)
    # 目盛り表記を強制的に整数にする
    plt.gca().get_yaxis().set_major_locator(ticker.MaxNLocator(integer=True))
    plt.grid()
    plt.title('Cumulative distribution')
    plt.savefig(basename + '_cumulative_distribution.jpg')
    plt.close()

def drawphase(scale):
    points = [np.array([])] * 2
    #point = np.array([ 0.95782, 0  ])#lambda = 3.0
    #point = np.array([ 0.809197, 0  ])#lambda = 1.2
    #point = np.array([5,0])
    #[1.06462819 1.37029825]    
#[0.06309614 0.99800745]
#angle between major axis and x axis= 1.5076582419255626
#angle between minor axis and x axis= -0.06313808486933388
    q  = np.random.random(1000) * scale
    p  = np.zeros_like(q) 
    point = np.array([q,p])
    for i in range(2500):
        #print(point)
        #print(i)
        point = U(point)
        points[0] = np.append(points[0] , point[0])
        points[1] = np.append(points[1] , point[1])
    q  = -np.random.random(1000) * 0.1 + scale
    p  = np.zeros_like(q) 
    point = np.array([q,p])

    for i in range(1000):
        #print(point)
        #print(i)
        point = U(point)
        points[0] = np.append(points[0] , point[0])
        points[1] = np.append(points[1] , point[1])
    print(type(points))
    fig = plt.figure(figsize = (14,14))
    ax = fig.add_subplot(111)
    #plot_ellipse_by_equation(ax, x[0] , x[1] , x[2])
    #plt.plot(points[0],points[1],'o')
    #matrix = np.array( [ [x[0], x[1]/2], [x[1]/2 , x[2] ] ])
    #eig_val, eig_vec = np.linalg.eig(matrix)
    #print(eig_val, eig_vec)
    #print(( 1/eig_val) ** 0.5 ) 
    #phase = np.loadtxt("phase_shearless_12.txt")
    plt.plot(points[0],points[1],",b")
    ax.set_xlim(-30,30)
    ax.set_ylim(-30,30)
    fig.subplots_adjust()
    #ax.get_xaxis().set_visible(False)
    #ax.get_yaxis().set_visible(False)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    plt.savefig("test{}.png".format(str(round(scale,2)).replace(".","")))
    #plt.show()
    plt.close()

def main():
    fig = plt.figure(figsize = (14,14))
    ax = fig.add_subplot(111)
    points = [np.array([])]  * 2
    data = np.arange(15, 16, 0.01)
    data = np.round(data, 1)
    numbers = np.array([])
    radius = np.array([])
    n = 0
    for i in data:
        drawphase(i)
        n = quantumnumber(i)
        print(np.abs(n - np.around(n, decimals=0, out=None)) )
        if n  != None:
            if  np.abs(n - np.around(n, decimals=0, out=None)) < 10 ** (-1) :
                q = i   
                p = 0
                qp = np.array([q,p])
                points = [np.array([])]  * 2
                for j in range(1,10000):
                    qp = U(qp)
                    points[0] = np.append(points[0],qp[0])
                    points[1] = np.append(points[1],qp[1])
                    #print(points)
                    ax.text(i,0,"{}".format(np.around(n,decimals = 0,out = None)))
                plt.plot(points[0],points[1],",k")        
                numbers = np.append(numbers,n)
                radius = np.append(radius,i)
                arrays = np.array([numbers,radius])
                np.savetxt("torus.txt",arrays)
    plt.show()

if __name__ == '__main__':
        main()

# In[ ]: