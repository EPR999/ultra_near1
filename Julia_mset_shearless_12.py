import sys
import os.path
#sys.path.append(r"/Users/koda/PyModules/SimpleQmap")
#sys.path.append(r"/Users/koda/PyModules")
#sys.path.append(r"C:\PyModules\SimpleQmap")
#sys.path.append(r"C:\PyModules")
import numpy as np
import matplotlib.pyplot as plt
#import SimpleQmap as sq
#from maps import StandardMap as st
from mpl_toolkits.mplot3d import Axes3D
#import sympy as sym
#from sympy import Array
#from numba import jit,autojit
import math
import time
#import psutil
#import plot2
from numpy.fft import fftfreq, fft, ifft
import scipy.special
import scipy
import numba as nb
#import jax.numpy as jnp
#import jax
from functools import partial
#from jax.experimental import loops
from matplotlib.colors import Normalize
from tqdm import tqdm
from math import factorial
import math
#from google.colab import files
import scipy

twopi = 2.0*np.pi
omega = 1
initp = 0
imsize = 3
resize = 5
reorigin = 0.0
imorigin = 0.0 #Origin on the complex surface is (reorigin,imorigin)
realscale =1
imscale = 1 #(realscale,imscale)=(1,1) is used as usual.
step = 9
orbit = [np.array([])] * 2
orbits = [[],[]] * 2
diskcount = True
originseed = []
p_hist = []
origin = np.array([])
arg2 = 0
switch = False
switch3 = False
radius = 0.0001
stradius = 0.001#普通の半径
cradius = 0.01#普通の半径にするためのもの
orbitnumber = 17
curvecount = 0
curve_smooth =  10
diskgrid = 200
a_orb =  0.20334038#0.05027134 0.20334038
b_orb =0.05027134 
t = 1.0740275527854348
timestep = step
stdrad = 0.08#初期面の半径.
stat_mset2 = False
#
#
a_orb = 0.44511123
b_orb = 0.99929173
t = 1.5331572184
k = 3.0
xf = 1.2
xb = 1.0
e2 = k * xf * (np.exp(-8 * xb * (2 * xf - xb)) / (1 - np.exp(-32 * xf * xb)))
a_orb = 8
b_orb = 8
t = 0
#
hbar = 1
b_orb = 0.44515836
a_orb = 0.57929427
t = 1.5096941616
tau = 0.05
b_orb = 1.04079259
a_orb = 1.10344608
t = 1.337145721
lambda_1 = 1.2
if lambda_1 == 1.2:
    b_orb,a_orb = [0.95545155 , 1.45685233]
    t =  1.5269150456843639
    text = "phase_shearless_12.txt"
elif lambda_1 == 3.0:
    b_orb,a_orb = [0.95610473, 1.06019203]
    t = 1.4357197061513873
    text = "phase_shearless_30.txt"

#[ 1.09022468 -0.05451134  0.89337822]#\lambda = 3.0の場合
#[ 1.09022468 -0.05451134  0.89337822]
#[1.09392884 0.88967406] [[ 0.99089102  0.13466623]
# 0.13466623 0.99089102
#angle between major axis and x axis= 1.4357197061513873
#angle between minor axis and x axis= -0.1350766206435093



#================================================
#theta,q_im(初期条件) →　q,pの実部
def init(theta):#角の情報を座標の情報に変換する,
    q = a_orb * np.sin(theta)
    p = b_orb * np.cos(theta)
    return np.array([q,p]) #初期運動量もthetaとともに確定する.
#===================================================
def I(point):#Iはqです
     return point[0]  



def cp(point):#複素面上の点が与えたれたら複素数をかえす
    z = point[0] + 1j * point[1]
    return z
def arginit(point,arg,radius):
    x = point[0] + radius * np.cos(arg)
    y = point[1] + radius * np.sin(arg)
    return np.array([x,y])
#=================================================
def arginitcp(point,radius):
    x = point[0] + radius * np.cos(arg)
    y = point[1] + radius * np.sin(arg)
    z = cp(x,y)
    return z

def dotV(x):
    return k * x * np.exp(-8 * x**2) - e2 * (np.exp(-8 * pow(x - xb, 2)) - np.exp(-8 * pow(x + xb, 2)))


def dotV(x):
    return x + 2/lambda_1 * np.sin(x/lambda_1)

def V(x):
    return np.array(x) ** 2/2 - 2 * np.cos(x/lambda_1) 

class KScatteringMap:
    def __init__(self,k,xf,xb):
        self.k = k
        self.xf = xf
        self.xb = xb
    def ScattMapt(self,qp):
        for i in range(step):
            qp[0],qp[1] =  U(qp)
        return qp[0],qp[1]
        #正の時間方向の写像
    def U(self,qp):
        return np.array([qp[0] + tau * qp[1] - (tau ** 2) *  dotV(qp[0]) , qp[1] -  tau  * dotV(qp[0])  ])
    
    #逆写像
    def Ui(self,qp):
            return np.array([qp[0] - qp[1] - 0.5 * dotV(qp[0]), qp[1] + 0.5 * dotV(qp[0]) + 0.5 * dotV(qp[0] - qp[1] - 0.5 * dotV(qp[0]))])

class KScatteringMap2:
    def __init__(self,k,xf,xb):
        self.k = k
        self.xf = xf
        self.xb = xb
    #正の時間方向の写像
    def U(self,qp):
        return np.array([qp[0] + tau  * qp[1] - (tau ** 2) * dotV(qp[0]) , qp[1] - tau * dotV(qp[0])  ])
    
    #逆写像
    def Ui(self,qp):
        return np.array([qp[0] - qp[1], qp[1] -  dotV(qp[0] -  qp[1]) ])


def ScattMapt(qp,step,cmap):
    for i in range(step):
        qp[0],qp[1] =  cmap.U(qp)
    return np.array([qp[0],qp[1]])
def ScattMapti(qp,step,cmap):
    for i in range(step):
        qp[0],qp[1] =  cmap.Ui(qp)
    return np.array([qp[0],qp[1]])

def tMap( z,step,func):#複素数zを与えたら変数を買える様に進化させた．このプログラムいじり倒して遊んでやろうと思っている。
    qp = init(z)
    qp_rotated = rotateinitialmanifold(a_orb,b_orb,t,qp)
    for i in range(step):
        qp_rotated[0],qp_rotated[1] = func(qp_rotated)
    return np.array([qp_rotated[0],qp_rotated[1]])

def rotateinitialmanifold(a,b,t,points):
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    R = np.array([[cos_t,-sin_t],[sin_t,cos_t]])
    rotated = np.dot(R,points)
    return rotated


#--------------------------------------------------------------------
#以下はクリックした時の挙動.
#--------------------------------------------------------------------
def onclick(event):#右クリック二回で取得
    global stat
    global x1,y1
    global x_pres,y_pres
    global x1,y1
    global xy
    global origins
    x1,y1 = event.xdata,event.ydata
    

def on_press(event):
    global stat_mset2
    if event.key == "d":
        plt.title("get mode ")
        stat_mset2 =  True


def mset2_Release(event):
    global x1_mset,y1_mset,x2_mset,y2_mset,stat_mset
    x2_mset = event.xdata
    y2_mset = event.ydata
    if event.button == 2:
       # exit()
        mset2(x1_mset,x2_mset,y1_mset,y2_mset)
    stat_mset = False 

def mset2_onclick(event):
    global x1_mset,y1_mset
    global stat_mset
    print(event.button)
    if event.button == 2 :
        x1_mset = event.xdata
        y1_mset = event.ydata
        #stat_mset = True
#--------------------------------------------------------------------
#以下は軌道を取得する部分.f
def action2(x_cross):
    global text
    S = 0
    i = 0
    x_cross0 = x_cross
    prepoint  = x_cross
    theta_0 = gettheta(x_cross0)
    #print(x_cross)
    #print(period)
    fig = plt.figure(figsize = (16,12))
    fig2 = plt.figure(figsize = (18,12))
    ax = fig.add_subplot(111,projection = "3d")
    ax2 = fig2.add_subplot(111)
    plt.subplots_adjust(wspace=0.6, hspace=2)
    actions = np.array([])
    iterates = np.array([])
    import copy 
    initpoint = np.array([x_cross])
    #initpoint[0] = np.append(initpoint[0],x_cross[0])
    #initpoint[1] = np.append(initpoint[1],x_cross[1])
    initpoint = np.array([x_cross,x_cross])
    print(initpoint)
    #ax2 = fig.add_subplot(122)
    plt.ylabel("Re(S)",fontsize = 27)
    plt.xlabel("Time step",fontsize = 27)
    ax.set_xlabel(r"${\rm Re}(q)$",labelpad = 20)
    ax.set_ylabel(r"${\rm Re}(p)$",labelpad = 20)
    ax.set_zlabel(r"${\rm Im}(q)$",labelpad = 30)
    #$plt.ylim(-100,100)
    #plt.ylim(0,0.1)
    colormap = plt.get_cmap("tab10")
    points = [np.array([])] * 2
    distancefrom_real = np.array([])
    while i <4000:
        points[0] = np.append(points[0], x_cross[0])
        points[1] = np.append(points[1], x_cross[1])
        iterates = np.append(iterates, i)
        actions = np.append(actions, S.real)
        #print("actions = ",actions)
        i +=  1
        #jacob = Jacobian_henon.strictJ2(cmap.U,periodicpoint,period)
        #w,v = np.linalg.eig(jacob)
        #wabs0 = w[0] * np.conj(w[0])
        #wabs1 = w[1] * np.conj(w[1])
        #if wabs0 > wabs1:
        #    tmp = wabs1
        #    wabs1 = wabs0
        #    wabs0 = tmp
        #print("stability =",wabs1 ** 0.5)
        #plt.title("stability = {}".format(str(round(wabs1 ** 0.5, 4))))
        distancefrom_real = np.append(distancefrom_real, np.sqrt(x_cross[0].imag ** 2 + x_cross[1].imag ** 2 ) )
        prepoint = x_cross
        theta = gettheta(x_cross0)
        preI = x_cross[0] ** 2 + x_cross[1] ** 2
        prethetaforsin = CASIN( x_cross[1] / (preI ** 0.5) )
        #print(x_cross)
        x_cross =  cmap.U(x_cross)
        #theta = gettheta(x_cross0)
        #print(theta)
        #print("x_cross =",x_cross)
        #print(period)
        # S += 0.5 * x_cross[1] ** 2 + V(x_cross) - x_cross[1] * ( - 0.5 * dotV(x_cross[0]) - 0.5 * dotV(x_cross[0] + x_cross[1] - 0.5 * dotV(x_cross[0]) ) )#終わりから始めを引け
        S += (0.5 * tau * (x_cross[1] ** 2) + tau * V(prepoint[0]) - x_cross[1] * ( x_cross[0] - prepoint[0] ) )
        #actioans = np.append(actions, S.imag)
        #print("S +=",0.5 * (x_cross[1] ** 2) + V(prepoint) - x_cross[1] * ( x_cross[0] - prepoint[0] ) )
        #print("S = ",S)
        if i == 1:
            #S -= V(prepoint)
            S -= generatorfunc2( theta_0)
            #S -= generatorfunc1(preI,prethetaforsin)
        #iterates = np.append(iterates, i)
        #print(i)
        #print('x_cross = ',x_cross)
        #print(np.linalg.norm(x_cross - periodicpoint))    
    ax2.plot(iterates,actions,"-o",color = "red")
    ax2.plot(iterates[0],actions[0],"ro",color = "red",markersize = 12)
    periodicpoint = np.loadtxt("complex_periodicpoint_12.txt",dtype = np.complex128)

    #ax.plot(periodicpoint[:,0].real,periodicpoint[:,1].real,periodicpoint[:,0].imag,"o",color = "purple")
    print(iterates.shape)
    ax.plot(points[0].real , points[1].real , points[0].imag,  "-o")
    ax.plot(initpoint[:,0].real,initpoint[:,1].real,initpoint[:,0].imag,"ro",markersize = 12)
    #boolean = np.abs(points[0].imag ) < 0.1
    boolean = ( np.diff( np.sign(points[0].imag)  ) != 0 )
    boolean = np.append(boolean,False)
    print(boolean.shape)
    points_array = np.array(points)
    near_real = points_array[:,boolean]
    phase = np.loadtxt(text)
    fig2 = plt.figure(figsize = (12,12))
    ax_proj =  fig2.add_subplot(111)
    main3(ax_proj)
    ax_proj.set_xlabel(r"${\rm Re}(q)$")
    ax_proj.set_ylabel(r"${\rm Re}(p)$")
    ax_proj.set_xlim( -60,60)
    ax_proj.set_ylim( -60,60)
    ax2.set_ylim(-200,200)
    xi_str = str(round(theta_0.real,6))
    eta_str = str(round(theta_0.imag,6))
    ax.set_title("xi = {},eta = {}".format(xi_str, eta_str))
    #if lambda_1 == 3.0:
    #   ax_proj.set_xlim( -30,30)
    #    ax_proj.set_ylim( -30,30)
    ax_proj.plot(phase[0],phase[1],",",zorder = 0)
    ax_proj.plot(initpoint[:,0].real,initpoint[:,1].real,"ro",markersize = 12)
    for i in range(len(near_real[0])):
        ax_proj.plot(near_real[0][i].real , near_real[1][i].real, "o", color = colormap(2),markersize = 10,zorder = 2)
        #ax.plot(near_real[0].real , near_real[1].real, near_real[0].imag,color = "o",markersize = 10)
        ax2.plot(iterates[boolean][i],actions[boolean][i],"o",markersize = 10,color = colormap(2))
    iterates_str = np.array(iterates[boolean],dtype = np.int32)
    print(iterates)
    iterates_bool = iterates[boolean]
    actions_bool = actions[boolean]
    argminimum  = np.argmin(distancefrom_real)
    ax2.plot(iterates[argminimum],distancefrom_real[argminimum],"o",markersize= 12)
    points = np.array(points)
    boolean_dist =  distancefrom_real < distancefrom_real[argminimum] + 0.3
    print(boolean_dist[boolean_dist == True])
    ax_proj.plot(points[0,boolean_dist ].real, points[1,boolean_dist].real,"o",markersize= 12,zorder = 100,color = "blue")
    for i in range(len(iterates_str)):
        ax2.text(iterates_bool[i]+1,actions_bool[i] + 1,iterates_str[i],fontsize = 20)
    #print(near_real[:,0])
    ax_proj.plot(points[0].real, points[1].real, "-o",zorder = 1)
    plt.show()
#----------------------------------------------------------------------------
def Release():#Get x1,y1(initialoriginx,initialoriginy) Iはp
    global x1,y1,x2,y2
    global seed
    global orbit,switch
    radius = 0.001
    maxI = 60
    counter = False
    I_hist = np.array([])
    I_saved = 10000
    theta_hist = []
    origin_hist = [np.array([])] * 2
    orbit = [np.array([])] * 2
    switch  = False
    #x1...theta/y...q_im
    origin = np.array([x1,y1]).T
    pradius = radius
    theta0 = cp(origin)
    start = time.time()
    func = cmap.U
    print(theta0)

    #最初の原点を求める．
    radius,origin = initialdisksize(theta0,radius,func)
    print(radius,origin)
    #exit()
    theta_hist = theta_hist.extend([theta0])

    print(origin)
    originsave(origin,origin_hist)
     
        
    origin0 = origin #args0[0]の方向のpと比較して向きを定める．

    print(origin0)

    theta0 = cp(origin)#一旦進んだ後に戻るべき原点．
    
    theta =  theta0
    
    originsave(origin,origin_hist)#原点の保存
    
    I_real = I(origin).real

    I_hist = np.append(I_hist,I_real)
    
    print(origin0)

    #ここまできたら半径を小さくしても良い．
    #radius  = radius 
    print("radius = ",radius)
    i  = 0 
    while abs(I_real) < maxI  :#でかい方向
        i += 1
        print(theta)
        
        preradius = radius #新しいチェック機能．


        radius,origin = initialdisksize(theta,radius,func)
        print("aradius = ",radius)

        if np.any(origin) == None or radius == None:
            print('this')
            break

        originsave(origin,origin_hist)

        mappedorigin = tMap(theta,step,func)

        theta = cp(origin)
        
        plt.plot(origin[0],origin[1],'.',color = 'red')


        if radius  <  preradius and counter == False: #ここからもう一回とれ． 
            I_saved = I_real
            print("back")
            switch = not switch 
            counter = not counter
            radius = radius/3
    #print(switch)
        I_real = I(mappedorigin).real

        if I_saved < I_real and counter == True :#だんだんI_realが小さくなるはずなので必要以上に小さくなってはならない．
            print("forward")
            print(I_saved)
            print(I_real)
            switch = not switch 
            counter = False
            radius = radius * 3

        I_hist = np.append(I_hist,I_real)


    #逆に向かう方向のもの. 
    print("reverse")
    switch = True
     
    radius = cradius/5
    
    radius,origin = initialdisksize(theta0,radius,func)
    print("aradius = ",radius)
    originsave(origin,origin_hist)
    
    theta = cp(origin)

    mappedorigin = tMap(theta,step,func)

    I_real = I(mappedorigin).real

    print(I_real)

    while abs(I_real) < maxI :
        print(I_real)
        radius,origin = initialdisksize(theta,radius,func)
        
        if np.any(origin) == None or radius == None:
            break
        print("aradius = ",radius)
        #現在の原点の位置を保存しておく.後でpの増減の符号が変わらない様に取らせる.減少か増加いずれかの方向に単調に動く様に
        #次の原点を決める角の決定を行う.
        
        originsave(origin,origin_hist)

        mappedorigin = tMap(theta,step,func)

        theta = cp(origin)
        
        plt.plot(origin[0],origin[1],'.',color = 'red')

        if radius * 5 <  preradius and counter == False: #ここからもう一回とれ． 
            I_saved = I_real
            print("back")
            switch = not switch 
            counter = not counter
            print(switch)
        
        I_real = I(mappedorigin).real

        if I_saved > I_real and counter == True :#だんだんI_realが小さくなるはずなので必要以上に小さくなってはならない．
            print("foward")
            print(I_saved)
            print(I_real)
            switch = not switch 
            counter = False

    print('end')
    a_str = format(str(round(a_orb,3))).replace('.','')
    b_str = format(str(round(b_orb,3))).replace('.','')
    number = 3
    
    with open('orbitdistortion_a{}_b{}_{}_{}.txt'.format(a_str,b_str,step,number),'w') as o_hand:
        vec = np.array([origin_hist[0],origin_hist[1]])
        vec = vec.T
        #np.savetxt(o_hand,vec)
    
    return origin_hist
def CACOS(z):# cosの値から複素のcomplex thetaを返す
    xi = 2 * math.atan(np.real((1 - z) ** 0.5) /np.real((1 + z) ** 0.5) )
    eta = math.asinh( np.imag( (np.conj(1 +z) ** 0.5) *  ((1-z) ** 0.5)  )) 
    return xi + 1j * eta


def CASIN(z):
    xi = math.atan( np.real(z) / np.real( (1-z) ** 0.5 * ( 1 +z ) ** 0.5 )) 
    eta = math.asinh( np.imag( np.conj(1-z) ** 0.5 *  (1 +z) ** 0.5 ) )
    return xi + 1j * eta
#----------------------------------------------------------------------------------------------------------------------

#次の円の大きさを考える.
#ここで次の原点でのpの値を求めておく.

def initialdisksize(theta,radius,func): #this method get one of the directions of the two."Switch" determine it.
    global orbit
    count = 0
    disk = []
    originseed = np.array([])
    originseed2 = np.array([])
    originseedx = 0
    originseedy = 0
    dargs = twopi/diskgrid * np.arange(0 , diskgrid + 1 , 1)
    while True:
        disk = theta.real + radius * np.cos(dargs) + 1j  *( radius * np.sin(dargs) + theta.imag)
        Mappeddisk = np.array(tMap(disk,step,func))
        I_s = I(Mappeddisk)
        start = time.time()
        idx = np.argwhere(  ( np.diff(np.sign(np.imag(I_s)) )  != 0 ) & (np.diff(np.sign(np.imag(I_s)) ) != np.nan ) ).flatten()
        args =  dargs[1] * idx
        #print(np.imag(Mappeddisk[1]))
        if len(args) > 2 :
            radius = radius - radius/10
        elif len(args) == 2 :
            break
        elif len(args) < 2 :
            print('?')
            return
        #return
    origins = bisectionOndisk2(theta,args,radius,func)
     
    origin =  decidepoint(origins,func)  
    print(origin)
    
    return radius,origin

#def bisectionOndisk(origin,radius,arg,func):#bisectionOndisk
#    #print(arg)
 #   global orbit
 ##   plt.plot(origin[0],origin[1],'.',color ='green')
 #   epsilon = twopi/diskgrid #epsilon is  angular variable
 #   upperarg =  arg  + epsilon
 #   downerarg = arg  - epsilon
 #   #plt.plot(originx + radius * math.cos(upperarg),originy + radius * math.sin(upperarg),'.')
 #   i = 0
 ##   while i < 100:
 #       print(i)
 #       # print("startarg(bisectionOndisk) = %f, endarg(bisectionOndisk) = %f" %(startarg,endarg))
 #       arg = (upperarg + downerarg)/2
 #       px =  origin[0]   + radius * np.exp(1j * arg).real #point on circle
 #       py =  origin[1] + radius * np.exp(1j * arg).imag #upointx,dpointx..etc means the poit on the disk near Im0p point
 #       upx = origin[0] + radius * np.exp(1j * upperarg).real
 #       dpx = origin[0]  + radius * np.exp(1j * downerarg).real
 #       upy = origin[1] + radius * np.exp(1j * upperarg).imag
 ##       dpy = origin[1] + radius * np.exp(1j * downerarg).imag
 #       plt.plot(px, py,'.',color = 'yellow')
 #       dinittheta = dpx + 1j * dpy
 #       uinittheta = upx + 1j * upy
 #       midinittheta = px + 1j * py
 #       u_mapped = tMap(uinittheta,step,func)
 #       d_mapped = tMap(dinittheta,step,func)
 #       mid_mapped = tMap(midinittheta,step,func)
 #       u_I = I(u_mapped)
 #       d_I = I(d_mapped)
 #       mid_I = I(mid_mapped) 
 #       if  u_I.imag * d_I.imag >= 0 :
 #           upperarg += epsilon
 ##           downerarg -= epsilon
#            i = 0
        #print('a')
 #       elif mid_I.imag * u_I.imag < 0:#ScattMapt(theta,p,step,cmap):
#            downerarg = arg
        #print('b')
 #       else:
#            upperarg = arg
        #print('c')
 #       if abs(upperarg - downerarg) < 0.5**10:
            #print('break')
 #           x = origin[0] + radius * np.cos(arg)
 #           y = origin[1] + radius * np.sin(arg)
 #           plt.plot(x,y,'.',color = 'midnightblue')
 #           orbitsave(np.array([x,y]),orbit)  
 #           return x,y
        #print(arg)
 #       i += 1
 #   return x,y
#=============================

def bisectionOndisk2(theta,args,radius,func):#bisectionOndisk2では正しい偏角を二つ返す.
    #print(arg)
    global orbits
    origins = [np.array([])] * 2 
    epsilon = twopi/diskgrid #epsilon is  angular variable
    #plt.plot(originx + radius * math.cos(upperarg),originy + radius * math.sin(upperarg),'.')
    i = 0
    for j in range(len(args)):
        upperarg =  args[j] + epsilon
        downerarg = args[j] - epsilon
        while i < 1000:
             
            # print("startarg(bisectionOndisk) = %f, endarg(bisectionOndisk) = %f" %(startarg,endarg))
            args[j] = (upperarg + downerarg)/2
            px =  theta.real   + radius * np.exp(1j * args[j]).real #point on circle
            py =  theta.imag + radius * np.exp(1j * args[j]).imag #upointx,dpointx..etc means the point on the disk near Im0p point
            upx = theta.real + radius * np.exp(1j * upperarg).real
            dpx = theta.real  + radius * np.exp(1j * downerarg).real
            upy = theta.imag + radius * np.exp(1j * upperarg).imag
            dpy = theta.imag + radius * np.exp(1j * downerarg).imag
            dinittheta = dpx + 1j * dpy
            uinittheta = upx + 1j * upy
            midinittheta = px + 1j * py
            u_mapped = tMap(uinittheta,step,func)
            d_mapped = tMap(dinittheta,step,func)
            mid_mapped = tMap(midinittheta,step,func)
            u_I = I(u_mapped)
            d_I = I(d_mapped)
            mid_I = I(mid_mapped)
             
            if u_I.imag * d_I.imag >= 0 :
                upperarg += epsilon
                downerarg -= epsilon
                i = 0
            #print('a')
            elif mid_I.imag * u_I.imag < 0:#ScattMapt(theta,p,step,cmap):
                downerarg = args[j]
            #print('b')
            else:
                upperarg = args[j]
            #print('c')
            if abs(upperarg - downerarg) < 10 ** (-7):
                #print('break')
                x = theta.real + radius * np.cos(args[j]) 
                y = theta.imag + radius * np.sin(args[j])
                origins[0] = np.append(origins[0],x)
                origins[1] = np.append(origins[1],y)
                plt.plot(x,y,'.',color = 'midnightblue') 
                break
            #print(arg)
            i += 1
    return origins
#Q============================
#def Bindisk(origin,radius,):
#=============================
def decidepoint(points,func):#I.realの向きを考えて，
    global switch 
    print(switch)
    points = np.array(points)
    theta1 = cp(points[:,0])
    theta2 = cp(points[:,1])
    origin1 = np.array([theta1.real,theta1.imag]) 
    origin2 = np.array([theta2.real,theta2.imag])
    mappedorigin1 = tMap(theta1,step,func)
    mappedorigin2 = tMap(theta2,step,func)
    I1_real = I(mappedorigin1).real
    I2_real = I(mappedorigin2).real
    print(I1_real)
    print(I2_real)
    if I1_real < I2_real and switch == True: 
        return origin2 
    elif I1_real > I2_real and switch == False :
        return origin2 
    else: 
        return origin1 



def getpoint(origins,radius,pradius,cmap,switch):
    global orbitnumber,curvecount,x1,y1
    a = 0
    b = 0
    count = 0
    #print(switch)#
    #############################################
    startarg = startangle(np.array([origins[0][-1]-origins[0][-2],origins[1][-1] - origins[1][-2] ]))
    endarg =  startarg 
    subgetp(startarg,radius,origins)

def subgetp(startarg,radius,origins):#上の関数のサブルーチン
    seeds = int(radius * 1000)
    for l in range(8):
        startarg = startarg + np.pi/4 * l  
        endarg =  startarg + np.pi
        for i in range(seeds):
            Slice = (i/seeds) * radius 
            theta = np.pi/2 - math.acos(Slice/radius)
            x = origins[0][-2] + radius * np.cos( startarg + theta )
            y = origins[1][-2] + radius * np.sin( startarg + theta )
            x3 = origins[0][-2] + radius * np.cos( endarg - theta  )
            y3 = origins[1][-2] + radius * np.sin(endarg - theta )

            a,b = bisection(x,y,x3,y3,ScattMapt)
            plt.plot(a,b,'.',color = 'blue')
            orbitsave(np.array([a,b]),orbit)  
    return
#count += 1
def angle(vec1,vec2):#判定用の内積.
    cs = np.dot(vec1,vec2)/(np.linalg.norm(vec1) * np.linalg.norm(vec2))
    if abs(cs) > 1:
        cs = - np.sign(cs) *  10 ** (-10) * 2
    return math.acos(cs)
def curve_judge(origins,radius,origin):#次の原点が決まる前に、見直しをする.bisectionOndiskの後,p_hist保存の前.動くのが一回限りなので単なる応急処置.
    global p_hist
    print(radius)
    radius = stradius
    threshold = np.pi * (7/9)
    if len(origins[0]) > 3:
        vec1 =np.array( [origin[0] - origins[0][-1],origin[1] - origins[1][-1]])
        vec2 = np.array([origins[0][-2]-origins[0][-1],origins[1][-2] - origins[1][-1]])
        inpdang = angle(vec1,vec2)
        if inpdang < threshold  :
            print('曲がり')
            radius = radius/2 #いい案がないので、だいたい正しく動きそうなような.
            radius,args = initialdisksize(origin,radius,diskgrid,cmap)
            genuinearg = decidearg(radius,args,p_hist,origin)
            x,y = bisectionOndisk(origin,radius,genuinearg,cmap)
            origin = np.array([x,y])
            print(origin,radius)
            return
    return radius,origin

#=================================
#def curve_judge2(origins):
#===============================-
def bisection(x,y,x3,y3,func):
    epsilon = 0.001
    i = 0
    if x > x3:
        tmpx = x
        tmpy = y
        x = x3
        x3 = tmpx
        y =y3
        y3 = tmpy
    #print("x= %f, y= %f"%(x,y))
    while i < 400:
        midx = (x+x3)/2
        midy = (y+y3)/2
        inittheta = x + 1j * y
        midinittheta = midx + 1j * midy
        if func(init(x,y),step,cmap)[1].imag * func(init(x3,y3),step,cmap)[1].imag > 0:#ScattMapt(theta,p,step,cmap)
            return [np.nan,np.nan]
        if  func(init(x3,y3),step,cmap)[1].imag *  func(init(midx,midy),step,cmap)[1].imag < 0 :
            x = midx
            y = midy
        else:
            x3 = midx
            y3 = midy
        if math.sqrt((x-x3)**2+(y-y3)**2) < 0.5 ** 20:
            break
            i += 1
    return [midx,midy]
#--------------------------------------
#原点と次の原点を結ぶベクトルについて.
def rotate(vec):#90度回転.
    rotate = np.array([[0.,1.],[-1.,0.]])
    newv = np.dot(vec,rotate)
    return newv
def startangle(vec):#スタートの角を返す.
    newv = rotate(vec)
    if newv[0] > 0 and newv[1] > 0 :
        print('1')
        a = np.array([1,0])
        cs = np.dot(a,newv)/np.linalg.norm(newv)
        theta = math.acos(cs) + np.pi
    elif newv[0] < 0 and newv[1] > 0 :
        print('2')
        a = np.array([-1,0])
        cs = np.dot(a,newv)/np.linalg.norm(newv)
        theta = - math.acos(cs)
    elif newv[0] < 0 and newv[1] < 0:
        print('3')
        a = np.array([-1,0])
        cs = np.dot(a,newv)/np.linalg.norm(newv)
        theta =  math.acos(cs)
    elif newv[0] > 0 and newv[1] < 0:
        print('4')
        a = np.array([1,0])
        cs = np.dot(a,newv)/np.linalg.norm(newv)
        theta = np.pi - math.acos(cs)
    return theta
#--------------------------------------
#====================================================================-============
def Itaction(x_cross,periodicpoint,period):
    S = 0
    i = 0
    prepoint  = x_cross
    print(x_cross)
    print(period)
    while True:
        prepoint = x_cross
        theta = gettheta(x_cross)
        pretheta = theta
        preI = x_cross[0] ** 2 + x_cross[1] ** 2  
        prethetaforsin = CASIN(x_cross[1] /(preI  ** 0.5))   
        #print(x_cross) 
        x_cross =  cmap.U(x_cross)
        theta = gettheta(x_cross)
        #print(theta)
        I = x_cross[0] ** 2 +  x_cross[1] ** 2
        #print("x_cross =",x_cross)
        # S += 0.5 * x_cross[1] ** 2 + V(x_cross) - x_cross[1] * ( - 0.5 * dotV(x_cross[0]) - 0.5 * dotV(x_cross[0] + x_cross[1] - 0.5 * dotV(x_cross[0]) ) )#終わりから始めを引け
        S += 0.5 * (x_cross[1] ** 2) + V(prepoint) - x_cross[1] * ( x_cross[0] - prepoint[0] ) 
        print("S +=",0.5 * (x_cross[1] ** 2) + V(prepoint) - x_cross[1] * ( x_cross[0] - prepoint[0] ) )
        print("S = ",S)
        if i == 0:
            S -= V(prepoint)
            S -= generatorfunc(preI,prethetaforsin)

        i += 1
        #print(i)
        print('x_cross = ',x_cross)
        if   np.linalg.norm(np.imag(x_cross)) < 10 ** (-7) * 2  and step !=1:
            print("theta = ",np.linalg.norm(np.imag(x_cross)))
            #S -= V(prepoint)           #print("Im(Smin)= " ,S.imag)
            #print('x_cross = ',x_cross)
            return S.imag
        elif period == 1 and  (np.linalg.norm(x_cross - periodicpoint)  < 10 ** (-5) or np.linalg.norm(np.imag(x_cross)) < 5 * 10 ** (-5)):
            print("theta? = ",np.linalg.norm(np.imag(x_cross)))
            #S -= V(prepoint)
            #print("Im(Smin)= " ,S.imag)
            #print('x_cross = ',x_cross)
            return S.imag
        if np.linalg.norm(x_cross) > 10:
            return np.inf
        if i > 1000:
            return np.inf           

#--------------------------------------
def originsave(origin,origin_hist):
    origin_hist[0] =  np.append(origin_hist[0], origin[0])
    origin_hist[1] =  np.append(origin_hist[1], origin[1])

def orbitsave(point,orbit):
    orbit[0] =  np.append(orbit[0], point[0])
    orbit[1] =  np.append(orbit[1], point[1])
#originseed = (radius,arg), 2 be that of another side point
def mset():#Julia集合を描く
    originx = 0
    originy = 0
    #originx = 2.00
    #originy = 2.58
    #height  = 2.73  - 2.588
    #width = 2.218 -  2.03
    height = 4.8
    width = 2 * np.pi
    grid = 400
    grid_mset = 1000
    grid2 = grid - 1
    step = 4000
    step2 = 157 #msetを図示する用
    step3 = 189 #msetを図示する用
    fig = plt.figure(figsize = (12,12))
    ax = fig.add_subplot(111)
    xi = np.linspace(originx, originx +  width ,grid)
    eta = np.linspace(originy, originy + height,grid)
    xi_mset = np.linspace(originx, originx +  width ,grid_mset)
    eta_mset = np.linspace(originy, originy +  height ,grid_mset)
    xi2,eta2 = np.meshgrid(xi,eta)
    xi2_mset,eta2_mset = np.meshgrid(xi_mset,eta_mset)
    theta = xi2 + eta2 * 1j    
    theta_mset = xi2_mset + eta2_mset * 1j    
    q = a_orb * np.sin(theta)
    p = b_orb * np.cos(theta)
    q_mset = a_orb * np.sin(theta_mset)
    p_mset = b_orb * np.cos(theta_mset)
    qp = np.array([q,p])
    qp_mset = np.array([q_mset,p_mset])
    qp_rotated = np.array([q * np.cos(t) - p * np.sin(t),  q * np.sin(t) + p * np.cos(t)])
    qp_rotated_mset = np.array([q_mset * np.cos(t) - p_mset * np.sin(t),  q_mset * np.sin(t) + p_mset * np.cos(t)])
    div_numbers = np.full_like(qp_rotated[0],step)
    div_numbers_checkbool  =  np.linalg.norm(qp_rotated,axis = 0)  < 1000 #すでに発散したかどうか
    cm = plt.cm.get_cmap('seismic')
    plt.xlabel(r"$\xi$",fontsize = 20)
    plt.ylabel(r"$\eta$",fontsize = 20,rotation  = "horizontal",labelpad = 20)
    print(qp_rotated.shape)
    xi2_mset = xi2_mset[:,:-1]
    eta2_mset = eta2_mset[:,:-1]
    print(eta2)
    for i in range(step):
        qp_rotated = cmap.U(qp_rotated)
        boolean = np.logical_or(np.linalg.norm(qp_rotated,axis = 0) > 1000, np.isnan(np.linalg.norm(qp_rotated,axis = 0)) == True)#発散してるところ
        #print(np.linalg.norm(qp_rotated,axis = 0) )
        #print(div_numbers_checkbool)
        #print(boolean)
        boolean =  np.logical_and(div_numbers_checkbool, boolean) #発散していてなおかつまだ記していない。
        #print(boolean)
        div_numbers_checkbool[boolean] = False
        #print("b = ",div_numbers_checkbool)
        #if i < step3:
        #    qp_rotated_mset = cmap.U(qp_rotated_mset)
        #    if i == step2 - 1 :
        #        boolean_mset =  np.logical_and(np.diff( np.sign(qp_rotated_mset[0].imag) ) != 0, qp_rotated_mset[0][:,:-1] < 10 ** 5)
        #        plt.plot(xi2_mset[boolean_mset],eta2_mset[boolean_mset],',g',zorder = 3)
        #    elif i == step3 - 1:
        #        boolean_mset =  np.logical_and(np.diff( np.sign(qp_rotated_mset[0].imag) ) != 0, qp_rotated_mset[0][:,:-1] < 10 ** 5)
        #        plt.plot(xi2_mset[boolean_mset],eta2_mset[boolean_mset],',k',zorder = 3)
            #elif i == 126 - 1:
            #    boolean_mset =  np.logical_and(np.diff( np.sign(qp_rotated_mset[0].imag) ) != 0, qp_rotated_mset[0][:,:-1] < 10 ** 5)
            #    plt.plot(xi2_mset[boolean_mset],eta2_mset[boolean_mset],',r',zorder = 3)
            #elif i == 94 -1 :
            #    boolean_mset =  np.logical_and(np.diff( np.sign(qp_rotated_mset[0].imag) ) != 0, qp_rotated_mset[0][:,:-1] < 10 ** 5)
            #    plt.plot(xi2_mset[boolean_mset],eta2_mset[boolean_mset],',b',zorder = 3)

        #print(boolean[0])
        #
        print(i)
        div_numbers[boolean] =  i 
        #print(div_numbers[4])
        
        #print(div_numbers)
    #print(div_numbers)
    #print(boolean.shape)
    #xi2 = xi2[:,:-1]
    #eta2 = eta2[:,:-1]
    #print(mset2_Release)
    #exit(
    print(div_numbers)
    plt.scatter(xi2,eta2, s=0.1, c=div_numbers, marker = '.',cmap='seismic', vmax =step,vmin = 0,alpha = 1)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label("Time step",size = 25)
    plt.rcParams["font.size"] = 30
    #plt.xlim(0,2 * np.pi)
    plt.tick_params(labelsize=20)
    #plt.ylim(0,height-0.2)
    #plt.xlim(xi2[0,0],xi2[-1,-1])
    #plt.plot(xi2[boolean],eta2[boolean],',k',zorder = 1)
    #plotperiods(textpath)
    fig.canvas.mpl_connect('key_press_event', on_press)
    fig.canvas.mpl_connect('button_press_event',mset2_onclick)#(なぜonkeyが使えないのか...)
    fig.canvas.mpl_connect('button_release_event',mset2_Release)
    fig.canvas.mpl_connect('button_release_event',mset2_dblclick) 
    plt.show()    
    plt.close()
def mset2(x1,x2,y1,y2):
    if x1 > x2:
        tmp = x2
        x2 = x1
        x1 = tmp
    if y1 > y2:
        tmp = y2
        y2 = y1
        y1 = tmp
    originx = x1
    originy = y1
    width = x2 - x1
    height = y2 - y1
    grid =400
    step = 4000
    step2 = 157 #msetを図示する用
    step3 = 189 #msetをもう一つ図示する用
    grid2 = grid - 1
    grid_mset = 600
    fig = plt.figure(figsize = (12,12))
    ax = fig.add_subplot(111)
    xi = np.linspace(originx, originx +  width ,grid)
    eta = np.linspace(originy, originy + height,grid)
    xi_mset = np.linspace(originx, originx +  width ,grid_mset)
    eta_mset = np.linspace(originy, originy +  height ,grid_mset)
    xi2,eta2 = np.meshgrid(xi,eta)
    xi2_mset,eta2_mset = np.meshgrid(xi_mset,eta_mset)
    theta = xi2 + eta2 * 1j    
    theta_mset = xi2_mset + eta2_mset * 1j
    q = a_orb * np.sin(theta)
    p = b_orb * np.cos(theta)
    q_mset = a_orb * np.sin(theta_mset)
    p_mset = b_orb * np.cos(theta_mset)
    qp_rotated = np.array([q * np.cos(t) - p * np.sin(t),  q * np.sin(t) + p * np.cos(t)])
    qp_rotated_mset = np.array([q_mset * np.cos(t) - p_mset * np.sin(t),  q_mset * np.sin(t) + p_mset * np.cos(t)])
    qp_rotated = np.array([q * np.cos(t) - p * np.sin(t),  q * np.sin(t) + p * np.cos(t)])
    div_numbers = np.full_like(qp_rotated[0],step)
    div_numbers_checkbool  =  np.linalg.norm(qp_rotated,axis = 0)  < 1000 #すでに発散したかどうか
    cm = plt.cm.get_cmap('seismic')
    print(qp_rotated.shape)
    xi3_mset = xi2_mset[:,:-1]
    eta3_mset = eta2_mset[:,:-1]
    xi4_mset = xi2_mset[:-1,:]
    eta4_mset = eta2_mset[:-1,:]
    for  i in range(step):
        qp_rotated = cmap.U(qp_rotated)
        boolean = np.logical_or(np.linalg.norm(qp_rotated,axis = 0) > 1000, np.isnan(np.linalg.norm(qp_rotated,axis = 0)) == True)#発散してるところ
        #print(np.linalg.norm(qp_rotated,axis = 0) )
        #print(div_numbers_checkbool)
        #print(boolean)
        boolean =  np.logical_and(div_numbers_checkbool, boolean) #発散していてなおかつまだ記していない。
        #print(boolean)
        div_numbers_checkbool[boolean] = False
        #print("b = ",div_numbers_checkbool)
        print(i)
        #print(boolean[0])
        #
        div_numbers[boolean] =  i 
        #if i < step3:
        #    qp_rotated_mset = cmap.U(qp_rotated_mset)
           #if i == step2 - 1 :
           #     boolean_mset =  np.logical_and(np.diff( np.sign(qp_rotated_mset[0].imag) ) != 0, qp_rotated_mset[0][:,:-1] < 10 ** 5)
           #     plt.plot(xi3_mset[boolean_mset],eta3_mset[boolean_mset],',g',zorder = 3)
           #     boolean_mset =  np.logical_and(np.diff( np.sign(qp_rotated_mset[0].imag) , axis = 0 ) != 0, qp_rotated_mset[0][:-1,:] < 10 ** 5)
           #     plt.plot(xi4_mset[boolean_mset],eta4_mset[boolean_mset],',g',zorder = 3)
            #if i == step3 - 1:
            #    boolean_mset =  np.logical_and(np.diff( np.sign(qp_rotated_mset[0].imag) ) != 0, qp_rotated_mset[0][:,:-1] < 10 ** 5)
            #    plt.plot(xi3_mset[boolean_mset],eta3_mset[boolean_mset],',k',zorder = 3)
            #    boolean_mset =  np.logical_and(np.diff( np.sign(qp_rotated_mset[0].imag) , axis = 0 ) != 0, qp_rotated_mset[0][:-1,:] < 10 ** 5)
            #    plt.plot(xi4_mset[boolean_mset],eta4_mset[boolean_mset],',k',zorder = 3)
            #elif i == 126 - 1:
            #    boolean_mset =  np.logical_and(np.diff( np.sign(qp_rotated_mset[0].imag) ) != 0, qp_rotated_mset[0][:,:-1] < 10 ** 5)
            #    plt.plot(xi3_mset[boolean_mset],eta3_mset[boolean_mset],',r',zorder = 3)
           # elif i == 94 -1:
           #     boolean_mset =  np.logical_and(np.diff( np.sign(qp_rotated_mset[0].imag) ) != 0, qp_rotated_mset[0][:,:-1] < 10 ** 5)
           #     plt.plot(xi2_mset[boolean_mset],eta2_mset[boolean_mset],',b',zorder = 3)



        #print(div_numbers[4])
        
        #print(div_numbers)
    #print(div_numbers)
    #print(boolean.shape)
    #xi2 = xi2[:,:-1]
    #eta2 = eta2[:,:-1]
    #print(mset2_Release)
    #exit(
    print(div_numbers)
    plt.rcParams["font.size"] = 30
    #plt.xlim(0,2 * np.pi)
    plt.tick_params(labelsize=20)
    plt.scatter(xi2,eta2, s=0.1, c=div_numbers, marker = '.',cmap='seismic', vmax = step,vmin = 0,alpha = 1)
    cbar = plt.colorbar()
    plt.xlabel(r"$\xi$",labelpad = 20,fontsize = 20)
    plt.ylabel(r"$\eta$",rotation = "horizontal",fontsize = 20,labelpad = 20)
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label("Time step",size = 25)
    #plt.ylim(eta2[0,0],eta2[-1,-1])
    fig.canvas.mpl_connect('button_release_event',Release)
    fig.canvas.mpl_connect('key_press_event', on_press)
    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('button_press_event',mset2_onclick)
    fig.canvas.mpl_connect('button_press_event',mset2_dblclick)
    fig.canvas.mpl_connect('button_release_event',mset2_Release)

    #stat_mset = False 
    #plotperiods(textpath)
    plt.show()    
def branches(x1,y1,x2,y2):
    originx = 0
    originy = 0
    height = -6.0
    grid = 1000
    grid2 = grid - 1
    width = 2 * np.pi
    fig = plt.figure(figsize = (10,10))
    ax = fig.add_subplot(111)
    xi = np.linspace(originx, originx +  width ,grid)
    eta = np.linspace(originy, originy + height,grid)
    xi2,eta2 = np.meshgrid(xi,eta)
    theta = xi2 + eta2 * 1j    
    q = a_orb * np.sin(theta)
    p = b_orb * np.cos(theta)
    qp = np.array([q,p])
    qp_rotated = np.array([q * np.cos(t) - p * np.sin(t),  q * np.sin(t) + p * np.cos(t)])
    mapped = ScattMapt(qp_rotated,step,cmap)
    boolean = np.diff( np.sign(mapped[0].imag) ) != 0
    print(boolean.shape)

    xi3 = xi2[:,:-1]
    eta3 = eta2[:,:-1]
    xi4 = xi2[ :-1,:]
    eta4 = eta2[ :-1 , :]
    print(mset2_Release)
    #exit()
    #print(eta3[boolean2].shape)
    boolean = np.logical_and(np.diff( np.sign(mapped[0].imag) ) != 0, mapped[0][:,:-1] < 10 ** 5)
    boolean2 = np.logical_and(np.diff( np.sign(mapped[0].imag),axis = 0 ) != 0, mapped[0][:-1,:] < 10 ** 5)
    print(boolean.shape)
    print(xi4.shape)
    plt.ylim(eta2[0,0],eta2[-1,-1])
    plt.xlim(xi2[0,0],xi2[-1,-1])
    plt.plot(xi3[boolean],eta3[boolean],',k',zorder = 1)
    plt.plot(xi4[boolean2],eta4[boolean2],',k',zorder = 1)
    print(mset2_Release)
    #e
    ax.set_xlabel(r"$\xi$",fontsize = 20)
    ax.set_ylabel(r"$\eta$",fontsize = 20,rotation  = "horizontal",labelpad = 20)
    plt.ylim(eta2[0,0],eta2[-1,-1])
    plt.xlim(xi2[0,0],xi2[-1,-1])
    plt.rcParams["font.size"] = 30
    #plt.xlim(0,2 * np.pi)
    plt.tick_params(labelsize=20)
    #plt.plot(xi2[boolean],eta2[boolean],',k',zorder = 1)
    fig.canvas.mpl_connect('button_press_event',mset2_dblclick)
    #plotperiods(textpath)
    #fig.canvas.mpl_connect('key_press_event', on_press)
    fig.canvas.mpl_connect('button_press_event',mset2_onclick)#(なぜonkeyが使えないのか...)
    fig.canvas.mpl_connect('button_release_event',mset2_Release)
    plt.show()    
    
def gettheta(cpoint):#cpoint から　thetaを得る． Kahan, W: Branch cuts for complex elementary functionを参照
    print("a = ",cpoint)
    cpoint_rotated = rotateinitialmanifold(a_orb,b_orb,-t,cpoint)
    sine =  cpoint_rotated[0]/a_orb
    cosine = cpoint_rotated[1]/b_orb
    #print(cosine **2 + sine ** 2)
    theta = CACOS(cosine) #まずcosineに処理を施す
    theta2 = CASIN(sine) 
    #print("acos =",theta)
    #print("asin =",theta2)
    if  theta.real < np.pi/2 and theta2.real < 0:
        theta_conf = theta2.real + 2 *  np.pi + 1j * theta.imag
    elif theta.real > np.pi/2 and theta2.real < 0:
        theta_conf = theta.real - 2 * theta2.real + 1j * theta.imag
        #print("theta = {}".format(theta_conf))
    else:
        theta_conf = theta
    confirm_point = np.array([a_orb * np.sin(theta_conf),b_orb * np.cos(theta_conf)])
    confirm_point = rotateinitialmanifold(a_orb,b_orb,t,confirm_point)
    if  np.sign(confirm_point[0].imag) != np.sign(cpoint[0].imag):
        theta_conf = theta_conf.real - 1j * theta_conf.imag 
    #print(theta_conf)
    confirm_point = np.array([a_orb * np.sin(theta_conf),b_orb * np.cos(theta_conf)])
    confirm_point = rotateinitialmanifold(a_orb,b_orb,t,confirm_point)
    print("b = ",confirm_point)
    return theta_conf

def on_press(event):
    global stat_mset2
    if event.key == "d":
        plt.title("get mode ")
        stat_mset2 =  True


def mset2_Release(event):
    global x1_mset,y1_mset,x2_mset,y2_mset,stat_mset
    x2_mset = event.xdata
    y2_mset = event.ydata
    if event.button == 2:
       # exit()
        mset2(x1_mset,x2_mset,y1_mset,y2_mset)
    stat_mset = False 

def mset2_onclick(event):
    global x1_mset,y1_mset
    global stat_mset
    print(event.button)
    if event.button == 2 :
        x1_mset = event.xdata
        y1_mset = event.ydata
        #stat_mset = True
def mset2_dblclick(event):
    global x1_mset,y1_mset
    global stat_mset
    if event.dblclick == 1:
        xi = event.xdata
        eta = event.ydata
        theta = xi + 1j * eta 
        qp = np.array([a_orb * np.sin(theta) , b_orb * np.cos(theta)])
        qp = rotateinitialmanifold(a_orb,b_orb,t,qp)
        action2(qp)
def generatorfunc2(theta):#母関数
    return  (a_orb*b_orb/2) * (theta + (1/2) * np.sin(2 * theta) * np.cos(2 * t)) - (1/8) * (a_orb ** 2 + b_orb ** 2) * np.cos(2 * theta) * np.sin(2 * t)

def plotactions(textpath):#その点から軌道に沿った作用の大きさで色分け
    #stdrad = rad
    #print(stdrad)
    text = np.loadtxt(textpath,dtype = np.complex128)
    #print(textpath)
    superdata1 = text
    #fig = plt.figure(figsize = (10,8))
    #ax = fig.add_subplot(1,1,1)
   # plt.xlabel(r'$Re(\theta_0)$',fontsize = 20)
   # plt.ylabel(r'$Im(\theta_0)$',fontsize = 20)
    #plt.title("radius = {} ".format(str(round(stdrad,3))), fontsize = 33)

    #plt.semilogx()
    allpoints = [np.array([])]*2
    allaxispoint = np.array([])#基準となる交点の数々.
    iterates = text[:,5]
    initpointRe = text[:,10]
    initpointIm = text[:,11]
    global initpoint_array
    initpoint_array = np.array([initpointRe,initpointIm])
    axis = text[:,5]
    axis3 = text[:,4]
    ImS  = np.abs(text[:, 2])#print(vec.s hape)
    #plotarray = np.array([axis,axis3,ImS])#初期化
    #print(ImS.shape)
    #print(text[:,10].shape)
    #plt.plot(text[:,10],text[:,11],"o")
    a_str = str(round(a_orb,3)).replace('.','')
    b_str = str(round(b_orb,3)).replace('.','')
    #tree  = np.loadtxt("test_3_a_{}_b_{}.txt".format(a_str,b_str))#"test_10_a_$a_orb","_b_$b_orb",".txt" mset_40_4_a_025_b_03.txt
    #ree = np.loadtxt("mset_40_1_a_030_b_008.txt")
    #plt.plot(tree[:,0],tree[:,1],",k",zorder = 1)
    #plt.scatter(initpointRe, initpointIm, s = 100, c = ImS, alpha = 1, cmap="jet_r", marker = "o", zorder = 2)
    cbar = plt.colorbar(shrink= 0.89,label = "log10(ImS)")
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("|Im(S)|",size = 20)
    plt.clim(vmin=-0.0,vmax=0.1)

    #plt.ylim(0.0,4.0)
    #plt.title("a = {}, b = {},t = {} ".format(str(round(a_orb,3)),str(round(b_orb,3)),timestep), fontsize = 33)
    #plt.plot(initpointRe[0],initpointIm[0],"o", color = 'red')
    #for i in [1,4,5,6,7,8]:#それぞれの半径について．凡例を作るため。
        #extractarray = plotarray[:,  plotarray[0] == i ]
        #ax.scatter(  extractarray[1],extractarray[2] ,s = 100 ,label = "period{}".format(i))
        #print(vec)
        #np.savetxt(p_hand,vec,newline = "\r\n", fmt = '%.16f%+.16fj ' * 7)
        #plt.scatter(axis3,ImS , s = 100, c = axis, alpha = 1, cmap="jet", marker = "o", zorder = 2)
        #plt.plot(text[:,0], ImS, 'o',markersize = 12)
        #cbar = plt.colorbar(shrink= 0.89

    #   from matplotlib.colors import LogNorm
        #bar=ax.imshow(stdall,norm=LogNorm())
        ##cbar.ax.set_ylabel('period', fontsize = 33, weight="bold")
        #cbar.ax.set_yticklabels(np.arange(cbar_min, cbar_max+cbar_step, cbar_step), fontsize=16, weight='bold',label = "period")
    #   ax.legend(loc = "upper left",fontsize = 14)
#       plt.tick_params(labelsize = 20)
#       plt.ylim(-0.001,0.03)
    plt.rcParams["font.size"] = 30
    #plt.xlim(0,2 * np.pi)
    plt.tick_params(labelsize=20)

    #plt.savefig('mset30_40_rad{}.png'.format(str(round(stdrad,3)).replace('.',''))) 
    #cid = fig.canvas.mpl_connect('button_press_event', onkey)
    #plt.connect('button_press_event', Press)
    #plt.connect('motion_notify_event',Drag)
    #plt.connect('button_release_event',Release)
    #plt.show()
 #cid = fig.canvas.mpl_connect('button_press_event', 


def plotperiods(textpath):#その点から軌道に沿った作用の大きさで色分け
    #stdrad = rad
    #print(stdrad)
    text = np.loadtxt(textpath,dtype = np.complex128)
    print(text)
    #print(textpath)
    superdata1 = text
    #fig = plt.figure(figsize = (10,8))
    #ax = fig.add_subplot(1,1,1)
   # plt.xlabel(r'$Re(\theta_0)$',fontsize = 20)
   # plt.ylabel(r'$Im(\theta_0)$',fontsize = 20)
    #plt.title("radius = {} ".format(str(round(stdrad,3))), fontsize = 33)

    #plt.semilogx()
    allpoints = [np.array([])]*2
    allaxispoint = np.array([])#基準となる交点の数々.
    iterates = text[:,5]
    initpointRe = text[:,10]
    initpointIm = text[:,11]
    global initpoint_array
    initpoint_array = np.array([initpointRe,initpointIm])
    axis = text[:,5]
    axis3 = text[:,4]
    ImS  = np.abs(text[:, 0])#print(vec.s hape)
    #plotarray = np.array([axis,axis3,ImS])#初期化
    #print(ImS.shape)
    #print(text[:,10].shape)
    #plt.plot(text[:,10],text[:,11],"o")
    a_str = str(round(a_orb,3)).replace('.','')
    b_str = str(round(b_orb,3)).replace('.','')
    #tree  = np.loadtxt("test_3_a_{}_b_{}.txt".format(a_str,b_str))#"test_10_a_$a_orb","_b_$b_orb",".txt" mset_40_4_a_025_b_03.txt
    #ree = np.loadtxt("mset_40_1_a_030_b_008.txt")
    #plt.plot(tree[:,0],tree[:,1],",k",zorder = 1)
    #plt.scatter(initpointRe, initpointIm, s = 100, c = ImS, alpha = 1, cmap="jet_r", marker = "o", zorder = 2)
    #cbar = plt.colorbar(shrink= 0.89,label = "log10(ImS)")
    #cbar.ax.tick_params(labelsize=14)
    #cbar.set_label("|Im(S)|",size = 20)
    #plt.clim(vmin=-0.0,vmax=0.1)
    #plt.ylim(0.0,4.0)
    #plt.title("a = {}, b = {},t = {} ".format(str(round(a_orb,3)),str(round(b_orb,3)),timestep), fontsize = 33)
    #plt.plot(initpointRe[0],initpointIm[0],"o", color = 'red')
    for i in [1,4,5,6,7,8]:#それぞれの半径について．凡例を作るため。
        print(text[:,0])
        extractarray = text[text[:,0] == i,:]
        print(extractarray.shape)
        print(extractarray[:,10])
        #plt.plot(  extractarray[:,10],extractarray[:,11] ,'o',markersize = 10,label = "period{}".format(i))
        #print(vec)
        #np.savetxt(p_hand,vec,newline = "\r\n", fmt = '%.16f%+.16fj ' * 7)
        #plt.scatter(axis3,ImS , s = 100, c = axis, alpha = 1, cmap="jet", marker = "o", zorder = 2)
        #plt.plot(text[:,0], ImS, 'o',markersize = 12)
        #cbar = plt.colorbar(shrink= 0.89)

        #from matplotlib.colors import LogNorm
        #bar=ax.imshow(stdall,norm=LogNorm())
        ##cbar.ax.set_ylabel('period', fontsize = 33, weight="bold")
        #cbar.ax.set_yticklabels(np.arange(cbar_min, cbar_max+cbar_step, cbar_step), fontsize=16, weight='bold',label = "period")
    #   ax.legend(loc = "upper left",fontsize = 14)
        #plt.tick_params(labelsize = 20)
        #plt.ylim(-0.001,0.03)
    plt.rcParams["font.size"] = 30
    #plt.xlim(0,2 * np.pi)
    plt.tick_params(labelsize=20)

def extract_pointsbyperiod(period):
    points = np.loadtxt("periodicpoint2.txt")
    print(points[:,2] == period)
    points = points[points[:,2] == period,0:2]
    return  points


cmap = KScatteringMap2(k,xf,xb)
seed = [[],[]]
xy = [np.array([])]*2
origins = np.array([])
#v(p,cmap)
stat = 1#マウスイベントの状態.
stat_mset  = False
period = 40  
a_str = format(str(round(a_orb,3))).replace('.','')
b_str = format(str(round(b_orb,3))).replace('.','')
#seed = np.loadtxt('test_10_a_{}_b_{}.txt'.format(a_str,b_str)) #ここに読み込みたいMsetのファイルを持ってくる。
x1=0
y1=0
x1_mset = 0
y1_mset = 0
x2_mset = 0
y2_mset = 0
x_pres = 0
y_pres = 0
DragFlag = False

BC = 12 * np.pi
QMIN, QMAX = -BC, BC
N = 2**10
dQ = (QMAX - QMIN) / (N - 1)
T_kick = 0.05
k = 3.0
xf = 1.2
xb = 1.0
twopi = 2 * np.pi
step = 0
e2 = k * xf * (np.exp(-8 * xb * (2 * xf - xb)) / (1 - np.exp(-32 * xf * xb)))

# 量子BCH近似のそれぞれの項．（5次の項まで書かれている．）
def _bchHamitonianTerm(x, y, order):
    if order == 1:
        return x + y
    elif order == 2:
        return (x@y - y@x)/2
    elif order == 3:
        return x@x@y/12 - x@y@x/6 + x@y@y/12 + y@x@x/12 - y@x@y/6 + y@y@x/12
    elif order == 4:
        return x@x@y@y/24 - x@y@x@y/12 + y@x@y@x/12 - y@y@x@x/24
    elif order == 5:
        return (
            - x@x@x@x@y/720 + x@x@x@y@x/180 + x@x@x@y@y/180 - x@x@y@x@x/120 - x@x@y@x@y/120 - x@x@y@y@x/120 +
            x@x@y@y@y/180 + x@y@x@x@x/180 - x@y@x@x@y/120 + x@y@x@y@x/30 - x@y@x@y@y/120 - x@y@y@x@x/120 - x@y@y@x@y/120 +
            x@y@y@y@x/180 - x@y@y@y@y/720 - y@x@x@x@x/720 + y@x@x@x@y/180 - y@x@x@y@x/120 - y@x@x@y@y/120 - y@x@y@x@x/120 +
            y@x@y@x@y/30 - y@x@y@y@x/120 + y@x@y@y@y/180 + y@y@x@x@x/180 - y@y@x@x@y/120 - y@y@x@y@x/120 - y@y@x@y@y/120 +
            y@y@y@x@x/180 + y@y@y@x@y/180 - y@y@y@y@x/720
        )
    elif order == 6:
        return (
            (-1/60) * y @ x @ y @ x @ y @ x + (-1/240) * x @ x @ y @ x @ x @ y + (-1/240) * x @ x @ y @ x @ y @ y + (-1/240) * x @ x @ y @ y @ x @ y + (-1/240) * x @ y @ x @ x @ y @ y + (-1/240) * x @ y @ y @ x @ x @ y + (-1/240) * x @ y @ y @ x @ y @ y + (-1/360) * y @ x @ x @ x @ y @ x + (-1/360) * y @ x @ y @ x @ x @ x + (-1/360) * y @ x @ y @ y @ y @ x + (-1/360) * y @ y @ y @ x @ x @ x + (-1/360) * y @ y @ y @ x @ y @ x + (-1/1440) * x @ x @ x @ x @ y @ y + (-1/1440) * x @ x @ y @ y @ y @ y +
            (1/1440) * y @ y @ x @ x @ x @ x + (1/1440) * y @ y @ y @ y @ x @ x + (1/360) * x @ x @ x @ y @ x @ y + (1/360) * x @ x @ x @ y @ y @ y + (1/360) * x @ y @ x @ x @ x @ y + (1/360) * x @ y @ x @ y @ y @ y + (1/360) * x @ y @ y @ y @ x @ y +
            (1/240) * y @ x @ x @ y @ x @ x + (1/240) * y @ x @ x @ y @ y @ x + (1/240) * y @ x @ y @ y @ x @ x + (1/240) *
            y @ y @ x @ x @ y @ x + (1/240) * y @ y @ x @ y @ x @ x + (
                1/240) * y @ y @ x @ y @ y @ x + (1/60) * x @ y @ x @ y @ x @ y
        )
    elif order == 7:
        return ((-1/140) * x @ y @ x @ y @ x @ y @ x + (-1/140) * y @ x @ y @ x @ y @ x @ y + (-1/630) * x @ x @ x @ y @ x @ y @ x + (-1/630) * x @ y @ x @ x @ x @ y @ x + (-1/630) * x @ y @ x @ y @ x @ x @ x + (-1/630) * x @ y @ x @ y @ y @ y @ x + (-1/630) * x @ y @ y @ y @ x @ y @ x + (-1/630) * y @ x @ x @ x @ y @ x @ y + (-1/630) * y @ x @ y @ x @ x @ x @ y + (-1/630) * y @ x @ y @ x @ y @ y @ y + (-1/630) * y @ x @ y @ y @ y @ x @ y + (-1/630) * y @ y @ y @ x @ y @ x @ y + (-1/1120) * x @ x @ y @ x @ x @ y @ y + (-1/1120) * x @ x @ y @ y @ x @ x @ y + (-1/1120) * x @ x @ y @ y @ x @ y @ y + (-1/1120) * x @ y @ y @ x @ x @ y @ y + (-1/1120) * y @ x @ x @ y @ y @ x @ x + (-1/1120) * y @ y @ x @ x @ y @ x @ x + (-1/1120) * y @ y @ x @ x @ y @ y @ x + (-1/1120) * y @ y @ x @ y @ y @ x @ x + (-1/1512) * x @ x @ x @ y @ x @ x @ x + (-1/1512) * x @ x @ x @ y @ y @ y @ x + (-1/1512) * x @ y @ y @ y @ x @ x @ x + (-1/1512) * y @ x @ x @ x @ y @ y @ y + (-1/1512) * y @ y @ y @ x @ x @ x @ y + (-1/1512) * y @ y @ y @ x @ y @ y @ y + (-1/5040) * x @ x @ x @ x @ x @ y @ x + (-1/5040) * x @ x @ x @ x @ x @ y @ y + (-1/5040) * x @ x @ x @ y @ x @ x @ y + (-1/5040) * x @ x @ x @ y @ x @ y @ y + (-1/5040) * x @ x @ x @ y @ y @ x @ x + (-1/5040) * x @ x @ x @ y @ y @ x @ y + (-1/5040) * x @ x @ y @ x @ x @ x @ y + (-1/5040) * x @ x @ y @ x @ y @ y @ y + (-1/5040) * x @ x @ y @ y @ x @ x @ x + (-1/5040) * x @ x @ y @ y @ y @ x @ x + (-1/5040) * x @ x @ y @ y @ y @ x @ y + (-1/5040) * x @ x @ y @ y @ y @ y @ y + (-1/5040) * x @ y @ x @ x @ x @ x @ x + (-1/5040) * x @ y @ x @ x @ x @ y @ y + (-1/5040) * x @ y @ x @ x @ y @ y @ y + (-1/5040) * x @ y @ y @ x @ x @ x @ y + (-1/5040) * x @ y @ y @ x @ y @ y @ y + (-1/5040) * x @ y @ y @ y @ x @ x @ y + (-1/5040) * x @ y @ y @ y @ x @ y @ y + (-1/5040) * x @ y @ y @ y @ y @ y @ x + (-1/5040) * y @ x @ x @ x @ x @ x @ y + (-1/5040) * y @ x @ x @ x @ y @ x @ x + (-1/5040) * y @ x @ x @ x @ y @ y @ x + (-1/5040) * y @ x @ x @ y @ x @ x @ x + (-1/5040) * y @ x @ x @ y @ y @ y @ x + (-1/5040) * y @ x @ y @ y @ x @ x @ x + (-1/5040) * y @ x @ y @ y @ y @ x @ x + (-1/5040) * y @ x @ y @ y @ y @ y @ y + (-1/5040) * y @ y @ x @ x @ x @ x @ x + (-1/5040) * y @ y @ x @ x @ x @ y @ x + (-1/5040) * y @ y @ x @ x @ x @ y @ y + (-1/5040) * y @ y @ x @ x @ y @ y @ y + (-1/5040) * y @ y @ x @ y @ x @ x @ x + (-1/5040) * y @ y @ x @ y @ y @ y @ x + (-1/5040) * y @ y @ y @ x @ x @ y @ x + (-1/5040) * y @ y @ y @ x @ x @ y @ y + (-1/5040) * y @ y @ y @ x @ y @ x @ x + (-1/5040) * y @ y @ y @ x @ y @ y @ x + (-1/5040) * y @ y @ y @ y @ y @ x @ x + (-1/5040) * y @ y @ y @ y @ y @ x @ y + (1/30240) * x @ x @ x @ x @ x @ x @ y + (1/30240) * x @ y @ y @ y @ y @ y @ y + (1/30240) * y @ x @ x @ x @ x @ x @ x + (1/30240) * y @ y @ y @ y @ y @ y @ x + (1/3780) * x @ x @ x @ x @ y @ y @ y + (1/3780) * x @ x @ x @ y @ y @ y @ y + (1/3780) * y @ y @ y @ x @ x @ x @ x + (1/3780) * y @ y @ y @ y @ x @ x @ x + (1/2016) * x @ x @ x @ x @ y @ x @ x + (1/2016) * x @ x @ x @ x @ y @ x @ y + (1/2016) * x @ x @ x @ x @ y @ y @ x + (1/2016) * x @ x @ y @ x @ x @ x @ x + (1/2016) * x @ x @ y @ y @ y @ y @ x + (1/2016) * x @ y @ x @ x @ x @ x @ y + (1/2016) * x @ y @ x @ y @ y @ y @ y + (1/2016) * x @ y @ y @ x @ x @ x @ x + (1/2016) * x @ y @ y @ y @ y @ x @ x + (1/2016) * x @ y @ y @ y @ y @ x @ y + (1/2016) * y @ x @ x @ x @ x @ y @ x + (1/2016) * y @ x @ x @ x @ x @ y @ y + (1/2016) * y @ x @ x @ y @ y @ y @ y + (1/2016) * y @ x @ y @ x @ x @ x @ x + (1/2016) * y @ x @ y @ y @ y @ y @ x + (1/2016) * y @ y @ x @ x @ x @ x @ y + (1/2016) * y @ y @ x @ y @ y @ y @ y + (1/2016) * y @ y @ y @ y @ x @ x @ y + (1/2016) * y @ y @ y @ y @ x @ y @ x + (1/2016) * y @ y @ y @ y @ x @ y @ y + (1/840) * x @ x @ y @ x @ x @ y @ x + (1/840) * x @ x @ y @ x @ y @ x @ x + (1/840) * x @ x @ y @ x @ y @ x @ y + (1/840) * x @ x @ y @ x @ y @ y @ x + (1/840) * x @ x @ y @ y @ x @ y @ x + (1/840) * x @ y @ x @ x @ y @ x @ x + (1/840) * x @ y @ x @ x @ y @ x @ y + (1/840) * x @ y @ x @ x @ y @ y @ x + (1/840) * x @ y @ x @ y @ x @ x @ y + (1/840) * x @ y @ x @ y @ x @ y @ y + (1/840) * x @ y @ x @ y @ y @ x @ x + (1/840) * x @ y @ x @ y @ y @ x @ y + (1/840) * x @ y @ y @ x @ x @ y @ x + (1/840) * x @ y @ y @ x @ y @ x @ x + (1/840) * x @ y @ y @ x @ y @ x @ y + (1/840) * x @ y @ y @ x @ y @ y @ x + (1/840) * y @ x @ x @ y @ x @ x @ y + (1/840) * y @ x @ x @ y @ x @ y @ x + (1/840) * y @ x @ x @ y @ x @ y @ y + (1/840) * y @ x @ x @ y @ y @ x @ y + (1/840) * y @ x @ y @ x @ x @ y @ x + (1/840) * y @ x @ y @ x @ x @ y @ y + (1/840) * y @ x @ y @ x @ y @ x @ x + (1/840) * y @ x @ y @ x @ y @ y @ x + (1/840) * y @ x @ y @ y @ x @ x @ y + (1/840) * y @ x @ y @ y @ x @ y @ x + (1/840) * y @ x @ y @ y @ x @ y @ y + (1/840) * y @ y @ x @ x @ y @ x @ y + (1/840) * y @ y @ x @ y @ x @ x @ y + (1/840) * y @ y @ x @ y @ x @ y @ x + (1/840) * y @ y @ x @ y @ x @ y @ y + (1/840) * y @ y @ x @ y @ y @ x @ y)
        '''
        return (
            (-1/140) * x @ y @ x @ y @ x @ y @ x + (-1/140) * y @ x @ y @ x @ y @ x @ y + (-1/630) * x @ x @ x @ y @ x @ y @ x + (-1/630) * x @ y @ x @ x @ x @ y @ x + (-1/630) * x @ y @ x @ y @ x @ x @ x + (-1/630) * x @ y @ x @ y @ y @ y @ x + (-1/630) * x @ y @ y @ y @ x @ y @ x + (-1/630) * y @ x @ x @ x @ y @ x @ y + (-1/630) * y @ x @ y @ x @ x @ x @ y + (-1/630) * y @ x @ y @ x @ y @ y @ y + (-1/630) * y @ x @ y @ y @ y @ x @ y + (-1/630) * y @ y @ y @ x @ y @ x @ y + (-1/1120) * x @ x @ y @ x @ x @ y @ y + (-1/1120) * x @ x @ y @ y @ x @ x @ y + (-1/1120) * x @ x @ y @ y @ x @ y @ y + (-1/1120) * x @ y @ y @ x @ x @ y @ y + (-1/1120) * y @ x @ x @ y @ y @ x @ x + (-1/1120) * y @ y @ x @ x @ y @ x @ x + (-1/1120) * y @ y @ x @ x @ y @ y @ x + (-1/1120) * y @ y @ x @ y @ y @ x @ x + (-1/1512) * x @ x @ x @ y @ x @ x @ x + (-1/1512) * x @ x @ x @ y @ y @ y @ x + (-1/1512) * x @ y @ y @ y @ x @ x @ x + (-1/1512) * y @ x @ x @ x @ y @ y @ y + (-1/1512) * y @ y @ y @ x @ x @ x @ y + (-1/1512) * y @ y @ y @ x @ y @ y @ y + (-1/5040) * x @ x @ x @ x @ x @ y @ x + (-1/5040) * x @ x @ x @ x @ x @ y @ y + (-1/5040) * x @ x @ x @ y @ x @ x @ y + (-1/5040) * x @ x @ x @ y @ x @ y @ y + (-1/5040) * x @ x @ x @ y @ y @ x @ x + (-1/5040) * x @ x @ x @ y @ y @ x @ y + (-1/5040) * x @ x @ y @ x @ x @ x @ y + (-1/5040) * x @ x @ y @ x @ y @ y @ y + (-1/5040) * x @ x @ y @ y @ x @ x @ x + (-1/5040) * x @ x @ y @ y @ y @ x @ x + (-1/5040) * x @ x @ y @ y @ y @ x @ y + (-1/5040) * x @ x @ y @ y @ y @ y @ y + (-1/5040) * x @ y @ x @ x @ x @ x @ x + (-1/5040) * x @ y @ x @ x @ x @ y @ y + (-1/5040) * x @ y @ x @ x @ y @ y @ y + (-1/5040) * x @ y @ y @ x @ x @ x @ y + (-1/5040) * x @ y @ y @ x @ y @ y @ y + (-1/5040) * x @ y @ y @ y @ x @ x @ y + (-1/5040) * x @ y @ y @ y @ x @ y @ y + (-1/5040) * x @ y @ y @ y @ y @ y @ x + (-1/5040) * y @ x @ x @ x @ x @ x @ y + (-1/5040) * y @ x @ x @ x @ y @ x @ x + (-1/5040) * y @ x @ x @ x @ y @ y @ x + (-1/5040) * y @ x @ x @ y @ x @ x @ x + (-1/5040) * y @ x @ x @ y @ y @ y @ x + (-1/5040) * y @ x @ y @ y @ x @ x @ x + (-1/5040) * y @ x @ y @ y @ y @ x @ x + (-1/5040) * y @ x @ y @ y @ y @ y @ y + (-1/5040) * y @ y @ x @ x @ x @ x @ x + (-1/5040) * y @ y @ x @ x @ x @ y @ x + (-1/5040) * y @ y @ x @ x @ x @ y @ y + (-1/5040) * y @ y @ x @ x @ y @ y @ y + (-1/5040) * y @ y @ x @ y @ x @ x @ x + (-1/5040) * y @ y @ x @ y @ y @ y @ x + (-1/5040) * y @ y @ y @ x @ x @ y @ x + (-1/5040) * y @ y @ y @ x @ x @ y @ y +
            (-1/5040) * y @ y @ y @ x @ y @ x @ x + (-1/5040) * y @ y @ y @ x @ y @ y @ x + (-1/5040) * y @ y @ y @ y @ y @ x @ x + (-1/5040) * y @ y @ y @ y @ y @ x @ y + (1/30240) * x @ x @ x @ x @ x @ x @ y + (1/30240) * x @ y @ y @ y @ y @ y @ y + (1/30240) * y @ x @ x @ x @ x @ x @ x + (1/30240) * y @ y @ y @ y @ y @ y @ x + (1/3780) * x @ x @ x @ x @ y @ y @ y + (1/3780) * x @ x @ x @ y @ y @ y @ y + (1/3780) * y @ y @ y @ x @ x @ x @ x + (1/3780) * y @ y @ y @ y @ x @ x @ x + (1/2016) * x @ x @ x @ x @ y @ x @ x + (1/2016) * x @ x @ x @ x @ y @ x @ y + (1/2016) * x @ x @ x @ x @ y @ y @ x + (1/2016) * x @ x @ y @ x @ x @ x @ x + (1/2016) * x @ x @ y @ y @ y @ y @ x + (1/2016) * x @ y @ x @ x @ x @ x @ y + (1/2016) * x @ y @ x @ y @ y @ y @ y + (1/2016) * x @ y @ y @ x @ x @ x @ x + (1/2016) * x @ y @ y @ y @ y @ x @ x + (1/2016) * x @ y @ y @ y @ y @ x @ y + (1/2016) * y @ x @ x @ x @ x @ y @ x + (1/2016) * y @ x @ x @ x @ x @ y @ y + (1/2016) * y @ x @ x @ y @ y @ y @ y + (1/2016) * y @ x @ y @ x @ x @ x @ x + (1/2016) * y @ x @ y @ y @ y @ y @ x + (1/2016) * y @ y @ x @ x @ x @ x @ y + (1/2016) * y @ y @ x @ y @ y @ y @ y + (1/2016) * y @ y @ y @ y @ x @ x @ y + (1/2016) * y @ y @ y @ y @ x @ y @ x + (1/2016) *
            y @ y @ y @ y @ x @ y @ y + (1/840) * x @ x @ y @ x @ x @ y @ x + (1/840) * x @ x @ y @ x @ y @ x @ x + (1/840) * x @ x @ y @ x @ y @ x @ y + (1/840) * x @ x @ y @ x @ y @ y @ x + (1/840) * x @ x @ y @ y @ x @ y @ x + (1/840) * x @ y @ x @ x @ y @ x @ x + (1/840) * x @ y @ x @ x @ y @ x @ y + (1/840) * x @ y @ x @ x @ y @ y @ x + (1/840) * x @ y @ x @ y @ x @ x @ y + (1/840) * x @ y @ x @ y @ x @ y @ y + (1/840) * x @ y @ x @ y @ y @ x @ x + (1/840) * x @ y @ x @ y @ y @ x @ y + (1/840) * x @ y @ y @ x @ x @ y @ x + (1/840) * x @ y @ y @ x @ y @ x @ x + (1/840) * x @ y @ y @ x @ y @ x @ y + (1/840) * x @ y @ y @ x @ y @ y @ x + (
                1/840) * y @ x @ x @ y @ x @ x @ y + (1/840) * y @ x @ x @ y @ x @ y @ x + (1/840) * y @ x @ x @ y @ x @ y @ y + (1/840) * y @ x @ x @ y @ y @ x @ y + (1/840) * y @ x @ y @ x @ x @ y @ x + (1/840) * y @ x @ y @ x @ x @ y @ y + (1/840) * y @ x @ y @ x @ y @ x @ x + (1/840) * y @ x @ y @ x @ y @ y @ x + (1/840) * y @ x @ y @ y @ x @ x @ y + (1/840) * y @ x @ y @ y @ x @ y @ x + (1/840) * y @ x @ y @ y @ x @ y @ y + (1/840) * y @ y @ x @ x @ y @ x @ y + (1/840) * y @ y @ x @ y @ x @ x @ y + (1/840) * y @ y @ x @ y @ x @ y @ x + (1/840) * y @ y @ x @ y @ x @ y @ y + (1/840) * y @ y @ x @ y @ y @ x @ y
        ) '''
    elif order == 8:
        return ((-1/280) * x @ y @ x @ y @ x @ y @ x @ y + (-1/1260) * x @ x @ x @ y @ x @ y @ x @ y + (-1/1260) * x @ y @ x @ x @ x @ y @ x @ y + (-1/1260) * x @ y @ x @ y @ x @ x @ x @ y + (-1/1260) * x @ y @ x @ y @ x @ y @ y @ y + (-1/1260) * x @ y @ x @ y @ y @ y @ x @ y + (-1/1260) * x @ y @ y @ y @ x @ y @ x @ y + (-1/1680) * y @ x @ x @ y @ x @ x @ y @ x + (-1/1680) * y @ x @ x @ y @ x @ y @ x @ x + (-1/1680) * y @ x @ x @ y @ x @ y @ y @ x + (-1/1680) * y @ x @ x @ y @ y @ x @ y @ x + (-1/1680) * y @ x @ y @ x @ x @ y @ x @ x + (-1/1680) * y @ x @ y @ x @ x @ y @ y @ x + (-1/1680) * y @ x @ y @ x @ y @ y @ x @ x + (-1/1680) * y @ x @ y @ y @ x @ x @ y @ x + (-1/1680) * y @ x @ y @ y @ x @ y @ x @ x + (-1/1680) * y @ x @ y @ y @ x @ y @ y @ x + (-1/1680) * y @ y @ x @ x @ y @ x @ y @ x + (-1/1680) * y @ y @ x @ y @ x @ x @ y @ x + (-1/1680) * y @ y @ x @ y @ x @ y @ x @ x + (-1/1680) * y @ y @ x @ y @ x @ y @ y @ x + (-1/1680) * y @ y @ x @ y @ y @ x @ y @ x + (-1/2240) * x @ x @ y @ y @ x @ x @ y @ y + (-1/3024) * x @ x @ x @ y @ x @ x @ x @ y + (-1/3024) * x @ x @ x @ y @ x @ y @ y @ y + (-1/3024) * x @ x @ x @ y @ y @ y @ x @ y + (-1/3024) * x @ y @ x @ x @ x @ y @ y @ y + (-1/3024) * x @ y @ y @ y @ x @ x @ x @ y + (-1/3024) * x @ y @ y @ y @ x @ y @ y @ y + (-1/4032) * y @ x @ x @ x @ x @ y @ x @ x + (-1/4032) * y @ x @ x @ x @ x @ y @ y @ x + (-1/4032) * y @ x @ x @ y @ x @ x @ x @ x + (-1/4032) * y @ x @ x @ y @ y @ y @ y @ x + (-1/4032) * y @ x @ y @ y @ x @ x @ x @ x + (-1/4032) * y @ x @ y @ y @ y @ y @ x @ x + (-1/4032) * y @ y @ x @ x @ x @ x @ y @ x + (-1/4032) * y @ y @ x @ y @ x @ x @ x @ x + (-1/4032) * y @ y @ x @ y @ y @ y @ y @ x + (-1/4032) * y @ y @ y @ y @ x @ x @ y @ x + (-1/4032) * y @ y @ y @ y @ x @ y @ x @ x + (-1/4032) * y @ y @ y @ y @ x @ y @ y @ x + (-23/120960) * y @ y @ y @ y @ x @ x @ x @ x + (-1/10080) * x @ x @ x @ x @ x @ y @ x @ y + (-1/10080) * x @ x @ x @ x @ x @ y @ y @ y + (-1/10080) * x @ x @ x @ y @ x @ x @ y @ y + (-1/10080) * x @ x @ x @ y @ y @ x @ x @ y + (-1/10080) * x @ x @ x @ y @ y @ x @ y @ y + (-1/10080) * x @ x @ x @ y @ y @ y @ y @ y + (-1/10080) * x @ x @ y @ x @ x @ x @ y @ y + (-1/10080) * x @ x @ y @ x @ x @ y @ y @ y + (-1/10080) * x @ x @ y @ y @ x @ x @ x @ y + (-1/10080) * x @ x @ y @ y @ x @ y @ y @ y + (-1/10080) * x @ x @ y @ y @ y @ x @ x @ y + (-1/10080) * x @ x @ y @ y @ y @ x @ y @ y + (-1/10080) * x @ y @ x @ x @ x @ x @ x @ y + (-1/10080) * x @ y @ x @ y @ y @ y @ y @ y + (-1/10080) * x @ y @ y @ x @ x @ x @ y @ y + (-1/10080) * x @ y @ y @ x @ x @ y @ y @ y + (-1/10080) * x @ y @ y @ y @ x @ x @ y @ y + (-1/10080) * x @ y @ y @ y @ y @ y @ x @ y + (-1/60480) * y @ y @ x @ x @ x @ x @ x @ x + (-1/60480) * y @ y @ y @ y @ y @ y @ x @ x + (1/60480) * x @ x @ x @ x @ x @ x @ y @ y + (1/60480) * x @ x @ y @ y @ y @ y @ y @ y + (1/10080) * y @ x @ x @ x @ x @ x @ y @ x + (1/10080) * y @ x @ x @ x @ y @ y @ x @ x + (1/10080) * y @ x @ x @ y @ y @ x @ x @ x + (1/10080) * y @ x @ x @ y @ y @ y @ x @ x + (1/10080) * y @ x @ y @ x @ x @ x @ x @ x + (1/10080) * y @ x @ y @ y @ y @ y @ y @ x + (1/10080) * y @ y @ x @ x @ x @ y @ x @ x + (1/10080) * y @ y @ x @ x @ x @ y @ y @ x + (1/10080) * y @ y @ x @ x @ y @ x @ x @ x + (1/10080) * y @ y @ x @ x @ y @ y @ y @ x + (1/10080) * y @ y @ x @ y @ y @ x @ x @ x + (1/10080) * y @ y @ x @ y @ y @ y @ x @ x + (1/10080) * y @ y @ y @ x @ x @ x @ x @ x + (1/10080) * y @ y @ y @ x @ x @ y @ x @ x + (1/10080) * y @ y @ y @ x @ x @ y @ y @ x + (1/10080) * y @ y @ y @ x @ y @ y @ x @ x + (1/10080) * y @ y @ y @ y @ y @ x @ x @ x + (1/10080) * y @ y @ y @ y @ y @ x @ y @ x + (23/120960) * x @ x @ x @ x @ y @ y @ y @ y + (1/4032) * x @ x @ x @ x @ y @ x @ x @ y + (1/4032) * x @ x @ x @ x @ y @ x @ y @ y + (1/4032) * x @ x @ x @ x @ y @ y @ x @ y + (1/4032) * x @ x @ y @ x @ x @ x @ x @ y + (1/4032) * x @ x @ y @ x @ y @ y @ y @ y + (1/4032) * x @ x @ y @ y @ y @ y @ x @ y + (1/4032) * x @ y @ x @ x @ x @ x @ y @ y + (1/4032) * x @ y @ x @ x @ y @ y @ y @ y + (1/4032) * x @ y @ y @ x @ x @ x @ x @ y + (1/4032) * x @ y @ y @ x @ y @ y @ y @ y + (1/4032) * x @ y @ y @ y @ y @ x @ x @ y + (1/4032) * x @ y @ y @ y @ y @ x @ y @ y + (1/3024) * y @ x @ x @ x @ y @ x @ x @ x + (1/3024) * y @ x @ x @ x @ y @ y @ y @ x + (1/3024) * y @ x @ y @ y @ y @ x @ x @ x + (1/3024) * y @ y @ y @ x @ x @ x @ y @ x + (1/3024) * y @ y @ y @ x @ y @ x @ x @ x + (1/3024) * y @ y @ y @ x @ y @ y @ y @ x + (1/2240) * y @ y @ x @ x @ y @ y @ x @ x + (1/1680) * x @ x @ y @ x @ x @ y @ x @ y + (1/1680) * x @ x @ y @ x @ y @ x @ x @ y + (1/1680) * x @ x @ y @ x @ y @ x @ y @ y + (1/1680) * x @ x @ y @ x @ y @ y @ x @ y + (1/1680) * x @ x @ y @ y @ x @ y @ x @ y + (1/1680) * x @ y @ x @ x @ y @ x @ x @ y + (1/1680) * x @ y @ x @ x @ y @ x @ y @ y + (1/1680) * x @ y @ x @ x @ y @ y @ x @ y + (1/1680) * x @ y @ x @ y @ x @ x @ y @ y + (1/1680) * x @ y @ x @ y @ y @ x @ x @ y + (1/1680) * x @ y @ x @ y @ y @ x @ y @ y + (1/1680) * x @ y @ y @ x @ x @ y @ x @ y + (1/1680) * x @ y @ y @ x @ y @ x @ x @ y + (1/1680) * x @ y @ y @ x @ y @ x @ y @ y + (1/1680) * x @ y @ y @ x @ y @ y @ x @ y + (1/1260) * y @ x @ x @ x @ y @ x @ y @ x + (1/1260) * y @ x @ y @ x @ x @ x @ y @ x + (1/1260) * y @ x @ y @ x @ y @ x @ x @ x + (1/1260) * y @ x @ y @ x @ y @ y @ y @ x + (1/1260) * y @ x @ y @ y @ y @ x @ y @ x + (1/1260) * y @ y @ y @ x @ y @ x @ y @ x + (1/280) * y @ x @ y @ x @ y @ x @ y @ x)
    else:
        raise ValueError("order > 8 does not implement")




def bchHamiltonian(matT, matV, order):
    s = -1.j / hbar * T_kick
    x = matV
    y = matT
    out = sum([s ** (i - 1) * _bchHamitonianTerm(x, y, i)
               for i in range(1, order + 1)])
    return out

    '''
ハミルトニアンの量子BCH近似式をorderの項までの書き下し．
H = V + K + T/(i*hbar) * 1/2 * [V, K] + T^2/(i*habr)^2 * [V - K, [V , K]] + ...
'''


# 位置と運動量の定義


def get_qp(n=N):
    q = np.linspace(-BC, BC, n)
    nu = fftfreq(n, d=dQ)  # nuは周波数
    p = 2 * np.pi * nu  * hbar  # 周波数から運動量に変換
    return q, p


# ポテンシャルの定義
#def V(q):
#    return .5 * q ** 2 - 2 * np.cos(q) - np.sqrt(np.pi) * scipy.special.erf(q) / 2

def V(q):
    return - (k/16) * np.exp(-8 * (np.array(q) ** 2)) - ((e2/(2 * (8 ** 0.5))) * np.sqrt(np.pi)) * (scipy.special.erf( (8 ** 0.5)* ( np.array(q) - xb)) - scipy.special.erf((8 ** 0.5) *( np.array(q) + xb)))

lamb =1.2
def V(q):
    return q ** 2 /2 - 2 * np.cos(q/lamb)
# 運動エネルギーの定義
def T(p):
    return 0.5 * p ** 2


# 規格化
def renormalize(psi):
    prob = np.real(np.conj(psi) * psi)  # 波動関数から確率へ変換
    renorm = np.sqrt((prob.sum() * dQ) + 1e-10)
    out = psi / renorm
    return out


# 初期コヒーレント状態
def init_state(q):
    d = q #- 2
    psi = np.power(np.pi, -1/4) * np.exp(-.5 * d ** 2 / hbar)
    psi = psi.astype(np.complex_)
    return renormalize(psi)


# TVオーダーの行列計算
def TVevolve(psi, dt, q, p):
    psi = np.exp(-V(q) * dt * 1.j / hbar) * psi
    psi = np.exp(-T(p) * dt * 1.j / hbar) * fft(psi)
    return ifft(psi)

# TVオーダーの行列計算
def TVevolvewithabsorb(psi, dt, q, p):
    psi = np.exp(-V(q) * dt * 1.j / hbar) * psi
    psi = np.exp(-T(p) * dt * 1.j / hbar) * fft(psi)
    psi = ifft(psi)
    psi = tanh_abs2(q, 1.2, 10) * psi
    print(psi)
    psi = tanh_abs(q, -1.2, 10) * psi
    print(psi)
    return psi


def tanh_abs(x,x_c,beta):
    xx = beta*((x-x_c))
    return (np.tanh(xx)+1)/2

def tanh_abs2(x,x_c,beta):
    xx = beta*(-(x-x_c))
    return (np.tanh(xx)+1)/2

def exp_abs(x, x1,x2,alpha,beta):
    x1 = (x - x1)
    x2 = (x - x2)
    theta0= (1 - np.tanh(x1*beta))/2
    theta1= (1 + np.tanh(x2*beta))/2
    w = (x1**2*theta0 +x2**2*theta1)
    p = np.exp(-w/alpha)
    return p

# VTオーダーの行列計算
def VTevolve(psi, dt, q, p):
    psi = jnp.exp(+T(p) * dt * 1.j / hbar) * fft(psi)
    psi = jnp.exp(+V(q) * dt * 1.j / hbar) * ifft(psi)
    return psi



# ポテンシャルの行列表現．対角に並べただけ．
def matrix_V():
    q, p = get_qp()
    return np.diag(V(q))


# 運動エネルギーの行列表現．


def matrix_T():
    q, p = get_qp()
    matT = np.zeros((N, N), dtype=np.complex_)
    for j in range(N):
        psi_j = np.zeros(N, dtype=np.complex_)
        psi_j[j] = 1.0
        pvec = fft(psi_j)
        pvec = T(p) * pvec  # WHY HANADA?
        matT[j] = ifft(pvec)
    return matT.T


# ユニタリ行列の対角化
def unitary_eig(matrix):
    R, V = scipy.linalg.schur(matrix, output="complex")
    return R.diagonal(), V.T


# BCH波動関数の計算
def bch_evolve(evals, evecs, phi0, n):
    phi = np.zeros(N, dtype=nb.complex128)
    for j in range(N):
        u_j = np.exp(-1.j * evals[j] / hbar * n)
        psi_j = evecs[j]
        phi += u_j * np.dot(np.conj(psi_j), phi0) * \
            psi_j  # u^n_j <psi_j|phi_0>|psi_j>
    return renormalize(phi)

def main3(ax):#量子状態の図示
    n_steps = 0
    dt = T_kick
    q, p = get_qp()
    phi0 = init_state(q).astype(np.complex_)
    matT = matrix_T()
    matV = matrix_V()
    #H = bchHamiltonian(matT, matV, 3)
    H = matT + matV
    #U = np.exp(-i * H / hbar)bb
    bch_evals1, bch_evecs1 = np.linalg.eigh(H)
    bch_evecs1 = bch_evecs1.T
    h = np.arange(0,N,1)
    print(bch_evals1,h)
    #fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xlim(-BC, BC)
    ax.set_ylim(-BC, BC)
    ax.set_title(r'$N = {}, \tau = {}, BC = {}, grid = {}$'.format(
        n_steps, T_kick, QMAX - QMIN, N),fontsize = 20)
    phi0 = np.array(phi0)
    #bch_prob1 = psi * np.conj(psi)
    np.max(phi0)
    bch_evals1 = np.array(bch_evals1)
    bch_evecs1 = np.array(bch_evecs1)
    a = np.linspace(-BC,BC, 10 ** 3)
    b = np.linspace(-BC,BC, 10 ** 3)
    A,B = np.meshgrid(a,b)
    z = T(A) + V(B)
    #bch_psi1 = bch_evolve(bch_evals1, bch_evecs1, bch_evecs1[0], n_steps * T_kick)
    for i in range(0,500):
        psi = bch_evecs1[i]
        density  = psi * np.conj(psi)
        print( np.abs(q[np.argmax(density)]),h[i])
        #if i %5 == 0:
        #    cntr = ax.contour(A,B,z,levels = [bch_evals1[i]],color = "blue")
        #    ax.clabel(cntr)
        if i == 125 or i == 251 or i == 375 or i == 189:
            cntr = ax.contour(A,B,z,levels = [bch_evals1[i]],color = "red")
            ax.clabel(cntr)
    ax.set_xlabel(r"$q$",fontsize = 25)
    ax.set_ylabel(r"$p$",fontsize = 25)
    plt.tick_params(labelsize = 20)
    #ax.set_ylim(0.000,0.013)
    #plt.show()

#seed = np.loadtxt('step{}_11_a_{}_b_{}.txt'.format(step,a_str,b_str)) #ここに読み込みたいMsetのファイルを持ってくる。
#theta0 = np.loadtxt('crosspoint.txt')
#theta0 = theta0.T
#fig.canvas.mpl_connect('button_press_event',onclick)
textpath = 'initpoint_10_13_test14_a{}_b{}.txt'.format(str(round(a_orb,3)).replace('.',''), str(round(b_orb,3)).replace('.',''))

mset()
#point = np.loadtxt("ctheta_section_period{}.txt".format(period),dtype = np.complex128)
#fig = plt.figure(figsize = (6,6))
#plot2.main() 

#plt.plot(seed[:,0],seed[:,1],',k',zorder = 1)
#plt.plot(6.101014,0.57610,'o')
#plt.xlim(0.,3.14)
#plt.ylim(2.76,5.02)


#textpath = 'initpoint_10_13_test14_a_{}_b_{}.txt'.format(str(round(a_orb,3)).replace('.','z'), str(round(b_orb,3)).replace('.','z'))
textpath = 'initpoint_10_13_test14_a_{}_b_{}.txt'.format(str(round(a_orb,3)).replace('.','z'), str(round(b_orb,3)).replace('.','z'))



#fig.canvas.mpl_connect('button_press_event',onclick)#(なぜonkeyが使えないのか...)
#plt.show()
