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
b_orb = 0.44515836
a_orb = 0.57929427
t = 1.5096941616
tau = 0.05
b_orb = 1.04079259
a_orb = 1.10344608
t = 1.337145721
lambda_1 = 1.2
if lambda_1 == 1.2:
    b_orb = 1.53306952
    a_orb = 0.65363539
    t = 1.527207
    text = "phase_shearless_12.txt"
elif lambda_1 == 3.0:
    b_orb = 1.04079259
    a_orb = 1.10344608
    t = 1.337145721
    text = "phase_shearless_30.txt"

#[1.04079259 1.10344708] #\lambda = 3.0の場合
#[0.23153047 0.97282765]
#angle between major axis and x axis= 1.3371457211538098
#angle between minor axis and x axis= -0.23365060564108692
#[1.06462819 1.37029825] lambda = 1.2　の場合
#[0.06309614 0.99800745]
#angle between major axis and x axis= 1.5076582419255626
#angle between minor axis and x axis= -0.06313808486933388



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
    plt.ylabel("Im(S)",fontsize = 27)
    plt.xlabel("Time step",fontsize = 27)
    ax.set_xlabel(r"${\rm Re}(q)$",labelpad = 20)
    ax.set_ylabel(r"${\rm Re}(p)$",labelpad = 20)
    ax.set_zlabel(r"${\rm Im}(q)$",labelpad = 30)
    #$plt.ylim(-100,100)
    #plt.ylim(0,0.1)
    colormap = plt.get_cmap("tab10")
    points = [np.array([])] * 2
    while i < 500:
        points[0] = np.append(points[0], x_cross[0])
        points[1] = np.append(points[1], x_cross[1])
        iterates = np.append(iterates, i)
        actions = np.append(actions, S.imag)
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
        S += 0.5 * tau * (x_cross[1] ** 2) + tau * V(prepoint[0]) - x_cross[1] * ( x_cross[0] - prepoint[0] ) 
        #actions = np.append(actions, S.imag)
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
    ax_proj.set_xlabel(r"${\rm Re}(q)$")
    ax_proj.set_ylabel(r"${\rm Re}(p)$")
    ax_proj.set_xlim( -60,60)
    ax_proj.set_ylim( -60,60)
    ax2.set_ylim(-50,190)
    xi_str = str(round(theta_0.real,6))
    eta_str = str(round(theta_0.imag,6))
    ax.set_title("xi = {},eta = {}".format(xi_str, eta_str))
    if lambda_1 == 3.0:
        ax_proj.set_xlim( -30,30)
        ax_proj.set_ylim( -30,30)
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
    height  = 4.8
    grid =300
    grid_mset = 1500
    grid2 = grid - 1
    step = 3000
    step2 = 157 #msetを図示する用
    step3 = 189 #msetを図示する用
    width = 2 * np.pi
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
    div_numbers_checkbool  =  np.linalg.norm(qp_rotated,axis = 0)  < 100 #すでに発散したかどうか
    cm = plt.cm.get_cmap('seismic')
    plt.xlabel(r"$\xi$",fontsize = 20)
    plt.ylabel(r"$\eta$",fontsize = 20,rotation  = "horizontal",labelpad = 20)
    print(qp_rotated.shape)
    xi2_mset = xi2_mset[:,:-1]
    eta2_mset = eta2_mset[:,:-1]
    for i in range(step):
        qp_rotated = cmap.U(qp_rotated)
        boolean = np.logical_or(np.linalg.norm(qp_rotated,axis = 0) > 100, np.isnan(np.linalg.norm(qp_rotated,axis = 0)) == True)#発散してるところ
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
             #   boolean_mset =  np.logical_and(np.diff( np.sign(qp_rotated_mset[0].imag) ) != 0, qp_rotated_mset[0][:,:-1] < 10 ** 5)
             #   plt.plot(xi2_mset[boolean_mset],eta2_mset[boolean_mset],',r',zorder = 3)
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
    plt.scatter(xi2,eta2, s=0.1, c=div_numbers, marker = '.',cmap='seismic', vmax =step,vmin = 0,alpha = 0.5)
    cbar = plt.colorbar()
    cbar.ax.tick_params(labelsize=20)
    cbar.set_label("Time step",size = 25)
    plt.rcParams["font.size"] = 30
    #plt.xlim(0,2 * np.pi)
    plt.tick_params(labelsize=20)
    plt.ylim(0,height-0.2)
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
    grid = 300
    step = 3000
    step2 = 157 #msetを図示する用
    step3 = 189 #msetをもう一つ図示する用
    grid2 = grid - 1
    grid_mset = 1500
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
    div_numbers_checkbool  =  np.linalg.norm(qp_rotated,axis = 0)  < 100 #すでに発散したかどうか
    cm = plt.cm.get_cmap('seismic')
    print(qp_rotated.shape)
    xi2_mset = xi2_mset[:,:-1]
    eta2_mset = eta2_mset[:,:-1]
    for  i in range(step):
        qp_rotated = cmap.U(qp_rotated)
        boolean = np.logical_or(np.linalg.norm(qp_rotated,axis = 0) > 100, np.isnan(np.linalg.norm(qp_rotated,axis = 0)) == True)#発散してるところ
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
        #    if i == step2 - 1 :
        #        boolean_mset =  np.logical_and(np.diff( np.sign(qp_rotated_mset[0].imag) ) != 0, qp_rotated_mset[0][:,:-1] < 10 ** 5)
        #        plt.plot(xi2_mset[boolean_mset],eta2_mset[boolean_mset],',g',zorder = 3)
        #    elif i == step3 - 1:
        #        boolean_mset =  np.logical_and(np.diff( np.sign(qp_rotated_mset[0].imag) ) != 0, qp_rotated_mset[0][:,:-1] < 10 ** 5)
        #        plt.plot(xi2_mset[boolean_mset],eta2_mset[boolean_mset],',k',zorder = 3)
           # elif i == 126 - 1:
           #     boolean_mset =  np.logical_and(np.diff( np.sign(qp_rotated_mset[0].imag) ) != 0, qp_rotated_mset[0][:,:-1] < 10 ** 5)
           #     plt.plot(xi2_mset[boolean_mset],eta2_mset[boolean_mset],',r',zorder = 3)
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
    plt.scatter(xi2,eta2, s=0.1, c=div_numbers, marker = '.',cmap='seismic', vmax = step,vmin = 0,alpha = 0.7)
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
    return  (a_orb*b_orb/2) * (theta + (1/2) * np.sin(2 * theta) * np.cos(2 * t)) - (1/4) * (a_orb ** 2 + b_orb ** 2) * np.cos(2 * theta) * np.sin(2 * t)

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
