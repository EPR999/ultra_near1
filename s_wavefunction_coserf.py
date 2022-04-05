# -*- coding:utf-8 -*-
import numpy as np
import sys
import matplotlib.pyplot as plt
import numpy as np
import pylab
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import special
from colorama import Fore, Back, Style
import math
import os


Seeds = [[],[],[]]
psi = [[],[],[]]#[key,value,p]
density = [[],[]]#[value,p]
densityhist = [[],[],[]]
twopi = 2.0*np.pi
grid=500
regrid = 500
pgrid = 2000
omega = 1
initp = 0
imsize = 3
resize = 4
realscale = 1
imscale = 1 #(realscale,imscale)=(1,1) is used as usual.
step =7#42これ任意のステップでなんとかなるように.実際に出力したマップとの整合性に気をつけて.
radius = 0
step = 120
diskgrid = 200
key= 0
originseed = []
originseed2 =[]
origin = np.array([])
arg2 = 0
radius = 0.02
addorbit = True
a_orb = 0.22
b_orb = 0.06
t = np.pi/3
b_orb = 1.06462819
a_orb = 1.37029825
t = 1.50765824
timestep = step
planckconstant = 0.01
T = 0.05
np.set_printoptions(threshold =  5000)
###################

stdrad = 0.02000001#初期面の半径.
k = 3.0
xf = 1.2
xb = 1.0
e2 = k * xf * (np.exp(-8 * xb * (2 * xf - xb)) / (1 - np.exp(-32 * xf * xb)))
    
#==================

#tau =  0.04
#k = 

#これ単体で動かすことはない.
def V(x): 
    q = x[0] 
    p = x[1]
    return - (k/16) * np.exp(-8 * (q ** 2)) - ((e2/(2 * (8 ** 0.5))) * np.sqrt(np.pi)) * (special.erf( (8 ** 0.5)* (q - xb)) - special.erf((8 ** 0.5) *( q+ xb)))

def V(x):
    q = x[0] 
    p = x[1]
    return q ** 2/ 2 - 2 *  np.cos(q)
def dotV(q):
    return q + 2 *  np.sin(q) 

def ddotV(q):
    return 1 + 2 * np.cos(q)

lambda_1 = 1.2
def dotV(x):
    return x +1/lambda_1   * np.sin(x/lambda_1)
def V(x):
    return np.array(x) ** 2 - np.cos(x/lambda_1) 

class KScatteringMap:
    def __init__(self,k,xf,xb):
        self.k = k
        self.xf = xf
        self.xb = xb
    #正の時間方向の写像
    def U(self,qp):
        return np.array([qp[0] + qp[1] - 0.5 * dotV(qp[0]), qp[1] - 0.5 * dotV(qp[0]) - 0.5 * dotV(qp[0] + qp[1] - 0.5 * dotV(qp[0])) ])
    
    #逆写像
    def Ui(self,qp):
        return np.array([qp[0] - qp[1] - 0.5 * dotV(qp[0]), qp[1] + 0.5 * dotV(qp[0]) + 0.5 * dotV(qp[0] - qp[1] - 0.5 * dotV(qp[0]))])

def ScattMapt(qp,iterate):
    for i in range(step):
        qp[0],qp[1] =  cmap.U(qp)
    return np.array([qp[0],qp[1]])
def ScattMapti(qp,iterate):
    for i in range(step):
        qp[0],qp[1] =  cmap.Ui(qp)
    return np.array([qp[0],qp[1]])
##================================================
class KScatteringMap2:
    def __init__(self,k,xf,xb):
        self.k = k
        self.xf = xf
        self.xb = xb
    #正の時間方向の写像
    def U(self,qp):
        return np.array([qp[0] + qp[1] - dotV( qp[0]) , qp[1] - dotV(qp[0])  ])
    
    #逆写像
    def Ui(self,qp):
        return np.array([qp[0] - qp[1], qp[1] -  dotV(qp[0] - qp[1]) ])

class KScatteringMap3:
    def __init__(self,k,xf,xb):
        self.k = k
        self.xf = xf
        self.xb = xb
    #正の時間方向の写像
    def U(self,qp):
        return np.array([qp[0] + qp[1] - dotV( qp[0]) , qp[1] - dotV(qp[0])  ])
    
    #逆写像
    def Ui(self,qp):
        return np.array([qp[0] - qp[1], qp[1] -  dotV(qp[0] - qp[1]) ])

def ScattMapt(qp,iterate):
    for i in range(step):
        qp[0],qp[1] =  cmap.U(qp)
    return np.array([qp[0],qp[1]])
def ScattMapti(qp,iterate):
    for i in range(step):
        qp[0],qp[1] =  cmap.Ui(qp)
    return np.array([qp[0],qp[1]])


#================================================
#theta,q_im(初期条件) →　q,pの実部
def init(theta):#角の情報を座標の情報に変換する,
    q = a_orb * np.sin(theta)
    p = b_orb * np.cos(theta)
    return np.array([q,p]) #初期運動量もthetaとともに確定する.
#===================================================

def cp(x):#実部と虚部から複素数をかえす
    z = x[:,0] + 1j * x[:,1]
    return z
#=====================================================

#確率密度関数を求める
def probdensity(psi):
    #orbit has information of inittheta
    #print(psi)
    density = psi * np.conj(psi)
    #print(density)
    #exit()
    #print(density[::10])
    return density

#点の情報を代入して作用を求める.

def WaveFunction(S):
    #print(np.imag(S))
    #planck = 10 ** 34
    #print(S)
    #print(S.imag /planckconstant)
    #exit()
    #exit()
    psi = np.exp(-( S.imag /planckconstant ))
    #rint(psi)
    return psi

def rotateinitialmanifold(a,b,t,points):
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    R = np.array([[cos_t,sin_t],[-sin_t,cos_t]])
    rotated = np.dot(R,points)
    return rotated

def action(x,iter,func):
    S = np.zeros_like(x[:,0],dtype = np.complex128)
    theta_0 = cp(x)
    points = init(theta_0)
    points = rotateinitialmanifold(a_orb,b_orb,t,points)
    #print(points)
    for i in range(iter):
        #preI = points[0] ** 2 + points[1] ** 2   
        prepoints = points
        points =  func(points)
       #prethetaforsin = CASIN(points[1] /(preI  ** 0.5))   
        #I = points[0,:] ** 2 +  points[1,:] ** 2
        # S += 0.5 * x[1] ** 2 + V(x) - x[1] * ( - 0.5 * dotV(x[0]) - 0.5 * dotV(x[0] + x[1] - 0.5 * dotV(x[0]) ) )#終わりから始めを引け
        S += 0.5 * (points[1] ** 2) + V(prepoints[0]) - points[1] * ( points[0] - prepoints[0] ) 
        #print("Im(S)=",np.abs(S.imag).min())
        if i == 0:
           S -= generatorfunc2(theta_0)
        #print(  "S + = ",0.5 * (points[1] ** 2) + V(prepoints) - points[1] * ( points[0] - prepoints[0] ) )
        print("point = ",points)

    print('points = ',points[0])
    return S,points[0].real
#===============================================
def gettheta(cpoint):#cpoint から　thetaを得る． Kahan, W: Branch cuts for complex elementary functionを参照
    theta_confs = np.array([])
    I = cpoint[0] ** 2 + cpoint[1] ** 2
    cosine =  cpoint[0]/(I ** (1/2) )
    sine = cpoint[1]/(I ** (1/2))
    #print(cosine)

    theta = CACOS(cosine) #まずcosineに処理を施す
    theta2 = CASIN(sine)
    for i in range(len(theta)):
        if theta.real[i] > 0 and theta2.real[i] < 0:
            theta_conf = theta2.real[i] + 2 * np.pi + 1j * np.abs(theta.imag[i])
        else :
            theta_conf = theta[i]
        theta_conf = theta_conf.real + 1j * theta_conf.imag     
        theta_confs = np.append(theta_conf,theta_confs)
    #print(theta_confs)
    return theta_conf


def gettheta(cpoint):#cpoint から　thetaを得る． Kahan, W: Branch cuts for complex elementary functionを参照
    #print("a = ",cpoint)
    cpoint_rotated = rotateinitialmanifold(a_orb,b_orb,t,cpoint)
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
    confirm_point = rotateinitialmanifold(a_orb,b_orb,-t,confirm_point)
    if  np.sign(confirm_point[0].imag) != np.sign(cpoint[0].imag):
        theta_conf = theta_conf.real - 1j * theta_conf.imag 
    #print(theta_conf)
    confirm_point = np.array([a_orb * np.sin(theta_conf),b_orb * np.cos(theta_conf)])
    confirm_point = rotateinitialmanifold(a_orb,b_orb,-t,confirm_point)
    #print("b = ",confirm_point)
    return theta_conf

def CACOS(z):#mountain Oock cosの値から複素のcomplex thetaを返すよ   
    xi = 2 * np.arctan(np.real((1 - z) ** 0.5) /np.real((1 + z) ** 0.5) )
    eta = np.arcsinh( np.imag( (np.conj(1 +z) ** 0.5) *  ((1-z) ** 0.5)  )) 
    return xi + 1j * eta


def CASIN(z):
    xi = np.arctan( np.real(z) / np.real( (1-z) ** 0.5 * ( 1 +z ) ** 0.5 )) 
    eta = np.arcsinh( np.imag( np.conj(1-z) ** 0.5 *  (1 +z) ** 0.5 ) )
    return xi + 1j * eta
#===============================================
def draw(array):
    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(111)
    plt.plot(array[0],array[1],',k')
    plt.xlim(-2,2)
    plt.show()

def inter(density,q):  
    array  = np.real(np.array([q,density]).T)
    array = np.real(np.array(sorted(array , key = lambda x: np.inf if np.isnan(x[0]) else x[0])))
    #print(array[:,1])
    print(array.shape)
    print(array)
    f = interpolate.interp1d(np.real(array[:,0]),np.real(array[:,1]),assume_sorted = False)
    p1 = np.linspace(-60,60, 10 ** 5)
    print(f(p1))
    #exit()
    return f 
def inter2(density,q):  #バグの出所がわからないため
    array = np.real(np.array([q,density]).T)
    print(array.shape)
    print(array[:,0].shape)
    print(array[1].shape)
    exit()
    array = np.real(np.array(sorted(array , key = lambda x: np.inf if np.isnan(x[0]) else x[0])))
    #print(array[:,1])
    print(array[0].shape)
    print(array[1].shape)
    #exit()
    f = interpolate.interp1d(np.real(array[:,0]),np.real(array[:,1]),assume_sorted = False,kind = 'linear')
    return f 
def generatorfunc(I,theta):#母関数√
    ##print(I)
    return  (I/2) * (theta - (1/2) * np.sin(2 * theta))    
def generatorfunc2(theta):#母関数
    return  (a_orb*b_orb/2) * (theta + (1/2) * np.sin(2 * theta) * np.cos(2 * t)) - (1/8) * (a_orb ** 2 + b_orb ** 2) * np.cos(2 * theta) * np.sin(2 * t)

def sortbranch(key,count):
    largestpoints = np.array([])
    branches = [np.array([])] * count
    branchnumbers = np.array([])
    for i in range(len(count)):
       branch = np.loadtxt("chain{}_branch{}_step{}_a_{}_b_{}.txt".format(chainnumber,step,a_str,b_str))
       largestpoint = np.append(largestpoints,largestpoint)

    jointarray = np.array([largestpoints,branchnumbers])
    jointarray = np.real(np.array(sorted(jointarray , key = lambda x: np.inf if np.isnan(x[0]) else x[0])))
    return jointarray[1,:]

def counttext(chainnumber):
    import glob
    count = 0
    
    while True:
        if (os.path.isfile("chain{}_step{}_a{}_b{}_{}.txt".format(chainnumber,step,a_str,b_str,count)) == True ) or ( os.path.isfile("chain{}_step{}_a{}_b{}_{}.txt".format(chainnumber,step,a_str,b_str,count) ) == True):  
            count += 1
        else:
            break
    return count



def branchcut_pair(f1,f2):
    p1 = np.linespace(-60,60, 10 ** 5)
    p2 = np.linespace(-60,60,10 ** 5)
    h1 = f1-f2
    h1_array = h1(p1)
    f1_array = f1(p1)
    f2_array = f2(p2)
    cutidx  = np.argwhere(  np.sign(np.diff(h1_array) != 0))
    if np.sign(h1_array[0]) > 0:
        f1_cut = np.append(f1_array[idx:],np.zeros_like(f1_array[:,idx]))
        f2_cut = np.append(f2_array[:idx],np.zeros_like(f2_array[:,idx]))
    else:
        f1_cut = np.append(f1_array[idx:],np.zeros_like(f1_array[:,idx]))
        f2_cut = np.append(f2_array[:idx],np.zeros_like(f2_array[:,idx]))
    return f1,f2

def branchcut(density,event):
    p1 = np.linspace(-60,60, 10 ** 5)
    print(density)
    #exit()
    #cutidx  = np.argwhere(  np.diff(np.sign(density) ) != 0 )
    cutidx = np.searchsorted(p1,event.xdata)
    print(cutidx)
    print(event.xdata)
    if density[0] < density[cutidx]:
        density_cut = np.append(density[ : cutidx],np.full_like(density[cutidx :],  - np.inf ))
    else:
        density_cut = np.append(np.full_like(density[: cutidx],-np.inf),density[cutidx :])
    print(density_cut.shape)
    f1_cut = inter(density_cut,p1)
    fig = plt.figure()
    plt.plot(p1,10 ** density_cut)
    plt.semilogy()
    plt.show()
    plt.close()
    return density_cut
    #exit() 

def summingupdensity():
    import glob
    count = 0
    x = np.linspace(-60,60, 10 ** 5)
    while True:
        if (os.path.isfile("chain{}_step{}_a{}_b{}_{}_cutted.txt".format(chainnumber,step,a_str,b_str,count)) == True ) or ( os.path.isfile("chain{}_step{}_a{}_b{}_{}_cutted.txt".format(chainnumber,step,a_str,b_str,count) ) == True):  
            count += 1
        else:
            break
    array = np.loadtxt("chain{}_step{}_a{}_b{}_0_cutted.txt".format(chainnumber,step,a_str,b_str))
    summed_density = np.zeros_like(array[1])
    for i in range(count):
        array = np.loadtxt("chain{}_step{}_a{}_b{}_{}_cutted.txt".format(chainnumber,step,a_str,b_str,i))
        summed_density += 10 ** array[1]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.semilogy()
    ax.plot(x,summed_density,'-')
    plt.show()
    return summed_density 


def wavedensityofchain(chainnumber,):
    countchain = counttext([])
    sortnumber = sortbranch()
    functions = np.append([])
    for i in range(countchain):
        number = sortnumber[i:i+1]
        branch1 = np.loadtxt("chain{}_branch{}_step{}_a{}_b{}.txt".format(chainnumber,number[0],step,a_str,b_str))
        branch2 = np.loadtxt("chain{}_branch{}_step{}_a{}_b{}.txt".format(chainnumber,number[1],step,a_str,b_str))
        f1 = wavedensity(branch1)
        f2 = wavedensity(brahch2)
        if i != 0:
            cutted1,cutted2 = branchcut_pair(f1,f2)
            p1 = np.linespace(-60,60, 10 ** 5)
            p2 = np.linespace(-60,60,10 ** 5)
            f1_cutted = inter(cutted1,p1)
            f2_cutted = inter(cutted2,p2)
            functions = np.append(functions,f2_cutted)
        else: 
            cutted1,cutted2 = branchcut_pair(f1,f2)
            p1 = np.linespace(-60,60, 10 ** 5)
            p2 = np.linespace(-60,60,10 ** 5)
            f1_cutted = interpolate(cutted1,p1)
            f2_cutted = interpolate(cutted2,p2)
            functions = np.append(functions,f1)
            functions = np.append(functions,f2_cutted)
    summed_density = summingupdensity(fs)
    return summed_density        

def wavedensity(seed):
    S,q = action(seed,step,func)
    psi = WaveFunction(S)
    #print("Im(S)=",(np.abs(S.imag/planckconstant)))
    density = probdensity(psi) 
    #print(density)
    f = inter(density,q) #pに対する補間式.
    return density


def onclick(event,x,density,i):#branchcutする
    if event.dblclick == 1:
        index = np.searchsorted(x,event.xdata)
        density = branchcut(density,event)
        plt.clf() 
        plt.close()
        print(density.shape)
        print(x.shape)
        #exit()
        #text = "chain{}_branch{}_step{}_a_{}_b_{}_cutted.txt".format(i + 1,step,a_str,b_str)
        text = "chain{}_step{}_a{}_b{}_{}_cutted.txt".format(chainnumber,step,a_str,b_str,i) 
        with open(text,mode = 'w') as f:
            np.savetxt(text,np.array([x,density]))

def cutting():
    global fs 
    number = counttext(chainnumber)
    print(number)
    #exit()
    for i in range(0, number):
        #seed = np.loadtxt('orbitdistortion_a{}_b{}_{}_{}.txt'.format(a_str,b_str,step,number))
        branch = np.loadtxt("chain{}_step{}_a{}_b{}_{}.txt".format(chainnumber,step,a_str,b_str,i))
        S,q = action(branch,step,func)
        psi = WaveFunction(S)
#print("Im(S)=",(np.abs(S.imag/planckconstant)))
        density = probdensity(psi) 
        density = np.log(density)
        f = inter(density,q) #pに対する補間式.
        #print(f)
        #print(f)
        x = np.linspace(-60,60,10 ** 5)
        #作用の配列
        fig = plt.figure()
        ax = fig.add_subplot(111)
        #plt.plot(q,density,'.k')#density[1] = p,density[0] = psi(p)
        plt.title("$step= $ {}".format(step))
        plt.xlabel("$q$")
        plt.ylabel(r"$|\psi(q)|^2$")
        #plt.xlim(-0.3,0.3)
        #ax.semilogy()
        #plt.show()
        #plt.plot(seed[:,0],seed[:,1],',k')
        #plt.show()
        #plt.plot(np.arange(0,len(S),1),S.real,'-k')

        density = f(x) 
        plt.plot(x,10 ** density,'-')
        print(density)
        ax.semilogy()
        fig.canvas.mpl_connect('button_press_event', lambda event:onclick(event,x,density,i) )
        ax.set_ylim(10 ** ( -8), 10 ** 0)
        ax.set_xlim(-3,3)
        plt.show()
        #np.savetxt('chain{}_step{}_a{}_b{}_{}_cutted.txt'.format(chainnumber,step,a_str,b_str,number),np.array([x,density]))
#summingupdensity(x,density)



state = 1
cmap = KScatteringMap2(k,xf,xb)
period = 2 
func = cmap.U
a_str = format(str(round(a_orb,3))).replace('.','')
b_str = format(str(round(b_orb,3))).replace('.','')
number = 3
chainnumber = 1
seed = np.loadtxt('orbit_a{}_b{}_{}_{}.txt'.format(a_str,b_str,step,number))
number = 1
S,q = action(seed,step,func)
psi = WaveFunction(S)
#print("Im(S)=",(np.abs(S.imag/planckconstant)))
#density = probdensity(psi) 
#density = np.log(density)
#print(density)
#f = inter(density ,q) #pに対する補間式.
#print(f)
#x = np.linspace(-60,60,10 ** 5)
#作用の配列
#fig = plt.figure()
#ax = fig.add_subplot(111)
#plt.plot(q,density,'.k')#density[1] = p,density[0] = psi(p)
#plt.title("$step= $ {}".format(step))
#plt.xlabel("$q$")
#plt.ylabel(r"$|\psi(q)|^2$")
#plt.xlim(-0.3,0.3)
#ax.semilogy()
#plt.show()
#plt.plot(seed[:,0],seed[:,1],',k')
#plt.show()
#plt.plot(np.arange(0,len(S),1),S.real,'-k')

#density = f(x) 
#plt.plot(x,10 ** density,'-')
#print(density)
#ax.semilogy()
#fig.canvas.mpl_connect('button_press_event', lambda event:onclick(event,x,density) )
#plt.show()
#print(densityhist[1])
if __name__ == '__main__':
    cutting()
    #density = summingupdensity()
    print(density)
    print(density)
