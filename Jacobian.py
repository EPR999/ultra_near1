import numpy as np
from mpmath import *
import Jacobian

#線形化を行うために写像についてヤコビアンを求める.
#This program evaluate Jacobian of the map at fixed and periodic orbits in order to do linearlization around a fixed or periodicorbit.

k = 3.0
xf = 1.2
xb = 1.0
e2 = k * xf * (np.exp(-8 * xb * (2 * xf - xb)) / (1 - np.exp(-32 * xf * xb)))
twopi = 2 * np.pi

k = 3.0
xf = 1.2
xb = 1.0
e2 = k * xf * (np.exp(-8 * xb * (2 * xf - xb)) / (1 - np.exp(-32 * xf * xb)))
dimension = 2
step = 1

def dotV(x):  #dotV(x)
     return k * x * np.exp(-8 * x**2) - e2 * (np.exp(-8 * pow(x - xb, 2)) - np.exp(-8 * pow(x + xb, 2)))


#def dotV(x):  #cosine function 
#     return q ** 2 - np.cos(q)
#def ddotV(x): #
#   return 

#def dotV(x):  # kicked scattering 
#     return k * x * np.exp(-8 * x**2) - e2 * (np.exp(-8 * pow(x - xb, 2)) - np.exp(-8 * pow(x + xb, 2)))
 

#derivative of V"
def ddotV(x):
    return k*(1 - 16*x**2)*np.exp(-8*x**2) - 16*e2*((x - xb)*np.exp(-8*(x - xb)**2) - (x + xb)*np.exp(-8*(x + xb)**2))
def dotV_s(x):
    return k * x * exp(-8 * x**2) - e2 * (exp(-8 * (x - xb)**2) - exp(-8 * (x + xb) ** 2))
def ddotV_s(x):
    return k*(1 - 16*x**2)*exp(-8*x**2) - 16*e2*((x - xb)*exp(-8*(x - xb)**2) - (x + xb)*exp(-8*(x + xb)**2))
    #逆写像
def U_s(self,qp):
    return np.array([qp[0] + qp[1]  -  dotV_s(qp[0]), qp[1] - dotV_s(qp[0] ) ])

def U(qp): #正の時間方向の写像
    return np.array([qp[0] + qp[1] - dotV(qp[0]), qp[1] -  dotV(qp[0]) ])
def Ui(qp): #inverse of
     return np.array([qp[0] - qp[1] , qp[1] + dotV(qp[0] - qp[1])])
def f(x):#テスト関数
    return x[0]**2 + x[1]**3- 1,x[0]**2



def derivative(func,x,i):#多変数関数の微分(i番目が微分され戻る.) 
    h = 0.1 ** 6

def derivative(func,x,i):#多変数関数の微分(i番目が微分され戻る.)
    h = 0.1 ** 7
    h_vec = np.zeros_like(x)
    h_vec[i] = h
    xp = x + h_vec
    xm = x - h_vec
    #print((.array(func(xp))-.array(func(xm)))/(2 * h))
    return (np.array(func(xp))-np.array(func(xm)))/(2 * h)

def ScattMapt(qp):#stepは周期点の周期
    #step = periodic.minimumperiodcheck(qp[0],qp[1],U)
    for i in range(step):
        qp =  U(qp)
    return qp

def strictJ(qp):
    DU = np.array([[1 - ddotV(qp[0]) , 1 ],[ - ddotV(qp[0]) , 1   ]])
    return DU
#def strictJ(qp):#override
 #   DU = np.array([[1 ,  - 1 ],[ 1 - ddotV(qp[0] ) , ddotV(qp[0] - qp[1] )  ]])
 #   return DU


## strictJ(qp);##


def strictJ2(func,qp,step):  
    Jacob = strictJ(qp)
    for i in range(step-1):
        qp = func(qp)
        #print('sJ =',strictJ(qp))
        Jacob = np.dot(Jacob , strictJ(qp))
        #print('J=',Jacob)
    return Jacob

#def strictJ_s(qp):
#    return np.array([[mpf('1.0') -  ddotV_s(qp[0]) , mpf('1') ],\
#        [- mpf('0.5') * ddotV_s(qp[0]) - mpf('0.5') * (mpf('1.0') - mpf('0.5') * ddotV_s(qp[0])) * ddotV_s(qp[0]+ qp[1] - 0.5 * dotV_s(qp[0])), 1 - 0.5 * ddotV_s(qp[0] + qp[1]  - 0.5 * dotV_s(qp[0])) ]])
def strictJ_s(qp):
    return np.array([mpf('1.0') - ddotV_s(qp[0]), mpf('1')], [   - ddotV_s(qp[0]) , mpf('1')])
def strictJ_s(qp):
    return np.array([mpf('1.0'), - mpf('1.0') ], [  mpf("1.0") - ddotV_s(qp[0] - qp[1]) , ddotV_s(qp[0] - qp[1])])



def strictJ2_s(func,qp,step):
    Jacob = strictJ_s(qp)
    #print(Jacob)
    for i in range(step-1):
        qp = func(qp)
        #print('sJ =',strictJ_s(qp))
        Jacob = np.dot(Jacob , strictJ_s(qp))
        #print(type(strictJ_s(qp)))
        #print('J=',Jacob)
    #print(Jacob)
    return Jacob

def main():
    #x = [Re(q),Re(p)] 不動点・周期点を受け取る.
    mp.dps = 30
    x = np.array([  -0.09172528047368282971134959952547662207295697028013291493785923369644609803097491  ,  -4.134740337300568162589317258041903660460787480222040257029801662182853210489195e-14 ])  
    jacobian2 = [np.array([]),np.array([])]
    #print(strictJ(x))
    step = 7
    jacobian = strictJ2(U,x,step)
    #print(jacobian)
    w,v = np.linalg.eig(jacobian)#wは固有値,vは固有ベクトル.
    print(w,v)
    exit()
    return w,v

if __name__ == '__main__':
    main()
    

#def main():
#    x = np.array([1.2,0.0]) #x = [Re(q),Re(p)] 不動点・周期点を受け取る.
#    step = 1
#    jacobian = strictJ2(cmap.U,x,step)       
#    w,v = np.linalg.eig(jacobian)#wは固有値,vは固有ベクトル.
#    print(v)

