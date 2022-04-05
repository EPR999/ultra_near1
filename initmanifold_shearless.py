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


def main():
	points = [np.array([])] * 2
	point = np.array([ 0.95782, 0  ])
	#point = np.array([ 0.809197, 0  ])#lambda = 1.2
	#[1.06462819 1.37029825]
#[0.06309614 0.99800745]
#angle between major axis and x axis= 1.5076582419255626
#angle between minor axis and x axis= -0.06313808486933388
	for i in range(10000):
		point = U(point)
		points[0] = np.append(points[0] , point[0])
		points[1] = np.append(points[1] , point[1])
	print(type(points))
	A =  np.array([ points[0]**2 , points[0] * points[1] , points[1] ** 2]).T
	b =   np.ones_like(points[0]) 
	print(A.shape)
	x = np.linalg.lstsq(A,b)[0]
	print(x)
	fig = plt.figure(figsize = (14,14))
	ax = fig.add_subplot(111)
	print(x)
	plot_ellipse_by_equation(ax, x[0] , x[1] , x[2])
	#plt.plot(points[0],points[1],'o')
	matrix = np.array( [ [x[0], x[1]/2], [x[1]/2 , x[2] ] ])
	eig_val, eig_vec = np.linalg.eig(matrix)
	#print(eig_val, eig_vec)
	print(( 1/eig_val) ** 0.5 ) 
	phase = np.loadtxt("phase_shearless_30.txt")
	plt.plot(phase[0,:],phase[1,:],",b")
	plt.xlim(-40.0,40.0)
	plt.ylim(-40.0,40.0)
	ax.set_xlabel(r"$q$",fontsize = 25)
	ax.set_ylabel(r"$p$",fontsize = 25)
	plt.tick_params(labelsize = 20)
	plt.title("lambda = {}".format(round(lamb,1)),fontsize = 20)
	plt.show()
	# 長軸、短軸を求める。
	idx1, idx2 = eig_val.argsort()
	major_axis = eig_vec[:, idx1]  # 固有値が小さいほうの固有ベクトルが major axis
	minor_axis = eig_vec[:, idx2]  # 固有値が大きいほうの固有ベクトルが minor axis
	print(major_axis)
	theta1 =np.arctan(major_axis[1] / major_axis[0])
	theta2 = np.arctan(minor_axis[1] / minor_axis[0])
	print(f"angle between major axis and x axis= {theta1}")
	print(f"angle between minor axis and x axis= {theta2}")
	ax.set_xlabel(r"$q$",fontsize = 25)
	ax.set_ylabel(r"$p$",fontsize = 25)
	plt.tick_params(labelsize = 20)
	plt.title("lambda = ",)
	#theta1 = np.rad2deg(np.arctan())
	exit()
	#ax.scatter(points[0], points[1], label= "Data Points", color="g" )
	#ax.scatter(points)
	#plt.show()
	#print(ret)



	fig = plt.figure(figsize = (12,12))
	ax = fig.add_subplot(111)
	deg = ret[2]
	theta = np.linspace( 0 , 2 * np.pi, 10 ** 3 )
	a_orb = ret[1,1]
	b_orb = ret[1,2]
	line = np.array( [ a_orb * np.sin(theta), b_orb * np.cos(theta) ] )
	elipse =  np.array( [ line[0] * np.cos(deg)  - line[1] * np.sin(deg) , line[0] * np.sin(deg) + line[1] * np.cos(deg) ] )
	one = np.array
	plt.plot(points[0], points[1], ',k')
	plt.plot(elipse[0],elipse[1],',k')
	exit()

def plot_ellipse_by_equation(ax, a , b , c):
	x = np.linspace( - 6, 6 , 10 ** 3.5)
	y = np.linspace( - 6, 6 , 10 ** 3.5)
	X, Y  = np.meshgrid(x,y)
	eqn = a * X ** 2 + b * X * Y + c *  Y ** 2
	#print(eqn)
	Z = 1
	ax.contour(X, Y, eqn, levels = [Z], colors=["r"] ,linewidths = 3 )
	





if __name__ == '__main__':
	main()