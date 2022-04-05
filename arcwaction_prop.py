import sys
import os.path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sym
from sympy import Array
import pylab
#from numba import jit,autojit
from mpmath import *
import math
import Jacobian 
#import japanize_matplotlib
#実験用.散乱写像で不動点から安定多様体と不安定多様体
#parameterz
dimension = 2
k = 3.0
xf = 1.2
xb = 1.0
a_orb =  0.20334038#0.05027134 0.20334038
b_orb =0.05027134 
t = 1.0740275527854348
#a_orb =  0.20910305#0.05027134 0.20334038
#b_orb = 0.05040667 
#t = 1.0757887088699023
#a_orb = 0.21
#b_orb = 0.05
#t = 1.0740275527854348
stdrad = 0.08
e2 = k * xf * (np.exp(-8 * xb * (2 * xf - xb)) / (1 - np.exp(-32 * xf * xb)))
W_s = [[],[]]
W_u = [[],[]]
traj = [np.array([]),np.array([])]
step = 1
timestep = 7
number = 3
DragFlag  = False
initpoint_array = np.array([])
lns = []
actions = np.array([])
iterates = np.array([])
textpath = 'initpoint_10_13_test14_a{}_b{}.txt'.format(str(round(a_orb,4)).replace('.',''), str(round(b_orb,4)).replace('.',''))
#kicking potentialの時間微分
def dotV(x):
    return k * x * np.exp(-8 * x**2) - e2 * (np.exp(-8 * pow(x - xb, 2)) -  np.exp(-8 * pow(x + xb, 2)))

def dotV_s(x):
    return k * x * exp(-8 * x**2) - e2 * (exp(-8 * pow(x - xb, 2)) -  exp(-8 * pow(x + xb, 2)))

def V(x):
    return - (k/16) * np.exp(-8 * (x[0] ** 2)) - ((e2/(2 * (8 ** 0.5))) * np.sqrt(np.pi)) * (math.erf( (8 ** 0.5)* (x[0] - xb)) - math.erf((8 ** 0.5) *( x[0] + xb)))
#
class KScatteringMap:
    def __init__(self,k,xf,xb):
        self.k = k
        self.xf = xf
        self.xb = xb
    #正の時間方向の写像
    def U(self,qp):
        return np.array([ qp[0] + qp[1] - dotV(qp[0]) , qp[1] -  dotV(qp[0])  ])
    #逆写像
    def Ui(self,qp):
        return (qp[0] - qp[1] - 0.5 * dotV(qp[0]), qp[1] + 0.5 * dotV(qp[0]) + 0.5 * dotV(qp[0] - qp[1] - 0.5 * dotV(qp[0])))
    def U_arb(self,qp):
        return np.array([mpf(qp[0] - qp[1] - 0.5 * dotV_s(qp[0])),mpf(qp[1] - 0.5 * dotV_s(qp[0]) - 0.5 * dotV_s(qp[0] + qp[1] - 0.5 * dotV_s(qp[0])))])
#
def CACOS(z):#cosの値から複素のcomplex thetaを返す
	xi = 2 * math.atan(np.real((1 - z) ** 0.5) /np.real((1 + z) ** 0.5) )
	eta = math.asinh( np.imag( (np.conj(1 +z) ** 0.5) *  ((1-z) ** 0.5)  )) 
	return xi + 1j * eta
#
def CASIN(z):
    xi = math.atan( np.real(z) / np.real( (1-z) ** 0.5 * ( 1 +z ) ** 0.5 )) 
    eta = math.asinh( np.imag( np.conj(1-z) ** 0.5 *  (1 +z) ** 0.5 ) )
    return xi + 1j * eta
#
def ScattMapt(qp):
    for i in range(step):
        qp =  cmap.U(qp)
    return qp

def load():
	data = np.loadtxt("theta.txt")



def rotateinitialmanifold(a,b,t,points):
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    R = np.array([[cos_t,sin_t],[-sin_t,cos_t]])
    rotated = np.dot(R,points)
    return rotated

def gettheta(cpoint):#cpoint から　thetaを得る． Kahan, W: Branch cuts for complex elementary functionを参照
	print("a = ",cpoint)
	print(t)
	cpoint_rotated = rotateinitialmanifold(a_orb,b_orb,t,cpoint)
	sine =  cpoint_rotated[0]/a_orb
	cosine = cpoint_rotated[1]/b_orb
	#print(cosine **2 + sine ** 2)
	theta = CACOS(cosine) #まずcosineに処理を施す
	theta2 = CASIN(sine) 
	#print("acos =",theta)
	#print("asin =",theta2)
	if theta.real < np.pi/2 and theta2.real < 0:
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
	print("b = ",confirm_point)
	return theta_conf

def CACOS(z):#mountain Oock cosの値から複素のcomplex thetaを返すよ   
    xi = 2 * math.atan(np.real((1 - z) ** 0.5) /np.real((1 + z) ** 0.5) )
    eta = math.asinh( np.imag( (np.conj(1 +z) ** 0.5) *  ((1-z) ** 0.5)  )) 
    return xi + 1j * eta


def CASIN(z):
    xi = math.atan( np.real(z) / np.real( (1-z) ** 0.5 * ( 1 +z ) ** 0.5 )) 
    eta = math.asinh( np.imag( np.conj(1-z) ** 0.5 *  (1 +z) ** 0.5 ) )
    return xi + 1j * eta    

def connector(theta):#本来なら最初に書くべき
    q = a_orb * np.sin(theta)
    p = b_orb * np.cos(theta)
    return np.array([q,p])

def generatorfunc(I,theta):#母関数
	return - (I/2) * (theta - (1/2) * np.sin(2 * theta))

def generatorfunc2(I,q):#️多分正しい方
	print("G(I,q) = {}".format(q *np.arcsin(q/(I ** 0.5)) + (I - q ** 2) ** 0.5))
	return np.sqrt(I) *(q *np.arcsin(q/(I ** 0.5)) + (I - q ** 2) ** 0.5)


def generatorfunc2(I,q):#母関数
	return   (I/2) * (np.arcsin(q/(I ** 0.5)) + (1/2) *  np.sin(2 * np.arcsin(q / ( I ** 0.5 ) )))

def generatorfunc2(I,theta):#母関数
	print("theta = ",theta )
	print("G(I,q) =",( I/2) * (theta + (1/2) * np.sin(2 * theta)))
	return  (I/2) * (theta + (1/2) * np.sin(2 * theta))

def imagchange(point):
	newpoint = np.real(point) - 1j * np.imag(point)
	return newpoint

def generatorfunc2(theta):#母関数
	print(theta)
	aa  =  (a_orb*b_orb/2) * (theta + (1/2) * np.sin(2 * theta) * np.cos(2 * t)) - (1/8) * (a_orb** 2 + b_orb ** 2) * np.cos(2 * theta) * np.sin(2 * t)
	print("G(q,I) = {}".format(aa))
	return  (a_orb*b_orb/2) * (theta + (1/2) * np.sin(2 * theta) * np.cos(2 * t)) - (1/8) * (a_orb** 2 + b_orb ** 2) * np.cos(2 * theta) * np.sin(2 * t)

def action(x_cross,periodicpoint,period):
	S = 0
	i = 0
	theta = gettheta(x_cross)
	x_cross_t = np.array([a_orb * np.sin(theta),b_orb * np.cos(theta)]) 
	x_cross = np.array(x_cross)
	print(x_cross_t)
	print(x_cross)
	prepoint  = x_cross
	#print(x_cross)
	#print(period)
	while True:
		prepoint = x_cross
		theta_0 = theta 
		preI = x_cross[0] ** 2 + x_cross[1] ** 2	
		prethetaforsin = CASIN(x_cross[1]/(preI ** 0.5))
		#print(x_cross)
		x_cross =  cmap.U(x_cross)
		#print(theta)

		#print("x_cross =",x_cross)
		# S += 0.5 * x_cross[1] ** 2 + V(x_cross) - x_cross[1] * ( - 0.5 * dotV(x_cross[0]) - 0.5 * dotV(x_cross[0] + x_cross[1] - 0.5 * dotV(x_cross[0]) ) )#終わりから始めを引け
		S += 0.5 * (x_cross[1] ** 2) + V(prepoint) - x_cross[1] * ( x_cross[0] - prepoint[0] ) 
		print("S +=",0.5 * (prepoint[1] ** 2) + V(prepoint) - prepoint[1] * ( x_cross[0] - prepoint[0] ) )
		print("S = ",S)
		if i == 0:
			## -= V(prepoint)
			S -= generatorfunc2(theta_0)
			print()
		i += 1
		#print(i)
		#print('x_cross = ',x_cross)
		if   np.linalg.norm(np.linalg.norm(x_cross - periodicpoint)) < 10 ** (-6)    and period !=1:
			print("theta = ",np.linalg.norm(np.imag(x_cross)))
			#S -= V(prepoint)			#print("Im(Smin)= " ,S.imag)
			#print('x_cross = ',x_cross)
			return S.imag,i	
		elif period == 1 and ( (np.linalg.norm(x_cross - periodicpoint)  < 10 ** (-4) or np.linalg.norm(np.imag(x_cross)) < 5 * 10 ** (-4)) ):
			print("theta? = ",np.linalg.norm(np.imag(x_cross)))
			#S -= V(prepoint)
			#print("Im(Smin)= " ,S.imag)
			#print('x_cross = ',x_cross)
			return S.imag,i	
		if np.linalg.norm(x_cross) > 10:
			return np.nan,0
		if i > 1000:
			return np.nan,0 
#
def extract_pointsbyperiod(period):
    points = np.loadtxt("periodicpoint.txt")
    #print(points[:,2] == period)
    points = points[points[:,2] == period,0:2]
    return  points
#
def drawingactions(rad,textpath):#点を描画する
	stdrad = rad
	#print(stdrad)
	text = np.loadtxt(textpath.format(str(round(stdrad,4)).replace('.','z')),dtype = np.complex128)
	#print(text)
	superdata1 = text
	fig = plt.figure(figsize = (12,12))
	ax = fig.add_subplot(1,1,1)
	plt.xlabel(r'$stability$',fontsize = 33,labelpad = -7)
	plt.ylabel(r'$|Im(S)|$',fontsize = 33)
	plt.title("radius = {} ".format(str(round(stdrad,4))), fontsize = 33)
	plt.semilogx()
	allpoints = [np.array([])]*2
	allaxispoint = np.array([])#基準となる交点の数々.
	iterates = text[:,4]
	print(text[:,0])
	axis = text[:,0]
	axis3 = text[:,3]
	ImS  = np.abs(text[:,1])#print(vec.shape)
	plotarray = np.array([axis,axis3,ImS])#初期化
	print(ImS)
	for i in [1,4,5,6,7,8,10]:#それぞれの半径について．凡例を作るため。
		extractarray = plotarray[:,  plotarray[0] == i ]
		ax.scatter(  extractarray[1],extractarray[2] ,s = 100 ,label = "period{}".format(i))
		#print(vec)
		#np.savetxt(p_hand,vec,newline = "\r\n", fmt = '%.16f%+.16fj ' * 7)
		#plt.scatter(axis3,ImS , s = 100, c = axis, alpha = 1, cmap="jet", marker = "o", zorder = 2)
		#plt.plot(text[:,0], ImS, 'o',markersize = 12)
		#cbar = plt.colorbar(shrink= 0.89
		from matplotlib.colors import LogNorm
		#bar=ax.imshow(stdall,norm=LogNorm())
		##cbar.ax.set_ylabel('period', fontsize = 33, weight="bold")
		#cbar.ax.set_yticklabels(np.arange(cbar_min, cbar_max+cbar_step, cbar_step), fontsize=16, weight='bold',label = "period")
		ax.legend(loc = "upper left",fontsize = 14)
		plt.tick_params(labelsize = 16)
		plt.ylim(-0.001,0.3)
	plt.show()
		#plt.savefig('result_rad{}.png'.format(str(round(stdrad,3)).replace('.','z'))) 
 #cid = fig.canvas.mpl_connect('button_press_event', 
#
def plotactions(rad,textpath):#点を描画する
	stdrad = rad
	#print(stdrad)
	text = np.loadtxt(textpath.format(str(round(stdrad,3)).replace('.','z')),dtype = np.complex128)
	print(textpath)
	superdata1 = text
	fig = plt.figure(figsize = (12,12))
	ax = fig.add_subplot(1,1,1)
	plt.xlabel(r'$Re(\theta_0)$',fontsize = 20)
	plt.ylabel(r'$Im(\theta_0)$',fontsize = 20)
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
	ImS  = np.abs(text[:,1])#print(vec.s hape)
	#plotarray = np.array([axis,axis3,ImS])#初期化
	#print(ImS.shape)
	#print(text[:,10].shape)
	#plt.plot(text[:,10],text[:,11],"o")
	a_str = str(round(a_orb,4)).replace('.','')
	b_str = str(round(b_orb,4)).replace('.','')
	#tree  = np.loadtxt("step{}_{}_a_{}_b_{}.txt".format(timestep,number,a_str,b_str))#"test_10_a_$a_orb","_b_$b_orb",".txt" mset_40_4_a_025_b_03.txt
	#ree = np.loadtxt("mset_40_1_a_030_b_008.txt")
	#plt.plot(tree[:,0],tree[:,1],",k",zorder = 1)
	plt.scatter(initpointRe, initpointIm, s = 100, c = ImS, alpha = 1, cmap="jet_r", marker = "o", zorder = 2)
	cbar = plt.colorbar(shrink= 0.89,label = "log10(ImS)")
	cbar.ax.tick_params(labelsize=20)
	#cbar.set_label("|Im(S)|",size = 25)
	#cbar.set_label("stability",size = 25)
	#cplt.clim(vmin=-0.02,vmax=0.1)

	#plt.clim(vmin=-0.0,vmax=2000)
	#plt.clim(vmin=-0.0,vmax=0.04)
	plt.ylim(tree[0,1],tree[-1,1])
	plt.xlim(tree[0,0],tree[-1,0])
	plt.title("a = {}, b = {}, t = {} ".format(str(round(a_orb,4)),str(round(b_orb,4)),timestep), fontsize = 33)
def plotactions(rad,textpath):#点を描画する
	stdrad = rad
	#print(stdrad)
	text = np.loadtxt(textpath.format(str(round(stdrad,4)).replace('.','z')),dtype = np.complex128)
	print(textpath)
	superdata1 = text
	fig = plt.figure(figsize = (12,12))
	ax = fig.add_subplot(1,1,1)
	plt.xlabel(r'$Re(\theta_0)$',fontsize = 20)
	plt.ylabel(r'$Im(\theta_0)$',fontsize = 20)
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
	ImS  = np.abs(text[:,4])#print(vec.s hape)
	print(ImS)
	#plotarray = np.array([axis,axis3,ImS])#初期化
	#print(ImS.shape)
	#print(text[:,10].shape)
	#plt.plot(text[:,10],text[:,11],"o")
	a_str = str(round(a_orb,3)).replace('.','')
	b_str = str(round(b_orb,3)).replace('.','')
	#tree  = np.loadtxt("step{}_{}_a_{}_b_{}.txt".format(timestep,number,a_str,b_str))#"test_10_a_$a_orb","_b_$b_orb",".txt" mset_40_4_a_025_b_03.txt
	#ree = np.loadtxt("mset_40_1_a_030_b_008.txt")
	#plt.plot(tree[:,0],tree[:,1],",k",zorder = 1)
	plt.scatter(initpointRe, initpointIm, s = 100, c = ImS, alpha = 1, cmap="jet_r", marker = "o", zorder = 2)
	cbar = plt.colorbar(shrink= 0.89,label = "log10(ImS)")
	cbar.ax.tick_params(labelsize=20)
	#cbar.set_label("|Im(S)|",size = 25)
	cbar.set_label("stability",size = 25)
	#cplt.clim(vmin=-0.02,vmax=0.1)
	#plt.clim(vmin=-0.0,vmax=2000)
	#plt.clim(vmin=-0.0,vmax=0.04)
	#plt.ylim(tree[0,1],tree[-1,1])
	#plt.xlim(tree[0,0],tree[-1,0])
	plt.title("a = {}, b = {}, t = {} ".format(str(round(a_orb,3)),str(round(b_orb,3)),timestep), fontsize = 33)
	#plt.plot(initpointRe[0],initpointIm[0],"o", color = 'red')
	#plt.plot(initpointRe[0],initpointIm[0],"o", color = 'red')
	#for i in [1,4,5,6,7,8]:#それぞれの半径について．凡例を作るため。
		#extractarray = plotarray[:,  plotarray[0] == i ]
		#ax.scatter(  extractarray[1],extractarray[2] ,s = 100 ,label = "period{}".format(i))
		#print(vec)
		#np.savetxt(p_hand,vec,newline = "\r\n", fmt = '%.16f%+.16fj ' * 7)
		#plt.scatter(axis3,ImS , s = 100, c = axis, alpha = 1, cmap="jet", marker = "o", zorder = 2)
		#plt.plot(text[:,0], ImS, 'o',markersize = 12)
		#cbar = plt.colorbar(shrink= 0.89

	#	from matplotlib.colors import LogNorm
		#bar=ax.imshow(stdall,norm=LogNorm())
		##cbar.ax.set_ylabel('period', fontsize = 33, weight="bold")
		#cbar.ax.set_yticklabels(np.arange(cbar_min, cbar_max+cbar_step, cbar_step), fontsize=16, weight='bold',label = "period")
	#	ax.legend(loc = "upper left",fontsize = 14)
#		plt.tick_params(labelsize = 20)
#		plt.ylim(-0.001,0.03)
	plt.rcParams["font.size"] = 22
	#plt.xlim(0,2 * np.pi)
	plt.tick_params(labelsize=16)
	plt.savefig('mset30_40_rad{}.png'.format(str(round(stdrad,4)).replace('.',''))) 
	#cid = fig.canvas.mpl_connect('button_press_event', onkey)
	plt.connect('button_press_event', Press)
	plt.connect('motion_notify_event',Drag)
	plt.connect('button_release_event',Release)
	plt.show()
 #cid = fig.canvas.mpl_connect('button_press_event',

def plotperiods(textpath):#その点から軌道に沿った作用の大きさで色分け
    #stdrad = rad
    #print(stdrad)
    text = np.loadtxt(textpath,dtype = np.complex128)
    print(text)

    #print(textpath)
    superdata1 = text
    fig = plt.figure(figsize = (10,8))
    ax = fig.add_subplot(1,1,1)
    plt.xlabel(r'$\xi$',fontsize = 20)
    plt.ylabel(r'$\eta$',fontsize = 20)
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
        print(extractarray)
        plt.plot(  extractarray[:,10],extractarray[:,11] ,'o',markersize = 10,label = "period{}".format(i),zorder = 8 - i )
        #print(vec)
        #np.savetxt(p_hand,vec,newline = "\r\n", fmt = '%.16f%+.16fj ' * 7)
        #plt.scatter(axis3,ImS , s = 100, c = axis, alpha = 1, cmap="jet", marker = "o", zorder = 2)
        #plt.plot(text[:,0], ImS, 'o',markersize = 12)
        #cbar = plt.colorbar(shrink= 0.89)

        #from matplotlib.colors import LogNorm
        #bar=ax.imshow(stdall,norm=LogNorm())
        ##cbar.ax.set_ylabel('period', fontsize = 33, weight="bold")
        #cbar.ax.set_yticklabels(np.arange(cbar_min, cbar_max+cbar_step, cbar_step), fontsize=16, weight='bold',label = "period")
        plt.legend(loc = "upper left",fontsize = 14)
        #plt.tick_params(labelsize = 20)
        #plt.ylim(-0.001,0.03)
    plt.rcParams["font.size"] = 22
    #plt.xlim(0,2 * np.pi)
    plt.tick_params(labelsize=20)
    plt.connect('button_press_event', Press)
    plt.connect('motion_notify_event',Drag)
    plt.connect('button_release_event',Release)
    plt.show()

def plt_graph3d(angle):
	ax.view_init(azim)


def plotperiodicpoint(rad,textpath):#点を描画する
	stdrad = rad
	#print(stdrad)
	text = np.loadtxt(textpath.format(str(round(stdrad,4)).replace('.','z')),dtype = np.complex128)
	print(textpath)
	superdata1 = text
	fig = plt.figure(figsize = (10,8))
	ax = fig.add_subplot(1,1,1)
	plt.xlabel(r'$Re(\theta_0)$',fontsize = 20)
	plt.ylabel(r'$Im(\theta_0)$',fontsize = 20)
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
	ImS  = np.abs(text[:,4])#print(vec.s hape)
	#plotarray = np.array([axis,axis3,ImS])#初期化
	#print(ImS.shape)
	#print(text[:,10].shape)
	#plt.plot(text[:,10],text[:,11],"o")
	a_str = str(round(a_orb,4)).replace('.','')
	b_str = str(round(b_orb,4)).replace('.','')
	tree  = np.loadtxt("step{}_3_a_{}_b_{}.txt".format(timestep,a_str,b_str))#"test_10_a_$a_orb","_b_$b_orb",".txt" mset_40_4_a_025_b_03.txt
	#ree = np.loadtxt("mset_40_1_a_030_b_008.txt")
	#plt.plot(tree[:,0],tree[:,1],",k",zorder = 1)
	plt.scatter(initpointRe, initpointIm, s = 100, c = ImS, alpha = 1, cmap="jet_r", marker = "o", zorder = 2)
	cbar = plt.colorbar(shrink= 0.89,label = "log10(ImS)")
	cbar.ax.tick_params(labelsize=20)
	#cbar.set_label("|Im(S)|",size = 25)
	cbar.set_label("stability",size = 25)
	#plt.clim(vmin=-0.0,vmax=0.04)
	plt.clim(vmin=-0.0,vmax=2000)
	#plt.clim(vmin=-0.0,vmax=0.04)
	plt.ylim(tree[0,1],tree[-1,1])
	plt.xlim(tree[0,0],tree[-1,0])
	print(tree[0,0],tree[-1,0])
	plt.title("a = {}, b = {}, t = {} ".format(str(round(a_orb,4)),str(round(b_orb,4)),timestep), fontsize = 33)
	#plt.plot(initpointRe[0],initpointIm[0],"o", color = 'red')
	#for i in [1,4,5,6,7,8]:#それぞれの半径について．凡例を作るため。
		#extractarray = plotarray[:,  plotarray[0] == i ]
		#ax.scatter(  extractarray[1],extractarray[2] ,s = 100 ,label = "period{}".format(i))
		#print(vec)
		#np.savetxt(p_hand,vec,newline = "\r\n", fmt = '%.16f%+.16fj ' * 7)
		#plt.scatter(axis3,ImS , s = 100, c = axis, alpha = 1, cmap="jet", marker = "o", zorder = 2)
		#plt.plot(text[:,0], ImS, 'o',markersize = 12)
		#cbar = plt.colorbar(shrink= 0.89

	#	from matplotlib.colors import LogNorm
		#bar=ax.imshow(stdall,norm=LogNorm())
		##cbar.ax.set_ylabel('period', fontsize = 33, weight="bold")
		#cbar.ax.set_yticklabels(np.arange(cbar_min, cbar_max+cbar_step, cbar_step), fontsize=16, weight='bold',label = "period")
	#	ax.legend(loc = "upper left",fontsize = 14)
#		plt.tick_params(labelsize = 20)
#		plt.ylim(-0.001,0.03)
	plt.rcParams["font.size"] = 22
	#plt.xlim(0,2 * np.pi)
	plt.tick_params(labelsize=16)
	plt.savefig('mset30_40_rad{}.png'.format(str(round(stdrad,3)).replace('.',''))) 
	#cid = fig.canvas.mpl_connect('button_press_event', onkey)
	plt.connect('button_press_event', Press)
	plt.connect('motion_notify_event',Drag)
	plt.connect('button_release_event',Release)
	plt.show()
 #cid = fig.canvas.mpl_connect('button_press_event', 
################################################################
def action2(x_cross,periodicpoint,period):
	global actions, iterates
	S = 0
	i = 0
	x_cross0 = x_cross
	prepoint  = x_cross
	theta_0 = gettheta(x_cross0)
	#print(x_cross)
	#print(period)
	actions = np.array([])
	iterates = np.array([])
	#plt.subplot(1,2,2)
	plt.ylabel(r"${\rm Im}(S)$",fontsize = 27)
	plt.xlabel("iterates",fontsize = 27)
	plt.ylim(-0.012,0.012)
	plt.xlim(-1,25) 
	colormap = plt.get_cmap("tab10")
	plt.plot(0,0,'o',color = colormap(period - 3),markersize = 12,zorder = 100 - period)
	while True:
		iterates = np.append(iterates, i)
		actions = np.append(actions, S.imag)
		#print("actions = ",actions)
		i +=  1
		jacob = Jacobian.strictJ2(cmap.U,periodicpoint,period)
		w,v = np.linalg.eig(jacob)
		wabs0 = w[0] * np.conj(w[0])
		wabs1 = w[1] * np.conj(w[1])
		if wabs0 > wabs1:
			tmp = wabs1
			wabs1 = wabs0
			wabs0 = tmp
		#print("stability =",wabs1 ** 0.5)
		
		prepoint = x_cross
		theta = gettheta(x_cross0)
		preI = x_cross[0] ** 2 + x_cross[1] ** 2
		prethetaforsin = CASIN( x_cross[1] / (preI ** 0.5) )
		#print(x_cross)
		x_cross =  cmap.U(x_cross)
		#plt.plot(x_cross)
		#exit()
		#theta = gettheta(x_cross0)
		#print(theta)
		#print("x_cross =",x_cross)
		print(period)
		# S += 0.5 * x_cross[1] ** 2 + V(x_cross) - x_cross[1] * ( - 0.5 * dotV(x_cross[0]) - 0.5 * dotV(x_cross[0] + x_cross[1] - 0.5 * dotV(x_cross[0]) ) )#終わりから始めを引け
		S += 0.5 * (x_cross[1] ** 2) + V(prepoint) - x_cross[1] * ( x_cross[0] - prepoint[0] )
		#actions = np.append(actions, S.imag)
		#print("S +=",0.5 * (x_cross[1] ** 2) + V(prepoint) - x_cross[1] * ( x_cross[0] - prepoint[0] ) )
		#print("S = ",S)
		print(periodicpoint)
		if i == 1:
			print()
			#S -= V(prepoint)
			S -= generatorfunc2( theta_0)
			#S -= generatorfunc1(preI,prethetaforsin)
		#iterates = np.append(iterates, i)
		#print(i)
		#print('x_cross = ',x_cross)
		print(np.linalg.norm(x_cross - periodicpoint))
		if  np.linalg.norm(x_cross - periodicpoint) < 10 ** (-6) and period !=1 or i == 15:
			#print("theta = ",np.linalg.norm(np.imag(x_cross)))
			#plt.title("stability = {}".format(str(round(wabs1 ** 0.5, 4))))
			plt.plot(iterates,actions,"-o",color = colormap(period - 3),zorder = 100 - period)
			#plt.plot(iterates[70:90], actions[70:90],'-o',zorder = 2,color = "aqua" )
			#plt.plot(iterates[220:300], actions[220:300],'-o',zorder = 2,color = "limegreen" )
			#plt.plot(iterates[0:90], actions[0:90],'-o',zorder = 2,color = "aqua" )
			#plt.plot(iterates[90:300], actions[90:300],'-o',zorder = 2,color = "limegreen" )
			print(np.linalg.norm(x_cross - periodicpoint))
			print(periodicpoint)
			print("S = ",S)
			#S -= V(prepoint)			#print("Im(Smin)= " ,S.imag)
			#print('x_cross = ',x_cross)
			return wabs1 ** 0.5 #S.imag,i	#不安定性を返します．
		elif period == 1 and  (np.linalg.norm(x_cross - periodicpoint)  < 10 ** (-5) or np.linalg.norm(np.imag(x_cross)) < 5 * 10 ** (-6)):
			#print("theta? = ",np.linalg.norm(np.imag(x_cross)))
			#plt.title("stability = {}".format(	str(round(wabs1 ** 0.5, 4))))
			plt.plot(iterates,actions,"-o",color = colormap(period),zorder = 1)
			#plt.plot(theta,,"-o",color = "red",zorder = 1)
			plt.plot(theta.real, theta.real ,"-o",color = "green",zorder = 1)
			print("S = ",S)
			#S -= V(prepoint)
			#print("Im(Smin)= " ,S.imag)
			#print('x_cross = ',x_cross)
			#plt.plot(iterates,actions,"-o")
			return wabs1 ** 0.5
		if np.linalg.norm(x_cross) > 10:
			return np.inf
		if i > 1000:
			return np.inf

def getsection_on_thetaplane(xi,eta,x1,x2,y1,y2):
	global textpath,a_orb,b_orb	
	data =	np.loadtxt( textpath , dtype = np.complex128)
	xi_point = xi[(x1 < xi) & (xi < x2)]
	eta_point = eta[ (y1< eta) & (eta < y2)]
	allpoint = np.array([xi,eta])
	position_onrealx = np.array(data[:,3])
	position_onrealy = np.array(data[:,4])
	print(data.shape)
	boolean = (x1 < xi) &(xi < x2 ) & (y1< eta) & (eta < y2)
	position_onrealx = data[boolean,8] 
	position_onrealy = data[boolean,9]
	periods = np.array(data[boolean,0],dtype = np.int32)
	periodicpoint = np.array([position_onrealx,position_onrealy])
	theta =  xi[boolean]+ 1j * eta[boolean]
	q = data[boolean,6]
	p = data[boolean,7]
	qp = np.array([q,p])
	print(qp)
	#fig = plt.figure(figsize = (12,12))
	#ax = fig.add_subplot(111)
	#qp = rotateinitialmanifold(a_orb,b_orb,-t,qp)
	#plt.plot(position_onrealx,position_onrealy,'o')
	#phase = np.loadtxt('scatteringphase_original.txt')
	#plt.plot(phase[0,0:80000],phase[1,0:80000],',k',zorder = 2)
	#plt.show()
	fig  = plt.figure(figsize =  (14.402,8.38889))
	ax = fig.add_subplot(111, projection = '3d')
	#plt.subplot(1,2,2)
	#qp = rotateinitialmanifold(a_orb,b_orb,-t,qp)
	for i in range(len(qp[1,:])):
		print(len(qp[1,:]))
		print(periods[i])
		drawingpoint(qp[:,i],periodicpoint[:,i],periods[i],ax)
	plt.show()
	return allpoint
def Press(event):
    global x1,y1,DragFlag
    if (event.xdata is None) or (event.ydata is None):
        return
    cx = event.xdata
    cy = event.ydata

    x1 = cx
    y1 = cy
    if event.button == 2:
    	print('tt')
    	DragFlag = True
    else:
        return


def Drag(event):
    global x1,y1,x2,y2,DragFlag
    if DragFlag == False:
        return

    if (event.xdata is None) or (event.ydata is None):
        return

    cx = event.xdata
    cy = event.ydata

    x2 = cx
    y2 = cy

    ix1,ix2 = sorted([x1,x2])
    iy1,iy2 = sorted([y1,y2])
    print('o')
    DrawRect(ix1,ix2,iy1,iy2)
    plt.draw()

def Release(event):
    global x1,y1,x2,y2
    global DragFlag
    global initpoint_array
    DragFlag = False 
    if event.button == 2:
        DragFlag = False
        ix1,ix2 = sorted([x1,x2])
        iy1,iy2 = sorted([y1,y2])
        getsection_on_thetaplane(initpoint_array[0],initpoint_array[1],ix1,ix2,iy1,iy2)         



def DrawRect(x1,x2,y1,y2):
    Rect = [ [ [x1,x2],[y1,y1] ],
             [ [x2,x2],[y1,y2] ],
             [ [x1,x2],[y2,y2] ],
             [ [x1,x1],[y1,y2] ] ]
    #print(Rect)
    #print(lns)
    #print(lns)
    #for rect in Rect:
    	#ln, = plt.plot(rect[0],rect[1],color='g',lw=1)
    	#lns.append(ln)
    
    #for i,rect in enumerate(Rect):
    #   ln = lns[i].set_data(rect[0],rect[1])
    #   lns.append(ln)

def onkey(event):
    return


def drawingpoint(point,periodicpoint,period,ax):
	global actions, iterates,textpath
	clearnumber = 6
	print(periodicpoint)
	#fig  = plt.figure(figsize = (15,15))
	#ax = fig.add_subplot(121, projection = '3d')
	#ax = fig.add_subplot(121)
	#ax.yaxis.set_label_coords(-0.5, 0.5)
	#ax.xaxis.set_label_coords(-0.5, 0.5)
	#ax.zaxis.set_label_coords(-0.00, 0.06)
	periodicpoint = np.array(periodicpoint,dtype = np.float64)
	#ax.set_title(r'perx = ${}$, pery = ${}$'.format(str(round(periodicpoint[0],4)),str(round(periodicpoint[1],4))),fontsize = 30)
	#plt.plot(point[0].real,point[1].real ,'o',zorder = 1)
	points = [np.array([])] * 3
	points_dist = np.array([])
	periodicpoint_array = np.array([]) 
	#plt.plot(point[0].real , point[1].real , (np.imacg(point[0]) ** 2 + np.imag(point[1]) **2 ) ** 0.5 , label = 'start')
	colormap = plt.get_cmap("tab10")
	colormap2 = plt.get_cmap("tab20")
	import copy
	pointed = copy.deepcopy(point)
	print(point)
	prepoint = point
	#print(linepoints)
	#Z = f(X, Y)
	#ax.plot_surface(X, Y, Z,alpha = 0.1)
	initpoint = [np.array([])] * 3
	initpoint[0] = np.append(initpoint[0], point[0].real)
	initpoint[1] = np.append(initpoint[1], point[1].real)
		#points[2] = np.append(points[2], (np.imag(point[0]) ** 2 + np.imag(point[1]) ** 2) ** 0.5 )
	initpoint[2] = np.append(initpoint[2], (np.imag(point[0])  ))
	initpoint[0] = np.append(initpoint[0], point[0].real)
	initpoint[1] = np.append(initpoint[1], point[1].real)
		#points[2] = np.append(points[2], (np.imag(point[0]) ** 2 + np.imag(point[1]) ** 2) ** 0.5 )
	initpoint[2] = np.append(initpoint[2], (np.imag(point[0])  ))
	periodicpoint_array = [np.array([])] * 2
	#periodicpoint_array = np.array(periodicpoint) 
	periodicpoint_array[0] = np.append(periodicpoint_array[0],periodicpoint[0])
	periodicpoint_array[1] = np.append(periodicpoint_array[1],periodicpoint[1])
	periodicpoint_array[0] = np.append(periodicpoint_array[0],periodicpoint[0])
	periodicpoint_array[1] = np.append(periodicpoint_array[1],periodicpoint[1])
	for i in range(300):
		#if i  == 0:
		#	plt.plot(point[0].real, point[1].real ,(np.imag(point[0]) ** 2 + np.imag(point[1]) ** 2) ** 0.5 ,"o",zorder = 3,markersize = 13,color = "red" )
		points_dist = np.append(points_dist,(np.imag(point[0]) ** 2 + np.imag(point[1])  ** 2) ** 0.5)
		points[0] = np.append(points[0], point[0].real)
		points[1] = np.append(points[1], point[1].real)
		#points[2] = np.append(points[2], (np.imag(point[0]) ** 2 + np.imag(point[1]) ** 2) ** 0.5 )
		points[2] = np.append(points[2], (np.imag(point[1])  ))
		prepoint = point
		point = cmap.U(point)
		line = np.linspace(0, 1, 10 ** 3)
		diffvector  = prepoint - point
		print(np.ones_like(line) * (point[0].real)  +  line  * diffvector[0].real )
		linepoints = np.array([np.ones_like(line) * (point[0].real)  +  line  * diffvector[0].real  , np.ones_like(line) * point[1].real  + line * diffvector[1].real , np.ones_like(line) * point[0].imag + line * diffvector[0].imag ])
		print(linepoints)
		plt.tight_layout()
		if np.all(linepoints[2] > 0) == False:
			boolean_minusz = linepoints[2] > -0.001
			boolean_plusz = linepoints[2] < -0.001
			if np.all( np.abs(linepoints[2]) < 0.003 )  == False:
				ax.plot(linepoints[0,boolean_minusz],linepoints[1,boolean_minusz],linepoints[2,boolean_minusz],'-',zorder = 100 - period,color = colormap(period - 3),markersize = 12, lw = 5 )
				ax.plot(linepoints[0, boolean_plusz],linepoints[1,boolean_plusz],linepoints[2,boolean_plusz],'-',zorder = 100 - period,color =  colormap(period - 3 ),markersize = 12, lw = 5,linestyle = "dashed")
			else:
				ax.plot(linepoints[0,boolean_minusz],linepoints[1,boolean_minusz],linepoints[2,boolean_minusz],'-',zorder = 100 - period,color = colormap(period - 3),markersize = 12, lw = 5,alpha = 0.3 )
				ax.plot(linepoints[0, boolean_plusz],linepoints[1,boolean_plusz],linepoints[2,boolean_plusz],'-',zorder = 100 - period,color =  colormap(period - 3 ),markersize = 12, lw = 5,linestyle = "dashed",alpha = 0.3)
		else :
			if np.all( np.abs(linepoints[2]) < 0.003 )  == False:
				ax.plot(linepoints[0],linepoints[1],linepoints[2],'-',zorder = 100 - period,color = colormap(period - 3),markersize = 12, lw = 5 )
			else:
				ax.plot(linepoints[0],linepoints[1],linepoints[2],'-',zorder = 100 - period,color = colormap(period - 3),markersize = 12, lw = 5 ,alpha = 0.3)
		print(point)
		print("real = ",np.real(point[0])** 2 + np.real(point[1]) **2)
		#print("imag = ",np.imag(point[0] )** 2 + np.imag(point[1]) **2)
		if np.linalg.norm(point - periodicpoint) < 10 ** (-6) or np.linalg.norm( np.sqrt(np.imag(point[0]) ** 2 + np.imag(point[1]) ** 2) ) < 5 * 10 ** (-6):
			print(pointed)
			#exit()
			#print("")
			break
	phase = np.loadtxt('scatteringphase_original_3.txt')
	#stability = action2(pointed,periodicpoint,period)
	ax.plot(phase[0,0:80000],phase[1,0:80000],',b',zorder = 1, alpha = 0.5)
	#for i in range(1):
	#	fig  = plt.figure(figsize = (12,12))
	#	ax = fig.add_subplot(121, projection = '3d')
	#	ax2 = fig.add_subplot(122)
	points = np.array(points)
	print(points_dist)
	#exit()
	arg = np.argwhere(  np.diff( np.sign(points_dist - 0.001) ) != 0  )[0,0]
	points_complex = points[:,points_dist > 0.01]
	points_real = points[:,points_dist < 0.01]
	print(points_complex)
	#exit()_complex
	ax.view_init(elev=14, azim=-72)
	#ax.plot(points[0, 0 :  arg + 2],points[1, 0 : arg + 2 ],points[2, 0 : arg + 2 ],'-',zorder = 100 - period,color = colormap(period-3),lw = 5 )
	#ax.plot(points[0,arg :],points[1,arg : ] ,points[2, arg :],'-',zorder = 100 - period,color = colormap(period-3) , alpha = 0.4,lw = 5)
	#ax.plot(points[0, 0 :  arg + 3],points[1, 0 : arg + 3 ],points[2, 0 : arg + 3 ],'-o',zorder = 100 - period,color = "red",lw = 5 )
	#ax.plot(points[0,arg + 2 :],points[1,arg  + 2: ] ,points[2, arg  +  2:],'-o',zorder = 100 - period,color = "red", alpha = 0.1,lw = 5)
	#ax.plot(points[0,6:],points[1,6:],points[2,6:],'-o',zorder = 2,color = colormap(period-3), alpha = 0.4 )
	#ax.plot(initpoint[0],initpoint[1],initpoint[2],'o',zorder = 100 - period,color = colormap(period-3),markersize = 10, lw = 5 )
	horizonx = [0, - 2]
	ax.plot(initpoint[0],initpoint[1],initpoint[2],'o',zorder = 100 - period,color = colormap(period - 3),markersize = 12, lw = 5 )
	#ax.plot(periodicpoint_array[0],periodicpoint_array[1],"o",markersize = 12,color = colormap(period - 3))
		#ax.plot(points[0][60:90],points[1][60:90],points[2][60:90],'-o',zorder = 2,color = "aqua" )
	#ax.plot(points[0][200:300],points[1][200:300],points[2][200:300],'-o',zorder = 2,color = "limegreen" )
	#ax.plot(points[0][0:60],points[1][0:60],points[2][0:60],'-o',zorder = 2,color = "aqua" )
	#ax.plot(points[0][90:200],points[1][90:200],points[2][90:200],'-o',zorder = 2,color = "limegreen" )
	#ax.plot(points[0][70:90],points[1][70:90],'o',zorder = 2,color = "aqua" )
	#ax.plot(points[0][220:300],points[1][220:300],'o',zorder = 2,color = "limegreen" )
	#ax.plot(points[0][90:300],points[1][90:300],'o',zorder = 2,color = "limegreen" )
	#print(points[2][200:300])
	#	ax.plot(phase[0,:],phase[1,:],',k',zorder = 1)
	#	ax2.set_ylabel("Im(S)",fontsize = 27)
	#	ax2.set_xlabel("stability",fontsize = 27)
	#	#plt.tick_params(labelsize = 18)
	#	plt.rcParams["font.size"] = 18
	#ax.set_xlabel(r"${\rm Re }(q)$",fontsize = 25,labelpad = 26)
	#ax.set_ylabel(r"${\rm Re }(p)$", fontsize = 25,labelpad = 26)
	ax.set_xlabel(r"${\rm Re }(q)$",fontsize = 25,labelpad = 26)
	ax.set_ylabel(r"${\rm Re }(p)$", fontsize = 25,labelpad = 26)

	Igfont = {"family":"IPAexGothic"}
	plt.rcParams['font.family'] = 'IPAGothic'
	ax.set_zlabel(r"${\rm Im}(q)$", fontsize = 25, labelpad = 30,fontname ="IPAexGothic")
	#ax.set_zlim(0,0.05)
#		q	plt.xticks(fontsize=16)
#		plt.yticks(fontsize=16)
#		ax.tick_params(axis='z', labelsize= 16)
	ax.set_xlim(-0.5,0.5) 
	ax.set_ylim(-0.5,0.5)
	ax.set_zlim(-0.01,0.15)
#		ax2.plot(iterates,actions,"-o",color = "red")
		#plt.title("a = {}, b = {} ".format(str(round(a_orb,3)),str(round(b_orb,3))), fontsize = 33)
		#ax.set_zlabel("Distance from real plane", fontsize = 24)
		#plt.tick_params(labelbottom=False,
		#            labelleft=False,
		#           labelright=False,
		#           labeltop=False)z
		#plt.tick_params(bottom=False,
		#           left=False,
		#           right=False,
		#           top=False)
#		stability = action2(pointed,periodicpoint,period)
#		ax2.title.set_text("stability = {}".format( str( round(stability , 4) ) ) )
	#plt.title("stability = {}".format( str( round(stability , 4) ) ) )
#		plt.tight_layout()
	#plt.plot(t)
	plt.tight_layout()
	#plt.show()
#	
cmap = KScatteringMap(k,xf,xb)
def main():
	global step,stdrad
	stdrad  =  0.08
	#np.arange(0.0,0.15,0.04)
	for l in range(1,2):#2から7というのは番号で半径を表している．
		#stdrad = rad_array[l-1]
		print(stdrad)
		points_save  = [np.array([])] * 2
		ppoints_save = [np.array([])] * 2
		periodnumbers = np.array([])
		numbers = np.array([])
		thetas = np.array([]) 
		acts = np.array([])
		stability = [np.array([])] * 2
		stabilityfp = [np.array([])] * 2
		periods  = np.array([])
		iterates = np.array([])
		initpointthetas = [np.array([])] * 2
		a_str = format(str(round(a_orb,4))).replace('.','')
		b_str = format(str(round(b_orb,4))).replace('.','')
		textpath = 'initpoint_10_13_test14_a{}_b{}.txt'.format(str(round(a_orb,4)).replace('.',''), str(round(b_orb,4)).replace('.',''))
		if os.path.isfile(textpath) == True:
			os.remove(textpath)
		if os.path.isfile(textpath) == False:
		#if True:
			for k in [1,4,5,6,7,8]:
				period = k
				step = k
				#for s in range(len(rad_array)):
				#	stdrad = rad_array[s]
			#		print(stdrad)
			##			exit()
				#data = np.loadtxt("sectios113_s_period{}_rad{}.txt".format(period, str(round(stdrad,3)).replace('.','z')), dtype = np.complex128)
				data = np.loadtxt("section117_s_period{}_a{}_b{}.txt".format(period,a_str,b_str),dtype = np.complex128)
				points = np.array([data[:,4],data[:,5]])
				#print(points[0] ** 2 + points[1] ** 2)
				#print(points[0,:].shape)
				if len(data) != 0 :
					for i in range(len(points[0,:])):
						#print(i)
						numbers = np.append(numbers,i + 1)
						numbers = np.append(numbers,i + 1)
						point = points[:,i]
						point2 = imagchange(point)
						print(point2)
						print(point)
						periodicpoint = np.array([data[i,0],data[i,1]]) 
						#print(point)
						#theta  = np.array([data[i,2],data[i,3]])
						points_save[0] = np.append(points_save[0],point[0])
						points_save[1] = np.append(points_save[1],point[1])
						points_save[0] = np.append(points_save[0],point2[0])
						points_save[1] = np.append(points_save[1],point2[1])
						ppoints_save[0] = np.append(ppoints_save[0],periodicpoint[0])
						ppoints_save[1] = np.append(ppoints_save[1],periodicpoint[1])
						ppoints_save[0] = np.append(ppoints_save[0],periodicpoint[0])
						ppoints_save[1] = np.append(ppoints_save[1],periodicpoint[1])
						print(point)
						theta = gettheta(point) 
						theta2 = gettheta(point2)
						#exit()
						initpointthetas[0] = np.append(initpointthetas[0],theta.real)
						initpointthetas[1] = np.append(initpointthetas[1],theta.imag)
						initpointthetas[0] = np.append(initpointthetas[0],theta2.real)
						initpointthetas[1] = np.append(initpointthetas[1],theta2.imag)
						#print("theta =",theta.real % np.pi)
						#print("point= ", point)
						thetas = np.append(thetas, theta)
						thetas = np.append(thetas, theta2)
						#print("thetas = ", thetas[i])
						#print(periodicpoint)
						act = action(point, periodicpoint,period)
						act2 = action(point2, periodicpoint,period)
						print("act = {}".format(act))
						#exit()
						#print(acts)
						#print(type(act))
						if act != np.inf:
							acts = np.append(acts,act[0])
							#print(theta) 
							iterates = np.append(iterates,act[1])
							#print(periodicpoint)
							#print(periodicpoint)
							import Jacobian2
							jacob = Jacobian.strictJ2(cmap.U,periodicpoint,period)
							jacobforpoint = Jacobian2.strictJ2(cmap.U,periodicpoint,1)
							periods = np.append(periods, period)
							w,v = np.linalg.eig(jacob)
							wfp,vfp = np.linalg.eig(jacobforpoint)
							wfpabs0 = wfp[0] * np.conj(wfp[0])
							wfpabs1 = wfp[1] * np.conj(wfp[1])
							wabs0 = w[0] * np.conj(w[0])
							wabs1 = w[1] * np.conj(w[1])
							if wabs0 > wabs1:
								tmp = wabs1
								wabs1 = wabs0
								wabs0 = tmp
							if wfpabs0 > wfpabs1:
								tmp = wfpabs1
								wfpabs1 = wfpabs0
								wfpabs0 = tmp
							stability[0] = np.append(stability[0], ( wabs0 ** 0.5 ))
							stability[1] = np.append(stability[1],( wabs1 ** 0.5 ) )

							stabilityfp[0] = np.append(stabilityfp[0], ( wfpabs0 ** 0.5 ))
							stabilityfp[1] = np.append(stabilityfp[1],( wfpabs1 ** 0.5 ) )
						if act2 != np.inf:
							acts = np.append(acts,act2[0])
							#print(theta) 
							iterates = np.append(iterates,act2[1])
							#print(periodicpoint)
							#print(periodicpoint)
							jacob2 = Jacobian.strictJ2(cmap.U,periodicpoint,period)
							jacobforpoint = Jacobian2.strictJ2(cmap.U,periodicpoint,1)
							periods = np.append(periods, period)
							w2,v2 = np.linalg.eig(jacob2)
							wfp2,vfp2 = np.linalg.eig(jacobforpoint)
							wfpabs02 = wfp2[0] * np.conj(wfp2[0])
							wfpabs12 = wfp2[1] * np.conj(wfp2[1])
							wabs02 = w2[0] * np.conj(w2[0])
							wabs12 = w2[1] * np.conj(w2[1])
							if wfpabs02 > wfpabs12 :
								tmp = wfpabs12
								wfpabs12 = wfpabs02
								wfpabfs02 = tmp
							if wabs02 > wabs12 :
								tmp = wabs12
								wabs12 = wabs02
								wabs02 = tmp
							stability[0] = np.append(stability[0], ( wabs02 ** 0.5 ))
							stability[1] = np.append(stability[1],( wabs12 ** 0.5 ) )	
							stabilityfp[0] = np.append(stabilityfp[0], ( wfpabs02 ** 0.5 ))
							stabilityfp[1] = np.append(stabilityfp[1],( wfpabs12 ** 0.5 ) )
					#print(stability[0])
						acts = np.abs(acts)
						#print(acts)
			#print(acts)
			with open(textpath,'a')as p_hand:#zのやつを開いてください
				print(np.array(initpointthetas[0]).shape)
				print(np.array(numbers).shape)
				vec = np.array([ periods,numbers, acts, stability[0],stability[1],iterates, points_save[0], points_save[1], ppoints_save[0],ppoints_save[1],initpointthetas[0],initpointthetas[1],stabilityfp[0],stabilityfp[1]])#i perio
				vec2 = np.array([ periods, numbers, acts, stability[0],stability[1],iterates, thetas,ppoints_save[0],ppoints_save[1],stabilityfp[0],stabilityfp[1]])
				#print(vec.shape)
				vec = vec.T
				vec2 = vec2.T
				vec = np.array( sorted(vec, key =lambda x : abs(x[1]) ))
				#print(abs(vec[:,8] - 0 ) > 10 ** (-6))
				vec2 = np.array( sorted(vec2, key =lambda x : abs(x[1]) ))
				for m in range(len(vec)):
					print(vec[m].shape)
					#print(vec[10].shape)
				vec = vec[ abs(vec[:,3]- 1) > 10 ** (-7) ]
				vec = vec[ np.logical_or( abs(vec[:,4]) > 1.60,  abs(vec[:,4])  < 1.3  )]
				print(vec)
				#vec = vec[abs(vec[:,8] - 0 ) > 10 ** (-6)]
				vec2 = vec2[ abs(vec2[:,7]- 1) > 10 ** (-7) ]
				#print(vec)
				np.savetxt(p_hand,vec,newline = "\r\n", fmt = '%.16f%+.16fj ' * 14)
		#plotactions(stdrad,textpath)
		plotperiods(textpath)
		#textpath = '2020_8_11_rad{}_test2.txt'.format(str(round(stdrad,3)).replace('.','z'))
		#drawingactions(stdrad,textpath)
		#終状態としているの周期の情報と

	#number = 15
	#point = np.array([vec[number,5],vec[number,6]])
	#print(point)
	#print("this = ",vec[number,:])
	#print(vec[0,0])
	#period = int(vec[number,0])
	#periodicpoint = np.array([vec[number,3],vec[number,4]])
	#print(periodicpoint)
	#periodic = np.array([vec2[number,6],vec2[number,7]])
		#for j in range(period):
		#	plt.plot(periodic[0],periodic[1],'-o',zorder= 2,markersize = 12,color = 'green')
		#	periodic = cmap.U(periodic)
		#act = action(point,period )
	#tree = np.loadtxt("treeall.txt")
	#plt.plot(vec2[number,5].real,vec2[number,5].imag,'o',zorder =2)
	#act = action(point, periodicpoint,vec2[number,0])
	#plt.plot(tree[:,0],tree[:,1],',k',zorder = 1)
	#print(vec2[number,1])
	#plt.show()n
	#plt.plot(vec2[number,6],vec2[number,7],'o',zorder= 2,markersize = 12,color = 'green')
	#phase = np.loadtxt("scatteringphase_small.txt")
	#plt.plot(phase[0,:],phase[1,:],',k')
	#plt.xlabel("q",fontsize = 33)
	#plt.ylabel("p", fontsize = 33)	
	#plt.show()
	#idx = (vec[:,4]).argmin()
	#rint(idx)
	#print(arg)
	#fnp.loadtxt("section_s_period{}.txt".format(vec[idx,0]),dtype = np.complex128)
	#print(vec)
	#繋げて並び替えYO

	#繋げて並び替えYO


if __name__ == "__main__":
	main()