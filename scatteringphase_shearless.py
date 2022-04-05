import numpy as np
import matplotlib.pyplot as plt
#from numba import jit
import matplotlib.patches as pat
from mpl_toolkits.mplot3d import Axes3D 

scale_inv = 1
#definition of V"

def ddotV(x):
    return k*(1 - 16*x**2)*np.exp(-8*x**2) - 16*e2*((x - xb)*np.exp(-8*(x - xb)**2) - (x + xb)*np.exp(-8*(x + xb)**2))

#qpはvecter(q,p)
#正の時間方向の写像
def U(qp):
    return np.array([qp[0] + qp[1] - dotV( qp[0] ) ,  qp[1]  - dotV( qp[0] )])
#OVERLIDE
#def U(qp):
#    return qp[0] + qp[1] -  dotV(qp[0]), qp[1] - dotV(qp[0]) 

#逆写像
def Ui(qp):
    return (qp[0] - qp[1] - 0.5 * dotV(qp[0]), qp[1] + 0.5 * dotV(qp[0]) + 0.5 * dotV(qp[0] - qp[1] - 0.5 * dotV(qp[0])))

def U(qp):
        return np.array([qp[0] + tau * qp[1] - (tau ** 2) *  dotV(qp[0]) , qp[1] -  tau  * dotV(qp[0])  ])

def dotV(x):  #kicked potential の時間微分
        x_scaled = x/scale_inv
        return k * x_scaled * np.exp(-8 * x_scaled**2) - e2 * (np.exp(-8 * pow((x - xb)*scale_inv ,  2)) -  np.exp(-8 * pow( (x + xb) * scale_inv, 2)))

def dotV(x):
    return x + 2/lambda_1 * np.sin(x/lambda_1)


def ScattMapt(qp,step,cmap):
    for i in range(step):
        qp[0],qp[1] =  cmap.U([qp[0],qp[1]])
    return qp[0],qp[1]

def rotateinitialmanifold(a,b,t,points):
    sin_t = np.sin(t)
    cos_t = np.cos(t)
    R = np.array([[cos_t,sin_t],[-sin_t,cos_t]])
    print(points)
    rotated = np.dot(R,points)
    return rotated

k = 3.0
xf = 1.2
xb = 1.0
tmax = 5000
a_orb =  0.20334038#0.05027134 0.20334038
b_orb =0.05027134 
t = 1.0740275527854348

#b_orb = 0.44515836
#a_orb = 0.57929427
#t = 1.5096941616
#tau = 0.05
#b_orb = 1.04079259
#a_orb = 1.10344608
#t = 1.337145721
#lambda = 3.0
b_orb = 1.06462819
a_orb = 1.37029825
t = 1.50765824

#[1.04079259 1.10344708] #\lambda = 3.0の場合
#[0.23153047 0.97282765]
#angle between major axis and x axis= 1.3371457211538098
#angle between minor axis and x axis= -0.23365060564108692
#[1.06462819 1.37029825] lambda = 1.2　の場合
#[0.06309614 0.99800745]
#angle between major axis and x axis= 1.5076582419255626
#angle between minor axis and x axis= -0.06313808486933388

tau = 0.05
lambda_1 =3.0
e2 = k * xf * (np.exp(-8 * xb * (2 * xf - xb)) / (1 - np.exp(-32 * xf * xb)))
twopi = 2 * np.pi

def main():
    seed = 200
    q =  np.linspace(0,45,90)  
    p =  np.linspace(0,45,90)  
    #q =  -8 + (np.random.random(seed)  ) * 4 #* 0.001 + (-0.165532)
    #p =  -8 + (np.random.random(seed) ) * 4 #* 0.001 +  (-0.257289)
    #q2 =  8 + (np.random.random(seed)  ) * 4 #* 0.001 + (-0.165532)
    ##p2 =  8 + (np.random.random(seed) ) * 4 #* 0.001 +  (-0.257289)
    #q3 =   (np.random.random(seed)  ) * 24 #* 0.001 + (-0.165532)
    #p3 =   (np.random.random(seed) ) * 24 #* 0.001 +  (-0.257289)
    traj = [np.array([])] * 2 
    for i in range(tmax):
        print(i)
        #q,p = U([q,p])
        #q2,p2 = U([q2,p2])
        q,p = U([q,p])
        #traj[0] = np.append(traj[0],q)
        #traj[1] = np.append(traj[1],p)
        #traj[0] = np.append(traj[0],q2)
        #traj[1] = np.append(traj[1],p2)
        traj[0] = np.append(traj[0],q)
        traj[1] = np.append(traj[1],p)
        hantei = (traj[0] ** 2 + traj[1]  ** 2 ) ** 0.5
        #traj[0] = traj[0][abs(hantei) ]
        #traj[1] = traj[1][abs(hantei)  ]
    #np.savetxt('scatteringphase.txt',traj)
    #stdrad = np.arange(0.05,0.15,0.03)
    tau = np.linspace(0, 2*np.pi, 10 ** 4)
    q = a_orb * np.sin(tau)
    p = b_orb * np.cos(tau)
    qp = np.array([q,p])
    rotated = rotateinitialmanifold(a_orb,b_orb,-t,qp)
    #print(rotated)
    for l in range(1,2):
        #rad = stdrad[l-1]
        fig = plt.figure(figsize = (10,8))
        ax = fig.add_subplot(111)
        #for j in range(10):
        #    c = pat.Circle( xy=(0, 0), radius = rad-0.002+2*j/5000,fill = False, ec='r' ,zorder = 2)
        #    ax.add_patch(c)
        #    plt.title("radius = {} ".format(str(round(rad,3))), fontsize = 33)
        plt.plot(rotated[0],rotated[1],"-",color = "red",linewidth =  1.5)
        plt.plot(traj[0],traj[1],',k',zorder = 1)
        #c = pat.Circle(xy = (0.0,0.0),radius = 0.1,fc = "white",ec = "red")
        #ax.add_patch(c)
        plt.tick_params(labelsize = 18)
        plt.rcParams["font.size"] = 18
        plt.xlabel(r"$Re(q)$",fontsize = 33)
        plt.ylabel(r"$Re(p)$", fontsize = 33)
        plt.xlim(-40,40)
        plt.ylim(-40,40)
        #plt.xlim(-1.3, 1.3)
        #plt.ylim(-0.7,0.7)
        plt.xlabel(r"$q$", fontsize = 22)
        plt.ylabel(r"$p$", fontsize = 22)
        plt.tick_params(labelsize=18)
        #plt.plot(-1.2,0,'o',color = "green",markersize = 12)
        #plt.plot(1.2,0,'o',color = "green",markersize = 12)
        plt.rcParams["font.size"] = 22
        #plt.savefig("std_rad{}.png".format(str(round(rad,3)).replace('.','z')))
        #plt.title("a = {}, b = {} ".format(str(round(a_orb,3)),str(round(b_orb,3))), fontsize = 33)
        plt.tight_layout()
        #np.savetxt("scatteringphase_original_4.txt",traj)
        np.savetxt("phase_shearless_30.txt",traj)
    plt.show()

#np.savetxt('scatteringmap.txt',traj)

class ScatteringMap: #made by R.Kogawa
    def __init__(self,xb,k,e2):
        self.xb = xb
        self.k = k
        self.e2 = e2
    def U(self,qp): #正の時間方向の写像
        return [qp[0] + qp[1] - 0.5 * dotV(qp[0]), qp[1] - 0.5 * dotV(qp[0]) - 0.5 * dotV(qp[0] + qp[1] - 0.5 * dotV(qp[0]))]
    def Ui(self,qp): #inverse of
        return [qp[0] - qp[1] - 0.5 * dotV(qp[0]), qp[1] + 0.5 * dotV(qp[0]) + 0.5 * dotV(qp[0] - qp[1] - 0.5 * dotV(qp[0]))]

#def dotV(x):  #kicked potential の時間微分
#        return k * x * np.exp(-8 * x**2) - e2 * (np.exp(-8 * pow(x - xb, 2)) - np.exp(-8 * pow(x + xb, 2)))



if __name__ == '__main__':
        main()

