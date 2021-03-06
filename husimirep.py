import SimpleQmap as sq
import numpy as np
import matplotlib.pyplot as plt

def Traj(map,sample=50,tmax=500):
    q = np.random.random(sample)
    p = (np.random.random(sample) -0.5) * (pmax - pmin) * 2
    res = [np.array([])]*2
    for i in range(tmax):
        pp = p - map.func0(q)
        qq = q + map.func1(pp)
        q = qq - np.floor(qq)
        p = pp - np.floor(pp)
        res[0] = np.append(res[0],q)
        res[1] = np.append(res[1],p)
    return res


dim = 50
k = 1
qmin, qmax = 0, 1
pmin, pmax = 0, 1 

map = sq.StandardMap(k=k)

a_orb = 0.3
b_orb = 0.30
dimension = 2
k = 0.01
xf = 1.2
xb = 1.0
a_orb = 0.22#0.22
b_orb = 0.06#0.06
t = 0#60/180 * np.pi
stdrad = 0.08
e2 = k * xf * (np.exp(-8 * xb * (2 * xf - xb)) / (1 - np.exp(-32 * xf * xb)))


def dotV(x):
    return - k * np.sin(x)
def U(qp):
     return np.array([ qp[0] + qp[1] - dotV(qp[0]) , qp[1] -  dotV(qp[0]) ])
traj = Traj(map)


# -*- coding:utf-8 -*-
"""
state.py

author: Yasutaka Hanada (2013/05/17)

Domain Setting and Quantum states.


"""

import numpy
twopi = 2.0*numpy.pi


def loadtxt(title,verbose=True):
    import re
    from SimpleQmap import State,ScaleInfo
    with open(title, "r") as file:
        for i, line in enumerate(file):
            if re.search('DIM', line):
                dim = int(line.split(" ")[2])
            if re.search('QMIN', line):
                qmin = float(line.split(" ")[2])
            if re.search('QMAX', line):
                qmax = float(line.split(" ")[2])
            if re.search('PMIN', line):
                pmin = float(line.split(" ")[2])
            if re.search('PMAX', line):
                pmax = float(line.split(" ")[2])
            if re.search('REP', line):
                rep = line.split(" ")[2].replace("\n","")

    if rep not in ["q","p"]:
        raise TypeError("must be q-rep or p-rep text file")

    data = numpy.loadtxt(title).transpose()
    vec = data[2] + 1.j*data[3]
    scl = ScaleInfo(dim, [[qmin,qmax], [pmin, pmax]])
    state = State(scl, vec)
    if rep == "p":
        state = state.p2q()
    if verbose:
        t = "load:%s\n" % title
        t += "dim:%d\n" % dim
        t += "domain:[%f,%f]x[%f,%f]\n" % (qmin,qmax, pmin, pmax)
        t += "representation:q" if rep =="q" else "!!Warning!!\nconvert original data (p-rep.) to q-rep.\n"
        print(t)
    return state


class ScaleInfo(object):
    """

    ScaleInfo???????????????????????????????????????????????????class?????????

    Parameters
    ----------
    dim : int
        Hilbert Space dimension
    domain: list
        domain of phase space. expected 2 by 2 list, e.g., [[qmin,qmax], [pmin, pmax]]

    Examples
    ----------
    >>> from qmap import ScaleInfo
    >>> scl = ScaleInfo(10, [[0,1],[-0.5,0.5]])
    >>> print(scl.x[0])
    [ 0.   0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9]
    >>> print(scl.x[1])
    [-0.5 -0.4 -0.3 -0.2 -0.1  0.   0.1  0.2  0.3  0.4]
    >>> print(scl.h) # scl.getPlanck()
    0.1
    """
    def __init__(self, dim, domain):
        self.dim = dim
        self.__setDomain(domain)

    def __setDomain(self, domain):
        if domain[0][0] > domain[0][1]:
            raise ValueError("qmin > qmax")
        if domain[1][0] > domain[1][1]:
            raise ValueError("pmin < qmax")

        self.domain = domain
        self.x = [numpy.linspace(self.domain[i][0], self.domain[i][1], self.dim, endpoint=False) for i in range(2)]
        self.h = self.getPlanck()
        self.hbar = self.h/twopi

    def getPlanck(self):
        """ return effective planck constant """
        W = (self.domain[0][1] -self.domain[0][0])*(self.domain[1][1] - self.domain[1][0])
        return W/self.dim

    def getX(self):
        """ return [q, p] """
        return self.x

    def state(self, data=[]):
        if len(data) == 0:
            return State(self)
        else:
            return State(self, data)

class State(numpy.ndarray):
    """

    State ???ScaleInfo???????????????????????????????????? :math:`| \psi\\rangle` ?????????????????????
    ????????? :math:`q-` ????????????????????? :math:`\\langle q | \\psi\\rangle` ???????????????
    ???????????????????????????????????????
    numpy.ndarray???????????????????????????


    Parameters
    ----------
    scaleinfo : ScaleInfo instance
                input scaleinfo instance
    data     : State, optional
                 if data is None, return a new array of given scaleinfo, filled with zeros.
                 if data is not None, return a new array of given data,
                 Note that length data must be same scaleinfo dimnsion.

    See Also
    --------
    numpy.ndarray : subclassing ndarray <http://docs.scipy.org/doc/numpy/user/basics.subclassing.html>


    Examples
    ----------

    >>> from qmap import State, ScaleInfo
    >>> scl = ScaleInfo(10, [[0,1],[-0.5,0.5]])
    >>> State(scl, range(10))
    State([ 0.+0.j,  1.+0.j,  2.+0.j,  3.+0.j,  4.+0.j,  5.+0.j,  6.+0.j,
            7.+0.j,  8.+0.j,  9.+0.j])
    >>> vec = State(scl)
    >>> print(vec)
    [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j
      0.+0.j]
    >>> print(vec.scaleinfo.x[0])
    [ 0.   0.1  0.2  0.3  0.4  0.5  0.6  0.7  0.8  0.9]
    >>> newvec = vec.copy()
    >>> print(newvec)
    [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j
      0.+0.j]
    >>> print(newvec.scaleinfo.h)
    0.1
    >>> newvec[0] = 1.0+0.j
    >>> vec
    State([ 0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
            0.+0.j,  0.+0.j,  0.+0.j])
    >>> newvec
    State([ 1.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,  0.+0.j,
            0.+0.j,  0.+0.j,  0.+0.j])

    """

    def __new__(cls, scaleinfo, data=None):
        if not isinstance(scaleinfo , ScaleInfo): raise TypeError("expected type ScaleInfo", type(scaleinfo))
        if data is None:
            data = numpy.zeros(scaleinfo.dim)
        cls.scaleinfo = scaleinfo
        cls.x = scaleinfo.x
        cls.domain = scaleinfo.domain
        cls.hbar = scaleinfo.hbar
        obj = numpy.asarray(data, dtype=numpy.complex128).view(cls)
        return obj

    def __array_finalize__(self, obj):
        if obj is None: return
        #self.scaleinfo = getattr(obj, 'scaleinfo', None)

    def savetxt(self, filename, rep='q',**kwargs):
        """
        Save an state data to a text file

        Parameters
        ----------
        filename: str
            file name
        rep: str
            'q', 'p' or 'hsm'

        See Also
        ----------
        numpy.savetxt <http://docs.scipy.org/doc/numpy/reference/generated/numpy.savetxt.html>
        """
        ann = self.__annotate(rep)
#        abs2 = numpy.abs(self*numpy.conj(self))
        if rep in ["q", "p"]:
            x = self.scaleinfo.x[0] if rep == "q" else self.scaleinfo.x[1]
            state = self if rep == "q" else self.q2p()
            abs2 = state.abs2()
            data = numpy.array([x, abs2, state.real, state.imag])
            numpy.savetxt(filename, data.transpose(), header=ann)
        elif rep == "hsm":
            x,y,z = self.hsmrep(**kwargs)
            data = numpy.array([x,y,z])
            with open(filename, "bw") as of:
                for slice_data in data.T:
                    numpy.savetxt(of, slice_data)
                    of.write(b"\n")
        else:
            raise TypeError('rep is "p", "q" or "hsm"')


    def __annotate(self,rep):
        import datetime
        ann ="DATE: %s\n" % datetime.datetime.now()
        ann += "DIM %d\n" % self.scaleinfo.dim
        ann += "QMIN %s\n" % self.scaleinfo.domain[0][0]
        ann += "QMAX %s\n" % self.scaleinfo.domain[0][1]
        ann += "PMIN %s\n" % self.scaleinfo.domain[1][0]
        ann += "PMAX %s\n" % self.scaleinfo.domain[1][1]
        ann += "PMAX %s\n" % self.scaleinfo.domain[1][1]
        ann += "REP %s" % rep
        return ann

    def insert(self, i, x):
        """

        >>> from state import State, ScaleInfo
        >>> scl = ScaleInfo(10, [[0,1],[-0.5,0.5]])
        >>> state = State(scl)
        >>> state.insert(2,2.0+1.j)
        >>> print(state)
        [ 0.+0.j  0.+0.j  2.+1.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j  0.+0.j
          0.+0.j]

        """
        if not isinstance(i, int): raise ValueError("excepted: integer")
        self[i] = x

    def coherent(self, q_c, p_c, x):
        """

        minimum-uncertainty Gaussian wave packet centered at (q_c,p_c, x)

        .. math::

            \langle x | \psi \\rangle = \exp[-(x-q_c)^2/2\hbar + p_c(x-q_c)/\hbar]

        .. warning::

            ??????????????????????????????????????????????????????????????????????????????????????????????????????

        Parameters
        ----------

        q_c, p_c : float
            Centroid (q_c,p_c,x) of the wave packet
        x : array
        """

        re = -(x - q_c)*(x - q_c)*numpy.pi/(self.scaleinfo.h)
        im = (x - q_c)*p_c*twopi/self.scaleinfo.h
        res = State(self.scaleinfo, data = numpy.exp(re+ 1.j*im))
        norm2 = numpy.abs(res.inner(res))
        return res/numpy.sqrt(norm2)

    def cs(self, q_c, p_c):
        """
        create new State object which
        minimum-uncertainty Gaussian wave packet centered at (q_c,p_c) on periodic boundary condition.

        Parameters
        ----------

        q_c, p_c : float
            Centroid (q_c,p_c) of wave packet

        Examples
        ----------

        >>> from state import State, ScaleInfo
        >>> scl = ScaleInfo(10, [[0,1],[-0.5,0.5]])
        >>> state = State(scl)
        >>> state.cs(0.5,0.1)
        State([ -5.19214101e-04 +4.91950621e-47j,
                -3.55650243e-03 -2.58395027e-03j,
                -1.22265106e-02 -3.76293303e-02j,
                 5.88151479e-02 -1.81014412e-01j,
                 3.95164004e-01 -2.87103454e-01j,
                 6.68740103e-01 -8.71513377e-71j,
                 3.95164004e-01 +2.87103454e-01j,
                 5.88151479e-02 +1.81014412e-01j,
                -1.22265106e-02 +3.76293303e-02j,  -3.55650243e-03 +2.58395027e-03j])

        .. seealso::

            Module :state:`coherent`

        """

        qrange = self.scaleinfo.domain[0]
        d = qrange[1] - qrange[0]
        lqmin, lqmax = qrange[0] - 2*d, qrange[1] + 2*d
        long_q = numpy.linspace(lqmin, lqmax, 5*self.scaleinfo.dim, endpoint=False)

        coh_state = self.coherent(q_c, p_c, long_q)

        vec = numpy.zeros(self.scaleinfo.dim, dtype=numpy.complex128)
        m = int(len(coh_state)/self.scaleinfo.dim)
        coh_state = coh_state.reshape(m,self.scaleinfo.dim)

        for i in range(m):
            vec += coh_state[i][::1]
        norm2 = numpy.dot(vec, numpy.conj(vec))

        return State(self.scaleinfo, vec/numpy.sqrt(norm2))

    def qrep(self):
        """
        return :math:`\\langle q | \\psi\\rangle` as a State object
        """
        return State(self.scaleinfo, self)

    def prep(self):
        """
        return :math:`\\langle p | \\psi\\rangle` as a State object
        """
        return State(self.scaleinfo, self.q2p())


    def p2q(self):
        """
        Fourier (inverse) transformation from
        the momentum :math:`(p)` representation to the position :math:`(q)` representation.

        .. math::

            \\langle q | \\psi\\rangle = \\sum_p \\langle q | p\\rangle \\!\\langle p | \\psi \\rangle

        """

        data = State(self.scaleinfo, self)
        if self.scaleinfo.domain[1][0]*self.scaleinfo.domain[1][1] < 0:
            data = numpy.fft.fftshift(data)
        data = numpy.fft.ifft(data)*numpy.sqrt(self.scaleinfo.dim)#/numpy.sqrt(self.scaleinfo.dim)
        return State(self.scaleinfo, data)

    def q2p(self):
        """
        Fourier (forward) transformation from
        the position :math:`(q)` representation to the momentum :math:`(p)` representation.

        .. math::

            \\langle p | \\psi\\rangle = \\sum_q \\langle p | q\\rangle \\!\\langle q | \\psi \\rangle

        """
        data = numpy.fft.fft(self)/numpy.sqrt(self.scaleinfo.dim)
        if self.scaleinfo.domain[1][0]*self.scaleinfo.domain[1][1] < 0:
            data = numpy.fft.fftshift(data)
        return State(self.scaleinfo, data) #*numpy.sqrt(self.scaleinfo.dim)

    def hsmrep(self, col=50, row=50, region=[]):
        """

        Husimi (phase space) representation.

        Parameter
        ----------
        col, row: int
            mesh grid of husimi rep.
        region: 2 by 2 list
            Husimi plot range. expected 2 by 2 list, e.g., [[qmin,qmax], [pmin, pmax]]
        """
        if len(region) == 0:
            region = self.scaleinfo.domain
        #else:
        #    region = hsm_region
        import SimpleQmap
        from SimpleQmap.ctypes_wrapper import wrapper
        import os, glob
        #path = os.environ['PYTHONPATH'].split(":")

        for path in glob.glob(SimpleQmap.__path__[0] + "/shared/libhsm*.so"):
            if os.path.exists(path):
                break
        try:
            cw = wrapper.call_hsm_rep(path)
        except NameError:
            raise RuntimeError("libhsm.so not found.")

        hsm_imag = cw.husimi_rep(self, self.scaleinfo.dim, self.scaleinfo.domain, region, [row,col])

        x = numpy.linspace(region[0][0], region[0][1], row)
        y = numpy.linspace(region[1][0], region[1][1], col)

        X,Y = numpy.meshgrid(x,y)
        return X,Y,hsm_imag


    def abs2(self):
        """
        return :math:`|\\langle x | \\psi \\rangle|^2` where :math:`x` is :math:`q` or :math:`p` as numpy.array object
        """
        data = self*numpy.conj(self)
        return numpy.abs(data)

    def norm(self):
        """
        return :math:`\\sum_x |\\langle x | \\psi \\rangle|^2` where
        :math:`x` is :math:`q` or :math:`p` as real value constant (numpy.float64)
        """
        norm = numpy.abs(numpy.sum(self*numpy.conj(self)))
        return numpy.float64(norm)

    def inner(self, phi):
        """
        return :math:`\\langle \\phi | \\psi\\rangle` as complex value constant (numpy.complex128)
        """
        res = numpy.sum(self*numpy.conj(phi))
        return State(self.scaleinfo, res)


    def copy(self):
        return State(self.scaleinfo, self.toarray())

    def toarray(self):
        return numpy.array(self.tolist())


class Matrix(numpy.matrix):
    def __new__(cls, mat, scaleinfo):
        cls.scaleinfo = scaleinfo
        cls.hbar = cls.scaleinfo.hbar
        cls.x = cls.scaleinfo.x
        return super(Matrix, cls).__new__(cls, mat)

    def eigh(self):
        data = numpy.squeeze(numpy.asarray(self))
        (evals, V) = numpy.linalg.eigh(data)
        index = numpy.argsort(evals)
        evecs = []
        for i in index:
            evecs.append( State(self.scaleinfo, V[:, i]) )
        return evals, evecs

    def eig(self):
        data = numpy.squeeze(numpy.asarray(self))
        (evals, V) = numpy.linalg.eig(data)
        evecs = []
        for i in range(len(evals)):
            evecs.append( State(self.scaleinfo, V[:, i]))
        return  evals, evecs




def _test():
    import doctest
    doctest.testmod()

if __name__ == "__main__":
    _test()


domain = [[qmin,qmax],[pmin,pmax]]
qmap = sq.Qmap(map, dim, domain) # defines the quantum system
evals, evecs = qmap.eigen() # return eigenvalues and list of eigenvector of the system. 


for i, evec in enumerate(evecs):
    fig,axs = plt.subplots(1,2,figsize=(10,5))

    #evec.savetxt("data.dat",rep="p") # rep = "q", "p" or "hsm"
    
    theta = np.linspace(-np.pi, np.pi, 100)
    z = np.exp(1.j*theta)
    axs[0].plot(z.real, z.imag,'-g')
    axs[0].plot(evals.real, evals.imag, 'ob')
    axs[0].plot(evals[i].real, evals[i].imag, 'or', markersize=10)

    x,y,z = evec.hsmrep(row=100,col=100)
    axs[1].contour(x,y,z,100)
    axs[1].plot(traj[0],traj[1],',k')
    
    fig.suptitle("%d-th eigenstate" % i)
    plt.show()
    break 