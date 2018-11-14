import numpy as np

import matplotlib.pyplot as plt




def testplot(wlen, power, w, w_0):




    ## Create Field ##

    x = y = z = np.linspace(0,10,100)

    #x, y, z = np.mgrid(X,Y,Z)
    N=100

    #u = np.zeros(N,N)

    

    ## Constants ##

    D_0 = 6*(3.33e-30)

    hbar = 1.05e-34

    c = 3e+8

    eps_0 = 8.85e-12

    

    ## Calculated Consants

    Delta = w - w_0

    rho = x**2 + y**2




    ## Main calc ##

    u = (D_0**2)/(2*hbar*Delta)*(4*power)/(np.pi*w_0*c*eps_0)*((w_0/w)**2)*np.exp(-2*rho**2/w**2)

    return x,y,u




(x1,y1,u1) = testplot(940e-9, 1, 900e-9, 940e-9)




plt.contour([x1,y1],u1)

plt.show()
