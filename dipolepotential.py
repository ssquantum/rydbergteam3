import numpy as np
import matplotlib.pyplot as plt


def testplot(wlen, power, w, w_0):

    ## Create Field ##
    x = y = np.arange(-30, 30, 0.1)
    x, y = np.meshgrid(x, y)

    ## Constants ##
    D_0 = 6 * (3.33e-30)
    hbar = 1.05e-34
    c = 3e+8
    eps_0 = 8.85e-12

    ## Calculated Consants ##
    Delta = w - w_0
    rho = x**2 + y**2

    ## Main calc ##
    u = (D_0**2)/(2*hbar*Delta)*(4*power)/(np.pi*w_0*c*eps_0)*((w_0/w)**2)*np.exp(-2*(rho**2)/(w**2))
    print(u)
    return x, y, u


(x1, y1, u1) = testplot(940e5, 5000, 400, 390)
plt.contourf(x1, y1, u1,levels=100)
plt.colorbar()
plt.show()
