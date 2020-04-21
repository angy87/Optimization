import numpy as np
from numpy.linalg import norm as norm
from scipy.optimize import minimize_scalar as line_search
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def steepest_descent(gradf, x0, y0, tolerance, max_iters):
    
    """
    gradf = gradient of the objective function
    x0, y0: coordinates of the starting point
    tolerance: tolerance for the stopping criteria
    max_iters = max number of iterations allowed
    """
    
    
    iters = 0  # iteration counter
    
    all_x, all_y = [], []  # store all the moves
    
    all_x.append(x0)
    all_y.append(y0)

    while norm(gradf(x0, y0)) > tolerance and iters <= max_iters:
        
        s = - gradf(x0,y0) / norm(gradf(x0, y0))  # search direction vector
        
        # Set the 1-D optimization problem
        f_alpha = lambda alpha : f(x0 + alpha * s[0], y0 + alpha * s[1])
        
        # Golden section search method to minimize f_alpha
        alpha = line_search(f_alpha, method = 'Golden').x

        # New point
        x_new = x0 + alpha * s[0]
        y_new = y0 + alpha * s[1]
        
        # Update old point
        x0 = x_new
        y0 = y_new
        
        iters += 1   
        
        all_x.append(x_new)
        all_y.append(y_new)
                
    return x_new, y_new, all_x, all_y



f = lambda x,y : x**2 -2 * x * y + 4 * y**2

gradf = lambda x,y : np.array([(2*x - 2*y), (-2*x +8*y)])

# Starting point coordinates
x0 = -15;
y0 = -15;

tolerance = 10**(-3);  # stopping criterion
max_iters = 100  # maximum number of iterations

x_min, y_min, all_x, all_y = steepest_descent(gradf, x0, y0, tolerance, max_iters)

        
        
# --------------------------------------
# Fig 1

x = np.arange(-20, 21, 1)

y = np.arange(-20, 21, 1)

X, Y = np.meshgrid(x, y)

Z = f(X,Y)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot_surface(X, Y, Z)
ax.scatter(x_min, y_min, f(x_min, y_min))
plt.show()


# --------------------------------------
# Fig 2

fig = plt.figure()
plt.axis('equal')
plt.contour(X, Y, Z, 25, linestyles='dashed')
plt.plot(x_min, y_min, 'ro', markersize = 10)
plt.plot(all_x, all_y, 'x-')
plt.xlabel('x')
plt.ylabel('y')
fig.savefig('steepest_descent.pdf')
