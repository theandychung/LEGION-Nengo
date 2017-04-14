import numpy as np
import Constants


def VectorField(ax, f, xran=[-5, 5], yran=[-5, 5], grid=[21, 21], color='k'):
    """
    Plot the direction field for an ODE written in the form
        x' = F(x,y)
        y' = G(x,y)

    The functions F,G are defined in the list of strings f.

    Input
    -----
    f:    list of strings ["F(X,Y)", "G(X,Y)"
          F,G are functions of X and Y (capitals).
    xran: list [xmin, xmax] (optional)
    yran: list [ymin, ymax] (optional)
    grid: list [npoints_x, npoints_y] (optional)
          Defines the number of points in the x-y grid.
    color: string (optional)
          Color for the vector field (as color defined in matplotlib)
    """
    # I = .5
    # epsilon = .2
    # gamma = 6.0
    # beta = 0.1
    x = np.linspace(xran[0], xran[1], grid[0]*1.5)
    y = np.linspace(yran[0], yran[1], grid[1]*1.5)

    def dX_dt(X, Y, t=0): return map(eval, f)

    X, Y = np.meshgrid(x, y)  # create a grid
    DX, DY = dX_dt(X, Y)  # compute growth rate on the grid
    M = (np.hypot(DX, DY))  # Norm of the growth rate
    M[M == 0] = 1.  # Avoid zero division errors
    DX = DX / M  # Normalize each arrows
    DY = DY / M

    ax.quiver(X, Y, DX, DY, pivot='mid', color=color)
    # ax.xlim(xran), plt.ylim(yran)
    # ax.grid('on')


## Example

# # Simple Pendulum
# # The sin function should be passed within
# # the scope of numpy `np`
# vecterfield = ["3 * X - X**3 + 2 - Y", "epsilon * (gamma * (1 + np.tanh(X / beta)) - Y)"]
# plotdf(vecterfield, xran=[xmin, xmax], yran=[ymin, ymax])
#
# X = np.linspace(xmin,xmax)
# plt.plot(X,3.*X-X**3+2+I,label='x-nullcline',color='b')
# plt.plot(X,gamma * (1 + np.tanh(X / beta)),label='y-nullcline',color='r')
#
# plt.show()
# reference https://gist.github.com/nicoguaro/6767643
