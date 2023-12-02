import numpy as np
from matplotlib import pyplot, cm


def draw_plot(func, steps, x_lims=(-2, 4), y_lims=(-2, 2)):
    # define range for input
    x_r_min, x_r_max = x_lims
    y_r_min, y_r_max = y_lims

    # sample input range uniformly at 0.1 increments
    xaxis = np.arange(x_r_min, x_r_max, 0.1)
    yaxis = np.arange(y_r_min, y_r_max, 0.1)

    # create a mesh from the axis
    x, y = np.meshgrid(xaxis, yaxis)

    # compute targets
    z = func(x, y)

    # create a surface plot
    figure = pyplot.figure()
    figure.add_subplot(projection='3d')
    axis = figure.axes[0]
    # axis.plot_surface(x, y, z, cmap='jet')
    axis.plot_wireframe(x, y, z, rstride=10, cstride=10)
    axis.contour(x, y, z, cmap=cm.magma)
    # print(steps)
    axis.plot(steps[0], steps[1], steps[2], color='red', linewidth=3, label='optimization progress')

    # show the plot
    axis.set_xlabel('X')
    axis.set_ylabel('Y')
    axis.set_zlabel('Z')
    axis.legend()
    pyplot.show()