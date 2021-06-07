import numpy as np
import matplotlib.pyplot as plt
import random



def cost_func(theta0, theta1, x, y):
    """The cost function, J(theta0, theta1) describing the goodness of fit."""
    theta0 = np.atleast_3d(np.asarray(theta0))
    theta1 = np.atleast_3d(np.asarray(theta1))
    return np.average((y-hypothesis(x, theta0, theta1))**2, axis=2)/2

def hypothesis(x, theta0, theta1):
    """Our "hypothesis function", a straight line."""
    return theta0 + theta1*x


def gradient_descent(W, x, y):
    y_hat = x.dot(W).flatten()
    error = (y - y_hat)
    mse = (1.0 / len(x)) * np.sum(np.square(error))
    gradient = -(1.0 / len(x)) * error.dot(x)
    return gradient, mse

def sgd_plot():
    # The data to fit
    m = 100
    theta0_true = 2
    theta1_true = 0.5
    x = np.linspace(-1, 1, m)
    y = theta0_true + theta1_true * x + theta0_true ** 2 * x

    # The plot: LHS is the data, RHS will be the cost function.
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6.15))
    ax[0].scatter(x, y, marker='x', s=40, color='k')
    # First construct a grid of (theta0, theta1) parameter pairs and their
    # corresponding cost function values.
    theta0_grid = np.linspace(-1,4,101)
    theta1_grid = np.linspace(-5,5,101)
    J_grid = cost_func(theta0_grid[np.newaxis,:,np.newaxis],
                       theta1_grid[:,np.newaxis,np.newaxis],
                       x, y)

    # A labeled contour plot for the RHS cost function
    X, Y = np.meshgrid(theta0_grid, theta1_grid)
    contours = ax[1].contour(X, Y, J_grid, 30)
    ax[1].clabel(contours)
    # The target parameter values indicated on the cost function contour plot
    ax[1].scatter([theta0_true]*2,[theta1_true]*2,s=[50,10], color=['k','w'])

    # Take N steps with learning rate alpha down the steepest gradient,
    # starting at (theta0, theta1) = (0, 0).
    N = 5
    alpha = 0.7
    theta = [np.array((0,0))]
    J = [cost_func(*theta[0], x, y)[0]]

    for j in range(N-1):
        last_theta = theta[-1]
        this_theta = np.empty((2,))
        this_theta[0] = last_theta[0] - alpha / m * np.sum(
                                        (hypothesis(x, *last_theta) - y))
        this_theta[1] = last_theta[1] - alpha / m * np.sum(
                                        (hypothesis(x, *last_theta) - y) * x)
        theta.append(this_theta)

        J.append(cost_func(*this_theta, x, y))


    # Annotate the cost function plot with coloured points indicating the
    # parameters chosen and red arrows indicating the steps down the gradient.
    # Also plot the fit function on the LHS data plot in a matching colour.
    colors = ['b', 'g', 'm', 'c', 'orange']
    ax[0].plot(x, hypothesis(x, *theta[0]), color=colors[0], lw=2,
               label=r'$\theta_0 = {:.3f}, \theta_1 = {:.3f}$'.format(*theta[0]))
    for j in range(1,N):
        ax[1].annotate('', xy=theta[j], xytext=theta[j-1],
                       arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
                       va='center', ha='center')
        ax[0].plot(x, hypothesis(x, *theta[j]), color=colors[j], lw=2,
               label=r'$\theta_0 = {:.3f}, \theta_1 = {:.3f}$'.format(*theta[j]))
    ax[1].scatter(*zip(*theta), c=colors, s=40, lw=0)

    # Labels, titles and a legend.
    ax[1].set_xlabel(r'$\theta_0$')
    ax[1].set_ylabel(r'$\theta_1$')
    ax[1].set_title('Cost function')
    ax[0].set_xlabel(r'$x$')
    ax[0].set_ylabel(r'$y$')
    ax[0].set_title('Data and fit')
    axbox = ax[0].get_position()
    # Position the legend by hand so that it doesn't cover up any of the lines.
    ax[0].legend(loc=(axbox.x0+0.5*axbox.width, axbox.y0+0.1*axbox.height),
                 fontsize='small')

    plt.show()

def architecture_plot():
    x = [5.3, 7.8, 9.2, 12, 19, 30, 43, 66, 26, 44.4, 14, 89, 84]
    y = [77.1, 79.1, 80.1, 81.6, 82.9, 83.6, 84, 84.3, 76, 80, 76.2, 82.7, 80.9]
    networks = [
        'EfficientNet B0',
        'EfficientNet B1',
        'EfficientNet B2',
        'EfficientNet B3',
        'EfficientNet B4',
        'EfficientNet B5',
        'EfficientNet B6',
        'EfficientNet B7',
        'ResNet 50',
        'ResNet 101',
        'DenseNet 169',
        'NASNet',
        'ResNext 101',
        ]

    fig, ax = plt.subplots()
    ax.scatter(x, y)

    for i, txt in enumerate(networks):
        # ax.annotate(txt, (x[i], y[i]))
        if txt == 'ResNet 50':
            plt.text(x[i] + 0.9, y[i], txt, fontsize=9)
        elif txt == 'NASNet' or txt == 'ResNext 101':
            plt.text(x[i] - 6, y[i] + 0.15, txt, fontsize=9)
        elif txt == 'EfficientNet B5':
            plt.text(x[i] + 0.9, y[i]-0.1, txt, fontsize=9)
        else:
            plt.text(x[i]+0.1, y[i]+0.1, txt, fontsize=9)
    plt.grid(color='black', linestyle='-', linewidth=0.1)
    plt.plot(x, y, '*')
    plt.title('ImageNet performance of all Network Architectures')
    plt.ylabel('Top-1 Accuracy (%)')
    plt.xlabel('Number of parameters (millions)')
    plt.legend()
    fig.savefig(r'C:\Users\Bruger\OneDrive\DTU - General engineering\6. Semester\Bachelor\pictures\GFLOPS_plot.png')
    plt.show()

def adam_plot():
    # example of plotting the adam search on a contour plot of the test function
    from math import sqrt
    from numpy import asarray
    from numpy import arange
    from numpy.random import rand
    from numpy.random import seed
    from numpy import meshgrid
    from matplotlib import pyplot
    from mpl_toolkits.mplot3d import Axes3D

    # objective function
    def objective(x, y):
        return x ** 2.0 + y ** 2.0

    # derivative of objective function
    def derivative(x, y):
        return asarray([x * 2.0, y * 2.0])

    # gradient descent algorithm with adam
    def adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2, eps=1e-8):
        solutions = list()
        # generate an initial point
        x = bounds[:, 0] + rand(len(bounds)) * (bounds[:, 1] - bounds[:, 0])
        score = objective(x[0], x[1])
        # initialize first and second moments
        m = [0.0 for _ in range(bounds.shape[0])]
        v = [0.0 for _ in range(bounds.shape[0])]
        # run the gradient descent updates
        for t in range(n_iter):
            # calculate gradient g(t)
            g = derivative(x[0], x[1])
            # build a solution one variable at a time
            for i in range(bounds.shape[0]):
                # m(t) = beta1 * m(t-1) + (1 - beta1) * g(t)
                m[i] = beta1 * m[i] + (1.0 - beta1) * g[i]
                # v(t) = beta2 * v(t-1) + (1 - beta2) * g(t)^2
                v[i] = beta2 * v[i] + (1.0 - beta2) * g[i] ** 2
                # mhat(t) = m(t) / (1 - beta1(t))
                mhat = m[i] / (1.0 - beta1 ** (t + 1))
                # vhat(t) = v(t) / (1 - beta2(t))
                vhat = v[i] / (1.0 - beta2 ** (t + 1))
                # x(t) = x(t-1) - alpha * mhat(t) / (sqrt(vhat(t)) + ep)
                x[i] = x[i] - alpha * mhat / (sqrt(vhat) + eps)
            # evaluate candidate point
            score = objective(x[0], x[1])
            # keep track of solutions
            solutions.append(x.copy())
            # report progress
            print('>%d f(%s) = %.5f' % (t, x, score))
        return solutions

    # seed the pseudo random number generator
    seed(42)
    # define range for input
    bounds = asarray([[-1.0, 1.0], [-1.0, 1.0]])
    # define the total iterations
    n_iter = 100
    # steps size
    alpha = 0.02
    # factor for average gradient
    beta1 = 0.8
    # factor for average squared gradient
    beta2 = 0.999
    # perform the gradient descent search with adam
    solutions = adam(objective, derivative, bounds, n_iter, alpha, beta1, beta2)
    # sample input range uniformly at 0.1 increments
    xaxis = arange(bounds[0, 0], bounds[0, 1], 0.1)
    yaxis = arange(bounds[1, 0], bounds[1, 1], 0.1)
    # create a mesh from the axis
    x, y = meshgrid(xaxis, yaxis)
    # compute targets
    results = objective(x, y)
    # create a filled contour plot with 50 levels and jet color scheme
    pyplot.contourf(x, y, results, levels=50, cmap='jet')
    # plot the sample as black circles
    solutions = asarray(solutions)
    # for j in range(1,n_iter):
    #     pyplot.annotate('', xy=solutions[int(j)], xytext=solutions[int(j - 1)],
    #                    arrowprops={'arrowstyle': '->', 'color': 'r', 'lw': 1},
    #                    va='center', ha='center')
    pyplot.plot(solutions[:, 0], solutions[:, 1], '.-', color='w', label='100 iterations')
    # show the plot
    pyplot.title('Adam Optimizer applied to x ** 2 + y ** 2')
    pyplot.ylabel('y')
    pyplot.xlabel('x')
    pyplot.legend()
    pyplot.show()

sgd_plot()