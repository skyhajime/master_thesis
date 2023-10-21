import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import const as C

import eigensolvers

def round_complex_number(x, decimal):
    return complex(round(x.real, decimal),round(x.imag, decimal))

def round_complex_matrix(M, decimal=1):
    vec_func = np.vectorize(round_complex_number)
    return vec_func(M, decimal)

def generate_krylov(evolution_function, observable_function, initial_point, num_col):
    Kr = []
    Er = []
    curr_state = initial_point
    for _ in range(num_col):
        Er.append(curr_state)
        Kr.append(observable_function(curr_state))
        curr_state = evolution_function(curr_state)
    npK = np.array(Kr)
    npE = np.array(Er)
    if npK.ndim == 1:
        npK = np.expand_dims(npK, axis=1)
    if npE.ndim == 1:
        npE = np.expand_dims(npE, axis=1)
    return npK, npE

def plot_complex_on_unit_circle(data):
    plt.figure(figsize=(5,5))
    theta = np.linspace(0, 2 * np.pi, 200)
    radius = 1
    a = radius * np.cos(theta)
    b = radius * np.sin(theta)
    plt.plot(a, b)

    x = [d.real for d in data]
    y = [d.imag for d in data]
    plt.scatter(x, y)
    plt.ylabel('Imaginary')
    plt.xlabel('Real')
    plt.xlim(-1.5,1.5)
    plt.ylim(-1.5,1.5)
    plt.show()
    return

def set_subplot_complex_on_unit_circle(data, ax, add_color=False):
    theta = np.linspace(0, 2 * np.pi, 200)
    radius = 1
    a = radius * np.cos(theta)
    b = radius * np.sin(theta)
    ax.plot(a, b)

    x = [d.real for d in data]
    y = [d.imag for d in data]
    if add_color:
        ax.scatter(x, y, c=list(range(len(data))))
    else:
        ax.scatter(x, y)
    ax.set_ylabel('Imaginary')
    ax.set_xlabel('Real')
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-1.5,1.5)
    return

def plot_torus(data, radius=1/2, n=100):
    u=np.linspace(0,2*np.pi,n)
    v=np.linspace(0,2*np.pi,n)
    u,v=np.meshgrid(u,v)
    X = (1 + radius*np.cos(u)) * np.cos(v)
    Y = (1 + radius*np.cos(u)) * np.sin(v)
    Z = radius * np.sin(u)

    fig = plt.figure(figsize=(5,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x axis')
    ax.set_ylabel('y axis')
    ax.set_zlabel('z axis')
    ax.set_xlim(-1.5,1.5)
    ax.set_ylim(-1.5,1.5)
    ax.set_zlim(-1.5,1.5)
    ax.set_box_aspect((1,1,1))
    ax.plot_surface(X, Y, Z,alpha=0.5, cmap=cm.Wistia)
    ax.plot3D(data[:,0], data[:,1], data[:,2], 'blue')
    ax.scatter3D(data[:,0], data[:,1], data[:,2], 'blue', marker='.')
    # fig.savefig("torus.png", dpi=130,bbox_inches = 'tight',transparent=True)
    plt.show()
    return

def plot_on_unit_square(data):
    assert data.ndim == 2
    plt.figure(figsize=(8,6))
    plt.scatter(data[:,0], data[:,1], c=list(range(len(data))))
    plt.colorbar()
    plt.axvline(1.0)
    plt.axvline(0.0)
    plt.axhline(1.0)
    plt.axhline(0.0)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.xlim(-0.1,1.1)
    plt.ylim(-0.1,1.1)
    plt.show()
    return

def apply_arnoldi_and_plot(Matrix, decimal=1):
    L, V = eigensolvers.apply_arnoldi_iteration(Matrix)

    print("Eigenvalue")
    print(round_complex_matrix(L, decimal))
    plot_complex_on_unit_circle(L)

def get_zero_norm_ratio(V):
    N = V.shape[1]
    n = N
    for k in range(N):
        if np.linalg.norm(V[:,k]) < C.POWER_EPS:
            n -= 1
    return n/N