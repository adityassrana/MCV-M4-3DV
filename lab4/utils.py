import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D


def line_draw(line, canv, size):
    def get_y(t):
        return -(line[0] * t + line[2]) / line[1]

    def get_x(t):
        return -(line[1] * t + line[2]) / line[0]

    w, h = size

    if line[0] != 0 and abs(get_x(0) - get_x(w)) < w:
        beg = (get_x(0), 0)
        end = (get_x(h), h)
    else:
        beg = (0, get_y(0))
        end = (w, get_y(w))
    canv.line([beg, end], width=4)


def plot_img(img, do_not_use=[0]):
    plt.figure(do_not_use[0])
    do_not_use[0] += 1
    plt.imshow(img)

def optical_center(P):
    u, s, vh = np.linalg.svd(P)
    o = vh[:,-1]
    o = o[:3] / o[3]
   
    return o

def view_direction(P, x):
    v,resid,rank,s = np.linalg.lstsq(P[:,:3], x, rcond=None)
    
    return v
    
def plot_points(points, texture): 
    # Creating figure
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
 
    # Creating plot
    ax.scatter3D(points[:,0],points[:,1],points[:,2], c = texture)
 
    # show plot
    plt.show()
    
def plot_lines(points): 
    # Creating figure
    fig = plt.figure(figsize = (10, 7))
    ax = plt.axes(projection ="3d")
    
    
    #ax = fig.add_subplot(111, projection='3d')

    for l in points:
        ax.plot(xs=[l[0][0], l[1][0]], ys=[l[0][1], l[1][1]],zs=[l[0][2], l[1][2]])
    plt.show()
    #Axes3D.plot()

def plot_camera(P, w, h, scale):

    o = optical_center(P);
    p1 = o + view_direction(P, np.array([0, 0, 1])) * scale;
    p2 = o + view_direction(P, np.array([w, 0, 1])) * scale;
    p3 = o + view_direction(P, np.array([w, h, 1])) * scale;
    p4 = o + view_direction(P, np.array([0, h, 1])) * scale;
    
    points = np.array([[o,p1]])
    points = np.vstack ((points, np.array([[o,p2]])))
    points = np.vstack ((points, np.array([[o,p3]])))
    points = np.vstack ((points, np.array([[o,p4]])))
    #points = np.vstack ((points, np.array([[o,(p1+p2)/2]])))
    points = np.vstack ((points, np.array([[p1,p2]])))
    points = np.vstack ((points, np.array([[p2,p3]])))
    points = np.vstack ((points, np.array([[p3,p4]])))
    points = np.vstack ((points, np.array([[p4,p1]])))
            
    plot_lines(points);