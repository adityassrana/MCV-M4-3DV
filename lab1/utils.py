import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Tuple

def warp(H,p):
    """
    Warp a point based on a given Homography matrix
    
    Returns:
        Tuple of points
    """
    x1,x2 = p
    x1, x2, x3 = H @ np.array([x1,x2,1])
    if x3 != 0:
        return x1/x3, x2/x3
    return x1, x2


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

def plot_img(img, do_not_use=[0], title=None):
    plt.figure(do_not_use[0])
    do_not_use[0] += 1
    plt.imshow(img)
    if title is not None:
        plt.title(title)

def get_angle_lines(l, m):
    '''
    Computes the angle between lines l and m, both defined Projective Geometry way
    '''
    l /= l[2]
    m /= m[2]
    omega_inf = np.eye(3)
    omega_inf[2,2] = 0

    orth_l = omega_inf@l
    orth_m = omega_inf@m
    angle = np.arccos(orth_l.dot(m)/np.sqrt(orth_l.dot(orth_l)*orth_m.dot(orth_m)))
    return angle*180/np.pi

    
def get_offset(H, I):
    h,w,c = I.shape
    tl = warp(H,(0,0)) # top left
    tr = warp(H,(w,0)) # top right
    bl = warp(H,(0,h)) # bottom left
    br = warp(H,(w,h)) # bottom right
    corners = np.array([tl,tr,bl,br])

    #get min and max coordinates in the new space
    min_x = np.ceil(corners.min(axis=0)[0])
    min_y = np.ceil(corners.min(axis=0)[1])

    return min_x, min_y

def transform_lines(H, I, lines):
    """
    going for a non-efficient implementation
    to make sure lines are normalized to
    avoid numeric underflow
    """
    offset = get_offset(H, I)
    tx = -offset[0]
    ty = -offset[1]
    trans = np.array([1, 0, tx, 0, 1, ty, 0, 0, 1]).reshape(3,3)

    Hinvt =  np.transpose(np.linalg.inv(trans@H))
    lines_t = []
    for line in lines:
        line_t = Hinvt@line
        line_t = line_t/line_t[2]
        lines_t.append(line_t)
    return lines_t
    #return [( Hinvt@l for l in lines]


def crop(I):
    '''
    Crops the image to remove black blocks generated as a result of the transformations
    '''
    ys, xs = np.where((I[:,:,0] != 0) & (I[:,:,1] != 0) & (I[:,:,2] != 0))
    x0, y0, x, y = np.min(xs), np.min(ys), np.max(xs), np.max(ys)
    return I[y0:y, x0:x]