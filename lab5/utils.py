import sys
import cv2
import numpy as np

# render 2d/3d plots
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

# Keep global variables at minimum
# Used for debugging
    # -1: QUIET, don't print anything
    #  0: NORMAL, show steps performed
    #  1: INFO, show values for different methods
    #  2: VERBOSE, show relevant matrices of pipeline 
    #  3: INSANE, show all values of data structures
debug = 1

if debug > 2:
    np.set_printoptions(threshold=sys.maxsize)  # print full arrays

#debug_display = False
debug_display = True
normalise = True  # activate coordinate normalisation
opencv = True  # whether use opencv or matplot to display images
path_imgs = "path/to/lab5/Data/"

def read_image(n):
    # Read an image from file. This method assumes images are a numbered sequence
    # in a folder
    global path_imgs

    path = path_imgs + str(n).zfill(4) + '_s.png'
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("ERROR: Image", path, "not loaded")
        sys.exit()

    if debug >= 0:
        print("  Image", path, "loaded")

    return img

def read_image_colour(n):
    # Read an image from file. This method assumes images are a numbered sequence
    # in a folder
    global path_imgs

    path = path_imgs + str(n).zfill(4) + '_s.png'
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print("ERROR: Colour image", path, "not loaded")
        sys.exit()

    if debug >= 0:
        print("  Colour image", path, "loaded")

    return img

def read_sequence():
    img1 = cv2.imread('../datasets/castle_dense_large/urd/0000.png', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('../datasetsn/castle_dense_large/urd/0001.png', cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print("Images not loaded")
        sys.exit()

    if debug >= 0:
        print('Images read')

    return img1, img2


def draw_lines(img1, img2, lines, x1, x2):
    """ img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines """
    r, c = img1.shape
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, x1, x2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1.astype(int)), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2.astype(int)), 5, color, -1)

        #cv2.imshow('epipolar lines and matches at img1', img1)
        #cv2.imshow('epipolar lines and matches at img2', img2)
        ## ASCII(space) = 32
        #key = cv2.waitKey(0) & 0xFF 

    return img1, img2


def draw_matches(img1, img2, x1, x2):
    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for pt1, pt2 in zip(x1, x2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        img1 = cv2.circle(img1, tuple(pt1.astype(int)), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2.astype(int)), 5, color, -1)

    return img1, img2


def draw_matches_cv(img1, img2, x1, x2):
    kp1 = [cv2.KeyPoint(p[0], p[1], 1) for p in x1]
    kp2 = [cv2.KeyPoint(p[0], p[1], 1) for p in x2]
    matches = [cv2.DMatch(i, i, 0) for i in range(len(kp1))]
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, np.random.choice(matches, 100), None, flags=2)
    plt.imshow(img3), plt.show()


def display_epilines(img1, img2, x1, x2, F):
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    cv2.namedWindow('epipolar lines and matches at img1', cv2.WINDOW_NORMAL)
    cv2.namedWindow('epipolar lines and matches at img2', cv2.WINDOW_NORMAL)

    lines1 = cv2.computeCorrespondEpilines(x2.reshape(-1, 1, 2), 2, F)
    lines1 = lines1.reshape(-1, 3)
    img3, img4 = draw_lines(img1, img2, lines1, x1, x2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(x1.reshape(-1, 1, 2), 1, F)
    lines2 = lines2.reshape(-1, 3)
    img5, img6 = draw_lines(img2, img1, lines2, x2, x1)

    if opencv:
        cv2.imshow('epipolar lines and matches at img1', img3)
        cv2.imshow('epipolar lines and matches at img2', img5)
        # ASCII(q) = 113, ASCII(esc) = 27, ASCII(space) = 32
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 113 or key == 27:
                cv2.destroyWindow('epipolar lines and matches at img1')
                cv2.destroyWindow('epipolar lines and matches at img2')
                break
    else:
        plt.subplot(121), plt.imshow(img3)
        plt.subplot(122), plt.imshow(img5)
        plt.show()

    
def show_matches(img1, img2, x1, x2):
    # Draw matches between two images
    cv2.namedWindow('matches at img1', cv2.WINDOW_NORMAL)
    cv2.namedWindow('matches at img2', cv2.WINDOW_NORMAL)

    img3, img4 = draw_matches(img1, img2, x1, x2)

    if opencv:
        cv2.imshow('matches at img1', img3)
        cv2.imshow('matches at img2', img4)
        # ASCII(q) = 113, ASCII(esc) = 27, ASCII(space) = 32
        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == 113 or key == 27:
                cv2.destroyWindow('matches at img1')
                cv2.destroyWindow('matches at img2')
                break
    else:
        plt.subplot(121), plt.imshow(img3)
        plt.subplot(122), plt.imshow(img4)
        plt.show()


def display_3d_points(X, x, img):
    # Plot a 3d set of points
    x_img = (x[:,:2].astype(int))
    rgb_txt = (img[x_img[:,1], x_img[:,0]])/255

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(X[:, 0], X[:, 1], X[:, 2], c=rgb_txt)
    #ax.set_ylim([-750,250])
    #ax.set_xlim([-800,-400])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

def display_3d_points_2(v, c=None):
    # Plot a 3d set of points
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(v[:, 0], v[:, 1], v[:, 2], c=c)
    #ax.set_ylim([-750,250])
    #ax.set_xlim([-800,-400])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.show()

