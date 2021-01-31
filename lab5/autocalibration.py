import numpy as np
import scipy
import reconstruction as rc
import maths as mth
import fundamental as fd


def estimate_aff_hom(cams, vps):
    # compute 3D Vanishing points
    V = rc.estimate_3d_points(cams[0], cams[1], vps[0].T, vps[1].T)

    # compute plane at infinity
    p = mth.nullspace(V.T)
    p = p / p[3, :]

    # compute homography
    # np.r_ and np.c_ are convenience functions
    # for building rows and columns from existing arrays
    aff_hom = np.r_[
        np.c_[np.eye(3), np.zeros(3)],
        p.T
    ]
    return aff_hom

def estimate_euc_hom(cams, vps):
    # make points homogeneous
    vpsh = fd.make_homogeneous(vps)

    # build A
    u, v, z = vpsh
    A = np.array([[u[0]*v[0], u[0]*v[1] + u[1]*v[0], u[0]*v[2] + u[2]*v[0], u[1]*v[1], u[1]*v[2] + u[2]*v[1], u[2]*v[2]],
                  [u[0]*z[0], u[0]*z[1] + u[1]*z[0], u[0]*z[2] + u[2]*z[0], u[1]*z[1], u[1]*z[2] + u[2]*z[1], u[2]*z[2]],
                  [v[0]*z[0], v[0]*z[1] + v[1]*z[0], v[0]*z[2] + v[2]*z[0], v[1]*z[1], v[1]*z[2] + v[2]*z[1], v[2]*z[2]],
                  [0, 1, 0, 0, 0, 0],
                  [1, 0, 0, -1, 0, 0]])

    # find w_v
    w_v = mth.nullspace(A)
    w_v = np.squeeze(w_v)

    # build w
    w = np.array([[w_v[0], w_v[1], w_v[2]],
                  [w_v[1], w_v[3], w_v[4]],
                  [w_v[2], w_v[4], w_v[5]]])

    # obtain A
    M = cams[:, :3]
    A = scipy.linalg.cholesky(np.linalg.inv(M.T@w@M), lower=False)

    # build euc_hom
    euc_hom = np.r_[
        np.c_[np.linalg.inv(A), np.zeros(3)],
        np.array([[0, 0, 0, 1]])
    ]

    return euc_hom