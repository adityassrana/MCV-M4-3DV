import numpy as np
import scipy
import reconstruction as rc
import maths as mth
import fundamental as fd


def estimate_aff_hom(cams, vps):
    """"
    compute homography for 
    affine rectification from vanishing points
    """
    vanish_P = rc.estimate_3d_points(cams[0], cams[1], vps[0].T, vps[1].T)
    plane = mth.nullspace(vanish_P.T)
    plane = plane / plane[3, :]
    Hpa = np.identity(4)
    Hpa[3] = p.T
    return Hpa

def estimate_euc_hom(cams, vps):
    # make points homogeneous
    vpsh = fd.make_homogeneous(vps)

    def get_eqn(a,b,c):
        return [a[0]*b[0], a[0]*b[1] + a[1]*b[0], 
                a[0]*b[2] + a[2]*b[0], a[1]*b[1], 
                a[1]*b[2] + a[2]*b[1], a[2]*b[2]]

    u, v, z = vpsh
    
    Eqs = np.array([get_eqn(u,v,z),get_eqn(u,z,v), get_eqn(v,z,u),[0, 1, 0, 0, 0, 0],[1, 0, 0, -1, 0, 0]])

    w_v = mth.nullspace(Eqs)
    w_v = np.squeeze(w_v)
    
    C = np.array([[w_v[0], w_v[1], w_v[2]],
                  [w_v[1], w_v[3], w_v[4]],
                  [w_v[2], w_v[4], w_v[5]]])
    M = cams[:, :3]
    A = scipy.linalg.cholesky(np.linalg.inv(M.T@C@M), lower=False)
    euc_hom = np.r_[np.c_[np.linalg.inv(A), np.zeros(3)],np.array([[0, 0, 0, 1]])]
    return euc_hom