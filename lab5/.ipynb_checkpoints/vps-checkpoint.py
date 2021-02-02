# vanishing points
import vp_detection as vp 

import utils as h

def estimate_vps(img):
    # Find the vanishing points of the image, through Xiaohulu methoed
    length_thresh = 60                      # Minimum length of the line in pixels
    seed = None                             # Or specify whatever ID you want (integer) Ex: 1337
    vpd = vp.VPDetection(length_thresh=length_thresh, seed=seed)
    vps = vpd.find_vps(img)

    if h.debug >= 0:
        print ("  Vanishing points found")
    if h.debug > 1:
        print ("      vps coordinates:\n", vpd.vps_2D )
    if h.debug > 2: 
        print ("      length threshold:", length_thresh)
        print ("      principal point:", vpd.principal_point)
        print ("      focal length:", vpd.focal_length)
        print ("      seed:", seed)

    return vpd.vps_2D

# deprecated
def estimate_vps2(img):
    edgelets1 = vp2.compute_edgelets(img)

    vp1 = vp2.ransac_vanishing_point(edgelets1, num_ransac_iter=2000, threshold_inlier=5)
    vp1 = vp2.reestimate_model(vp1, edgelets1, threshold_reestimate=5)

    edgelets2 = vp2.remove_inliers(vp1, edgelets1, 10)
    vp2nd = vp2.ransac_vanishing_point(edgelets2, num_ransac_iter=2000, threshold_inlier=5)
    vp2nd = vp2.reestimate_model(vp2nd, edgelets2, threshold_reestimate=5)

    vps = np.vstack((vp1, vp2nd))

    return vps.T
