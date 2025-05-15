from scipy.stats import qmc
import numpy as np



def generate_lhs_points_with_holes(
    domain_bounds,                # [(x_min, x_max), (y_min, y_max)]
    holes,                        # list of (center_x, center_y, radius)
    n_points,                      # total number of valid points desired
    sampler
):


    # Oversample and filter
    factor = 1.5  # oversampling factor
    while True:
        m = int(n_points * factor)
        u = sampler.random(m)  # [m, 2] in unit cube
        pts = qmc.scale(u, [b[0] for b in domain_bounds], [b[1] for b in domain_bounds])
        
        # Filter out points inside any hole
        mask = np.ones(len(pts), dtype=bool)
        for cx, cy, r in holes:
            dist2 = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
            mask &= dist2 > r ** 2
        
        valid_pts = pts[mask]
        if len(valid_pts) >= n_points:
            return valid_pts[:n_points]
        else:
            # Increase oversampling factor and try again
            factor *= 1.5


def generate_points(n_d, n_b, domain_bounds, holes, seed):
    
    x_min, x_max = domain_bounds[0]
    y_min, y_max = domain_bounds[1]

    sampler_1d = qmc.LatinHypercube(d=1, seed=seed)
    sampler_2d = qmc.LatinHypercube(d=2, seed=seed)

    dom = generate_lhs_points_with_holes(domain_bounds, holes, n_d, sampler_2d)
    dom_x = dom[:, 0]
    dom_y = dom[:, 1]

    points = sampler_1d.random(n_b)

    left_x = np.ones(n_b) * x_min
    left_y = qmc.scale(points, y_min, y_max).flatten()

    top_x = qmc.scale(points, x_min, x_max).flatten()
    top_y = np.ones(n_b) * y_max

    right_x = np.ones(n_b) * x_max
    right_y = qmc.scale(points, y_min, y_max).flatten()

    down_x = qmc.scale(points, x_min, x_max).flatten()
    down_y = np.ones(n_b) * y_min

    cx, cy, r = holes[0]

    theta = qmc.scale(points, 0, 2*np.pi).flatten()
    hole_x = cx + r*np.cos(theta)
    hole_y = cy + r*np.sin(theta)


    dom_x = np.hstack((dom_x, hole_x, left_x, top_x, right_x, down_x))
    dom_y = np.hstack((dom_y, hole_y, left_y, top_y, right_y, down_y))

    mask_left = np.equal(dom_x, x_min)
    mask_top = np.equal(dom_y, y_max)
    mask_right = np.equal(dom_x, x_max)
    mask_down = np.equal(dom_y, y_min)
    rad1 = np.sqrt((dom_x - cx)**2 + (dom_y - cy)**2)
    mask_hole = np.isclose(rad1, r)
    masks = {'left': mask_left, 'top': mask_top, 'right': mask_right, 'down': mask_down, 'hole': mask_hole}

    return dom_x, dom_y, masks