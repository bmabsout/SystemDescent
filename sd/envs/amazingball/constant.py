import numpy as np

constants = {
    'g': 10.0,
    'm': 1.0,
    'dt': 0.05,
    'max_rot_x': np.pi / 6,
    'max_rot_y': np.pi / 6,
    'max_ball_pos_x': 10.0,
    'max_ball_pos_y': 10.0,
    'pl_vel': np.pi / 6,
    'render_stripe_margin': 20,
    'render_tiltline_thickness': 15
}

def scale(v, domain, range):
    """ transform scalar v in from (x,y) to a value in (tar_x, tar_y) uniformly"""
    x, y = domain
    tar_x, tar_y = range
    return tar_x + (tar_y - tar_x) * (v - x) / (y - x)    