import os

def get_urdf_path(urdf_name):
    return os.path.join(os.path.dirname(__file__), urdf_name)