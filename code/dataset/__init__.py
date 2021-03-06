from .cars import Cars
from .cub import CUBirds
from .import utils
from .base import BaseDataset


_type = {
    'cars': Cars,
    'cub': CUBirds
}

def load(name, root, mode, transform = None, noise = None, severity = None):
    return _type[name](root = root, mode = mode, transform = transform, noise = noise, severity = severity)
    
