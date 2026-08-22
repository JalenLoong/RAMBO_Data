"""Components shared by RAMBO's quadruped and biped tasks."""

from .camera import create_front_rgb_camera, make_front_rgb_camera_cfg
from .contact_generator import ContactGenerator

__all__ = ["ContactGenerator", "create_front_rgb_camera", "make_front_rgb_camera_cfg"]
