import pyglet
from gym.envs.classic_control import rendering

class Image(rendering.Geom):
    """
    Our own implementation of gym.envs.classic_control.rendering.Image 
    to render an image loaded from an ndarray instead of a file.
    """
    def __init__(self, arr):
        rendering.Geom.__init__(self)

        h, w, channels = arr.shape
        assert channels == 4, 'Image must be in RGBA format'
        pitch = -w * channels
        self.img = pyglet.image.ImageData(w, h, 'RGBA', arr.tobytes(), pitch=pitch)

    def render1(self):
        self.img.blit(0,0)
