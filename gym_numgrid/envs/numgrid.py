import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering

import numpy as np
import pyglet

from gym_numgrid import mnist_loader

class NumGrid(gym.Env):
    """
    An environment consisting of a grid of hand-written digits images
    loaded from the MNIST training database.
    It also holds a cursor respresenting the agent's local view on the world.
    """
    metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 30,
            'configure.required': True
            }

    def __init__(self):
        self._seed()
        self.viewer = None

    def _step(self, action):
        digit, pos = action
        reward = 0
        done = False
        info = {'out_of_bounds': False, 'digit': self.current_digit}

        if digit < 10:
            if digit != info['digit']:
                reward -= 3
            else:
                reward += 3
                # done = True

        if not self.position_space.contains(pos):
            info['out_of_bounds'] = True
        else:
            self.cursor_pos = np.array(pos)
            info['cursor'] = self.cursor

        return self.cursor_pos, reward, done, info

    def _reset(self):
        self.cursor_pos = np.array(self.position_space.sample())
        return self.cursor_pos

    def _render(self, mode='human', close=False):
        if close:
            return

        scale = (self.render_scale,) * 2

        world_size = np.array(self.world.shape[::-1])
        screen_size = (world_size * scale).astype(int)

        if self.viewer is None:
            scaling = rendering.Transform(scale=scale)
            self.viewer = rendering.Viewer(*screen_size)

            world = np.array([(x,x,x,x) for x in self.world.flatten()])
            world = world.reshape(self.world.shape + (4,)) # RGBA
            world = Image(world)
            world.add_attr(scaling)
            self.viewer.add_geom(world)

            w, h = self.cursor_size
            vertices = (0,0), (w,0), (w,-h), (0,-h), (0,0)
            cursor_bounds = rendering.make_polyline(vertices)
            cursor_bounds.set_color(1, 0, 0)
            cursor_bounds.set_linewidth(2)
            self.cursor_trans = rendering.Transform()
            cursor_bounds.add_attr(scaling)
            cursor_bounds.add_attr(self.cursor_trans)
            self.viewer.add_geom(cursor_bounds)

            x, y = (self.cursor_size/2).astype(int)
            l1, l2 = ((x-1,-y-1),(x+1,-y+1)), ((x-1,-y+1),(x+1,-y-1))
            cursor_center = (rendering.Line(*l1), rendering.Line(*l2))
            for line in cursor_center:
                line.set_color(1, 0, 0)
                line.add_attr(scaling)
                line.add_attr(self.cursor_trans)
                self.viewer.add_geom(line)

            if self.draw_grid:
                W, H = world_size
                h, w = self.images.shape[:-3:-1]
                grid = [rendering.Line((i*w,H),(i*w,0)) for i in range(self.size[0])]
                grid += [rendering.Line((0,H-j*h),(W,H-j*h)) for j in range(self.size[1])]
                for line in grid:
                    line.set_color(0, 1, 0)
                    line.add_attr(scaling)
                    self.viewer.add_geom(line)

        px, py = self.cursor_pos * scale
        self.cursor_trans.set_translation(px, screen_size[1] - py)

        return self.viewer.render(return_rgb_array=(mode == 'rgb_array'))

    def _close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _configure(self, size=(10,10), cursor_size=(10,10), cursor_pos=(0,0), \
            mnist_images_path='train-images-idx3-ubyte.gz', \
            mnist_labels_path='train-labels-idx1-ubyte.gz', \
            render_scale=2, draw_grid=False):
        """
        size -- dimensions of the grid in number of images as a (width,height) tuple
        cursor_size -- dimensions of the cursor in pixels as a (width,height) tuple
        cursor_pos -- (x,y) initial position of the cursor, with top-left origin

        mnist_images_path -- path to the MNIST images file, in IDX gzipped format
        mnist_labels_path -- path to the MNIST labels file, in IDX gzipped format

        render_scale -- scale to apply to the viewer's rendering of the world
        draw_grid -- whether the viewer should draw a grid delimiting digit images
        """
        self.size = np.array(size)
        self.cursor_size = np.array(cursor_size)
        self.cursor_pos = np.array(cursor_pos)

        self.render_scale = render_scale
        self.draw_grid = draw_grid

        self.images = mnist_loader.load_idx_data(mnist_images_path, size[::-1])
        self.labels = mnist_loader.load_idx_data(mnist_labels_path, size[::-1])

        H, W, h, w = self.images.shape
        self.world = self.images.swapaxes(1,2).reshape(H*h, W*w)

        # An action consists of a guess at the digit currently under the cursor,
        # plus a new cursor position; the agent might not want to try a guess yet,
        # in which case it should use the value 10 to indicate that the prediction
        # must be ignored

        self.digit_space = spaces.Discrete(11)

        world_bounds = np.array(self.world.shape[::-1]) - 1 - self.cursor_size
        self.position_space = spaces.MultiDiscrete(np.stack([(0,0), world_bounds], 1))

        self.action_space = spaces.Tuple((self.digit_space, self.position_space))

        # A smaller action space for the position consisting of the 4 orthogonal directions;
        # one such action must be converted into a real action (a position),
        # typically using the cursor_move method

        self.direction_space = Direction()

        # An observation is the cursor view on the world, therefore the observation
        # space is equivalent to the cursor position space (cursor size being fixed)

        self.observation_space = self.position_space

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @property
    def current_digit(self):
        """
        Returns the digit currently under the cursor.
        """
        image_size = self.images.shape[:-3:-1]
        i = (self.cursor_center/image_size)[::-1].astype(int)
        return self.labels[tuple(i)]

    @property
    def cursor(self):
        """
        Returns the cursor view on the world.
        """
        x, y = self.cursor_pos
        w, h = self.cursor_size
        return self.world[y:y+h,x:x+w]

    @property
    def cursor_center(self):
        """
        Returns the cursor's center position.
        """
        return self.cursor_pos + (self.cursor_size/2).astype(int)

    def cursor_move(self, direction, distance=1):
        """
        Returns the cursor position if it were moved
        in the given direction on the given distance.

        direction -- value in the Direction space
        distance -- scalar integer value in pixels
        """
        return self.cursor_pos + np.array(direction) * distance

class Direction(gym.Space):
    def __init__(self):
        self.values = [(-1,0), (1,0), (0,-1), (0,1)]

    def sample(self):
        spaces.prng.np_random.shuffle(self.values)
        return self.values[0]

    def contains(self, x):
        return x in self.values

class Image(rendering.Geom):
    """
    Our own implementation of gym.envs.classic_control.rendering.Image 
    to render an image loaded from an ndarray instead of a file.
    """
    def __init__(self, arr):
        rendering.Geom.__init__(self)

        h, w, channels = arr.shape
        assert channels == 4, 'Image must be in RGBA format'
        arr = 255 - arr # Image is rendered in negative for some reason,
                        # so we get it back to its original state
        pitch = -w * channels
        self.img = pyglet.image.ImageData(w, h, 'RGBA', arr.tobytes(), pitch=pitch)

    def render1(self):
        self.img.blit(0,0)
