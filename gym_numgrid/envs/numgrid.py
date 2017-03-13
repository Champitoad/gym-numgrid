import numpy as np

import gym
from gym import spaces
from gym.envs.classic_control import rendering

from gym_numgrid.utils import mnist_loader
from gym_numgrid.envs.rendering import Image

class NumGrid(gym.Env):
    """
    An environment consisting of a grid of hand-written digits images
    loaded from the MNIST training database.
    It also holds a cursor respresenting the agent's local view on the world.
    """
    metadata = {
            'render.modes': ['human', 'rgb_array'],
            'video.frames_per_second': 60,
            'configure.required': True
            }

    def __init__(self, size=(5,5), cursor_size=(10,10),\
            digits=set(range(10)),\
            mnist_images_path='train-images-idx3-ubyte.gz',\
            mnist_labels_path='train-labels-idx1-ubyte.gz'):
        """
        size -- dimensions of the grid in number of images as a (width, height) tuple
        cursor_size -- dimensions of the cursor in pixels as a (width, height) tuple

        digits -- set of digits we want to load from MNIST
        mnist_images_path -- path to the MNIST images file, in IDX gzipped format
        mnist_labels_path -- path to the MNIST labels file, in IDX gzipped format
        """
        self.size = np.array(size)
        self.cursor_size = np.array(cursor_size)
        self.cursor_pos = np.zeros(2)

        self.labels = mnist_loader.load_idx_data(mnist_labels_path)
        i = 0
        labels_i = []
        while len(labels_i) < np.prod(size):
            if self.labels[i] in digits:
                labels_i.append(i)
            i += 1
        self.images = mnist_loader.load_idx_data(mnist_images_path, (labels_i[-1] + 1,))

        self.labels = self.labels[labels_i].reshape(size[::-1] + self.labels.shape[1:])
        self.images = self.images[labels_i].reshape(size[::-1] + self.images.shape[1:])

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

        # An observation is the cursor view on the world, therefore the observation
        # space is equivalent to the cursor position space (cursor size being fixed)

        self.observation_space = self.position_space

        spaces.prng.np_random.seed() # For correct random reset of the cursor position
        self.viewer = None

    def _step(self, action):
        digit, pos = action
        reward = 0
        done = False
        info = {'out_of_bounds': False, 'digit': self.current_digit}

        if digit < 10:
            if digit != info['digit']:
                reward -= 3
                if not self.position_space.contains(pos):
                    info['out_of_bounds'] = True
                else:
                    self.cursor_pos = np.array(pos)
            else:
                reward += 3
                self.cursor_pos = np.array(self.position_space.sample())

        info['cursor'] = self.cursor

        self.steps += 1
        if self.steps >= self.num_steps:
            done = True

        return self.cursor_pos, reward, done, info

    def _reset(self):
        self.steps = 0
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

    def _configure(self, num_steps=100, render_scale=2, draw_grid=False):
        """
        num_steps -- number of steps to achieve in an episode

        render_scale -- scale to apply to the viewer's rendering of the world
        draw_grid -- whether the viewer should draw a grid delimiting digit images
        """
        self.num_steps = num_steps
        self.steps = 0 # Number of steps done in the current episode

        self.render_scale = render_scale
        self.draw_grid = draw_grid

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
