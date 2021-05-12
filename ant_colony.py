import functools
import random
from collections import namedtuple

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors

import cave_generator


class ColorMap:
    def __init__(self):
        rgb = namedtuple('rgb', ['name', 'r', 'g', 'b'])
        self.green_alpha = self.create_colormap(rgb('green_alpha', 0, 200, 80))
        self.red_alpha = self.create_colormap(rgb('red_alpha', 200, 0, 0))

    def create_colormap(self, color):
        N = 256
        colormap = np.ones((N, 4))
        colormap[:, 0] = np.linspace(color.r / N, color.r / N, N)
        colormap[:, 1] = np.linspace(color.g / N, color.g / N, N)
        colormap[:, 2] = np.linspace(color.b / N, color.b / N, N)
        colormap[:, 3] = np.linspace(0.0, 1.0, N)
        return colors.LinearSegmentedColormap.from_list(color.name, colormap, N)


class WorldColorMap:
    def __init__(self):
        Category = namedtuple('Category', ['name', 'rgba', 'value'])
        self.empty = Category('empty', [0, 0, 0, 0], 0)
        self.ant_foraging = Category('ant', [120 / 255, 0, 0, 1], 1)
        self.ant_returning = Category('ant', [0, 100 / 255, 0, 1], 2)
        self.nest = Category('nest', [1, 0, 1, 1], 3)
        self.food = Category('food', [0, 1, 0, 1], 4)
        self.wall = Category('wall', [200 / 255, 100 / 255, 60 / 255, 1], 5)

        self.color_list = [self.empty.rgba, self.ant_foraging.rgba, self.ant_returning.rgba, self.nest.rgba,
                           self.food.rgba, self.wall.rgba]
        self.cmap = colors.ListedColormap(self.color_list)
        self.bounds = [self.empty.value, self.ant_foraging.value, self.ant_returning.value, self.nest.value,
                       self.food.value, self.wall.value, self.cmap.N]
        self.norm = colors.BoundaryNorm(self.bounds, self.cmap.N)


class Direction:
    UP = 'up'
    DOWN = 'down'
    LEFT = 'left'
    RIGHT = 'right'
    UP_LEFT = 'up_left'
    UP_RIGHT = 'up_right'
    DOWN_LEFT = 'down_left'
    DOWN_RIGHT = 'down_right'

    direction_names = [UP, DOWN, LEFT, RIGHT, UP_LEFT, UP_RIGHT, DOWN_LEFT, DOWN_RIGHT]

    direction_names_clockwise = [UP, UP_RIGHT, RIGHT, DOWN_RIGHT, DOWN, DOWN_LEFT, LEFT, UP_LEFT, UP]

    positions = {UP: (-1, 0),
                 DOWN: (1, 0),
                 LEFT: (0, -1),
                 RIGHT: (0, 1),
                 UP_LEFT: (-1, -1),
                 UP_RIGHT: (-1, 1),
                 DOWN_LEFT: (1, -1),
                 DOWN_RIGHT: (1, 1)}

    detection_direction_names = {UP: (UP, UP_LEFT, UP_RIGHT),
                                 DOWN: (DOWN, DOWN_LEFT, DOWN_RIGHT),
                                 LEFT: (LEFT, UP_LEFT, DOWN_LEFT),
                                 RIGHT: (RIGHT, UP_RIGHT, DOWN_RIGHT),
                                 UP_LEFT: (UP_LEFT, UP, LEFT),
                                 UP_RIGHT: (UP_RIGHT, UP, RIGHT),
                                 DOWN_LEFT: (DOWN_LEFT, DOWN, LEFT),
                                 DOWN_RIGHT: (DOWN_RIGHT, DOWN, RIGHT)}

    @classmethod
    @functools.lru_cache(maxsize=4096)
    def degrees_to_direction_name(cls, degrees):
        i = round(degrees / 45)
        return cls.direction_names_clockwise[i]

    @staticmethod
    @functools.lru_cache(maxsize=204800)
    def bearing(self_pos, other_pos):
        y_len = self_pos[0] - other_pos[0]
        x_len = other_pos[1] - self_pos[1]
        angle = np.degrees(np.arctan2(x_len, y_len))
        if angle < 0:
            angle += 360
        return angle

    @staticmethod
    @functools.lru_cache(maxsize=4096)
    def opposite_angle(angle):
        opposite = angle - 180
        if opposite < 0:
            return opposite + 360
        return opposite

    @staticmethod
    @functools.lru_cache(maxsize=4096)
    def turn_by_angle(self_angle, angle_to_turn):
        new_angle = self_angle + angle_to_turn
        if new_angle < 0:
            return new_angle + 360
        elif new_angle >= 360:
            return new_angle - 360
        return new_angle


class Ant:
    __slots__ = 'y_pos', 'x_pos', 'pos', 'degrees', 'task_state', 'lifespan', 'long_term_memory'

    def __init__(self, y_pos, x_pos, task_state):
        self.y_pos = None
        self.x_pos = None
        self.pos = None
        self.degrees = None
        self.task_state = task_state
        self.lifespan = 6000  # number of frames before dying
        # self.long_term_memory = {}  # memory for position visited and the number of times visited
        self.long_term_memory = set()  # memory for position visited and the number of times visited
        self.set_pos(y_pos, x_pos)

    def set_pos(self, y_pos, x_pos):
        new_pos = (y_pos, x_pos)
        if self.pos is None:
            self.degrees = np.random.randint(8) * 45
        else:
            self.degrees = Direction.bearing(self.pos, new_pos)
        self.y_pos = y_pos
        self.x_pos = x_pos
        self.pos = new_pos
        # memorize position when set
        self.long_term_memory.add(new_pos)
        # if self.long_term_memory.get(self.pos) is None:
        #     self.long_term_memory[self.pos] = 1
        # else:
        #     self.long_term_memory[self.pos] += 1

    def pos_curiosity(self, pos):
        # curiosity is used to help the ant explore new areas
        # more frequently visited positions will be penalized, while never seen position is prioritized
        # times_visited = self.long_term_memory.get(pos)
        # if times_visited is None:
        #     return 2
        # else:
        #     return 10 / (times_visited + 10)
        if pos in self.long_term_memory:
            return 0.5
        else:
            return 2

    def clear_memory(self):
        # self.long_term_memory = {self.pos: 1}
        self.long_term_memory = {self.pos}

    def decrement_lifespan(self):
        self.lifespan -= 1

    def is_dead(self):
        return self.lifespan < 1

    def turn_adjacent(self):
        num = np.random.randint(2)
        if num == 0:
            self.degrees = Direction.turn_by_angle(self.degrees, 45)
        else:
            self.degrees = Direction.turn_by_angle(self.degrees, -45)


class Pheromone:
    __slots__ = 'y_pos', 'x_pos', 'pos', 'strength'

    def __init__(self, y_pos, x_pos, strength):
        self.y_pos = y_pos
        self.x_pos = x_pos
        self.pos = (y_pos, x_pos)
        self.strength = strength


class AntColonySimulator:
    # NOTE: numpy uses matrix coordinate system, so all coordinates are in the (y, x) format instead of Cartesian (x, y)
    grid_columns = 100
    grid_rows = 100
    grid_shape = (grid_rows, grid_columns)
    max_pheromone_strength = 100
    pheromone_evaporate_rate = 0.1
    pheromone_deposit_rate = 10
    pheromone_disperse_ksize = (3, 3)
    pheromone_disperse_sigma = 0.25
    ant_spawn_probability = 0.2
    max_ants_count = 250

    nest_pos = (1, 1)
    food_pos = (98, 98)

    colormap = ColorMap()
    world_colormap = WorldColorMap()
    cave = np.zeros(grid_shape)
    world_grid = np.zeros(grid_shape)
    foraging_pheromone = np.ones(grid_shape)
    returning_pheromone = np.ones(grid_shape)

    rng_upper_bound = np.floor(ant_spawn_probability ** -1)

    ants_list = []
    foraging_state = 0
    returning_state = 1

    food_count = 0

    fig, ax = plt.subplots()

    def init_world_grid(self):
        self.cave = cave_generator.generate_cave(self.grid_rows, self.grid_columns, self.nest_pos, self.food_pos,
                                                 self.world_colormap.wall.value, self.world_colormap.empty.value)
        self.cave[self.nest_pos] = self.world_colormap.nest.value
        self.cave[self.food_pos] = self.world_colormap.food.value
        self.reset_world_grid()

    def reset_world_grid(self):
        self.world_grid = self.cave.copy()

    def spawn_ant(self):
        if not self.ants_list:  # if no ants, we spawn one
            self.ants_list.append(Ant(*self.nest_pos, self.foraging_state))
            return
        if len(self.ants_list) >= self.max_ants_count:  # don't spawn ants if max count reached
            return
        # else spawn ants based on RNG
        num = np.random.randint(self.rng_upper_bound)
        if num == 0:
            self.ants_list.append(Ant(*self.nest_pos, self.foraging_state))

    @functools.lru_cache(maxsize=204800)
    def get_detection_positions(self, y_pos, x_pos, direction_name):
        detection_names = Direction.detection_direction_names[direction_name]
        detection_positions = []
        for name in detection_names:
            y_direction, x_direction = Direction.positions[name]
            detection_positions.append((y_pos + y_direction, x_pos + x_direction))
        return detection_positions

    def detect_pheromone_in_direction(self, ant):
        # get the detection positions for ant's current direction
        detection_positions = self.get_detection_positions(*ant.pos, Direction.degrees_to_direction_name(ant.degrees))
        detection_positions = self.remove_invalid_positions(detection_positions)
        task_state = ant.task_state
        if task_state == self.foraging_state:
            return [Pheromone(*pos, self.foraging_pheromone[pos]) for pos in detection_positions]
        elif task_state == self.returning_state:
            return [Pheromone(*pos, self.returning_pheromone[pos]) for pos in detection_positions]

    @functools.lru_cache(maxsize=102400)
    def euclidean_distance(self, pos_1, pos_2):
        pos_1_y, pos_1_x = pos_1
        pos_2_y, pos_2_x = pos_2
        return (pos_1_y - pos_2_y) ** 2 + (pos_1_x - pos_2_x) ** 2
        # return np.sqrt((pos_1_y - pos_2_y) ** 2 + (pos_1_x - pos_2_x) ** 2)

    @functools.lru_cache(maxsize=4096)
    def direction_sense(self, value):
        if value >= 0:
            return 2
        else:  # smaller than 0, negative values
            return 0.5
        # return 0.5 / (1 + np.exp(-5 * value)) + 0.5

    def move_ant(self, ant):
        detected_pheromone = self.detect_pheromone_in_direction(ant)
        if not detected_pheromone:  # if no move available, turn to adjacent tile
            ant.turn_adjacent()
            return
        if len(detected_pheromone) == 1:  # don't do random sampling if only one sample
            ant.set_pos(*detected_pheromone[0].pos)  # update ant location to the only pheromone
            self.deposit_pheromone(ant)
            return
        task_state = ant.task_state
        ant_distance = self.euclidean_distance(ant.pos, self.nest_pos)
        # multiply pheromone strength with sense of direction
        if task_state == self.foraging_state:
            for pheromone in detected_pheromone:
                pheromone_pos = pheromone.pos
                direction = 1  # move randomly if no pheromone
                if pheromone.strength > 1:  # follow pheromone and move away from nest if exist
                    pheromone_distance = self.euclidean_distance(pheromone_pos, self.nest_pos)
                    direction = pheromone_distance - ant_distance
                pheromone.strength *= self.direction_sense(direction) * ant.pos_curiosity(pheromone_pos)
        elif task_state == self.returning_state:
            for pheromone in detected_pheromone:
                pheromone_pos = pheromone.pos
                direction = 1  # move randomly if no pheromone
                if pheromone.strength > 1:  # follow pheromone and move away from nest if exist
                    pheromone_distance = self.euclidean_distance(pheromone_pos, self.nest_pos)
                    direction = ant_distance - pheromone_distance  # move towards nest
                pheromone.strength *= self.direction_sense(direction) * ant.pos_curiosity(pheromone_pos)
        selected_pheromone = self.weighted_random_choice(detected_pheromone)
        ant.set_pos(*selected_pheromone.pos)  # update ant location to the selected pheromone
        self.deposit_pheromone(ant)

    def weighted_random_choice(self, detected_pheromone):
        weights = [pheromone.strength for pheromone in detected_pheromone]
        return detected_pheromone[self.weighted_choice_index(weights)]

    def weighted_choice_index(self, weights):
        rnd = random.random() * sum(weights)
        for i, w in enumerate(weights):
            rnd -= w
            if rnd < 0:
                return i

    def remove_invalid_positions(self, detection_positions):
        return [pos for pos in detection_positions if not self.is_colliding(*pos)]

    def is_colliding(self, y_pos, x_pos):
        if self.is_out_of_bounds(y_pos, x_pos):
            return True
        world_value = self.world_grid[y_pos, x_pos]
        return world_value == self.world_colormap.ant_foraging.value or \
               world_value == self.world_colormap.ant_returning.value or \
               world_value == self.world_colormap.wall.value
        # return world_value == self.world_colormap.wall.value

    @functools.lru_cache(maxsize=102400)
    def is_out_of_bounds(self, y_pos, x_pos):
        return not (0 <= y_pos < self.grid_rows and 0 <= x_pos < self.grid_columns)

    def evaporate_pheromone(self):
        # reduce pheromone strength by defined rate, but not lower than 1
        self.foraging_pheromone = np.maximum(self.foraging_pheromone - self.pheromone_evaporate_rate, 1)
        self.returning_pheromone = np.maximum(self.returning_pheromone - self.pheromone_evaporate_rate, 1)

    def disperse_pheromone(self):
        self.foraging_pheromone = cv2.GaussianBlur(self.foraging_pheromone,
                                                   self.pheromone_disperse_ksize,
                                                   self.pheromone_disperse_sigma)
        self.returning_pheromone = cv2.GaussianBlur(self.returning_pheromone,
                                                    self.pheromone_disperse_ksize,
                                                    self.pheromone_disperse_sigma)

    def deposit_pheromone(self, ant):
        ant_pos, task_state = ant.pos, ant.task_state
        if task_state == self.foraging_state:
            self.returning_pheromone[ant_pos] = np.minimum(self.max_pheromone_strength,
                                                           self.returning_pheromone[ant_pos]
                                                           + self.pheromone_deposit_rate)
        elif task_state == self.returning_state:
            self.foraging_pheromone[ant_pos] = np.minimum(self.max_pheromone_strength,
                                                          self.foraging_pheromone[ant_pos]
                                                          + self.pheromone_deposit_rate)

    def update_ants(self):
        self.reset_world_grid()
        updated_ants_list = []
        for ant in self.ants_list:
            y_pos, x_pos, task_state = ant.y_pos, ant.x_pos, ant.task_state
            self.move_ant(ant)
            ant.decrement_lifespan()
            if self.is_ant_found_food(y_pos, x_pos, task_state):
                ant.task_state = self.returning_state
                ant.clear_memory()
                ant.degrees = Direction.opposite_angle(ant.degrees)
            if self.is_ant_returned(y_pos, x_pos, task_state):
                self.food_count += 1
                print('food_count:', self.food_count)
            if not self.is_ant_returned(y_pos, x_pos, task_state) and not ant.is_dead():
                self.draw_ant(ant)
                updated_ants_list.append(ant)  # delete ants that returned, only keep ants that are still foraging
        self.ants_list = updated_ants_list

    def draw_ant(self, ant):
        ant_pos, task_state = ant.pos, ant.task_state
        if task_state == self.foraging_state:
            self.world_grid[ant_pos] = self.world_colormap.ant_foraging.value
        elif task_state == self.returning_state:
            self.world_grid[ant_pos] = self.world_colormap.ant_returning.value

    @functools.lru_cache(maxsize=102400)
    def is_ant_found_food(self, y_pos, x_pos, task_state):
        food_x_pos, food_y_pos = self.food_pos
        return y_pos == food_y_pos and x_pos == food_x_pos and task_state == self.foraging_state

    @functools.lru_cache(maxsize=102400)
    def is_ant_returned(self, y_pos, x_pos, task_state):
        nest_x_pos, nest_y_pos = self.nest_pos
        return y_pos == nest_y_pos and x_pos == nest_x_pos and task_state == self.returning_state

    def update(self):
        self.spawn_ant()
        self.disperse_pheromone()
        self.evaporate_pheromone()
        self.update_ants()

    def run(self):
        self.init_world_grid()
        self.ax.imshow(self.returning_pheromone, cmap=self.colormap.red_alpha, alpha=0.5)
        self.ax.imshow(self.foraging_pheromone, cmap=self.colormap.green_alpha, alpha=0.5)
        self.ax.imshow(self.world_grid, cmap=self.world_colormap.cmap, norm=self.world_colormap.norm)

        self.fig.canvas.draw()
        self.fig.show()

        frame = 0
        frame_skip = 200  # only visualise every N frames, to workaround graph drawing bottleneck
        # for i in range(20000):
        while True:
            self.update()
            if frame % 100 == 0:
                frame_skip = max(frame_skip - 1, 1)
            if frame % frame_skip == 0:
                im1 = self.ax.imshow(self.returning_pheromone, cmap=self.colormap.red_alpha, alpha=0.5)
                im2 = self.ax.imshow(self.foraging_pheromone, cmap=self.colormap.green_alpha, alpha=0.5)
                im3 = self.ax.imshow(self.world_grid, cmap=self.world_colormap.cmap, norm=self.world_colormap.norm)
                self.ax.draw_artist(self.ax.patch)
                self.ax.draw_artist(im1)
                self.ax.draw_artist(im2)
                self.ax.draw_artist(im3)
                self.fig.canvas.blit(self.ax.bbox)
                self.fig.canvas.flush_events()
            frame += 1


ant_colony_simulator = AntColonySimulator()
ant_colony_simulator.run()
