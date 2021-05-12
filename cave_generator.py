import cv2
import matplotlib.pyplot as plt
import numpy as np


def generate_cave(rows, cols, start_pos, end_pos, cave_value, space_value):
    # directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, 1), (1, 1), (1, -1), (-1, -1)]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    digger = start_pos
    goal = end_pos
    cave = np.full((rows, cols), cave_value)

    # initialize values
    cave[digger] = 0
    step_rng = rows // 2
    max_step = np.random.randint(step_rng)
    step = 0
    direction = directions[np.random.randint(len(directions))]

    while True:
        if step >= max_step:
            direction = directions[np.random.randint(len(directions))]
            max_step = np.random.randint(step_rng)
            step = 0
        move_y, move_x = direction
        digger_y, digger_x = digger
        new_y = min(rows - 2, max(1, move_y + digger_y))
        new_x = min(cols - 2, max(1, move_x + digger_x))
        digger = new_y, new_x
        cave[new_y - 1:new_y + 2, new_x - 1:new_x + 2] = space_value
        step += 1
        if digger == goal:
            break
    return cave
