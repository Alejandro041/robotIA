import math
import numpy as np
from config import LIDAR_RAYS, LIDAR_MAX_RANGE


def cast_ray(world, origin, theta, max_range, step=0.2):
    x, y = origin
    dist = 0.0
    while dist < max_range:
        x += math.cos(theta) * step
        y += math.sin(theta) * step
        dist += step
        xi, yi = int(x), int(y)
        if xi < 0 or yi < 0 or yi >= world.shape[0] or xi >= world.shape[1]:
            return dist
        if world[yi, xi] == 1:
            return dist
    return max_range


def lidar_scan(world, pose, rays=LIDAR_RAYS, max_range=LIDAR_MAX_RANGE):
    x, y, yaw = pose
    angles = np.linspace(-math.pi, math.pi, rays, endpoint=False)
    dists = []
    for a in angles:
        d = cast_ray(world, (x, y), yaw + a, max_range)
        dists.append(d)
    return angles, np.array(dists)
