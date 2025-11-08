import numpy as np
from config import START, GOAL


def make_world(w, h, density, keep_clear=(START, GOAL), min_clear_radius=3):
    world = np.zeros((h, w), dtype=np.uint8)
    # Estanterías tipo pasillos
    for x in range(6, w-6, 6):
        band_top = 5 + (x % 9)
        for y in range(h):
            if (y + band_top) % 8 in (0, 1):
                world[y, x] = 1
    # Obstáculos aleatorios
    n_rand = int(w*h*density*0.4)
    if n_rand > 0:
        rr = np.random.choice(w*h, size=n_rand, replace=False)
        for idx in rr:
            y, x = divmod(idx, w)
            world[y, x] = 1
    # Limpiar alrededor de puntos clave
    for p in keep_clear:
        y0, x0 = p[1], p[0]
        for yy in range(max(0, y0-min_clear_radius), min(h, y0+min_clear_radius+1)):
            for xx in range(max(0, x0-min_clear_radius), min(w, x0+min_clear_radius+1)):
                world[yy, xx] = 0
    return world


def inject_dynamic_obstacle(world, near_cell, radius=2):
    h, w = world.shape
    cx, cy = near_cell
    for yy in range(max(0, cy-radius), min(h, cy+radius+1)):
        for xx in range(max(0, cx-radius), min(w, cx+radius+1)):
            if (xx - cx)**2 + (yy - cy)**2 <= radius**2:
                world[yy, xx] = 1
