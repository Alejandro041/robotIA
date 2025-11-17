import math, heapq, itertools
from config import ROBOT_R

CLEARANCE = max(0, int(math.ceil(ROBOT_R)))
SMOOTH_SKIP = 6


def heuristic(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def neighbors(p, world):
    x, y = p
    h, w = world.shape
    for dx, dy in itertools.product([-1, 0, 1], repeat=2):
        if dx == 0 and dy == 0:
            continue
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            if dx != 0 and dy != 0:
                side_a = (x + dx, y)
                side_b = (x, y + dy)
                if world[side_a[1], side_a[0]] == 1 or world[side_b[1], side_b[0]] == 1:
                    continue
            yield (nx, ny)


def collision_cell(cell, world):
    x, y = cell
    if world[y, x] == 1:
        return True
    if CLEARANCE == 0:
        return False
    h, w = world.shape
    for yy in range(max(0, y - CLEARANCE), min(h, y + CLEARANCE + 1)):
        for xx in range(max(0, x - CLEARANCE), min(w, x + CLEARANCE + 1)):
            if world[yy, xx] == 1:
                return True
    return False


def a_star(world, start, goal):
    w, h = world.shape[1], world.shape[0]
    if collision_cell(start, world) or collision_cell(goal, world):
        return None

    open_set = []
    heapq.heappush(open_set, (0 + heuristic(start, goal), 0, start))
    came_from = {}
    g = {start: 0}

    while open_set:
        _, gcost, current = heapq.heappop(open_set)
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path
        for nb in neighbors(current, world):
            if collision_cell(nb, world):
                continue
            step_cost = math.hypot(nb[0]-current[0], nb[1]-current[1])
            tentative = gcost + step_cost
            if nb not in g or tentative < g[nb]:
                g[nb] = tentative
                came_from[nb] = current
                f = tentative + heuristic(nb, goal)
                heapq.heappush(open_set, (f, tentative, nb))
    return None


def path_to_continuous(path, spacing=0.5):
    if not path:
        return []
    pts = []
    prev = (path[0][0] + 0.5, path[0][1] + 0.5)
    pts.append(prev)
    for cell in path[1:]:
        target = (cell[0] + 0.5, cell[1] + 0.5)
        dx = target[0] - prev[0]
        dy = target[1] - prev[1]
        dist = math.hypot(dx, dy)
        steps = max(1, int(math.ceil(dist / max(spacing, 1e-3))))
        for i in range(1, steps + 1):
            t = i / steps
            pts.append((prev[0] + dx * t, prev[1] + dy * t))
        prev = target
    return pts


def _line_clear_cells(world, a, b):
    x0, y0 = a
    x1, y1 = b
    x0 = min(max(int(x0), 0), world.shape[1] - 1)
    y0 = min(max(int(y0), 0), world.shape[0] - 1)
    x1 = min(max(int(x1), 0), world.shape[1] - 1)
    y1 = min(max(int(y1), 0), world.shape[0] - 1)
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    while True:
        if world[y0, x0] == 1:
            return False
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
    return True


def smooth_path_cells(world, path):
    if not path:
        return []
    if len(path) <= 2:
        return path[:]
    smooth = [path[0]]
    i = 0
    n = len(path)
    while i < n - 1:
        j = i + 1
        while (
            j < n
            and (j - i) <= SMOOTH_SKIP
            and _line_clear_cells(world, path[i], path[j])
        ):
            j += 1
        next_i = j - 1
        smooth.append(path[next_i])
        i = next_i
    return smooth
