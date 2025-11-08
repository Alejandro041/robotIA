import math, heapq, itertools


def heuristic(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])


def neighbors(p, w, h):
    x, y = p
    for dx, dy in itertools.product([-1, 0, 1], repeat=2):
        if dx == 0 and dy == 0:
            continue
        nx, ny = x + dx, y + dy
        if 0 <= nx < w and 0 <= ny < h:
            yield (nx, ny)


def collision_cell(cell, world):
    x, y = cell
    return world[y, x] == 1


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
        for nb in neighbors(current, w, h):
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


def path_to_continuous(path):
    return [(p[0] + 0.5, p[1] + 0.5) for p in path]
