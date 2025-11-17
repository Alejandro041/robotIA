import math
from config import V_MAX, W_MAX, LOOKAHEAD_DIST, DT, ROBOT_R

COLLISION_RADIUS = max(0.35, ROBOT_R * 0.85)


def find_lookahead_point(path_xy, robot_xy, lookahead=LOOKAHEAD_DIST):
    for p in path_xy:
        if ((p[0]-robot_xy[0])**2 + (p[1]-robot_xy[1])**2)**0.5 >= lookahead:
            return p
    return path_xy[-1]


def unicycle_step(x, y, yaw, v_cmd, w_cmd, dt=DT):
    x = x + v_cmd * math.cos(yaw) * dt
    y = y + v_cmd * math.sin(yaw) * dt
    yaw = yaw + w_cmd * dt
    yaw = (yaw + math.pi) % (2*math.pi) - math.pi
    return x, y, yaw


def nearest_path_index(robot_xy, path_xy, start_idx=0):
    rx, ry = robot_xy
    best_i = start_idx
    best_d = 1e9
    for i in range(start_idx, len(path_xy)):
        d = (path_xy[i][0]-rx)**2 + (path_xy[i][1]-ry)**2
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def compute_controls(robot_state, path_xy, path_idx):
    x, y, yaw = robot_state
    # Punto de mira (pure pursuit)
    from_idx = max(0, path_idx)
    lookahead_pt = find_lookahead_point(path_xy[from_idx:], (x, y))
    dx, dy = lookahead_pt[0] - x, lookahead_pt[1] - y
    tgt_yaw = math.atan2(dy, dx)
    yaw_err = (tgt_yaw - yaw + math.pi) % (2*math.pi) - math.pi
    v_cmd = V_MAX * max(0.2, (1.0 - abs(yaw_err)/math.pi))
    w_cmd = max(-W_MAX, min(W_MAX, 2.5*yaw_err))
    return v_cmd, w_cmd


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


def _pose_hits(world, x, y, radius=COLLISION_RADIUS):
    h, w = world.shape
    rr = int(math.ceil(radius + 0.5))
    cx, cy = int(x), int(y)
    for yy in range(max(0, cy - rr), min(h, cy + rr + 2)):
        for xx in range(max(0, cx - rr), min(w, cx + rr + 2)):
            if world[yy, xx] != 1:
                continue
            x0, y0 = xx, yy
            x1, y1 = xx + 1.0, yy + 1.0
            nearest_x = _clamp(x, x0, x1)
            nearest_y = _clamp(y, y0, y1)
            dx = nearest_x - x
            dy = nearest_y - y
            if dx*dx + dy*dy <= radius*radius:
                return True
    return False


def segment_hits_obstacle(world, start_xy, end_xy):
    sx, sy = start_xy
    ex, ey = end_xy
    dist = math.hypot(ex - sx, ey - sy)
    steps = max(3, int(math.ceil(dist * 12)))
    for i in range(1, steps + 1):
        t = i / steps
        xi = sx + (ex - sx) * t
        yi = sy + (ey - sy) * t
        if _pose_hits(world, xi, yi):
            return True
    return False


def guarded_step(world, robot_state, v_cmd, w_cmd, dt=DT):
    next_state = unicycle_step(*robot_state, v_cmd, w_cmd, dt)
    if segment_hits_obstacle(world, (robot_state[0], robot_state[1]), (next_state[0], next_state[1])):
        return robot_state, True
    return next_state, False
