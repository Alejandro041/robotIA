import math
from config import V_MAX, W_MAX, LOOKAHEAD_DIST, DT


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
