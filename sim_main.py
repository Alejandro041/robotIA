import os, math, numpy as np, pandas as pd, matplotlib.pyplot as plt
from datetime import datetime
from config import (
    GRID_W, GRID_H, OBSTACLE_DENSITY, WORLD_GEN_RETRIES, START, GOAL,
    DT, MAX_STEPS, DYNAMIC_OBS_STEP, REPLAN_BLOCK_MARGIN,
    CSV_LOG, PLOT_TRAJ, PLOT_CURRENT, PLOT_CMDS, PLOT_LIDAR
)
from world import make_world, inject_dynamic_obstacle
from planner import a_star, path_to_continuous
from lidar import lidar_scan
from controller import nearest_path_index, unicycle_step, compute_controls
from maintenance import MaintenanceMonitor
from ai_agent import decide_action

np.random.seed(7)
os.makedirs("outputs", exist_ok=True)

# 1) Mundo y ruta inicial
world = None
path = None
world_attempts = 0
for attempt in range(1, WORLD_GEN_RETRIES + 1):
    world_attempts = attempt
    world = make_world(GRID_W, GRID_H, OBSTACLE_DENSITY)
    path = a_star(world, START, GOAL)
    if path is not None:
        break
if path is None:
    raise RuntimeError(
        f"No se encontró ruta inicial después de {WORLD_GEN_RETRIES} intentos. "
        "Reduce OBSTACLE_DENSITY o cambia START/GOAL."
    )
path_xy = path_to_continuous(path)

# 2) Estado del robot
robot = [START[0] + 0.5, START[1] + 0.5, 0.0]
path_idx = 0
mon = MaintenanceMonitor()

log = {
    "step": [], "x": [], "y": [], "yaw": [],
    "v_cmd": [], "w_cmd": [], "lidar_min": [],
    "path_len_left": [], "motor_current": [], "maintenance_alert": [],
    "ai_action": [], "path_blocked": []
}

# Aux
def path_blocked(world, path_xy, start_idx, margin):
    for (x, y) in path_xy[start_idx:]:
        xi, yi = int(x), int(y)
        if world[min(max(0, yi), world.shape[0]-1), min(max(0, xi), world.shape[1]-1)] == 1:
            return True
        rr = int(math.ceil(margin))
        for yy in range(max(0, yi-rr), min(world.shape[0], yi+rr+1)):
            for xx in range(max(0, xi-rr), min(world.shape[1], xi+rr+1)):
                if (xx - x)**2 + (yy - y)**2 <= margin**2 and world[yy, xx] == 1:
                    return True
    return False

replans = 0
for step in range(MAX_STEPS):

    # Obstáculo dinámico
    if step == DYNAMIC_OBS_STEP and path_idx < len(path_xy):
        tgt = (
            int(path_xy[min(len(path_xy)-1, path_idx+5)][0]),
            int(path_xy[min(len(path_xy)-1, path_idx+5)][1])
        )
        inject_dynamic_obstacle(world, tgt, radius=2)

    # Lidar y replanificación si bloqueado
    angs, dists = lidar_scan(world, robot)
    blocked_flag = path_blocked(world, path_xy, path_idx, REPLAN_BLOCK_MARGIN)
    if blocked_flag:
        replans += 1
        cur_cell = (int(robot[0]), int(robot[1]))
        new_path = a_star(world, cur_cell, GOAL)
        if new_path is not None:
            path_xy = path_to_continuous(new_path)
            path_idx = 0

    # Controles y avance guiados por IA
    path_idx = nearest_path_index((robot[0], robot[1]), path_xy, path_idx)
    path_len_left = max(0, len(path_xy) - path_idx)
    v_cmd, w_cmd = compute_controls(robot, path_xy, path_idx)
    base_v, base_w = v_cmd, w_cmd
    min_d = float(dists.min())
    if min_d < 1.2:
        v_cmd *= max(0.0, (min_d - 0.2))

    context = {
        "step": step,
        "lidar_min": float(min_d),
        "path_blocked": bool(blocked_flag),
        "path_len_left": path_len_left,
        "current_z": mon.zscore(),
        "v_cmd_nominal": float(base_v),
        "w_cmd_nominal": float(base_w)
    }
    decision = decide_action(context)
    ai_action = decision.get("accion", "seguir")

    if ai_action == "replanificar" and not blocked_flag:
        replans += 1
        cur_cell = (int(robot[0]), int(robot[1]))
        new_path = a_star(world, cur_cell, GOAL)
        if new_path is not None:
            path_xy = path_to_continuous(new_path)
            path_idx = nearest_path_index((robot[0], robot[1]), path_xy, 0)
            path_len_left = max(0, len(path_xy) - path_idx)
            v_cmd, w_cmd = compute_controls(robot, path_xy, path_idx)
            base_v, base_w = v_cmd, w_cmd
            if min_d < 1.2:
                v_cmd *= max(0.0, (min_d - 0.2))

    if ai_action == "reducir_velocidad":
        v_cmd *= 0.3
    elif ai_action == "pausa_mantenimiento":
        v_cmd = 0.0
        w_cmd = 0.0

    robot = list(unicycle_step(*robot, v_cmd, w_cmd))

    # Logs
    cur = mon.motor_current_model(v_cmd, w_cmd)
    mon.window.append(cur)
    alert = mon.alert()

    log["step"].append(step)
    log["x"].append(robot[0])
    log["y"].append(robot[1])
    log["yaw"].append(robot[2])
    log["v_cmd"].append(v_cmd)
    log["w_cmd"].append(w_cmd)
    log["lidar_min"].append(min_d)
    log["path_len_left"].append(path_len_left)
    log["motor_current"].append(cur)
    log["maintenance_alert"].append(alert)
    log["ai_action"].append(ai_action)
    log["path_blocked"].append(int(blocked_flag))

    # Llegada
    if (robot[0] - (GOAL[0] + 0.5))**2 + (robot[1] - (GOAL[1] + 0.5))**2 < 1.0**2:
        break

# Guardar CSV
Df = pd.DataFrame(log)
Df.to_csv(CSV_LOG, index=False)

# Graficar y guardar PNGs (sin estilos ni colores específicos)
plt.figure(figsize=(6,4))
plt.plot(Df["x"], Df["y"])
plt.title("Trayectoria del robot (x,y)")
plt.xlabel("x")
plt.ylabel("y")
plt.gca().set_aspect("equal", adjustable="box")
plt.tight_layout()
plt.savefig(PLOT_TRAJ, dpi=160)
plt.close()

plt.figure(figsize=(8,4))
plt.plot(Df["step"], Df["motor_current"], label="Corriente (A)")
alert_idx = Df.index[Df["maintenance_alert"] == 1].tolist()
if alert_idx:
    plt.scatter(Df.loc[alert_idx, "step"], Df.loc[alert_idx, "motor_current"], label="Alerta", s=18)
plt.title("Corriente y alertas")
plt.xlabel("Paso")
plt.ylabel("A")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_CURRENT, dpi=160)
plt.close()

plt.figure(figsize=(8,4))
plt.plot(Df["step"], Df["v_cmd"], label="v_cmd")
plt.plot(Df["step"], Df["w_cmd"], label="w_cmd")
plt.title("Comandos de control")
plt.xlabel("Paso")
plt.ylabel("Magnitud")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_CMDS, dpi=160)
plt.close()

plt.figure(figsize=(8,4))
plt.plot(Df["step"], Df["lidar_min"])
plt.title("Distancia mínima LiDAR")
plt.xlabel("Paso")
plt.ylabel("celdas")
plt.tight_layout()
plt.savefig(PLOT_LIDAR, dpi=160)
plt.close()

print("=== Sesión 2 — Prototipo Simulado ===")
print("Fecha:", datetime.now().isoformat(timespec='seconds'))
print("Pasos simulados:", len(Df))
print("Replanificaciones:", replans)
print("Intentos de mundo:", world_attempts)
print("CSV:", CSV_LOG)
print("PNG:", PLOT_TRAJ, PLOT_CURRENT, PLOT_CMDS, PLOT_LIDAR)
