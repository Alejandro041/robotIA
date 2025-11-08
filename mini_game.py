import math
from dataclasses import dataclass

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

from config import (
    GRID_W,
    GRID_H,
    OBSTACLE_DENSITY,
    WORLD_GEN_RETRIES,
    START,
    GOAL,
    DT,
    MAX_STEPS,
    DYNAMIC_OBS_STEP,
    REPLAN_BLOCK_MARGIN,
)
from world import make_world, inject_dynamic_obstacle
from planner import a_star, path_to_continuous
from lidar import lidar_scan
from controller import nearest_path_index, unicycle_step, compute_controls
from maintenance import MaintenanceMonitor
from ai_agent import decide_action


def path_blocked(world, path_xy, start_idx, margin):
    """Comprobación local para ver si el camino quedó tapado."""
    for (x, y) in path_xy[start_idx:]:
        xi, yi = int(x), int(y)
        if world[min(max(0, yi), world.shape[0] - 1), min(max(0, xi), world.shape[1] - 1)] == 1:
            return True
        rr = int(math.ceil(margin))
        for yy in range(max(0, yi - rr), min(world.shape[0], yi + rr + 1)):
            for xx in range(max(0, xi - rr), min(world.shape[1], xi + rr + 1)):
                if (xx - x) ** 2 + (yy - y) ** 2 <= margin ** 2 and world[yy, xx] == 1:
                    return True
    return False


@dataclass
class StepInfo:
    step: int
    action: str
    lidar_min: float
    current_z: float
    path_left: int
    blocked: bool


class MiniGame:
    def __init__(self):
        self.world = None
        self.path_xy = None
        self.robot = [START[0] + 0.5, START[1] + 0.5, 0.0]
        self.path_idx = 0
        self.replans = 0
        self.world_attempts = 0
        self.step = 0
        self.mon = MaintenanceMonitor()
        self.last_action = "seguir"
        self.last_lidar_min = 9.9
        self.running = True
        self.paused = True
        self.step_once = False
        self.manual_replan = False
        self.fig = None
        self.ax = None
        self.world_img = None
        self.path_line = None
        self.robot_dot = None
        self.heading = None
        self.status_text = None
        self.help_text = None
        self.history = []
        self._build_world_and_path()
        self._setup_plot()
        self.fig.canvas.mpl_connect("close_event", self._on_close)
        self.ani = animation.FuncAnimation(self.fig, self._tick, interval=int(DT * 1000))

    def _build_world_and_path(self):
        for attempt in range(1, WORLD_GEN_RETRIES + 1):
            world = make_world(GRID_W, GRID_H, OBSTACLE_DENSITY)
            path = a_star(world, START, GOAL)
            if path is not None:
                self.world_attempts = attempt
                self.world = world
                self.path_xy = path_to_continuous(path)
                return
        raise RuntimeError(
            f"No se encontró ruta inicial tras {WORLD_GEN_RETRIES} intentos. Ajusta config y vuelve a probar."
        )

    def _setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.set_title(
            "Mini juego robot • Controles: espacio=play/pausa, n=paso, r=replan, esc=cerrar"
        )
        self.ax.set_xlim(0, GRID_W)
        self.ax.set_ylim(0, GRID_H)
        extent = (0, GRID_W, 0, GRID_H)
        self.world_img = self.ax.imshow(
            self.world.T,
            origin="lower",
            extent=extent,
            cmap="Greys",
            alpha=0.35,
            interpolation="nearest",
        )
        path_x = [p[0] for p in self.path_xy]
        path_y = [p[1] for p in self.path_xy]
        (self.path_line,) = self.ax.plot(path_x, path_y, "b--", linewidth=1.4, label="Ruta")
        (self.robot_dot,) = self.ax.plot([], [], "ro", markersize=8, label="Robot")
        self.heading = self.ax.quiver(
            [self.robot[0]],
            [self.robot[1]],
            [math.cos(self.robot[2])],
            [math.sin(self.robot[2])],
            color="red",
            scale=10,
            scale_units="xy",
            angles="xy",
            width=0.008,
        )
        self.ax.scatter([START[0] + 0.5], [START[1] + 0.5], c="green", s=40, label="Inicio")
        self.ax.scatter([GOAL[0] + 0.5], [GOAL[1] + 0.5], c="purple", s=40, label="Meta")
        self.ax.legend(loc="upper right", fontsize=8)
        self.status_text = self.ax.text(
            0.01,
            1.02,
            "",
            transform=self.ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            color="black",
        )
        help_msg = (
            "Espacio: play/pausa • N: paso único • R: replanificar • Esc: salir\n"
            "Observa la acción IA, LIDAR y corriente para entender sus decisiones."
        )
        self.help_text = self.ax.text(
            0.5,
            -0.08,
            help_msg,
            transform=self.ax.transAxes,
            ha="center",
            va="top",
            fontsize=8,
            color="dimgray",
        )
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)
        self._update_plot()

    def _on_key(self, event):
        if event.key == " ":
            self.paused = not self.paused
        elif event.key == "n":
            self.step_once = True
            self.paused = False
        elif event.key == "r":
            self.manual_replan = True
        elif event.key == "escape":
            self._close_fig()

    def _on_close(self, _):
        self.running = False

    def _close_fig(self):
        if self.running:
            self.running = False
            plt.close(self.fig)

    def _replan_from_current(self, tag):
        cur_cell = (int(self.robot[0]), int(self.robot[1]))
        new_path = a_star(self.world, cur_cell, GOAL)
        if new_path is None:
            return False
        self.replans += 1
        self.path_xy = path_to_continuous(new_path)
        self.path_idx = 0
        path_x = [p[0] for p in self.path_xy]
        path_y = [p[1] for p in self.path_xy]
        self.path_line.set_data(path_x, path_y)
        self.ax.set_title(
            f"Mini juego robot • Ult. replan: {tag} • Controles: espacio=play/pausa, n=paso, r=replan, esc=cerrar"
        )
        return True

    def _sync_path_index(self):
        self.path_idx = nearest_path_index((self.robot[0], self.robot[1]), self.path_xy, 0)
        return max(0, len(self.path_xy) - self.path_idx)

    def _handle_dynamic_obstacle(self):
        if self.step == DYNAMIC_OBS_STEP and self.path_idx < len(self.path_xy):
            tgt = (
                int(self.path_xy[min(len(self.path_xy) - 1, self.path_idx + 5)][0]),
                int(self.path_xy[min(len(self.path_xy) - 1, self.path_idx + 5)][1]),
            )
            inject_dynamic_obstacle(self.world, tgt, radius=2)
            self.world_img.set_data(self.world.T)

    def _update_plot(self):
        self.robot_dot.set_data([self.robot[0]], [self.robot[1]])
        dx = math.cos(self.robot[2]) * 1.2
        dy = math.sin(self.robot[2]) * 1.2
        self.heading.set_offsets([[self.robot[0], self.robot[1]]])
        self.heading.set_UVC([dx], [dy])
        status = (
            f"Paso {self.step}/{MAX_STEPS} • Acción IA: {self.last_action} "
            f"• LIDAR min: {self.last_lidar_min:.2f} "
            f"• Replans: {self.replans} • Intentos mundo: {self.world_attempts}"
        )
        self.status_text.set_text(status)
        self.fig.canvas.draw_idle()

    def _goal_reached(self):
        return (
            (self.robot[0] - (GOAL[0] + 0.5)) ** 2
            + (self.robot[1] - (GOAL[1] + 0.5)) ** 2
            < 1.0 ** 2
        )

    def _tick(self, _frame):
        if not self.running:
            return
        if self.step >= MAX_STEPS or self._goal_reached():
            self._close_fig()
            return
        if self.paused and not self.step_once:
            return
        single_step = self.step_once
        self.step_once = False
        self._advance_step()
        if single_step:
            self.paused = True

    def _advance_step(self):
        self._handle_dynamic_obstacle()
        _, dists = lidar_scan(self.world, self.robot)
        blocked = path_blocked(self.world, self.path_xy, self.path_idx, REPLAN_BLOCK_MARGIN)
        blocked_flag = blocked
        self.path_idx = nearest_path_index((self.robot[0], self.robot[1]), self.path_xy, self.path_idx)
        path_len_left = max(0, len(self.path_xy) - self.path_idx)

        if blocked and self._replan_from_current("bloqueo"):
            path_len_left = self._sync_path_index()
            blocked = False

        if self.manual_replan:
            self.manual_replan = False
            if self._replan_from_current("manual"):
                path_len_left = self._sync_path_index()
                blocked = False

        v_cmd, w_cmd = compute_controls(self.robot, self.path_xy, self.path_idx)
        base_v, base_w = v_cmd, w_cmd
        min_d = float(dists.min())
        if min_d < 1.2:
            v_cmd *= max(0.0, (min_d - 0.2))

        context = {
            "step": self.step,
            "lidar_min": float(min_d),
            "path_blocked": bool(blocked_flag),
            "path_len_left": path_len_left,
            "current_z": self.mon.zscore(),
            "v_cmd_nominal": float(base_v),
            "w_cmd_nominal": float(base_w),
        }
        decision = decide_action(context)
        ai_action = decision.get("accion", "seguir")

        if ai_action == "replanificar" and not blocked_flag:
            if self._replan_from_current("IA"):
                path_len_left = self._sync_path_index()

        if ai_action == "reducir_velocidad":
            v_cmd *= 0.3
        elif ai_action == "pausa_mantenimiento":
            v_cmd = 0.0
            w_cmd = 0.0

        self.robot = list(unicycle_step(*self.robot, v_cmd, w_cmd))
        cur = self.mon.motor_current_model(v_cmd, w_cmd)
        self.mon.window.append(cur)

        info = StepInfo(
            step=self.step,
            action=ai_action,
            lidar_min=min_d,
            current_z=self.mon.zscore(),
            path_left=path_len_left,
            blocked=blocked_flag,
        )
        self.history.append(info)
        self.last_action = ai_action
        self.last_lidar_min = min_d
        self.step += 1

        self._update_plot()

        if self._goal_reached() or self.step >= MAX_STEPS:
            self._close_fig()

    def run(self):
        print("Mini juego iniciado. Usa la ventana para controlar la simulación.")
        plt.show()
        print(f"Pasos simulados: {self.step}")
        print(f"Replanificaciones: {self.replans}")
        print(f"Intentos de mundo: {self.world_attempts}")
        if self.history:
            last = self.history[-1]
            print(
                f"Última acción IA: {last.action} | LIDAR: {last.lidar_min:.2f} | "
                f"z-corriente: {last.current_z:.2f}"
            )


if __name__ == "__main__":
    game = MiniGame()
    game.run()
