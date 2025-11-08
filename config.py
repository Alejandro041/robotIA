# Parámetros globales de la simulación
GRID_W, GRID_H = 60, 40
OBSTACLE_DENSITY = 0.15
WORLD_GEN_RETRIES = 12
START = (2, 2)
GOAL = (55, 35)


LIDAR_RAYS = 16
LIDAR_MAX_RANGE = 8.0
DT = 0.15
ROBOT_R = 0.5
V_MAX = 2.0
W_MAX = 1.8
LOOKAHEAD_DIST = 1.2
REPLAN_BLOCK_MARGIN = 0.6
MAX_STEPS = 900
DYNAMIC_OBS_STEP = 250


# Mantenimiento predictivo (mock)
MOTOR_NOISE = 0.06
CURRENT_BASE = 2.5
CURRENT_LOAD_FACTOR = 0.7
ROLLING_WINDOW = 50
Z_ALERT = 2.5


# Salidas
CSV_LOG = "outputs/robot_log.csv"
PLOT_TRAJ = "outputs/plot_trayectoria.png"
PLOT_CURRENT = "outputs/plot_corriente_alertas.png"
PLOT_CMDS = "outputs/plot_comandos.png"
PLOT_LIDAR = "outputs/plot_lidar.png"
