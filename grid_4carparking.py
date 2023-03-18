import math
import utils.draw as draw
import time
from scripts.vehicle_obs import Vehicle
from scenarios.carpark import ParkEnv
from matplotlib.patches import Circle
from utils.draw_result import plot_result
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import carla
import sys
import os

try:
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__))))
except IndexError:
    pass
sys.path.append('../scripts')

scenario_name = 'grid-4'
save_result = True

park_entrance = carla.Transform(carla.Location(
    x=21.1, y=-30.8, z=1), carla.Rotation(yaw=270))
park_lot = carla.Transform(carla.Location(
    x=-6.7, y=-27.1, z=1), carla.Rotation(yaw=90))

pe = ParkEnv(plt.gca())

time_max = 100.0
simu_time = 0
dis_radius = 1.8  # obs dis_radius, just for display

dt = 0.1
T = 10

# spawn_point = [-15, -15, math.pi/2, 0]
# spawn_point = [-25, -30, math.pi/2, 0]
spawn_point = [-40, -15, math.pi/2, 0]
target_point = [-16, 8, math.pi/2]

ego_vehicle = Vehicle(spawn_point, horizon=T, carla=False)


x = [ego_vehicle.x]
y = [ego_vehicle.y]
yaw = [ego_vehicle.yaw]
v = [ego_vehicle.get_v()]
t = [0.0]
d = [0.0]
a = [0.0]

dist_stop = 0.1
speed_stop = 0.5
yaw_stop = 0.1

delta_opt, a_opt = 0.0, 0.0
a_exc, delta_exc = 0.0, 0.0

ego_vehicle.solver_basis(Q=np.diag([10, 10, 10, 1, 0]),
                         R=np.diag([0.1, 10]),
                         Rd=np.diag([0.5, 50]))
# ego_vehicle.solver_add_hard_obs(obs=pe.obs)

x_m = np.zeros((ego_vehicle.n_states, T+1)).T
z_ref = np.array([[target_point[0]]*T, [target_point[1]]*T, [target_point[2]]*T, [0]*T, [0]*T, [0]*T]).T
next_states = np.copy(x_m)
u0 = np.array([0, 0]*T).reshape(-1, 2).T
flag_Q = True
dist_goal = 10

log_data = []

while simu_time < time_max:
    obs = []
    for o in pe.obs:
        dist = np.hypot(ego_vehicle.x-o[0], ego_vehicle.y-o[1])
        if (dist < 10):
            obs.append(o)
    # print(obs)
    ego_vehicle.solver_add_cost()
    ego_vehicle.solver_add_soft_obs(obs, 2000)
    ego_vehicle.solver_add_bounds()

    # z0 = [-12.5, 3.5, 3.14, 0, 0, 0]
    z0 = np.array([ego_vehicle.x, ego_vehicle.y, ego_vehicle.yaw,
                  ego_vehicle.vx, ego_vehicle.vy, 0])
    # acc_optimal, steer_optimal, x_optimal, y_optimal, yaw_optimal, v_optimal
    start_time = time.time()
    if dist_goal < 4 and flag_Q == True:
        ego_vehicle.solver_basis(Q=np.diag([3000, 3000, 7000, 0, 0]),
                                 R=np.diag([0.1, 1]),
                                 Rd=np.diag([0.01, 100]))
        ego_vehicle.solver_add_cost()
        ego_vehicle.solver_add_soft_obs(obs, 150)
        ego_vehicle.solver_add_bounds()
        print("Change Q parameter!")
        flag_Q = False

    u_opt, xm_opt, cost_time = ego_vehicle.solve_MPC(
        z_ref, z0, next_states, u0)
    a_opt, delta_opt = u_opt[0, :]
    x_opt = xm_opt[:, 0]
    y_opt = xm_opt[:, 1]
    next_states = np.concatenate((xm_opt[:, 1:], xm_opt[:, -1:]), axis=1)
    u0 = np.concatenate((u_opt[:, 1:], u_opt[:, -1:]), axis=1)
    # end test point control

    # pick the first step
    if delta_opt is not None:
        delta_exc, a_exc = delta_opt, a_opt

    # print(a_exc, delta_exc)
    ego_vehicle.update(a_exc, delta_exc)
    simu_time += dt

    x.append(ego_vehicle.x)
    y.append(ego_vehicle.y)
    yaw.append(ego_vehicle.yaw)
    v.append(ego_vehicle.get_v())
    t.append(simu_time)
    d.append(delta_exc)
    a.append(a_exc)

    # dist = math.hypot(ego_vehicle.x - z_ref[0,-1], ego_vehicle.y - z_ref[1, -1])
    dist_goal = math.hypot(
        ego_vehicle.x - z_ref[0, 0], ego_vehicle.y - z_ref[0, 1])
    yaw_error = abs(ego_vehicle.yaw - z_ref[0, 2])
    log_data.append([simu_time, dist_goal, yaw_error,
                     a_exc, delta_exc, v[-1], cost_time])
    if dist_goal < dist_stop and yaw_error < yaw_stop:  # and \
        if save_result:
            plt.savefig(os.path.join('result', scenario_name+'-3.pdf'),
                        bbox_inches='tight', pad_inches=0)
        break
    steer = -delta_exc

    plt.cla()
    # pe.create_plot()
    pe.plot()
    draw.draw_car(ego_vehicle.x, ego_vehicle.y, ego_vehicle.yaw, steer)
    obs = np.array(pe.obs)
    for i in range(obs.shape[0]):
        # draw.draw_car(x=obs[i,0], y=obs[i,1], yaw=obs[0,2], steer = 0, color='red')
        c1, c2 = ego_vehicle.get_obs_centers(obs[i])
        circle1 = Circle(xy=c1, radius=dis_radius,
                         fc='none', ec='cornflowerblue')
        circle2 = Circle(xy=c2, radius=dis_radius,
                         fc='none', ec='cornflowerblue')
        plt.gca().add_patch(p=circle1)
        plt.gca().add_patch(p=circle2)

    plt.gcf().canvas.mpl_connect('key_release_event',
                                 lambda event:
                                 [exit(0) if event.key == 'escape' else None])

    if x_opt is not None:
        plt.plot(x_opt, y_opt, color='darkviolet', marker='o')
    plt.scatter(z_ref[4, 0], z_ref[4, 1], color='orange', marker='o')
    # whole traj
    plt.plot(x, y, '-b')
    plt.title("NMPC Set-point Car Parking, " + "v = " +
              str(round(ego_vehicle.get_v(), 2)))
    plt.pause(0.0001)

if save_result:
    plot_result(np.array(log_data), scenario_name, 'result')

# plt.title("Car Parking Complete! " + "p_error = " + str(round(dist_goal, 2))+"m, yaw_error = " + str(round(yaw_error, 2))+"rad")
# plt.savefig('car_parking_with_obs')
# plt.pause(0.0001)
# plt.show()
