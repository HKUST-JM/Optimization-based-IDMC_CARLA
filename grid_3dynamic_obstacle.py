from scripts.vehicle_obs import Vehicle
from matplotlib.patches import Circle
from utils.draw_result import plot_result
import utils.interpolate as itp
import utils.draw as draw
import numpy as np
import matplotlib.pyplot as plt
import math
import os

scenario_name = 'grid-3'
save_result = False

dist_stop = 1  # stop permitted when dist to goal < dist_stop
speed_stop = 0.5   # stop permitted when speed < speed_stop
steer_max = np.deg2rad(45.0)  # max steering angle [rad]
d_dist = 0.1
target_speed = 20 / 3.6  # target speed
speed_max = 20.0 / 3.6  # maximum speed [m/s]
speed_min = -10.0 / 3.6  # minimum speed [m/s]
time_max = 100.0  # max simulation time
dis_radius = 1.8  # obs dis_radius, just for display

N_IND = 10  # search index number

NX = 4  # state vector: z = [x, y, v, phi]
T = 10  # finite time horizon length
dt = 0.1  # time step
WB = 2.5  # [m] Wheel base


class PATH:
    def __init__(self, cx, cy, cyaw, ck):
        self.cx = cx
        self.cy = cy
        self.cyaw = cyaw
        self.ck = ck
        self.length = len(cx)
        self.ind_old = 0

    def nearest_index(self, node):
        """
        calc index of the nearest node in N steps
        :param node: current information
        :return: nearest index, lateral distance to ref point
        """

        dx = [
            node.x - x for x in self.cx[self.ind_old: (self.ind_old + N_IND)]]
        dy = [
            node.y - y for y in self.cy[self.ind_old: (self.ind_old + N_IND)]]

        dist = np.hypot(dx, dy)
        ind_in_N = int(np.argmin(dist))
        ind = self.ind_old + ind_in_N
        # if ind > self.ind_old:
        self.ind_old = ind

        rear_axle_vec_rot_90 = np.array([[math.cos(node.yaw + math.pi / 2.0)],
                                         [math.sin(node.yaw + math.pi / 2.0)]])

        vec_target_2_rear = np.array([[dx[ind_in_N]],
                                      [dy[ind_in_N]]])

        er = np.dot(vec_target_2_rear.T, rear_axle_vec_rot_90)
        er = er[0][0]

        return ind, er


def calc_ref_trajectory_in_T_step(node, ref_path, sp):
    """
    calc referent trajectory in T steps: [x, y, v, yaw]
    using the current velocity, calc the T points along the reference path
    :param node: current information
    :param ref_path: reference path: [x, y, yaw]
    :param sp: speed profile (designed speed strategy)
    :return: reference trajectory
    """

    z_ref = np.zeros((NX, T + 10))
    length = ref_path.length
    ind, _ = ref_path.nearest_index(node)

    z_ref[0, 0] = ref_path.cx[ind]
    z_ref[1, 0] = ref_path.cy[ind]
    z_ref[2, 0] = sp[ind]
    z_ref[3, 0] = ref_path.cyaw[ind]

    dist_move = 0.0

    for i in range(1, T + 10):
        dist_move += abs(node.get_v()) * dt

        ind_move = int(round(dist_move / d_dist))
        index = min(ind+i + ind_move, length - 1)

        z_ref[0, i] = ref_path.cx[index]
        z_ref[1, i] = ref_path.cy[index]
        z_ref[2, i] = sp[index]
        z_ref[3, i] = ref_path.cyaw[index]

    return z_ref, ind


if __name__ == '__main__':
    # scenario1
    ax = [0.0, 80.0]
    ay = [0.0, 0.0]

    # obs = np.array([[20.0,-18.0,1.57]])

    # # scenario2
    # ax = [0.0, 25.0, 40.0, 60.0, 80.0]
    # ay = [0.0, 40.0, 35.0, 40.0, 0.0]

    # scenario3
    # ax = [0, 80]
    # ay = [0.0, 80]

    # scenario 3 obs
    # obs = np.array([[32.0,30.0,1.57*0.5],[60, 64, 1.57*0.5]])

    cx, cy, cyaw, ck, s = itp.calc_spline_course(
        ax, ay, ds=d_dist)
    sp = itp.calc_speed_profile(cx, cy, cyaw, target_speed)

    ref_path = PATH(cx, cy, cyaw, ck)
    ego_vehicle = Vehicle((cx[0], cy[0], cyaw[0], 0.0), horizon=T, carla=False)

    simu_time = 0.0
    x = [ego_vehicle.x]
    y = [ego_vehicle.y]
    yaw = [ego_vehicle.yaw]
    v = [ego_vehicle.get_v()]
    t = [0.0]
    d = [0.0]
    a = [0.0]

    delta_opt, a_opt = 0.0, 0.0
    a_exc, delta_exc = 0.0, 0.0

    cnt = 200
    ego_vehicle.solver_basis(Q=np.diag([3, 3, 1, 10, 0]))

    next_states = np.zeros((ego_vehicle.n_states, T+1)).T
    u0 = np.array([0, 0]*T).reshape(-1, 2).T

    log_data = []
    obs_path = []

    while simu_time < time_max:
        z_ref, target_ind = \
            calc_ref_trajectory_in_T_step(ego_vehicle, ref_path, sp)
        # vx_ref = z_ref[2,:]*np.cos(z_ref[3,:])
        # vy_ref = z_ref[2,:]*np.sin(z_ref[3,:])

        z_ref = np.array([z_ref[0, :], z_ref[1, :], z_ref[3, :], z_ref[2, :], [
                         0]*z_ref.shape[1], [0]*z_ref.shape[1]]).T

        obs = np.array(
            [[ref_path.cx[cnt], ref_path.cy[cnt], ref_path.cyaw[cnt]]])
        obs_path.append(obs[0])
        cnt += 2
        ego_vehicle.solver_add_cost()
        ego_vehicle.solver_add_soft_obs(obs)
        ego_vehicle.solver_add_bounds()
        '''
        Use MPC solver with initialized u_opt from last iteration of MPC
        '''

        z0 = np.array([ego_vehicle.x, ego_vehicle.y,
                      ego_vehicle.yaw, ego_vehicle.vx, ego_vehicle.vy, 0])
        # acc_optimal, steer_optimal, x_optimal, y_optimal, yaw_optimal, v_optimal
        u_opt, xm_opt, cost_time = ego_vehicle.solve_MPC(
            z_ref[0:T+0, :], z0, next_states, u0)
        a_opt, delta_opt = u_opt[0, :]
        x_opt = xm_opt[:, 0]
        y_opt = xm_opt[:, 1]
        next_states = np.concatenate((xm_opt[:, 1:], xm_opt[:, -1:]), axis=1)
        u0 = np.concatenate((u_opt[:, 1:], u_opt[:, -1:]), axis=1)
        # end test point control

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

        dist = math.hypot(ego_vehicle.x - cx[-1], ego_vehicle.y - cy[-1])
        dist_error = math.hypot(
            ego_vehicle.x - z_ref[0, 0], ego_vehicle.y - z_ref[0, 1])
        yaw_error = abs(ego_vehicle.yaw - z_ref[0, 2])

        log_data.append([simu_time, dist_error, yaw_error,
                        a_exc, delta_exc, v[-1], cost_time])

        if dist < dist_stop and \
                abs(ego_vehicle.get_v()) < speed_stop:
            if save_result:
                plt.savefig(os.path.join('result', scenario_name+'-3.pdf'),
                            bbox_inches='tight', pad_inches=0)
            break

        steer = -delta_exc

        plt.cla()
        draw.draw_car(ego_vehicle.x, ego_vehicle.y, ego_vehicle.yaw, steer)
        for i in range(obs.shape[0]):
            draw.draw_car(x=obs[i, 0], y=obs[i, 1],
                          yaw=obs[i, 2], steer=0, color='red')
            c1, c2 = ego_vehicle.get_obs_centers(obs[i])
            circle1 = Circle(xy=c1, radius=dis_radius,
                             fc='white', ec='cornflowerblue')
            circle2 = Circle(xy=c2, radius=dis_radius,
                             fc='white', ec='cornflowerblue')
            plt.gca().add_patch(p=circle1)
            plt.gca().add_patch(p=circle2)

        plt.gcf().canvas.mpl_connect('key_release_event',
                                     lambda event:
                                     [exit(0) if event.key == 'escape' else None])

        if x_opt is not None:
            plt.plot(x_opt, y_opt, color='darkviolet', marker='*')
        plt.scatter(z_ref[4, 0], z_ref[4, 1], color='orange', marker='o')
        # whole traj
        plt.plot(cx, cy, color='gray')
        temp_path = np.array(obs_path)
        plt.plot(temp_path[:, 0], temp_path[:, 1], color='yellow', alpha=0.5)
        plt.plot(x, y, '-b')
        # plt.plot(cx[target_ind], cy[target_ind])
        plt.axis("equal")
        plt.title("NonLinear MPC, " + "v = " +
                  str(round(ego_vehicle.get_v(), 2)))
        plt.pause(0.0001)

if save_result:
    plot_result(np.array(log_data), scenario_name, 'result')
