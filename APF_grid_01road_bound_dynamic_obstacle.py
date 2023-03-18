from scripts.vehicle_v2x_obs import Vehicle
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
speed_stop = 4   # stop permitted when speed < speed_stop
steer_max = np.deg2rad(45.0)  # max steering angle [rad]
d_dist = 0.8
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
    z_ref = np.zeros((NX, T))
    length = ref_path.length
    ind, _ = ref_path.nearest_index(node)

    z_ref[0, 0] = ref_path.cx[ind]
    z_ref[1, 0] = ref_path.cy[ind]
    z_ref[2, 0] = node.get_v()
    z_ref[3, 0] = ref_path.cyaw[ind]

    dist_move = 0.0

    for i in range(1, T):
        dist_move += abs(sp[ind]) * dt
        ind_move = int(round(dist_move / d_dist))
        index = min(ind + ind_move, length - 1)

        z_ref[0, i] = ref_path.cx[index]
        z_ref[1, i] = ref_path.cy[index]
        z_ref[2, i] = sp[index]
        z_ref[3, i] = ref_path.cyaw[index]

    return z_ref, ind


if __name__ == '__main__':
    # start position and end position
    ax = [0.0, 80.0]
    ay = [0.0, 0.0]

    # ego_vehicle_path
    cx, cy, cyaw, ck, s = itp.calc_spline_course(
        ax, ay, ds=d_dist)
    sp = itp.calc_speed_profile(cx, cy, cyaw, target_speed)
    ref_path = PATH(cx, cy, cyaw, ck)

    # obs_vehicle_path
    cx, cy, cyaw, ck, s = itp.calc_spline_course(
        ax, ay, ds=0.1)
    ref_path_obs = PATH(cx, cy, cyaw, ck)
    obs_pos_cnt = int(ref_path_obs.length/3)

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

    ego_vehicle.solver_basis(Q=np.diag([3, 2, 0.5, 5, 0]))

    next_states = np.zeros((ego_vehicle.n_states, T+1)).T
    u0 = np.array([0, 0]*T).reshape(-1, 2).T

    log_data = []
    obs_path = []

    c_roads = [2.5, -2.5, 7.5]
    nc_roads = [[-2.5,-1], [7.5,1]]  

    while simu_time < time_max:
        z_ref, target_ind = \
            calc_ref_trajectory_in_T_step(ego_vehicle, ref_path, sp)

        cur_ref_path = np.vstack([z_ref[0, :], z_ref[1, :]]).T

        z_ref = np.array([z_ref[0, :], z_ref[1, :], z_ref[3, :], z_ref[2, :], [
                         0]*z_ref.shape[1], [0]*z_ref.shape[1]]).T

        obs = np.array(
            [[ref_path_obs.cx[obs_pos_cnt], ref_path_obs.cy[obs_pos_cnt], ref_path_obs.cyaw[obs_pos_cnt]]])
        obs_path.append(obs[0])
        obs_pos_cnt += 2

        ego_vehicle.solver_reset_cost()

        ego_vehicle.solver_reset_cost()
        ego_vehicle.solver_add_cost()
        ego_vehicle.solver_add_soft_obs(obs)
        # ego_vehicle.solver_add_hard_obs(obs=obs)
        
        '''
        Use MPC solver with initialized u_opt from last iteration of MPC
        '''

        z0 = np.array([ego_vehicle.x, ego_vehicle.y,
                      ego_vehicle.yaw, ego_vehicle.vx, ego_vehicle.vy, 0])
        # print("%.2f"% z0[1])
        ## ------------------> Add Road Bound to Solver <----------------------
        ego_vehicle.solver_add_c_road_pf(c_roads)
        ego_vehicle.solver_add_nc_road_pf(nc_roads)
        # ego_vehicle.solver_add_expnc_road_pf(nc_roads)

        ## For Debug
        # print("Left Marker")
        # # self.a_r *(dist - 1.5)**2
        # print("%.2f" %(10 * ((abs(z0[1] - nc_roads[0]) - 1.5)**2)))
        # print("Right Marker")
        # print("%.2f" %(10 * ((abs(z0[1] - c_roads[0]) - 1.5)**2)))
        # print("%.2f" %(1000 * (1/(z0[1] + 10)**2)))
        # self.ac_r * (1/(selfc[1] - road_pos))**2 + 10*(sqrt_dist-1)**2
        # print("DA")
        # print(0.5*(1/(z0[1] - c_roads[0]))**2 + ((z0[1] - c_roads[0])**2-1)**2)
        # print(0.5*(1/(z0[1] - c_roads[0]))**2)

        ## ------------------> Add Road Bound to Solver <----------------------
        ego_vehicle.solver_add_bounds()
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
        ## draw road lane marker 
        for road, dir in (nc_roads):
            plt.plot([(ax[0]), (ax[1])], [(ay[0]+road), (ay[1]+road)], color='k', linestyle='-')

        for road in (c_roads):
            plt.plot([(ax[0]), (ax[1])], [(ay[0]+road), (ay[1]+road)], color='b', linestyle='--')
        ## 
        if x_opt is not None:
            plt.plot(x_opt, y_opt, color='darkviolet', marker='*')
        # current ref traj
        plt.plot(cur_ref_path[:, 0], cur_ref_path[:, 1], color='green', marker='*')
        # whole traj
        plt.plot(cx, cy, color='gray')
        obs_dis_path = np.array(obs_path)
        # obs traj
        plt.plot(obs_dis_path[:, 0], obs_dis_path[:, 1], color='yellow', alpha=0.5)
        # ego vehicle traj
        plt.plot(x, y, '-b')
        # plt.plot(cx[target_ind], cy[target_ind])
        plt.axis("equal")
        plt.title("NonLinear MPC, " + "v = " +
                  str(round(ego_vehicle.get_v(), 2)))
        plt.pause(0.0001)

if save_result:
    plot_result(np.array(log_data), scenario_name, 'result')
