import carla
import math
import time

import numpy as np
from scipy.optimize import minimize

import casadi as ca

kf = -128916  # N/rad
kr = -85944  # N/rad
lf = 1.06  # m
lr = 1.85  # m
m = 1412  # kg
Iz = 1536.7  # kg*m2s
dt = None  # s
Lk = lf*kf - lr*kr  # = 22345.439999999973

n_input = 2  # acc, steering


class Vehicle:
    def __init__(self, state=None, actor=None, horizon=10, target_v=18, carla=True, delta_t=0.05):
        '''
        state: x=0.0, y=0.0, yaw=0.0, v=0.0, direct=1.0
        '''
        self.carla = carla
        global dt
        dt = delta_t
        if carla:
            self.actor = actor
            self.c_loc = self.actor.get_location()
            self.transform = self.actor.get_transform()
            yaw = self.transform.rotation.yaw * np.pi / 180
            self.x = self.c_loc.x
            self.y = self.c_loc.y
            self.yaw = yaw
            self.vx = 0
            self.vy = 0
            self.direct = 1
        else:
            assert not type(state) is None
            assert len(state) == 4
            self.x, self.y, self.yaw, v = state
            self.vx = v * math.cos(self.yaw)
            self.vy = v * math.sin(self.yaw)
            self.direct = 1

        # param of MPC
        self.horizon = horizon
        self.maxiter = 20
        # param of car state_dot
        self.omega = 0
        self.steer_bound = 0.6
        self.acc_lbound = -6
        self.acc_ubound = 3
        self.target_v = target_v / 3.6

    def get_state_carla(self):
        self.transform = self.actor.get_transform()
        self.x = self.transform.location.x
        self.y = self.transform.location.y
        self.z = self.transform.location.z
        yaw = self.transform.rotation.yaw
        self.yaw = yaw

        self.velocity = self.actor.get_velocity()
        self.vx = np.sqrt(self.velocity.x**2 + self.velocity.y**2)
        # self.vy = self.velocity.y
        self.vy = 0
        self.omega = self.actor.get_angular_velocity().z

        return [self.x, self.y, self.yaw, self.vx, self.vy, self.omega], self.z

    def get_v(self):
        # return self.vx/math.cos(self.yaw)
        if self.carla:
            self.get_state_carla()
            return self.get_direction() * np.sqrt(self.vx**2+self.vy**2)
        else:
            return np.sqrt(self.vx**2+self.vy**2)

    def get_direction(self):
        '''
        Get the direction of the vehicle's velocity
        '''
        yaw = np.radians(self.yaw)
        v_yaw = math.atan2(self.velocity.y, self.velocity.x)
        error = v_yaw - yaw
        if error < -math.pi:
            error += 2*math.pi
        elif error > math.pi:
            error -= 2*math.pi
        error = abs(error)
        if error > math.pi/2:
            return -1
        return 1

    def set_state(self, next_state):
        _, _, _, self.vx, self.vy, self.omega = next_state
        self.vy = -self.vy
        self.omega = -self.omega

    def set_target_velocity(self, target_v):
        self.target_v = target_v / 3.6

    def set_location_carla(self, location):
        if len(location) == 6:
            x, y, z, roll, pitch, yaw = location
        elif len(location) == 3:
            x, y, yaw = location
            z = roll = pitch = 0
        self.actor.set_transform(carla.Transform(carla.Location(x=x, y=y, z=z),
                                                 carla.Rotation(roll=roll, pitch=pitch, yaw=yaw)))

    def update(self, acc, steer):
        self.x += (self.vx*math.cos(self.yaw)-self.vy*math.sin(self.yaw))*dt
        self.y += (self.vy*math.cos(self.yaw)+self.vx*math.sin(self.yaw))*dt
        self.yaw += self.omega*dt
        self.vx += dt*acc
        self.vy = (m*self.vx*self.vy+dt*Lk*self.omega-dt*kf*steer *
                   self.vx-dt*m*(self.vx**2)*self.omega)/(m*self.vx-dt*(kf+kr))
        self.omega = (Iz*self.vx*self.omega+dt*Lk*self.vy-dt*lf *
                      kf*steer*self.vx)/(Iz*self.vx-dt*(lf**2*kf+lr**2*kr))

    def predict(self, prev_state, u):
        x, y, yaw, vx, vy, omega = prev_state
        acc = u[0]
        steer = u[1]

        x += (vx*math.cos(yaw)-vy*math.sin(yaw))*dt
        y += (vy*math.cos(yaw)+vx*math.sin(yaw))*dt
        yaw += omega*dt
        vx += dt*acc
        vy = (m*vx*vy+dt*Lk*omega-dt*kf*steer*vx -
              dt*m*(vx**2)*omega)/(m*vx-dt*(kf+kr))
        omega = (Iz*vx*omega+dt*Lk*vy-dt*lf*kf*steer*vx) / \
            (Iz*vx-dt*(lf**2*kf+lr**2*kr))

        return (x, y, yaw, vx, vy, omega)

    def solver_basis(self, Q=np.diag([3, 3, 1, 1, 0]), R=np.diag([0.1, 0.1]), Rd=np.diag([1, 10.0])):
        cx = ca.SX.sym('cx')
        cy = ca.SX.sym('cy')
        cyaw = ca.SX.sym('cyaw')
        cvx = ca.SX.sym('cvx')
        cvy = ca.SX.sym('cvy')
        comega = ca.SX.sym('comega')

        states = ca.vertcat(cx, cy)
        states = ca.vertcat(states, cyaw)
        states = ca.vertcat(states, cvx)
        states = ca.vertcat(states, cvy)
        states = ca.vertcat(states, comega)

        n_states = states.size()[0]
        self.n_states = n_states

        cacc = ca.SX.sym('cacc')
        csteer = ca.SX.sym('csteer')
        controls = ca.vertcat(cacc, csteer)
        n_controls = controls.size()[0]
        self.n_controls = n_controls

        rhs = ca.vertcat((cx + (cvx*ca.cos(cyaw)-cvy*ca.sin(cyaw))*dt),
                         (cy + (cvy*ca.cos(cyaw)+cvx*ca.sin(cyaw))*dt))
        rhs = ca.vertcat(rhs, cyaw + comega*dt)
        if self.carla:
            rhs = ca.vertcat(rhs, cvx + dt*cacc)
            rhs = ca.vertcat(rhs, (m*cvx*cvy+dt*Lk*comega-dt*kf*csteer*cvx -
                                   dt*m*(cvx*cvx)*comega)/(m*cvx-dt*(kf+kr)))
            rhs = ca.vertcat(rhs, (Iz*cvx*comega+dt*Lk*cvy-dt *
                             lf*kf*csteer*cvx)/(Iz*cvx-dt*(lf*lf*kf+lr*lr*kr)))
        else:
            rhs = ca.vertcat(rhs, cvx + dt*cacc)
            rhs = ca.vertcat(rhs, (m*cvx*cvy+dt*Lk*comega-dt*kf*csteer*cvx -
                                   dt*m*(cvx*cvx)*comega)/(m*cvx-dt*(kf+kr)))
            rhs = ca.vertcat(rhs, (Iz*cvx*comega+dt*Lk*cvy-dt *
                             lf*kf*csteer*cvx)/(Iz*cvx-dt*(lf*lf*kf+lr*lr*kr)))

        self.f = ca.Function('f', [states, controls], [rhs], [
            'input_state', 'control_input'], ['rhs'])

        # MPC
        self.U = ca.SX.sym('U', n_controls, self.horizon)
        self.X = ca.SX.sym('X', n_states, self.horizon+1)  # state
        self.P = ca.SX.sym('P', n_states, self.horizon+1)  # reference

        self.Q = Q
        self.R = R
        self.Rd = Rd

        self.opt_variables = ca.vertcat(ca.reshape(
            self.U, -1, 1), ca.reshape(self.X, -1, 1))
        
        self.Da = 1
        self.ac_r = 0.5
        self.anc_r = 100

    def solver_add_cost(self):
        self.obj = 0
        self.g = []  # equal constrains for multi-shooting
        self.lbg = []
        self.ubg = []
        self.lbx = []
        self.ubx = []

        self.g.append(self.X[:, 0]-self.P[:, 0])
        # add objective function and equality constraints
        for i in range(self.horizon):
            state_error = (self.X[0:5, i]-self.P[0:5, i])

            self.obj = self.obj + ca.mtimes([state_error.T, self.Q, state_error]) \
                       + ca.mtimes([self.U[:, i].T, self.R, self.U[:, i]])
            if i < (self.horizon-1):
                control_diff = self.U[:, i]-self.U[:, i+1]
                self.obj = self.obj + \
                    ca.mtimes([control_diff.T, self.Rd, control_diff])
            # x_next_ = self.predict(X[:,i], U[:,i])
            x_next_ = self.f(self.X[:, i], self.U[:, i])
            self.g.append(self.X[:, i+1]-x_next_)
        self.obj += ca.mtimes([(self.X[0:5, self.horizon]-self.P[0:5, self.horizon]).T, self.Q, (self.X[0:5, self.horizon]-self.P[0:5, self.horizon])])

    def get_obs_centers(self, ob_, radius=2.4, carla=False):
        '''
        [Right Coordinate System]
        Get the centers of the two circles that represent the obstacle
        Parameters:
            ob_ : [x, y, yaw]
            radius : radius of the obstacle
        Returns:
            obc1 : [x, y]
            obc2 : [x, y]
        '''
        if carla:
            obc1 = [ob_[0]+1.5*radius*np.cos(ob_[2]), ob_[1]+1.5*radius*np.sin(ob_[2])]
            obc2 = [ob_[0]-radius*np.cos(ob_[2]), ob_[1]-radius*np.sin(ob_[2])]
        else:
            obc1 = [ob_[0]+radius*np.cos(ob_[2]), ob_[1]+radius*np.sin(ob_[2])]
            obc2 = [ob_[0], ob_[1]]

        return obc1, obc2

    def solver_add_soft_obs(self, obs, ratio=370, expn = 1, carla=False):
        # self.obj = 0
        for i in range(self.horizon):
            for ob_ in obs:
                obc = self.get_obs_centers(ob_, carla)
                for obc_ in obc:
                    for selfc in self.get_obs_centers(self.X[:, i], carla):
                        # dist = ca.sqrt((selfc[0]-obc_[0])**2+\
                        #     (selfc[1]-obc_[1])**2)
                        dist = (selfc[0]-obc_[0])**2+(selfc[1]-obc_[1])**2
                        self.obj += (1/dist)**expn*ratio

    def soft_obs_apf(self, obs, ref_traj, ratio=370, expn=1, carla=False):
        obs_apf = 0
        for ob_ in obs:
            obc = self.get_obs_centers(ob_, carla)
            for obc_ in obc:
                for selfc in self.get_obs_centers(ref_traj[:, 0], carla):
                    # dist = ca.sqrt((selfc[0]-obc_[0])**2+\
                    #     (selfc[1]-obc_[1])**2)
                    dist = (selfc[0]-obc_[0])**2+(selfc[1]-obc_[1])**2
                    obs_apf += (1/dist)**expn*ratio
        
        return obs_apf

    # MPC Soler construction
    def solver_add_hard_obs(self, obs, carla=False):
        for i in range(self.horizon):
            for ob_ in obs:
                obc = self.get_obs_centers(ob_, carla)
                for obc_ in obc:
                    for selfc in self.get_obs_centers(self.X[:, i], carla):
                        dist = ca.sqrt((selfc[0]-obc_[0])**2 +
                                       (selfc[1]-obc_[1])**2)
                        self.g.append(dist)
                        self.ubg.append(np.inf)
                        self.lbg.append(0.5)


    def solver_add_c_road_pf(self, roads_pos, yaw=0, carla=False):
        '''
        [Right Coordinate System]
        Add the cost function for the crossable road lane
        Parameters:
            roads_pos : [y1, y2, y3, ...]
            yaw : yaw of the vehicle
        Returns:
            None
        '''
        for i in range(self.horizon):
            for road_pos in roads_pos:
                for selfc in self.get_obs_centers(self.X[:, i], carla):
                    selfc = [0, ca.sin(-yaw)*selfc[0]+ca.cos(-yaw)*selfc[1]]
                    dist = ca.fabs(selfc[1] - road_pos)

                # Standard Road Lane PF
                self.obj = self.obj + ca.if_else(
                    dist < 0.5,
                    self.ac_r * (dist-1)**2,
                    0)
                

    def c_road_pf(self, roads_pos, ref_traj, yaw=0, carla=False):
        road_pf = 0
        for road_pos in roads_pos:
            for selfc in self.get_obs_centers(ref_traj[:, 0], carla):
                selfc = [0, np.sin(-yaw)*selfc[0]+np.cos(-yaw)*selfc[1]]
                dist = np.fabs(selfc[1] - road_pos)

            # Standard Road Lane PF
            if dist < 0.5:
                road_pf += self.ac_r * (dist-1)**2

        return road_pf


    def solver_add_nc_road_pf(self, roads_pos, yaw=0, carla=False):
        '''
        [Right Coordinate System]
        Add the cost function for the noncrossable road lane
        Parameters:
            roads_pos : [y1, y2, y3, ...]
            yaw : yaw of the vehicle
        Returns:
            None
        '''
        for i in range(self.horizon):
            for road_pos, dir in roads_pos:
                for selfc in self.get_obs_centers(self.X[:, i], carla):
                    selfc = [0, (ca.sin(-yaw)*selfc[0]+ca.cos(-yaw)*selfc[1])]
                    dist = ca.fabs(selfc[1] - road_pos)
                    
                    ## Use reciprocal PF
                    self.obj = self.obj + \
                        ca.if_else(
                            dist < 1.5,
                            ca.if_else(dir == 1,
                                ca.if_else(selfc[1] > road_pos-0.2, 
                                        1000, 
                                        self.anc_r * (1/ca.fabs(selfc[1] - road_pos))**2),
                                ca.if_else(selfc[1] < road_pos+0.2, 
                                        1000, 
                                        self.anc_r * (1/ca.fabs(selfc[1] - road_pos))**2)
                                ),
                            0
                        )
        

    def nc_road_pf(self, roads_pos, ref_traj, yaw=0, carla=False):
        nc_road_pf = 0
  
        for road_pos, dir in roads_pos:
            for selfc in self.get_obs_centers(ref_traj[:, 0], carla):
                selfc = [0, (np.sin(-yaw)*selfc[0]+np.cos(-yaw)*selfc[1])]
                dist = np.fabs(selfc[1] - road_pos)
                
                ## Use reciprocal PF
                if dist < 1.5:
                    if dir == 1:
                        if selfc[1] > road_pos-0.2:
                            nc_road_pf += 1000
                        else:
                            nc_road_pf += self.anc_r * (1/np.fabs(selfc[1] - road_pos))**2
                    else:
                        if selfc[1] < road_pos+0.2:
                            nc_road_pf += 1000
                        else:
                            nc_road_pf += self.anc_r * (1/np.fabs(selfc[1] - road_pos))**2
        
        return nc_road_pf
                    

    def solver_add_single_tr_lgt_pf(self, light_pos):
        '''
        Add traffic light cost function
        Parameters:
            light_pos : [x] 
        '''
        for i in range(self.horizon):
            dist = -(self.X[:, i][0] - light_pos)
            dist_l = 1.5 - self.X[:, i][1]
            dist_r = self.X[:, i][1] + 1.5
            
            self.obj = self.obj + 200*(1/(dist)) + 1000*(1/(dist_l))**2 + 1000*(1/(dist_r))**2 
    

    def solver_add_single_tr_lgt_pf_carla(self, lane_center_y, yaw, tl_x):
        '''
        Add traffic light cost function in Carla
        Parameters: 
            lane_center_y : y coordinate of the lane center
            yaw : yaw of the lane center point
        Returns:
            None
        '''
        for i in range(self.horizon):
            selfc = [ca.cos(-yaw)*self.X[:, i][0] - ca.sin(-yaw)*self.X[:, i][1],
                     ca.sin(-yaw)*self.X[:, i][0] + ca.cos(-yaw)*self.X[:, i][1]]
            dist = -(selfc[0] - tl_x) + 1.5
            dist_l = 1.5 - (selfc[1]-lane_center_y)
            dist_r = (selfc[1]-lane_center_y) + 1.5

            self.obj += 200*(1/(dist)) + 1000*(1/(dist_l))**2 + 1000*(1/(dist_r))**2

    def traffic_pf(self, ref_traj, lane_center_y, yaw, tl_x):
        traffic_pf = 0
 
        selfc = [np.cos(-yaw)*ref_traj[0, 0] - np.sin(-yaw)*ref_traj[1, 0],
                    np.sin(-yaw)*ref_traj[0, 0] + np.cos(-yaw)*ref_traj[1, 0]]
        dist = -(selfc[0] - tl_x) + 2
        dist_l = 1.5 - (selfc[1]-lane_center_y)
        dist_r = (selfc[1]-lane_center_y) + 1.5

            # traffic_pf += 200*(1/(dist)) + 1000*(1/(dist_l))**2 + 1000*(1/(dist_r))**2
        traffic_pf = 200*(1/(dist))
        return traffic_pf
        

    def solver_add_bounds(self, obs=np.array([])):
        nlp_prob = {'f': self.obj, 'x': self.opt_variables, 'p': self.P,
                    'g': ca.vertcat(*self.g)}

        opts_setting = {'ipopt.max_iter': self.maxiter, 'ipopt.print_level': 0, 'print_time': 0,
                        'ipopt.acceptable_tol': 1e-8, 'ipopt.acceptable_obj_change_tol': 1e-6}

        self.solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts_setting)

        for _ in range(self.horizon+1):
            for _ in range(self.n_states):
                self.lbg.append(0.0)
                self.ubg.append(0.0)


        for _ in range((self.horizon+1)*obs.shape[0]*2):
            self.lbg.append(0)
            self.lbg.append(0)

            self.ubg.append(np.inf)
            self.ubg.append(np.inf)

        for _ in range(self.horizon):
            self.lbx.append(self.acc_lbound)
            self.lbx.append(-self.steer_bound)

            self.ubx.append(self.acc_ubound)
            self.ubx.append(self.steer_bound)

        for _ in range(self.horizon+1):
            self.lbx.append(-np.inf)
            self.lbx.append(-np.inf)
            self.lbx.append(-np.inf)
            self.lbx.append(-self.target_v)
            self.lbx.append(-np.inf)
            self.lbx.append(-np.inf)

            self.ubx.append(np.inf)
            self.ubx.append(np.inf)
            self.ubx.append(np.inf)
            self.ubx.append(self.target_v)
            self.ubx.append(np.inf)
            self.ubx.append(np.inf)

    '''
    MPC solver without initialized u_opt from last iteration of MPC
    '''
    def solve_MPC_wo(self, z_ref, z0, a_opt, delta_opt):
        xs = z_ref
        x0 = z0
        u0 = np.array([a_opt, delta_opt] *
                      self.horizon).reshape(-1, self.n_controls).T
        x_m = np.array(x0*(self.horizon+1)).reshape(-1, self.n_states).T

        c_p = np.vstack((x0, xs)).T
        init_control = np.concatenate((u0.reshape(-1, 1), x_m.reshape(-1, 1)))
        # unconstraint
        # res = self.solver(x0=init_control, p=c_p)

        # constraint
        res = self.solver(x0=init_control, p=c_p, lbg=self.lbg, lbx=self.lbx,
                          ubg=self.ubg, ubx=self.ubx)
        estimated_opt = res['x'].full()

        size_u0 = self.horizon*self.n_controls
        u0 = estimated_opt[:size_u0].reshape(self.horizon, self.n_controls)
        x_m = estimated_opt[size_u0:].reshape(self.horizon+1, self.n_states)

        if self.carla:
            return u0[0, 0], u0[0, 1], x_m
        else:
            return u0, x_m

    '''
    MPC solver with initialized u_opt from last iteration of MPC
    '''
    def solve_MPC(self, z_ref, z0, n_states, u0):
        xs = z_ref
        x_m = n_states

        c_p = np.vstack((z0.T, xs)).T
        init_control = np.concatenate((u0.reshape(-1, 1), x_m.reshape(-1, 1)))

        # unconstraint
        # res = self.solver(x0=init_control, p=c_p)

        # constraint
        start_time = time.time()
        res = self.solver(x0=init_control, p=c_p, lbg=self.lbg, lbx=self.lbx,
                          ubg=self.ubg, ubx=self.ubx)
        cost_time = time.time()-start_time
        estimated_opt = res['x'].full()

        size_u0 = self.horizon*self.n_controls
        u0 = estimated_opt[:size_u0].reshape(self.horizon, self.n_controls)
        x_m = estimated_opt[size_u0:].reshape(self.horizon+1, self.n_states)

        if self.carla:
            return u0[0, 0], u0[0, 1], x_m, cost_time
        else:
            return u0, x_m, cost_time
