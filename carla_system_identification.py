import sys
import os

try:
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'official'))
except IndexError:
    pass

import math
import carla
import atexit
import random
import logging
import argparse
import numpy as np

try:
    import pygame
    pygame.init()
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import queue  # python3
except ImportError:
    import Queue as queue  # python2

from official.basic_agent import BasicAgent


host = 'localhost'
port = 2000

NO_RENDERING = False
DT_ = 1/20  # [s] delta time step, = 1/FPS_in_server
RES_X = 720
RES_Y = 480

log_level = logging.DEBUG
logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

def display_image(surface, image_array, blend=False):
    image_surface = pygame.surfarray.make_surface(image_array.swapaxes(0, 1))
    if blend:
        image_surface.set_alpha(100)
    surface.blit(image_surface, (0, 0))

class CarlaSystemIdentification(object):
    def __init__(self, host, port, map_name='Town03'):
        # Init Carla Environment
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)

        self.world = self.client.get_world()
        self.original_settings = self.world.get_settings()
        
        # world in sync mode
        self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=NO_RENDERING,  # False for debug
            synchronous_mode=True,
            fixed_delta_seconds=DT_))
        
        atexit.register(self.clean)
        # Get Carla Map
        self.map = self.world.get_map()

        # Load Carla Map
        if str(self.map).split('/')[-1][:-1] == map_name:
            logging.info('Already load map {}, skipped'.format(map_name))
        else:
            self.world = self.client.load_world(map_name)

        self.time = 0
        self.clock = pygame.time.Clock()
        self.actor_list = []

        self.log_data = []
    
    def init_ego_vehicle(self, transform, target_speed=100, type='model3'):
        self.start_transform = transform
        # Init Ego Vehicle 
        blueprint_library = self.world.get_blueprint_library()
        ego_vehicle_type = blueprint_library.filter(type)[0]
        self.ego_vehicle = self.world.spawn_actor(ego_vehicle_type, self.start_transform)
        self.agent = BasicAgent(self.ego_vehicle, target_speed=target_speed)
        self.add_destroy_list(self.ego_vehicle)
        
        if self.ego_vehicle is not None:
            # setting camera information
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            camera_bp.set_attribute('image_size_x', '{}'.format(RES_X))
            camera_bp.set_attribute('image_size_y', '{}'.format(RES_Y))
            # camera_bp.set_attribute('fov', '110')
            self.camera = self.world.spawn_actor(
                camera_bp,
                carla.Transform(carla.Location(x=-5.5, z=2.8),
                                carla.Rotation(pitch=-15)),
                attach_to=self.ego_vehicle)
            self.camera.image_size_x = 720
            self.camera.image_size_x = 480
            self.actor_list.append(self.camera)
            self.image_queue = queue.Queue()
            self.camera.listen(self.image_queue.put)

        self.pg_display = pygame.display.set_mode(
            (RES_X, RES_Y),
            pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.clock.tick()
        self.world.tick()

    def add_destroy_list(self, actor):
        self.actor_list.append(actor)
    
    def set_destination(self, end_transform, start_transform):
        self.end_transform = end_transform
        self.agent.set_destination(self.end_transform.location, start_transform.location)

    def clean(self):
        self.world.apply_settings(self.original_settings)
        logging.info('destroying actors')
        self.client.apply_batch([carla.command.DestroyActor(x) for x in self.actor_list])
        logging.info('done.')

    def get_spawn_points(self):
        return self.map.get_spawn_points()
    
    def get_speed(self):
        vel = self.ego_vehicle.get_velocity()
        return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2)

    def get_state_data(self):
        veh_location = self.ego_vehicle.get_transform().location
        veh_rotation = self.ego_vehicle.get_transform().rotation
        veh_angvel = self.ego_vehicle.get_angular_velocity()
        return veh_location.x, veh_location.y, np.radians(veh_rotation.yaw), self.get_speed(), 0, np.radians(veh_angvel.z)

    def log_state_data(self, control=None):
        if control is not None:
            if control.brake > 0:
                control.throttle = -control.brake
            u = [control.throttle, control.steer]
        else:
            u = [0, 0]
        state_data = self.get_state_data()
        data = [state_data[0], state_data[1], state_data[2], state_data[3], state_data[4], state_data[5], u[0], u[1]]
        self.log_data.append(data)
        
    
    def run_step(self, control=None):
        self.clock.tick()
        self.world.tick()
        self.time += DT_

        image_rgb = self.image_queue.get()
        image_array = np.reshape(np.frombuffer(image_rgb.raw_data, dtype=np.dtype('uint8')),
                                 (image_rgb.height, image_rgb.width, 4))
        image_array = image_array[:, :, :3]
        image_array = image_array[:, :, ::-1]
        display_image(self.pg_display, image_array)

        pygame.display.flip()

        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            print(event.type)
            pygame.quit()
            exit()

        self.agent.set_target_speed(35 * math.sin(self.time/3.14) + 40)
        if control is None:
            control = self.agent.run_step()
        self.ego_vehicle.apply_control(control)

        if self.time > 8:
            self.log_state_data(control)
    
    def reach_destination(self):
        if self.ego_vehicle.get_location().distance(self.end_transform.location) < 2.0:
            return True
        else:
            return False
    
    def save_npy(self):
        np.save('data.npy', np.array(self.log_data))


def collect_data():
    CSI = CarlaSystemIdentification(host, port)

    # Assign the start position and end position
    s_transform = carla.Transform(carla.Location(x=233.7, y=33, z=0.2),\
                                carla.Rotation(pitch=0.0, yaw=90, roll=0.0))
    e_transform = random.choice(CSI.get_spawn_points())

    # Spawn Ego Vehicle and set destinationss
    CSI.init_ego_vehicle(s_transform)
    CSI.set_destination(e_transform, s_transform)

    while True:
        CSI.run_step()
        if CSI.time > 200 or CSI.agent.done():
            print("Reach destination")    
            break

    CSI.save_npy()
    pygame.quit()


def fit_data():
    data = np.load('data.npy')
    X = data[:, :-2]
    U = data[:, -2:]
    lamb = 1e-9

    def linear_regression(x, u, lamb):
        """Estimates linear system dynamics
        x, u: data used in the regression
        lamb: regularization coefficient
        """
        # Want to solve W^* = argmin sum_i ||W^T z_i - y_i ||_2^2 + lamb ||W||_F,
        # with z_i = [x_i u_i] and W \in R^{n + d} x n
        Y = x[2 : x.shape[0], :]
        X = np.hstack((x[1 : (x.shape[0] - 1), :], u[1 : (x.shape[0] - 1), :]))

        Q = np.linalg.inv(np.dot(X.T, X) + lamb * np.eye(X.shape[1]))
        b = np.dot(X.T, Y)
        W = np.dot(Q, b)

        A = W.T[:, 0:6]
        B = W.T[:, 6:8]

        error_matrix = np.dot(X, W) - Y
        error_max = np.max(error_matrix, axis=0)
        error_min = np.min(error_matrix, axis=0)
        error = np.vstack((error_max, error_min))

        return A, B, error

    A, B, error = linear_regression(X, U, lamb)
    print("A: ", A)
    print("B: ", B)
    print("Error: ", error)

    np.save("matrix_A.npy", A)
    np.save("matrix_B.npy", B)


def test_data():
    try:
        matrix_A = np.load("matrix_A.npy")
        matrix_B = np.load("matrix_B.npy")
    except:
        print("Load matrix data fail, or we can run --fit first")
        return

    # init CarlaSystemIdentification
    CSI = CarlaSystemIdentification(host, port)

    # Assign the start position and end position
    s_transform = carla.Transform(carla.Location(x=233.7, y=33, z=0.2),\
                                carla.Rotation(pitch=0.0, yaw=90, roll=0.0))
    
    # test straight line
    # Spawn Ego Vehicle and set destinationss
    state_errors = []
    CSI.init_ego_vehicle(s_transform)
    time = 0
    while time < 30:
        prev_state = CSI.get_state_data()
        cur_u = [0.3, 0]
        predict_state = np.dot(matrix_A, prev_state) + np.dot(matrix_B, cur_u)
        control = carla.VehicleControl()
        control.steer = cur_u[1]
        if cur_u[0] >= 0:
            control.throttle = cur_u[0]
            control.brake = 0
        else:
            control.throttle = 0
            control.brake = -cur_u[0]
        control.hand_brake = False
        control.manual_gear_shift = False
        CSI.run_step(control)
        cur_state = CSI.get_state_data()
        state_errors.append(abs(cur_state - predict_state))
        time += DT_
    CSI.clean()
    print('max error: ', np.max(state_errors, axis=0))
    print('mean error: ', np.mean(state_errors, axis=0))
    print('min error: ', np.min(state_errors, axis=0))

    # # test Right turn
    # Spawn Ego Vehicle and set destinationss
    state_errors = []
    CSI.init_ego_vehicle(s_transform)
    time = 0
    while time < 30:
        prev_state = CSI.get_state_data()
        cur_u = [0.5, 0.7]
        predict_state = np.dot(matrix_A, prev_state) + np.dot(matrix_B, cur_u)
        

        control = carla.VehicleControl()
        control.steer = cur_u[1]
        if cur_u[0] >= 0:
            control.throttle = cur_u[0]
            control.brake = 0
        else:
            control.throttle = 0
            control.brake = -cur_u[0]
        control.hand_brake = False
        control.manual_gear_shift = False
        CSI.run_step(control)
        cur_state = CSI.get_state_data()
        state_errors.append(abs(cur_state - predict_state))
        time += DT_
    CSI.clean()
    print('max error: ', np.max(state_errors, axis=0))
    print('mean error: ', np.mean(state_errors, axis=0))
    print('min error: ', np.min(state_errors, axis=0))

def main():
    argparser = argparse.ArgumentParser(
        description="CARLA System Identification")
    argparser.add_argument(
        '-c', '--collect',
        action='store_true',
    ) 
    argparser.add_argument(
        '-f', '--fit',
        action='store_true',
    )
    argparser.add_argument(
        '-t', '--test',
        action='store_true',
    )
    args = argparser.parse_args()

    if args.collect:
        collect_data()
    if args.fit:
        fit_data()
    if args.test:
        test_data()

if __name__ == '__main__':
    main()
