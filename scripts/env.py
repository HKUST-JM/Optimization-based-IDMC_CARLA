from collections import deque

import numpy as np
import random
import atexit
import logging
import os
import sys
import time

try:
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
except IndexError:
    pass

from tqdm import tqdm

try:
    import queue  # python3
except ImportError:
    import Queue as queue  # python2

import math
import carla
import cv2
from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions
from others_agent import OthersAgent

DT_ = None  # [s] delta time step, = 1/FPS_in_server
NO_RENDERING = False
BEV_RES_X = 200
RES_X = 720
RES_Y = 480

START_TIME = 3

DISPLAY_METHOD= 'spec' # 'pygame' or 'spec'

if DISPLAY_METHOD == 'pygame':
    import pygame
    pygame.init()

    def get_font():
        fonts = [x for x in pygame.font.get_fonts()]
        default_font = 'ubuntumono'
        font = default_font if default_font in fonts else fonts[0]
        font = pygame.font.match_font(font)
        return pygame.font.Font(font, 14)


    def display_image(surface, image_array, blend=False):
        image_surface = pygame.surfarray.make_surface(image_array.swapaxes(0, 1))
        if blend:
            image_surface.set_alpha(100)
        surface.blit(image_surface, (0, 0))


def draw_waypoints(world, waypoints, z=0.5, color=(255, 0, 0)):
    color = carla.Color(r=color[0], g=color[1], b=color[2], a=255)
    for w in waypoints:
        t = w.transform
        begin = t.location + carla.Location(z)
        world.debug.draw_points(begin, size=0.05, color=color, life_time=0.1)


class PIDAccelerationController():
    """
    PIDAccelerationController implements acceleration control using a PID.
    """

    def __init__(self, vehicle, K_P=1.0, K_I=0.0, K_D=0.0, dt=0.03):
        """
        Constructor method.

            :param vehicle: actor to apply to local planner logic onto
            :param K_P: Proportional term
            :param K_D: Differential term
            :param K_I: Integral term
            :param dt: time differential in seconds
        """
        self._vehicle = vehicle
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt
        self._error_buffer = deque(maxlen=10)

    def run_step(self, target_acc, debug=False):
        """
        Execute one step of acceleration control to reach a given target speed.

            :param target_acceleration: target acceleration in Km/h
            :param debug: boolean for debugging
            :return: throttle control
        """
        current_acc = self.get_acc()

        if debug:
            print('Current acceleration = {}'.format(current_acc))

        # print('err', current_acc, target_acc, target_acc-current_acc)
        return self._pid_control(target_acc, current_acc)

    def _pid_control(self, target_acc, current_acc):
        """
        Estimate the throttle/brake of the vehicle based on the PID equations

            :param target_speed:  target speed in Km/h
            :param current_speed: current speed of the vehicle in Km/h
            :return: throttle/brake control
        """
        error = target_acc - current_acc
        self._error_buffer.append(error)

        if len(self._error_buffer) >= 2:
            _de = (self._error_buffer[-1] - self._error_buffer[-2]) / self._dt
            _ie = sum(self._error_buffer) * self._dt
        else:
            _de = 0.0
            _ie = 0.0

        return np.clip((self._k_p * error) + (self._k_d * _de) + (self._k_i * _ie), -1.0, 1.0)

    def change_parameters(self, K_P, K_I, K_D, dt):
        """Changes the PID parameters"""
        self._k_p = K_P
        self._k_i = K_I
        self._k_d = K_D
        self._dt = dt

    def get_acc(self):
        # direction flag, 1: forward, -1: backward
        flag = 1

        yaw = np.radians(self._vehicle.get_transform().rotation.yaw)
        ax = self._vehicle.get_acceleration().x
        ay = self._vehicle.get_acceleration().y 
        acc_yaw = math.atan2(ay, ax)
        error = acc_yaw - yaw
        if error > math.pi:
            error -= 2 * math.pi
        elif error < -math.pi:
            error += 2 * math.pi
        error = math.fabs(error)
        if error > math.pi / 2:
            flag = -1

        return flag * np.sqrt(ax**2+ay**2)*0.1


class Env:
    def __init__(self, host="localhost", port=2000, map_id='05', birdeye_view=False, 
                 display_method='spec',recording=True, dt=0.05) -> None:
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.map = self.world.get_map()

        map_name = 'Town'+map_id
        if str(self.map).split('/')[-1][:-1] == map_name:
            logging.info('Already load map {}, skipped'.format(map_name))
        else:
            self.world = self.client.load_world(map_name)

        self.blueprint_library = self.world.get_blueprint_library()
        self.ego_vehicle_type = self.blueprint_library.filter("model3")[0]
        self.ego_vehicle = None

        global DT_, DISPLAY_METHOD
        DT_ = dt
        DISPLAY_METHOD = display_method
        # exit with cleaning all actors
        atexit.register(self.clean)

        self.original_settings = self.world.get_settings()
        # world in sync mode
        self.world.apply_settings(carla.WorldSettings(
            no_rendering_mode=NO_RENDERING,  # False for debug
            synchronous_mode=True,
            fixed_delta_seconds=DT_))

        # birdeye view setting informationget_obsget_obs
        self.birdview_on = birdeye_view
        if birdeye_view:
            self.font_res_x = BEV_RES_X
            PG_RES_X = BEV_RES_X + RES_X
            self.birdview_producer = BirdViewProducer(
                self.client,  # carla.Client
                target_size=PixelDimensions(width=BEV_RES_X, height=RES_Y),
                pixels_per_meter=6,
                crop_type=BirdViewCropType.FRONT_AND_REAR_AREA
            )
        else:
            self.font_res_x = 0
            PG_RES_X = RES_X

        # pygame setting
        if DISPLAY_METHOD == 'pygame':
            self.pg_display = pygame.display.set_mode(
                (PG_RES_X, RES_Y),
                pygame.HWSURFACE | pygame.DOUBLEBUF)
            self.font = get_font()
            self.clock = pygame.time.Clock()
            if self.recording:
                self.videoWriter = cv2.VideoWriter('mapping.mp4', cv2.VideoWriter_fourcc('M', 'P', 'E', 'G'),\
                                24, (BEV_RES_X+RES_X, RES_Y)) # fps
        elif DISPLAY_METHOD == 'spec':
            self.spectator = self.world.get_spectator()
        else:
            raise ValueError('display_method should be pygame or spec')
            exit()

        # update if spawn vehicles
        self.spawn_others_bool = False
        self.other_vehicles_auto = False

        self.recording = recording


    def reset(self, ego_transform=None):
        """
        initial environment
        ego_vehicle can be tranformed on the position by setting ego_transform

        ego_transform: carla.Transform
        """
        self.actor_list = []
        self.other_vehicles_list = []

        # choose a random point for generation
        if ego_transform is None:
            spawn_point = random.choice(
                self.world.get_map().get_spawn_points())
        else:
            spawn_point = ego_transform

        if self.ego_vehicle is None:
            self.ego_vehicle = self.world.spawn_actor(
                self.ego_vehicle_type, spawn_point)
            self.actor_list.append(self.ego_vehicle)
        else:
            self.ego_vehicle.set_transform(spawn_point)
        
        self._acc_controller = PIDAccelerationController(self.ego_vehicle, K_P=0.28, K_I=0.01, K_D=0.001, dt=DT_)

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

        self.time = 0

        self.world.tick()
        if DISPLAY_METHOD == 'pygame':
            self.clock.tick()

    def spawn_other_vehicles(self, transform_list, auto_mode=False):
        bp_lib = self.world.get_blueprint_library()
        bp_types = []
        self.spawn_other_agents = []
        self.other_vehicles_auto = auto_mode

        # bp_types.extend(bp_lib.filter('vehicle.nissan.*'))
        bp_types.extend(bp_lib.filter('vehicle.audi.*'))
        # bp_types.extend(bp_lib.filter('vehicle.tesla.model3'))

        for vehicle_i in range(len(transform_list)):
            bp = random.choice(bp_types)
            # if bp.has_attribute('color'):
            #     color = random.choice(bp.get_attribute('color').recommended_values)
            #     bp.set_attribute('color', color) Javier Alonso-Mora

            transform = transform_list[vehicle_i]
            vehicle = self.world.spawn_actor(bp, transform)
            self.world.tick()
            if vehicle is not None:
                if auto_mode:
                    vehicle.set_autopilot(True)
                    self.other_vehicles_list.append(vehicle)
                else:
                    agent = OthersAgent(vehicle)
                    self.spawn_other_agents.append(agent)
                    self.other_vehicles_list.append(vehicle)
            else:
                pass
               
        self.spawn_others_bool = True

    def update_other_agents(self):
        for agent in self.spawn_other_agents:
            control = agent.run_step()
            if control is None:
                control = carla.VehicleControl()

            if isinstance(control, carla.VehicleControl):
                agent._vehicle.apply_control(control)
                steer_, throttle_, brake_ = control.steer, control.throttle, control.brake
            else:
                if self.time >= START_TIME:  # starting time
                    steer_, throttle_, brake_ = action
                else:
                    steer_ = 0
                    throttle_ = 0.5
                    brake_ = 0

                agent._vehicle.apply_control(carla.VehicleControl(throttle=float(
                    throttle_), steer=float(steer_*20), brake=float(brake_)))

    def get_obs(self, cur_state=None, dist_threshold=20):
        obs = []
        heights = []
        for car in self.other_vehicles_list:
            location = car.get_location()
            transform = car.get_transform()
            phi = transform.rotation.yaw * np.pi / 180

            if cur_state is None:
                obs.append([location.x, -location.y, -phi])
            else:
                dist = np.sqrt((location.x-cur_state[0])**2 + (-location.y-cur_state[1])**2)
                if dist < dist_threshold:
                    obs.append([location.x, -location.y, -phi])
            heights.append(location.z)
        
        return obs, heights

    def get_state(self):
        """
        get the vehicle's state, TODO to fit our dynamic model
        """
        self.location = self.ego_vehicle.get_location()
        self.location_ = np.array(
            [self.location.x, self.location.y, self.location.z])

        self.transform = self.ego_vehicle.get_transform()
        phi = self.transform.rotation.yaw * np.pi / 180

        self.velocity = self.ego_vehicle.get_velocity()
        vx = self.velocity.x
        vy = self.velocity.y

        beta_candidate = np.arctan2(
            vy, vx) - phi + np.pi*np.array([-2, -1, 0, 1, 2])
        local_diff = np.abs(beta_candidate - 0)
        min_index = np.argmin(local_diff)
        beta = beta_candidate[min_index]

        # state = [self.velocity.x, self.velocity.y, self.yaw, self.angular_velocity.z]
        state = [
            self.location.x,  # x
            self.location.y,  # y
            np.sqrt(vx**2 + vy**2),  # v
            phi,  # phi
            beta,  # beta
        ]

        return np.array(state)

    def step(self, state, transform_mode=False):
        """
        excute one step of simulation

        state: if state is action, it's type is carla.VehicleControl or [steer_, throttle_, brake_]
               other is the transform data(carla.Transform or [x,y,z,roll,pitch,yaw])
        """
        if not transform_mode:
            steer_ = state[1] / 0.6

            for i in range(1):
                throttle_tmp = self._acc_controller.run_step(state[0])

                if throttle_tmp >= 0:
                    throttle_ = throttle_tmp
                    brake_ = 0
                    reverse_ = False
                else:
                    throttle_ = abs(throttle_tmp)
                    brake_ = 0
                    reverse_ = True

                self.ego_vehicle.apply_control(carla.VehicleControl(
                        throttle=float(throttle_), steer=float(-steer_), brake=float(brake_), reverse=reverse_))
            # self.world.tick(DT_/1)
        else:
            steer_ = state[1]
            accelerate_ = state[0]
            new_state = state[2]
            height = new_state[1]
            transform = new_state[0]
            self.ego_vehicle.set_transform(carla.Transform(carla.Location(
                x=transform[0], y=-transform[1], z=height), carla.Rotation(yaw=-np.degrees(transform[2]))))

        if self.spawn_others_bool:
            if not self.other_vehicles_auto:
                self.update_other_agents()

        if DISPLAY_METHOD == 'pygame':
            self.clock.tick()
        self.world.tick()
        self.time += DT_

        image_rgb = self.image_queue.get()
        image_array = np.reshape(np.frombuffer(image_rgb.raw_data, dtype=np.dtype('uint8')),
                                 (image_rgb.height, image_rgb.width, 4))
        image_array = image_array[:, :, :3]
        image_array = image_array[:, :, ::-1]

        if self.birdview_on:
            birdview = self.birdview_producer.produce(
                agent_vehicle=self.ego_vehicle  # carla.Actor (spawned vehicle)
            )
            rgb = BirdViewProducer.as_rgb(birdview)
            image_array = np.concatenate((rgb, image_array), axis=1)

            if self.recording:
                self.videoWriter.write(image_array[:,:,::-1])            

        if not transform_mode:
            vel = self.ego_vehicle.get_velocity()
            vel = np.sqrt(vel.x**2 + vel.y**2)
        else:
            vel = transform[3]

        if DISPLAY_METHOD == "pygame":
            display_image(self.pg_display, image_array)
            self.pg_display.blit(
                self.font.render(
                    'Velocity = {0:.2f} m/s'.format(vel), True, (255, 255, 255)),
                (self.font_res_x+8, 10))

            # pygame text display
            v_offset = 25
            bar_h_offset = self.font_res_x+75
            bar_width = 100
            if not transform_mode:
                display_item = {"steering": steer_,
                                "throttle": throttle_, "brake": brake_}
                for key, value in display_item.items():
                    rect_border = pygame.Rect(
                        (bar_h_offset, v_offset + 8), (bar_width, 6))
                    pygame.draw.rect(self.pg_display, (255, 255, 255), rect_border, 1)
                    if key == "steering":
                        rect = pygame.Rect((bar_h_offset + (1+value) * (bar_width)/2, v_offset + 8), (6, 6))
                    else:
                        rect = pygame.Rect(
                            (bar_h_offset + value * (bar_width), v_offset + 8), (6, 6))
                    pygame.draw.rect(self.pg_display, (255, 255, 255), rect)
                    self.pg_display.blit(self.font.render(
                        key, True, (255, 255, 255)), (self.font_res_x+8, v_offset+3))

                    v_offset += 18
            else:
                self.pg_display.blit(
                self.font.render(
                    'Accelerating = {0:.2f} m/s2'.format(accelerate_), True, (255, 255, 255)),
                (self.font_res_x+8, 28))

                self.pg_display.blit(
                self.font.render(
                    'Steering = {0:.2f} °'.format(-steer_*180/np.pi), True, (255, 255, 255)),
                (self.font_res_x+8, 46))
            
            pygame.display.flip()

            event = pygame.event.poll()
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        elif DISPLAY_METHOD == "spec":
            transform = self.ego_vehicle.get_transform()
            yaw = transform.rotation.yaw
            # self.spectator.set_transform(carla.Transform(transform.location + carla.Location(z=1.1), carla.Rotation(yaw=yaw, pitch=0)))
            self.spectator.set_transform(carla.Transform(transform.location + carla.Location(z=30), carla.Rotation(pitch=-90)))

    def clean(self):
        """
        restore carla's settings and destroy all created actors when the program exits
        """
        self.world.apply_settings(self.original_settings)
        logging.info('destroying actors')
        self.client.apply_batch([carla.command.DestroyActor(x)
                                for x in self.actor_list])
        self.client.apply_batch([carla.command.DestroyActor(x)
                                for x in self.other_vehicles_list])
        logging.info('done')

    def clean_all_actors(self):
        actors = self.world.get_actors().filter("*vehicle*" or "*walker*")
        for actor in actors:
            actor.destroy()


if __name__ == '__main__':
    log_level = logging.DEBUG
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    env = Env()
    env.reset()

    for _ in tqdm(range(500)):
        action = np.random.rand(3)
        action[0] = (action[0]-0.5) * 2
        action[1] = 1
        action[2] = 0

        env.step(action)
        if DISPLAY_METHOD == "pygame":
            event = pygame.event.poll()
            if event.type == pygame.QUIT:
                print(event.type)
                pygame.quit()
                exit()
    
    if DISPLAY_METHOD == "pygame":
        pygame.quit()
