import sys
import os

try:
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'official'))
except IndexError:
    pass

from scripts.x_v2x_agent import Xagent
from scripts.env import *
from scripts.vehicle_obs import Vehicle
from utils.draw_result import plot_result
from utils.carla_utils import spawn_vehicles_around_ego_vehicles
import random

simu_step = 0.05
gen_dis_max = 100
gen_dis_min = 10
target_v = 40
veh_num = 100

# logging 
log_level = logging.DEBUG 
logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

def generate_destination(ego_vehicle_transform):
    # generate destination
    destination = random.choice(spawn_points)
    while ego_vehicle_transform.location.distance(destination.location) < 200:
        # print(ego_vehicle_transform.location.distance(destination.location))
        destination = random.choice(spawn_points)
    return destination

# init environment
env = Env(map_id='03', display_method='spec', birdeye_view=False, dt=simu_step)
# clean all existing actors
env.clean_all_actors()
spawn_points = env.world.get_map().get_spawn_points()

# Senarios
senario = 'crossroad' # 'roundabout', 'multilaneACC', 'crossroad' and 'trafficControl'
# Roundabout
if senario is 'roundabout':
    start_transform = carla.Transform(carla.Location(x=1.93, y=109.09, z=0.27), carla.Rotation(pitch=0.0, yaw=-90.3625, roll=0.0))
    destination_transform = carla.Transform(carla.Location(x=-7.90, y=80.54, z=0.27), carla.Rotation(pitch=0.0, yaw=90.5864, roll=0.0))
    target_v = 35
elif senario is 'multilaneACC':
    start_transform = carla.Transform(carla.Location(x=-8.21, y=207.71, z=0.27), carla.Rotation(pitch=0.0, yaw=0.3625, roll=0.0))
    destination_transform = carla.Transform(carla.Location(x=238, y=110.7, z=0.27), carla.Rotation(pitch=0.0, yaw=-90.5864, roll=0.0))
elif senario is 'crossroad':
    traffic_lights = env.world.get_actors().filter('*traffic_light*')
    for traffic_light in traffic_lights:
        traffic_light.set_state(carla.TrafficLightState.Red)
        traffic_light.freeze(True)
    gen_dis_min = 80
    veh_num = 20
    target_v = 20
    start_transform = carla.Transform(carla.Location(x=8.914989, y=-96.977425, z=0.02), carla.Rotation(pitch=0.0, yaw=-88.5864, roll=0.0))
    destination_transform = carla.Transform(carla.Location(x=-46.57, y=-139.27, z=0.00), carla.Rotation(pitch=0.0, yaw=-180.5864, roll=0.0))
# elif senario is 'trafficControl':
#     traffic_lights = env.world.get_actors().filter('*traffic_light*')
#     for traffic_light in traffic_lights:
#         traffic_light.set_state(carla.TrafficLightState.Red)
#         traffic_light.freeze(True)
#     start_transform = carla.Transform(carla.Location(x=-10.03, y=47.31, z=0.02), carla.Rotation(pitch=0.0, yaw=92.00, roll=0.0))
#     destination_transform = carla.Transform(carla.Location(x=-11.42, y=207.50, z=0.27), carla.Rotation(pitch=0.0, yaw=-0.14, roll=0.0))
else:
    # random ego_vehicle's start position and destination
    start_transform = random.choice(spawn_points)
    destination_transform = generate_destination(start_transform)

# - Reset environment and the ego vehicle's position
env.reset(start_transform)

# - Dynamic model(for MPC control)
dynamic_model = Vehicle(actor=env.ego_vehicle, horizon=10, target_v=target_v, delta_t=simu_step)

# - Allocate Agent
agent = Xagent(env, dynamic_model, dt=simu_step)

# - Spawn other vehicles
transform_list = spawn_vehicles_around_ego_vehicles(start_transform, gen_dis_max, gen_dis_min, spawn_points, veh_num)
env.spawn_other_vehicles(transform_list, auto_mode=True)

# - Use internal A* to plan a route
print(start_transform, destination_transform)
agent.plan_route(start_transform, destination_transform)

# - Run simulation
cnt = 0
change_flag = True
# env.client.start_recorder(senario+'.log', True)
for _ in range(10000):
    try:
        new_state = agent.run_step()
        # - change traffic light
        if senario == 'crossroad':
            if cnt > 140 and change_flag:
                print('change traffic light')
                traffic_lights = env.world.get_actors().filter('*traffic_light*')
                for traffic_light in traffic_lights:
                    traffic_light.set_state(carla.TrafficLightState.Green)
                    traffic_light.freeze(True)
            
                change_flag = False
        env.step(new_state, transform_mode=False)
        cnt +=1
    except Exception as e:
        print(e)
        break

# env.client.stop_recorder()
# - save log data
np.savetxt(senario+'.txt', agent._log_data)
plot_result(senario)
