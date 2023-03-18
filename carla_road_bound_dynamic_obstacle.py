import sys
import os

try:
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'official'))
except IndexError:
    pass

from scripts.x_v2x_agent import Xagent
from scripts.env import *
from scripts.vehicle_v2x_obs import Vehicle
from utils.draw_result import plot_result


# logging 
log_level = logging.DEBUG 
logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

# init environment
env = Env(map_id='03', birdeye_view=False)
settings = env.world.get_settings()
settings.no_rendering_mode = True
env.world.apply_settings(settings)


# random ego_vehicle's start position 
spawn_points = env.world.get_map().get_spawn_points()
transform = random.choice(spawn_points)
env.reset(transform)

# dynamic model(for MPC control)
dynamic_model = Vehicle(actor=env.ego_vehicle, target_v=30)

# allocate Agent
agent = Xagent(env, dynamic_model, dt=0.1)

# spawn other vehicles


def spawn_vehicles_around_ego_vehicles(ego_vehicle_transform, radius, spawn_points, numbers_of_vehicles):
    # parameters:
    # ego_vehicle :: your target vehicle
    # radius :: the distance limitation between ego-vehicle and other free-vehicles
    # spawn_points :: the available spawn points in current map
    # numbers_of_vehicles :: the number of free-vehicles around ego-vehicle that you need
    np.random.shuffle(spawn_points) # shuffle all the spawn points
    ego_location = ego_vehicle_transform.location
    accessible_points = []
    for spawn_point in spawn_points:
        dis = math.sqrt((ego_location.x-spawn_point.location.x)**2 + (ego_location.y-spawn_point.location.y)**2)
        # it also can include z-coordinate,but it is unnecessary
        if dis < radius and dis > 5:
            accessible_points.append(spawn_point)

    transform_list = [] # keep the spawned vehicle in vehicle_list, because we need to link them with traffic_manager
    if len(accessible_points) < numbers_of_vehicles:
        # if your radius is relatively small,the satisfied points may be insufficient
        numbers_of_vehicles = len(accessible_points)

    for i in range(numbers_of_vehicles): # generate the free vehicle
        point = accessible_points[i]
        transform_list.append(point)

    return transform_list

transform_list = spawn_vehicles_around_ego_vehicles(transform, 300, spawn_points, 50)
env.spawn_other_vehicles(transform_list, auto_mode=True)

def generate_destination(ego_vehicle_transform):
    # generate destination
    destination = random.choice(spawn_points)
    while ego_vehicle_transform.location.distance(destination.location) < 200:
        print(ego_vehicle_transform.location.distance(destination.location))
        destination = random.choice(spawn_points)
    return destination

agent.plan_route(transform, generate_destination(transform))

for _ in range(10000):
    new_state = agent.run_step()
    env.step(new_state, transform_mode=True)
    event = pygame.event.poll()
    if event.type == pygame.QUIT:
        pygame.quit()
        exit()

pygame.quit()

# np.savetxt('test.txt', agent._log_data)
# plot_result(np.array(agent._log_data))

