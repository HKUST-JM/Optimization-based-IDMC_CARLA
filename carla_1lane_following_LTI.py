import sys
import os

try:
    sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'official'))
except IndexError:
    pass

from official.behavior_agent import BehaviorAgent
from official.basic_agent import BasicAgent
from scripts.others_agent import OthersAgent
from scripts.x_agent_carla import Xagent
from scripts.env import *
from scripts.lti_v2x import Vehicle
from scenarios.highway import transform_list, conj_list
from utils.draw_result import plot_result


# logging 
log_level = logging.DEBUG 
logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

# init environment
env = Env(birdeye_view=True)

# choose ego_vehicle's start position
start_position = conj_list[-2][1]

# set initial position in carla
env.reset(carla.Transform(start_position, carla.Rotation(yaw=5)))

# dynamic model(for MPC control)
dynamic_model = Vehicle(actor=env.ego_vehicle, target_v=30)

# allocate Agent
agent = Xagent(env, dynamic_model, dt=0.05)

# spawn other vehicles in Highway 
# env.spawn_other_vehicles(transform_list)

for _ in range(400):
    new_state = agent.run_step()
    env.step(new_state)
    event = pygame.event.poll()
    if event.type == pygame.QUIT:
        pygame.quit()
        exit()

pygame.quit()

# np.savetxt('test.txt', agent._log_data)
plot_result(np.array(agent._log_data))