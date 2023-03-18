import sys
import glob
import os
from scripts.env import *
import cv2

try:
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'official'))
except IndexError:
    pass

from official.behavior_agent import BehaviorAgent
from official.basic_agent import BasicAgent

log_level = logging.DEBUG 
logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

env = Env()
state = env.reset()
env.clock.tick()
env.world.tick()

traffic_manager = env.client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)
agent = BasicAgent(env.ego_vehicle)
spawn_points = env.map.get_spawn_points()
destination = random.choice(spawn_points).location
logging.info('Current state is {}'.format(env.get_state()))
logging.info('The destination is {}'.format(destination))

agent.set_destination(destination)


for _ in tqdm(range(500)):
    control = agent.run_step()
    control.manual_gear_shift = False
    logging.info('{}'.format(control))
    logging.info('Current state is {}'.format(env.get_state()))
    state = env.step(control)
    event = pygame.event.poll()
    if event.type == pygame.QUIT:
        pygame.quit()
        exit()

pygame.quit()

traffic_manager.set_synchronous_mode(False)
