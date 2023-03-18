import sys
# if 'win32' in sys.platform:
#     sys.path.append('C:\\Users\\chenk\\Desktop\\WindowsNoEditor\\PythonAPI\\carla\\dist')
# else:
#     assert 'incorrect system version'

# ref https://blog.csdn.net/qq_34809033/article/details/106208006#:~:text=python%E4%B8%ADwindow%E5%92%8Clinux%E4%B8%8B%E8%B7%AF%E5%BE%84%E5%85%BC%E5%AE%B9%201%20%E4%BD%BF%E7%94%A8os.path.join%20path%20%3D%20os.path.join%28os.path.split%28os.path.realpath%28__file__%29%29%5B0%5D%2C%20%27cmds%27%29%201,path%20%3D%20path.replace%28%27%5C%5C%27%2C%20%27%2F%27%29%201%204%20%E4%BD%BF%E7%94%A8%E6%9C%80%E6%96%B0%E7%9A%84pathlib%E6%A8%A1%E5%9D%97%20
# ref https://blog.csdn.net/wohu1104/article/details/125710603
import glob
import os
from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions

from scripts.env import *
import cv2


log_level = logging.DEBUG 
logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

env = Env()
state = env.reset()

birdview_producer = BirdViewProducer(
    env.client,  # carla.Client
    target_size=PixelDimensions(width=150, height=336),
    pixels_per_meter=4,
    crop_type=BirdViewCropType.FRONT_AND_REAR_AREA
)

for _ in tqdm(range(500)):
    action = np.random.rand(3)
    action[0] = (action[0]-0.5) * 2
    action[1] = 1
    action[2] = 0

    state = env.step(action)
    birdview = birdview_producer.produce(
        agent_vehicle=env.ego_vehicle  # carla.Actor (spawned vehicle)
    )
    rgb = BirdViewProducer.as_rgb(birdview)

    cv2.namedWindow('birdeye-view', cv2.WINDOW_NORMAL)
    cv2.imshow('birdeye-view', rgb)
    cv2.waitKey(10)

    event = pygame.event.poll()
    if event.type == pygame.QUIT:
        print(event.type)
        pygame.quit()
        exit()
    # pygame.display.flip()
# env.world.destroy()
cv2.destroyAllWindows()
pygame.quit()