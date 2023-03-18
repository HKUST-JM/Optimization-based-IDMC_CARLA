import carla
import random

# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
world = client.get_world()

# client.load_world('Town03')

# Retrieve the spectator object
spectator = world.get_spectator()

spectator.set_transform(carla.Transform(carla.Location(x=0, y=-0, z=50),
                                                    carla.Rotation(pitch=-90)))

world = client.get_world()
env_objs = world.get_environment_objects(carla.CityObjectLabel.Buildings)

objects_to_toggle = set()
for env_obj in env_objs:
    objects_to_toggle.add(env_obj.id)
world.enable_environment_objects(objects_to_toggle, False)

rp_flag = False
car_spawn_flag = True

if rp_flag == True:
    p1_location = carla.Transform(carla.Location(x=0, y=10, z=20), carla.Rotation(pitch=-90))
    # carla.DebugHelper.draw_point(p1_location, size=0.1, color=(255,0,0), life_time=0)
    # p1
    client.get_world().debug.draw_point(carla.Location(15., 15., 0.), size=0.05, life_time=0)
    # p2
    client.get_world().debug.draw_point(carla.Location(-15., 15., 0.), size=0.05, life_time=0)
    # p3
    client.get_world().debug.draw_point(carla.Location(15., -15., 0.), size=0.05, life_time=0)
    # p4
    client.get_world().debug.draw_point(carla.Location(-15., -15., 0.), size=0.05, life_time=0)
    # p5
    client.get_world().debug.draw_point(carla.Location(10., 30, 0.), size=0.05, life_time=0)
    # p6
    client.get_world().debug.draw_point(carla.Location(30., 10, 0.), size=0.05, life_time=0)

if car_spawn_flag == True:
    blueprint_library = world.get_blueprint_library()
    ego_vehicle_bp = blueprint_library.find('vehicle.mini.cooper_s')
    spawn_choices = world.get_map().get_spawn_points()

    print(spawn_choices)