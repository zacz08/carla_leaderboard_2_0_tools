#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Welcome to the capture sensor data script, a script that provides users with a baseline for data collection,
which they can later modify to their specific needs, easying the process of creating a database.

This script will start with a CARLA recorder log, spawning the desired sensor configuration at the ego vehicle,
and saving their data into a folder. The exact parameters can be found at the very top of the script and include:

- SENSORS: List of all the sensors tha will be spawned in the simulation
- WEATHER: Weather of the simulation
- RECORDER_INFO: List of all the CARLA recorder logs that will be run. Each recorder has four elements:
    Â· folder: path to the folder with the recorder files
    Â· name: name of the endpoint folder
    Â· start_time: start time of the recorder
    Â· duration: duration of the recorder. 0 to replay it until the end
- DESTINATION_FOLDER: folder where all sensor data will be stored
"""

import time
import os
import carla
import argparse
import random
import json
import threading
import glob

from queue import Queue, Empty
from leaderboard.envs.sensor_interface import SensorInterface
# from srunner.scenariomanager.timer import GameTime

################### User simulation configuration ####################
import numpy as np
import shutil
from data_tools.data_record_agent import DataRecorder

# from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
# from carla_birdeye_view import BirdViewProducer, BirdViewCropType, PixelDimensions


batch_generate = False
# 1) Choose the sensors
rgb_sensor_data = {"width": 800, "height": 600, "fov": 100}
SENSORS = [
    [
        'rgb_front',
        {
            'bp': 'sensor.camera.rgb',
            'image_size_x': rgb_sensor_data["width"], 
            'image_size_y': rgb_sensor_data["height"], 
            'fov': rgb_sensor_data["fov"],
            'x': 1.3, 'y': 0.0, 'z': 2.3, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0
        },
    ],
    [
        'rgb_rear',
        {
            'bp': 'sensor.camera.rgb',
            'image_size_x': rgb_sensor_data["width"], 
            'image_size_y': rgb_sensor_data["height"], 
            'fov': rgb_sensor_data["fov"],
            'x': -1.3, 'y': 0.0, 'z': 2.3, 'roll': 0.0, 'pitch': 0.0, 'yaw': 180.0
        },
    ],
    [
        'rgb_left',
        {
            'bp': 'sensor.camera.rgb',
            'image_size_x': rgb_sensor_data["width"], 
            'image_size_y': rgb_sensor_data["height"], 
            'fov': rgb_sensor_data["fov"],
            'x': 1.3, 'y': 0.0, 'z': 2.3, 'roll': 0.0, 'pitch': 0.0, 'yaw': -60.0
        },
    ],
    [
        'rgb_right',
        {
            'bp': 'sensor.camera.rgb',
            'image_size_x': rgb_sensor_data["width"], 
            'image_size_y': rgb_sensor_data["height"], 
            'fov': rgb_sensor_data["fov"],
            'x': 1.3, 'y': 0.0, 'z': 2.3, 'roll': 0.0, 'pitch': 0.0, 'yaw': 60.0
        },
    ],
    [
        'lidar',
        {
            'bp': 'sensor.lidar.ray_cast',
            'x': 1.3, 'y': 0.0, 'z': 2.5, 'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
            'range': 85, 'rotation_frequency': 10, 'channels': 64, 'upper_fov': 10,
            'lower_fov': -30
            # 'points_per_second': 600000, 'atmosphere_attenuation_rate': 0.004,
            # 'dropoff_general_rate': 0.45, 'dropoff_intensity_limit': 0.8, 'dropoff_zero_intensity': 0.4
        },
    ],
    [
        'gps',
        {
            'bp': 'sensor.other.gnss',
            'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            # 'sensor_tick': 0.01
            # 'noise_alt_stddev': 0.000005, 'noise_lat_stddev': 0.000005, 'noise_lon_stddev': 0.000005,
            # 'noise_alt_bias': 0.0, 'noise_lat_bias': 0.0, 'noise_lon_bias': 0.0
        }
    ],
    [
        'imu',
        {
            'bp': 'sensor.other.imu',
            'x': 0.0, 'y': 0.0, 'z': 0.0, 'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
            # 'sensor_tick': 0.05
            # 'noise_accel_stddev_x': 0.001, 'noise_accel_stddev_y': 0.001, 'noise_accel_stddev_z': 0.015,
            # 'noise_gyro_stddev_x': 0.001,'noise_gyro_stddev_y': 0.001, 'noise_gyro_stddev_z': 0.001
        }
    ],
]

# 2) Choose a weather
WEATHER = carla.WeatherParameters(
    sun_azimuth_angle=-1.0, sun_altitude_angle=70.0,
    cloudiness=30.0, precipitation=0.0, precipitation_deposits=80.0, wetness=15.0,
    wind_intensity=10.0,
    fog_density=2.0, fog_distance=0.0, fog_falloff=0.0)

# 3) Choose the recorder files

RECORDER_INFO = [
    # {
    #     'folder': "/home/zc/carla/lb2/leaderboard/data_tools/ScenarioLogs/Accident",
    #     # 'folder': "RouteLogs/0",
    #     'name': 'Accident',
    #     'start_time': 10,
    #     'duration': 20
    # },
    # {
    #     'folder': "/home/zc/carla/lb2/leaderboard/data_tools/ScenarioLogs/DynamicObjectCrossing_right_container",
    #     # 'folder': "RouteLogs/0",
    #     'name': 'DynamicObjectCrossing_right_container',
    #     'start_time': 5,
    #     'duration': 20
    # },
    {
        'folder': "/home/zc/carla/lb2/leaderboard/data_tools/ScenarioLogs/SignalizedJunctionLeftTurn_fast",
        # 'folder': "RouteLogs/0",
        'name': 'SignalizedJunctionLeftTurn_fast',
        'start_time': 5,
        'duration': 20
    }
]

# 4) Choose the destination folder
DESTINATION_FOLDER = "/home/zc/datasets/lb2_lmdrive"
################# End user simulation configuration ##################

FPS = 20
THREADS = 5
CURRENT_THREADS = 0
AGENT_TICK_DELAY = 10
END_POINT = ''


def create_folders(endpoint, sensors):
    for sensor_id, sensor_bp in sensors:
        sensor_endpoint = f"{endpoint}/{sensor_id}"
        if not os.path.exists(sensor_endpoint):
            os.makedirs(sensor_endpoint)

        if 'gnss' in sensor_bp:
            sensor_endpoint = f"{endpoint}/{sensor_id}/gnss_data.csv"
            with open(sensor_endpoint, 'w') as data_file:
                data_txt = f"Frame,Altitude,Latitude,Longitude\n"
                data_file.write(data_txt)

        if 'imu' in sensor_bp:
            sensor_endpoint = f"{endpoint}/{sensor_id}/imu_data.csv"
            with open(sensor_endpoint, 'w') as data_file:
                data_txt = f"Frame,Accelerometer X,Accelerometer y,Accelerometer Z,Compass,Gyroscope X,Gyroscope Y,Gyroscope Z\n"
                data_file.write(data_txt)

    # Create 3d_bbs folder
    folder = f"{endpoint}/3d_bbs"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Create lidar temp folder
    folder = f"{endpoint}/lidar_temp"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Create affordances folder
    folder = f"{endpoint}/affordances"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Create actors_data folder
    folder = f"{endpoint}/actors_data"
    if not os.path.exists(folder):
        os.makedirs(folder)

    # Create measurements folder
    folder = f"{endpoint}/measurements"
    if not os.path.exists(folder):
        os.makedirs(folder)


def add_listener(sensor, sensor_queue, sensor_id):
    sensor.listen(lambda data: sensor_listen(data, sensor_queue, sensor_id))


def sensor_listen(data, sensor_queue, sensor_id):
    sensor_queue.put((sensor_id, data.frame, data))
    return


def get_ego_id(recorder_file):
    found_lincoln = False
    found_id = None

    for line in recorder_file.split("\n"):

        # Check the role_name for hero
        if found_lincoln:
            if not line.startswith("  "):
                found_lincoln = False
                found_id = None
            else:
                data = line.split(" = ")
                if 'role_name' in data[0] and 'hero' in data[1]:
                    return found_id

        # Search for all lincoln vehicles
        if not found_lincoln and line.startswith(" Create ") and 'vehicle.lincoln' in line:
            found_lincoln = True
            found_id =  int(line.split(" ")[2][:-1])

    return found_id


def save_data_to_disk(sensor_id, frame, data, imu_data, endpoint):
    """
    Saves the sensor data into file:
    - 3D Bounding Box               ->              '.npy', one per frame, named as the frame id
    - Images                        ->              '.png', one per frame, named as the frame id
    - Lidar:                        ->              '.ply', one per frame, named as the frame id
    - SemanticLidar:                ->              '.ply', one per frame, named as the frame id
    - RADAR:                        ->              '.csv', one per frame, named as the frame id
    - GNSS:                         ->              '.csv', one line per frame, named 'gnss_data.csv'
    - IMU:                          ->              '.csv', one line per frame, named 'imu_data.csv'

    """
    global CURRENT_THREADS
    CURRENT_THREADS += 1
    if isinstance(data, carla.Image):
        sensor_endpoint = f"{endpoint}/{sensor_id}/{frame:04d}.jpg"
        data.save_to_disk(sensor_endpoint)

    elif isinstance(data, carla.LidarMeasurement):
        sensor_endpoint = f"{endpoint}/lidar_temp/{frame:04d}.npy"
        data.save_to_disk(sensor_endpoint)
        with open(sensor_endpoint, 'r') as file:
            # read and skip header, get number of vertices
            header = []
            num_vertices = 0
            line = file.readline().strip()
            while line != 'end_header':
                if line.startswith('element vertex'):
                    num_vertices = int(line.split()[2])
                line = file.readline().strip()

            # get vertices array
            vertices = []
            for _ in range(num_vertices):
                line = file.readline().strip()
                vertex = [float(coord) for coord in line.split()]
                vertices.append(vertex)
            vertex_array = np.array(vertices)

        # save lidar numpy file
        lidar_endpoint = f"{endpoint}/{sensor_id}/{frame:04d}.npy"
        np.save(lidar_endpoint, vertex_array)

    elif isinstance(data, carla.SemanticLidarMeasurement):
        sensor_endpoint = f"{endpoint}/{sensor_id}/{frame}.ply"
        data.save_to_disk(sensor_endpoint)

    elif isinstance(data, carla.GnssMeasurement):
        sensor_endpoint = f"{endpoint}/{sensor_id}/gnss_data.csv"
        with open(sensor_endpoint, 'a') as data_file:
            data_txt = f"{frame},{data.altitude},{data.latitude},{data.longitude}\n"
            data_file.write(data_txt)

    elif isinstance(data, carla.IMUMeasurement):
        sensor_endpoint = f"{endpoint}/{sensor_id}/imu_data.csv"
        with open(sensor_endpoint, 'a') as data_file:
            data_txt = f"{frame},{imu_data[0][0]},{imu_data[0][1]},{imu_data[0][2]},{data.compass},{imu_data[1][0]},{imu_data[1][1]},{imu_data[1][2]}\n"
            data_file.write(data_txt)

    else:
        print(f"WARNING: Ignoring sensor '{sensor_id}', as no callback method is known for data of type '{type(data)}'.")

    CURRENT_THREADS -= 1


def add_agent_delay(recorder_log):
    """
    The agent logs are delayed from the simulation recorder, which depends on the leaderboard setup.
    As the vehicle is stopped at the beginning, fake them with all 0 values, and the initial transform
    """

    init_tran = recorder_log['records'][0]['state']['transform']
    for _ in range(AGENT_TICK_DELAY):

        elem = {}
        elem['control'] = {
            'brake': 0.0, 'gear': 0, 'hand_brake': False, 'manual_gear_shift': False,
            'reverse': False, 'steer': 0.0, 'throttle': 0.0
        }
        elem['state'] = {
            'acceleration': {'value': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0},
            'angular_velocity': { 'value': 0.0, 'x': -0.0, 'y': 0.0, 'z': 0.0},
            'velocity': {'value': 0.0, 'x': 0.0, 'y': 0.0, 'z': 0.0},
            'transform': {
                'pitch': init_tran['pitch'], 'yaw': init_tran['yaw'], 'roll': init_tran['roll'],
                'x': init_tran['x'], 'y': init_tran['y'], 'z': init_tran['z']
            }
        }
        recorder_log['records'].insert(0, elem)

    return recorder_log


def extract_imu_data(recorder_logs):

    records = recorder_logs['records']
    log_data = []

    for record in records:
        acceleration_data = record['state']['acceleration']
        acceleration_vector = [acceleration_data['x'], acceleration_data['y'], acceleration_data['z']]

        # TODO: Remove this (don't use logs without angular velocity)
        if 'angular_velocity' in record['state']:
            angular_data = record['state']['angular_velocity']
            angular_vector = [angular_data['x'], angular_data['y'], angular_data['z']]
        else:
            angular_vector = [random.random(), random.random(), random.random()]

        log_data.append([acceleration_vector, angular_vector])

    return log_data


def save_recorded_data(endpoint, info, logs, start, duration):
    captured_logs = logs['records'][int(FPS*start):int(FPS*(start + duration))]
    saved_logs = {"records": captured_logs}

    with open(f'{endpoint}/ego_logs.json', 'w') as fd:
        json.dump(saved_logs, fd, indent=4)
    with open(f'{endpoint}/sensors.json', 'w') as fd:
        json.dump(SENSORS, fd, indent=4)
    with open(f'{endpoint}/simulation.json', 'w') as fd:
        simulation_info = info
        simulation_info.pop('name')
        simulation_info['input_data'] = simulation_info.pop('folder')
        simulation_info['weather'] = {
            'sun_azimuth_angle': WEATHER.sun_azimuth_angle, 'sun_altitude_angle': WEATHER.sun_altitude_angle,
            'cloudiness': WEATHER.cloudiness, 'wind_intensity': WEATHER.sun_azimuth_angle,
            'precipitation': WEATHER.precipitation, 'precipitation_deposits': WEATHER.precipitation_deposits, 'wetness': WEATHER.wetness,
            'fog_density':WEATHER.fog_density, 'fog_distance': WEATHER.fog_distance, 'fog_falloff': WEATHER.fog_falloff,
        }
        json.dump(simulation_info, fd, indent=4)


def set_endpoint(recorder_info):
    def get_new_endpoint(endpoint):
        i = 2
        new_endpoint = endpoint + "_" + str(i)
        while os.path.isdir(new_endpoint):
            i += 1
            new_endpoint = endpoint + "_" + str(i)
        return new_endpoint

    endpoint = f"{DESTINATION_FOLDER}/{recorder_info['name']}"
    if os.path.isdir(endpoint):
        old_endpoint = endpoint
        endpoint = get_new_endpoint(old_endpoint)
        print(f"\033[93mWARNING: Given endpoint already exists, changing {old_endpoint} to {endpoint}\033[0m")

    os.makedirs(endpoint)
    return endpoint


def save_additional_data(endpoint, frame, bbox, affordances, actors_data, measurements):
    # save bbox
    bbox_endpoint = f"{endpoint}/3d_bbs/{frame:04d}.npy"
    np.save(bbox_endpoint, bbox)

    # save affordances
    affordances_endpoint = f"{endpoint}/affordances/{frame:04d}.npy"
    np.save(affordances_endpoint, affordances)

    # save actors_data
    actors_data_endpoint = f"{endpoint}/actors_data/{frame:04d}.json"
    with open(actors_data_endpoint, 'w') as file:
        json.dump(actors_data, file, indent=4)

    # save measurements
    meas_endpoint = f"{endpoint}/measurements/{frame:04d}.json"
    with open(meas_endpoint, 'w') as file:
        json.dump(measurements, file, indent=4)


def capture_data():

    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--host', default='127.0.0.1', help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument('--port', default=2000, type=int, help='TCP port to listen to (default: 2000)')
    args = argparser.parse_args()
    print(__doc__)

    active_sensors = []
    # sensor_interface = SensorInterface()

    try:

        # Initialize the simulation
        client = carla.Client(args.host, args.port)
        client.set_timeout(120.0)
        world = client.get_world()

        for recorder_info in RECORDER_INFO:

            print(f"\n\033[1m> Getting the recorder information\033[0m")
            recorder_folder = recorder_info['folder']
            recorder_start = recorder_info['start_time']
            recorder_duration = recorder_info['duration']

            # recorder_path_list = glob.glob(f"{os.getcwd()}/{recorder_folder}/*.log")
            recorder_path_list = glob.glob(f"{recorder_folder}/*.log")
            if recorder_path_list:
                recorder_path = recorder_path_list[0]
                print(f"\033[1m> Running recorder '{recorder_path}'\033[0m")
            else:
                print(f"\033[91mCouldn't find the recorder file for the folder '{recorder_folder}'\033[0m")
                continue

            endpoint = set_endpoint(recorder_info)
            END_POINT = endpoint

            print(f"\033[1m> Preparing the world. This may take a while...\033[0m")
            
            # location of log praser cache
            cache_path = endpoint + '/cache.txt'
            # cache_path = endpoint + '/cache.json'
            with open(cache_path, 'w') as f:
                f.write(client.show_recorder_file_info(recorder_path, show_all=True))

            recorder_str = client.show_recorder_file_info(recorder_path, False)
            recorder_map = recorder_str.split("\n")[1][5:]
            world = client.load_world(recorder_map)
            world.tick()

            world.set_weather(WEATHER)
            settings = world.get_settings()
            settings.fixed_delta_seconds = 1 / FPS
            settings.synchronous_mode = True
            world.apply_settings(settings)

            world.tick()

            max_duration = float(recorder_str.split("\n")[-2].split(" ")[1])
            if recorder_duration == 0:
                recorder_duration = max_duration
            elif recorder_start + recorder_duration > max_duration:
                print("\033[93mWARNING: Found a duration that exceeds the recorder length. Reducing it...\033[0m")
                recorder_duration = max_duration - recorder_start
            if recorder_start >= max_duration:
                print("\033[93mWARNING: Found a start point that exceeds the recoder duration. Ignoring it...\033[0m")
                continue

            recorder_log_list = glob.glob(f"{os.getcwd()}/{recorder_folder}/log.json")
            recorder_log_path = recorder_log_list[0] if recorder_log_list else None
            if recorder_log_path:
                with open(recorder_log_path) as fd:
                    recorder_log = json.load(fd)
                recorder_log = add_agent_delay(recorder_log)
                imu_logs = extract_imu_data(recorder_log)
                save_recorded_data(endpoint, recorder_info, recorder_log, recorder_start, recorder_duration)
            else:
                imu_logs = None

            client.replay_file(recorder_path, recorder_start, recorder_duration, get_ego_id(recorder_str), False)
            # with open(f"{recorder_path[:-4]}.txt", 'w') as fd:
            #     fd.write(recorder_str)
            world.tick()

            hero = None
            while hero is None:
                possible_vehicles = world.get_actors().filter('vehicle.*')
                for vehicle in possible_vehicles:
                    if vehicle.attributes['role_name'] == 'hero':
                        hero = vehicle
                        break
                time.sleep(1)

            print(f"\033[1m> Creating the sensors\033[0m")
            create_folders(endpoint, [[s[0], s[1].get('bp')] for s in SENSORS])
            blueprint_library = world.get_blueprint_library()
            sensor_queue = Queue()
            for sensor in SENSORS:

                # Extract the data from the sesor configuration
                sensor_id, attributes = sensor
                blueprint_name = attributes.get('bp')
                sensor_transform = carla.Transform(
                    carla.Location(x=attributes.get('x'), y=attributes.get('y'), z=attributes.get('z')),
                    carla.Rotation(pitch=attributes.get('pitch'), roll=attributes.get('roll'), yaw=attributes.get('yaw'))
                )

                # Get the blueprint and add the attributes
                blueprint = blueprint_library.find(blueprint_name)
                for key, value in attributes.items():
                    if key in ['bp', 'x', 'y', 'z', 'roll', 'pitch', 'yaw']:
                        continue
                    blueprint.set_attribute(str(key), str(value))

                # Create the sensors and its callback
                sensor = world.spawn_actor(blueprint, sensor_transform, hero)
                add_listener(sensor, sensor_queue, sensor_id)
                active_sensors.append(sensor)

            for _ in range(10):
                world.tick()

            print(f"\033[1m> Running the replayer\033[0m")
            start_time = world.get_snapshot().timestamp.elapsed_seconds
            start_frame = world.get_snapshot().frame
            sensor_amount = len(SENSORS)

            max_threads = THREADS
            results = []

            ## init LMDrive data record agent
            save_agent = DataRecorder(world, hero)

            sensor_interface = SensorInterface()
            # input_data = sensor_interface.get_data(frame_diff)
            # save_agent._vehicle = hero
            # save_agent._init()
            # save_agent.birdview_producer = BirdViewProducer(
            #     client,  # carla.Client
            #     target_size=PixelDimensions(width=400, height=400),
            #     pixels_per_meter=4,
            #     crop_type=BirdViewCropType.FRONT_AND_REAR_AREA,
            #     )

            while True:
                current_time = world.get_snapshot().timestamp.elapsed_seconds
                current_duration = current_time - start_time
                if current_duration >= recorder_duration:
                    print(f">>>>>  Running recorded simulation: 100.00%  completed  <<<<<")
                    break

                completion = format(round(current_duration / recorder_duration * 100, 2), '3.2f')
                print(f">>>>>  Running recorded simulation: {completion}%  completed  <<<<<", end="\r")

                # Get and save the sensor data from the queue.
                missing_sensors = sensor_amount
                data_dict = {}
                save_add_data =False
                frame_diff = int(0)
                while True:

                    frame = world.get_snapshot().frame
                    try:
                        sensor_data = sensor_queue.get(True, 5.0)
                        if sensor_data[1] != frame: continue  # Ignore previous frame data
                        missing_sensors -= 1
                    except Empty:
                        raise ValueError("A sensor took too long to send their data")

                    # Get the data
                    sensor_id = sensor_data[0]
                    frame_diff = sensor_data[1] - start_frame
                    data = sensor_data[2]
                    imu_data = [[0,0,0], [0,0,0]] if not imu_logs else imu_logs[int(FPS*recorder_start + frame_diff)]
                    
                    # Add sensor data to sensor_data_dict
                    data_dict[sensor_id] = ((frame_diff, data))

                    res = threading.Thread(target=save_data_to_disk, 
                                           args=(
                                               sensor_id, 
                                               frame_diff, 
                                               data, 
                                               imu_data, 
                                               endpoint
                                               ))
                    results.append(res)
                    res.start()

                    if CURRENT_THREADS > max_threads:
                        for res in results:
                            res.join()
                        results = []

                    if missing_sensors <= 0:
                        save_add_data = True
                        break

                ################### ADD ###################
                if save_add_data:
                    # get 3d bbox, affordances, actors_data, measurements
                    bb_info = save_agent._get_3d_bbs()
                    aff_info = save_agent._get_affordances()
                    actors_info = save_agent.collect_actor_data()
                    measurements = save_agent.get_measurement(data_dict)

                    # save additional data of LMDrive
                    save_additional_data(endpoint, frame_diff, bb_info, aff_info, actors_info, measurements)
                ################### ADD ###################

                world.tick()

            for res in results:
                res.join()

            for sensor in active_sensors:
                sensor.stop()
                sensor.destroy()
            active_sensors = []

            for _ in range(50):
                world.tick()

    # End the simulation
    finally:
        # stop and remove cameras
        for sensor in active_sensors:
            sensor.stop()
            sensor.destroy()

        # set fixed time step length
        settings = world.get_settings()
        settings.fixed_delta_seconds = None
        settings.synchronous_mode = False
        world.apply_settings(settings)

        # delete temp files
        tmpdir = END_POINT + '/lidar_temp'
        shutil.rmtree(tmpdir)

if __name__ == '__main__':

    log_file_root = "/home/zc/carla/lb2/leaderboard/data_tools/ScenarioLogs/"
    
    if batch_generate:
        scenario_list = []
        for f in os.listdir(log_file_root):
            RECORDER_INFO[0]['folder'] = log_file_root + f
            RECORDER_INFO[0]['name'] = f
            scenario_list.append(f)
        # print(scenario_list)
        try:
            capture_data()
        except KeyboardInterrupt:
            pass
        finally:
            print('\ndone.')
    else:
        try:
            capture_data()
        except KeyboardInterrupt:
            pass
        finally:
            print('\ndone.')