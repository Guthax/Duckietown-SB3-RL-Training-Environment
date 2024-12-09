import os
import argparse

import gym
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import config
from gym_duckietown.src.gym_duckietown.envs.duckietown_env import DuckietownEnv
from test_gym import CustomGymEnv

config.set_config("16")

from stable_baselines3 import PPO, DDPG, SAC

from utils import VideoRecorder, parse_wrapper_class
from carla_env.state_commons import create_encode_state_fn, load_vae
from carla_env.rewards import reward_functions

from vae.utils.misc import LSIZE
from carla_env.wrappers import vector, get_displacement_vector
from carla_env.envs.carla_route_env import CarlaRouteEnv
from eval_plots import plot_eval, summary_eval


from config import CONFIG

from gym_duckietown.learning.utils.env import launch_env
from gym_duckietown.learning.utils.wrappers import NormalizeWrapper, ImgWrapper, DtRewardWrapper, ActionWrapper, \
    ResizeWrapper, MultiInputWrapper

"""
def run_eval(env, model, model_path=None, record_video=False):
    model_name = os.path.basename(model_path)
    log_path = os.path.join(os.path.dirname(model_path), 'eval')
    os.makedirs(log_path, exist_ok=True)
    video_path = os.path.join(log_path, model_name.replace(".zip", "_eval.avi"))
    csv_path = os.path.join(log_path, model_name.replace(".zip", "_eval.csv"))
    model_id = f"{model_path.split('/')[-2]}-{model_name.split('_')[-2]}"
    # vec_env = model.get_env()
    state = env.reset()
    rendered_frame = env.render(mode="rgb_array")

    columns = ["model_id", "episode", "step", "throttle", "steer", "vehicle_location_x", "vehicle_location_y",
               "reward", "distance", "speed", "center_dev", "angle_next_waypoint", "waypoint_x", "waypoint_y",
               "route_x", "route_y"]
    df = pd.DataFrame(columns=columns)

    # Init video recording
    if record_video:
        print("Recording video to {} ({}x{}x{}@{}fps)".format(video_path, *rendered_frame.shape,
                                                              int(env.fps)))
        video_recorder = VideoRecorder(video_path,
                                       frame_size=rendered_frame.shape,
                                       fps=env.fps)
        video_recorder.add_frame(rendered_frame)
    else:
        video_recorder = None

    episode_idx = 0
    # While non-terminal state
    print("Episode ", episode_idx)
    saved_route = False
    while episode_idx < 4:
        env.extra_info.append("Evaluation")
        action, _states = model.predict(state, deterministic=True)
        state, reward, dones, info = env.step(action)
        if env.step_count >= 150 and env.current_waypoint_index == 0:
            dones = True

        # Save route at the beginning of the episode
        if not saved_route:
            initial_heading = np.deg2rad(env.vehicle.get_transform().rotation.yaw)
            initial_vehicle_location = vector(env.vehicle.get_location())
            # Save the route to plot them later
            for way in env.route_waypoints:
                route_relative = get_displacement_vector(initial_vehicle_location,
                                                         vector(way[0].transform.location),
                                                         initial_heading)
                new_row = pd.DataFrame([['route', env.episode_idx, route_relative[0], route_relative[1]]],
                                       columns=["model_id", "episode", "route_x", "route_y"])
                df = pd.concat([df, new_row], ignore_index=True)
            saved_route = True

        vehicle_relative = get_displacement_vector(initial_vehicle_location, vector(env.vehicle.get_location()),
                                                   initial_heading)
        waypoint_relative = get_displacement_vector(initial_vehicle_location,
                                                    vector(env.current_waypoint.transform.location), initial_heading)

        new_row = pd.DataFrame(
            [[model_id, env.episode_idx, env.step_count, env.vehicle.control.throttle, env.vehicle.control.steer,
              vehicle_relative[0], vehicle_relative[1], reward,
              env.distance_traveled,
              env.vehicle.get_speed(), env.distance_from_center,
              np.rad2deg(env.vehicle.get_angle(env.current_waypoint)),
              waypoint_relative[0], waypoint_relative[1], None, None
              ]], columns=columns)
        df = pd.concat([df, new_row], ignore_index=True)

        # Add frame
        rendered_frame = env.render(mode="rgb_array")
        if record_video:
            video_recorder.add_frame(rendered_frame)
        if dones:
            state = env.reset()
            episode_idx += 1
            saved_route = False
            print("Episode ", episode_idx)

    # Release video
    if record_video:
        video_recorder.release()

    df.to_csv(csv_path, index=False)
    plot_eval([csv_path])
    summary_eval(csv_path)
"""
def create_duckietown():
    env = DuckietownEnv(
        seed=123,  # random seed
        map_name="loop_empty",
        max_steps=500001,  # we don't want the gym to reset itself
        domain_rand=False,
        camera_width=640,
        camera_height=480,
        accept_start_angle_deg=4,  # start close to straight
        full_transparency=True,
        distortion=True,
    )

    return env

def load_duckietown_env(vae):
    env = create_duckietown()
    print("Initialized environment")

    # Wrappers
    env = ResizeWrapper(env)
    env = MultiInputWrapper(vae=vae, env=env)
    print("Initialized Wrappers")
    return env
if __name__ == "__main__":
    model_path = "gym_duckietown/models/PPO_1733497542_id17/model_100000_steps.zip"

    algorithm_dict = {"PPO": PPO, "DDPG": DDPG, "SAC": SAC}
    if CONFIG["algorithm"] not in algorithm_dict:
        raise ValueError("Invalid algorithm name")

    AlgorithmRL = algorithm_dict[CONFIG["algorithm"]]

    if CONFIG["algorithm"] not in algorithm_dict:
        raise ValueError("Invalid algorithm name")

    vae = load_vae(f'./vae/log_dir/{CONFIG["vae_model"]}', LSIZE)
    low, high = [-1, 0], [1,1]
    obs = gym.spaces.Dict({
        "rgb_camera": gym.spaces.Box(low=0, high=255, shape=(3, 80, 160), dtype=np.uint8),
        "vae_latent": gym.spaces.Box(low=-4, high=4, shape=(64, ), dtype=np.float32)
        #"vehicle_measures": gym.spaces.Box(low=np.array(low), high=np.array(high),shape=(2, ), dtype=np.float32)
    })

    environment = load_duckietown_env(vae)
    #print(environment.observation_space["rgb_camera"].shape)
    #print(environment.observation_space["vae_latent"].shape)
    #print(environment.observation_space["vehicle_measures"].shape)

    #print(obs["rgb_camera"].shape)
    #print(obs["vae_latent"].shape)
    #print(obs["vehicle_measures"].shape)
    action = gym.spaces.Box(np.array([-1, -1]), np.array([1, 1]), dtype=np.float32)
    model = AlgorithmRL.load(model_path, env=environment,custom_objects={'observation_space': obs, 'action_space': action}, device='cpu')

    obs = environment.reset()
    done = False
    last_img = None
    while True:
        while not done:
            action, _states = model.predict(obs, deterministic=False)
            print(action)
            action = action[::-1]
            action[1] = action[1] * -1
            state, reward, dones, info = environment.step(action)
            current_img = state["rgb_camera"]
            if np.array_equal(current_img,last_img):
                print("True")
            last_img = current_img
            # Perform action
            #img = state["rgb_camera"]
            #plt.imshow(img)
            #@plt.show()
            environment.render()
        done = False
        obs = environment.reset()

