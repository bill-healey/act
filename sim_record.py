import os
import h5py
import numpy as np
import time
from dm_control import mujoco, viewer
from dm_control.rl import control
from teleop import TeleOpHandler
from plot_handler import PlotHandler
from constants import DT, XML_DIR, SIM_TASK_CONFIGS
from velocity_pickup_task import PickupTask


def test_sim_teleop(record=True):
    task_config = SIM_TASK_CONFIGS['sim_pickup_task']
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
    xml_path = os.path.join(XML_DIR, task_config['mujoco_xml'])
    physics = mujoco.Physics.from_xml_path(xml_path)
    task = PickupTask()
    teleop_handler = TeleOpHandler()
    teleop_handler.start()
    plot_handler = PlotHandler(camera_names)
    success = []
    env = control.Environment(physics, task, time_limit=episode_len, control_timestep=DT,
                              n_sub_steps=None, flat_observation=False)
    #ts = env.reset()
    #viewer.launch(environment_loader=lambda: env)
    for episode_idx in range(num_episodes):
        ts = env.reset()
        episode = [ts]
        action = np.zeros(task_config['action_len'])
        for t in range(episode_len):
            action = task.control_input_to_action(teleop_handler, action)
            ts = env.step(action)
            episode.append(ts)
            plot_handler.render_images(ts.observation['images'])
        if record:
            episode_return = np.sum([ts.reward for ts in episode[1:]])
            episode_max_reward = np.max([ts.reward for ts in episode[1:]])
            success_status = episode_max_reward == env.task.max_reward
            success.append(int(success_status))

            print(
                f"{episode_idx=} {'Successful' if success_status else 'Failed'} with reward max {episode_max_reward} sum {episode_return}")

            # Truncate episode
            episode = episode[:-1]

            data_dict = {
                '/observations/qpos': [],
                '/observations/qvel': [],
                '/action': [],
                **{f'/observations/images/{cam}': [] for cam in camera_names}
            }

            for ts in episode:
                data_dict['/observations/qpos'].append(ts.observation['qpos'])
                data_dict['/observations/qvel'].append(ts.observation['qvel'])
                data_dict['/action'].append(ts.observation['qpos'])
                for cam in camera_names:
                    data_dict[f'/observations/images/{cam}'].append(ts.observation['images'][cam])

            # HDF5
            succeeded_text = 'success' if success_status else 'fail'
            files = [f for f in os.listdir(dataset_dir) if f.endswith('.hdf5') and f'episode_{succeeded_text}' in f]
            next_index = max([int(f.split(f'_{succeeded_text}_')[1].split('.')[0]) for f in files], default=-1) + 1
            dataset_path = os.path.join(dataset_dir, f'episode_{succeeded_text}_{next_index}.hdf5')
            with h5py.File(dataset_path, 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                root.attrs['sim'] = True
                obs = root.create_group('observations')
                image = obs.create_group('images')
                for cam in camera_names:
                    image.create_dataset(cam, (episode_len, 480, 640, 3), dtype='uint8', chunks=(1, 480, 640, 3))
                obs.create_dataset('qpos', (episode_len, task_config['action_len']))
                obs.create_dataset('qvel', (episode_len, task_config['action_len']))
                root.create_dataset('action', (episode_len, task_config['action_len']))
                for name, array in data_dict.items():
                    root[name][...] = array
            del episode
    teleop_handler.stop()
    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')


if __name__ == '__main__':
    test_sim_teleop(record=True)
