import os
import h5py
import numpy as np
import collections
import time
from dm_control import mujoco
from dm_control.rl import control
from dm_control.suite import base
from dm_env import specs
from teleop import TeleOpHandler
from plot_handler import PlotHandler
from constants import DT, XML_DIR, SIM_TASK_CONFIGS


class PickupTask(base.Task):
    def initialize_episode(self, physics):
        super().initialize_episode(physics)
        random_pos = np.array([np.random.uniform(0, .15), np.random.uniform(.25, .75), 0.05])
        physics.named.data.qpos['red_box_joint'][:3] = random_pos
        #self.action = np.array([0, -0.96, 1.16, 0, -0.3, 0, 0.02239, -0.02239])
        self.action = np.zeros(5)  # Initialize action array
        self.max_reward = 5

    def before_step(self, action, physics):
        self.action = action
        physics.set_control(action)  # Apply actions to the physics
        super().before_step(action, physics)

    def after_step(self, physics):
        super().after_step(physics)

    @staticmethod
    def get_qpos(physics):
        qpos_raw = physics.data.qpos.copy()
        return qpos_raw

    @staticmethod
    def get_qvel(physics):
        qvel_raw = physics.data.qvel.copy()
        return qvel_raw

    @staticmethod
    def get_env_state(physics):
        env_state = physics.data.qpos.copy()[5:]
        return env_state

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos(physics)
        obs['qvel'] = self.get_qvel(physics)
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        for camera in SIM_TASK_CONFIGS['sim_pickup_task']['camera_names']:
            obs['images'][camera] = physics.render(height=480, width=640, camera_id=camera)
        return obs

    def get_reward(self, physics):
        # return whether gripper is holding the box
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)
        touch_gripper = ("red_box", "vx300s_right/10_right_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs
        reward = 0
        if touch_gripper:
            reward = 1
        if touch_gripper and touch_table:
            reward = 2
        if touch_gripper and not touch_table:
            reward = 4
        table_z = physics.named.data.xpos['table'][2]
        red_box_size = 0.02  # The height of the box from the MJCF definition
        red_box_current_z = physics.named.data.xpos['box'][2]
        if touch_gripper and not touch_table and red_box_current_z > table_z + 5 * red_box_size:
            reward = 5
        return reward

    def action_spec(self, physics):
        # Define the action specification
        return specs.BoundedArray(
            shape=(5,), dtype=np.float32, minimum=-1.0, maximum=1.0, name='action')


def get_action(teleop_handler, action):
    teleop_actions = teleop_handler.get_actions()
    action = [
        action[0] + teleop_actions['waist_rotation'],
        action[1] + teleop_actions['shoulder_elevation'],
        action[2] + teleop_actions['wrist_elevation'],
        # Claw has two hinges so claw action must be duplicated.  Model handles inversion of second hinge.
        # Also clamp claw ctrlrange to match model limitations
        max(-.4, min(action[3] + teleop_actions['gripper_rotation'], .4)),
        max(-.4, min(action[4] - teleop_actions['gripper_rotation'], .4))
    ]
    return action


def test_sim_teleop(record=True):
    task_config = SIM_TASK_CONFIGS['sim_pickup_task']
    dataset_dir = task_config['dataset_dir']
    num_episodes = task_config['num_episodes']
    episode_len = task_config['episode_len']
    camera_names = task_config['camera_names']
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
    xml_path = os.path.join(XML_DIR, f'single_viperx_pickup_cube.xml')
    physics = mujoco.Physics.from_xml_path(xml_path)
    task = PickupTask()
    teleop_handler = TeleOpHandler()
    teleop_handler.start()
    plot_handler = PlotHandler(camera_names)
    success = []
    env = control.Environment(physics, task, time_limit=episode_len, control_timestep=DT,
                              n_sub_steps=None, flat_observation=False)
    for episode_idx in range(num_episodes):
        ts = env.reset()
        episode = [ts]
        action = np.zeros(6)
        for t in range(episode_len):
            action = get_action(teleop_handler, action)
            ts = env.step(action)
            episode.append(ts)
            plot_handler.render_images(ts.observation['images'])
        if record:
            episode_return = np.sum([ts.reward for ts in episode[1:]])
            episode_max_reward = np.max([ts.reward for ts in episode[1:]])
            if episode_max_reward == env.task.max_reward:
                success.append(1)
                print(f"{episode_idx=} Successful with reward max {episode_max_reward} sum {episode_return}")
            else:
                success.append(0)
                print(f"{episode_idx=} Failed with reward max {episode_max_reward} sum {episode_return}")

            joint_traj = [ts.observation['qpos'] for ts in episode]
            # because the replaying, there will be eps_len + 1 actions and eps_len + 2 timesteps
            # truncate here to be consistent
            joint_traj = joint_traj[:-1]
            episode = episode[:-1]
            data_dict = {
                '/observations/qpos': [],
                '/observations/qvel': [],
                '/action': [],
            }
            for cam_name in camera_names:
                data_dict[f'/observations/images/{cam_name}'] = []
            while joint_traj:
                action = joint_traj.pop(0)
                ts = episode.pop(0)
                data_dict['/observations/qpos'].append(ts.observation['qpos'])
                data_dict['/observations/qvel'].append(ts.observation['qvel'])
                data_dict['/action'].append(action)
                for cam_name in camera_names:
                    data_dict[f'/observations/images/{cam_name}'].append(ts.observation['images'][cam_name])
            # HDF5
            t0 = time.time()
            succeeded_text = 'success' if episode_max_reward == env.task.max_reward else 'fail'
            files = [f for f in os.listdir(dataset_dir) if f.endswith('.hdf5')]
            indices = [int(f.split(f'_{succeeded_text}_')[1].split('.')[0]) for f in files if f'episode_{succeeded_text}' in f]
            next_index = max(indices) + 1 if indices else 0
            dataset_path = os.path.join(dataset_dir, f'episode_{succeeded_text}_{next_index}')
            with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                root.attrs['sim'] = True
                obs = root.create_group('observations')
                image = obs.create_group('images')
                for cam_name in camera_names:
                    _ = image.create_dataset(cam_name, (episode_len, 480, 640, 3), dtype='uint8',
                                             chunks=(1, 480, 640, 3), )
                # compression='gzip',compression_opts=2,)
                # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
                qpos = obs.create_dataset('qpos', (episode_len, 12))
                qvel = obs.create_dataset('qvel', (episode_len, 11))
                action = root.create_dataset('action', (episode_len, 12))
                for name, array in data_dict.items():
                    root[name][...] = array
            print(f'Saving: {time.time() - t0:.1f} secs\n')
            del episode
    teleop_handler.stop()
    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')


if __name__ == '__main__':
    test_sim_teleop(record=True)