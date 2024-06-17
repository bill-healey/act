import os

import h5py
import numpy as np
import collections
import matplotlib.pyplot as plt
import time
from dm_control import mujoco
from dm_control.rl import control
from dm_control.viewer import launch, viewer, user_input, renderer
from dm_control.suite import base
from dm_env import specs
from teleop import TeleOpHandler
from constants import DT, XML_DIR, START_ARM_POSE, MASTER_GRIPPER_POSITION_NORMALIZE_FN


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
        obs['images']['top'] = physics.render(height=480, width=640, camera_id='top')
        obs['images']['angle'] = physics.render(height=480, width=640, camera_id='angle')
        #obs['images']['vis'] = physics.render(height=480, width=640, camera_id='front_close')
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
        if touch_gripper and touch_table: # lifted
            reward = 2
        if touch_gripper and not touch_table: # lifted
            reward = 4
        table_z = physics.named.data.xpos['table'][2]
        red_box_size = 0.02  # The height of the box from the MJCF definition
        red_box_current_z = physics.named.data.xpos['box'][2]
        if touch_gripper and not touch_table and red_box_current_z > table_z + 5 * red_box_size: # lifted up
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
        action[2] + teleop_actions['elbow_elevation'],
        # Claw has two hinges so claw action must be duplicated.  Model handles inversion of second hinge.
        # Also clamp claw ctrlrange to match model limitations
        max(-.4, min(action[3] + teleop_actions['gripper_rotation'], .4)),
        max(-.4, min(action[4] + teleop_actions['gripper_rotation'], .4))
    ]
    return action


def test_sim_teleop(record=True, num_episodes=2):
    dataset_dir = 'data\excavator_dataset'
    camera_names = ['top']
    max_timesteps = 500
    if not os.path.isdir(dataset_dir):
        os.makedirs(dataset_dir, exist_ok=True)
    xml_path = os.path.join(XML_DIR, f'single_viperx_pickup_cube.xml')
    physics = mujoco.Physics.from_xml_path(xml_path)
    task = PickupTask()
    teleop_handler = TeleOpHandler()
    teleop_handler.start()
    success = []
    for episode_idx in range(num_episodes):
        env = control.Environment(physics, task, time_limit=60, control_timestep=DT,
                                  n_sub_steps=None, flat_observation=False)
        ts = env.reset()
        episode = [ts]
        ax = plt.subplot()
        plt_img = ax.imshow(ts.observation['images']['angle'])
        plt.ion()
        action = np.zeros(6)
        for t in range(max_timesteps):
            action = get_action(teleop_handler, action)
            ts = env.step(action)
            episode.append(ts)
            plt_img.set_data(ts.observation['images']['angle'])
            plt.pause(0.02)

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
            subtask_info = episode[0].observation['env_state'].copy()  # box pose at step 0
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
            max_timestamps = len(joint_traj)
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
            dataset_path = os.path.join(dataset_dir, f'episode_{episode_idx}')
            with h5py.File(dataset_path + '.hdf5', 'w', rdcc_nbytes=1024 ** 2 * 2) as root:
                root.attrs['sim'] = True
                obs = root.create_group('observations')
                image = obs.create_group('images')
                for cam_name in camera_names:
                    _ = image.create_dataset(cam_name, (max_timesteps, 480, 640, 3), dtype='uint8',
                                             chunks=(1, 480, 640, 3), )
                # compression='gzip',compression_opts=2,)
                # compression=32001, compression_opts=(0, 0, 0, 0, 9, 1, 1), shuffle=False)
                qpos = obs.create_dataset('qpos', (max_timesteps, 12))
                qvel = obs.create_dataset('qvel', (max_timesteps, 11))
                action = root.create_dataset('action', (max_timesteps, 12))

                for name, array in data_dict.items():
                    root[name][...] = array
            print(f'Saving: {time.time() - t0:.1f} secs\n')

            del env
            del episode

    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')




    teleop_handler.stop()


if __name__ == '__main__':
    test_sim_teleop(record=True)