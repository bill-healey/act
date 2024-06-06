import os
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
        self.max_reward = 4

    def before_step(self, action, physics):
        teleop_actions = teleop_handler.get_actions()
        self.action[:4] = list(teleop_actions.values())
        physics.set_control(self.action)  # Apply actions to the physics
        super().before_step(self.action, physics)
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

        touch_left_gripper = ("red_box", "vx300s_left/10_left_gripper_finger") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs

        reward = 0
        if touch_left_gripper:
            reward = 1
        if touch_left_gripper and touch_table: # lifted
            reward = 2
        if touch_left_gripper and not touch_table: # lifted
            reward = 4
        return reward


    def action_spec(self, physics):
        # Define the action specification
        return specs.BoundedArray(
            shape=(6,), dtype=np.float32, minimum=-1.0, maximum=1.0, name='action')

def test_sim_teleop():
    ts = env.reset()
    episode = [ts]
    ax = plt.subplot()
    plt_img = ax.imshow(ts.observation['images']['angle'])
    plt.ion()
    for t in range(1000):
        #action = get_action(master_bot_left, master_bot_right)
        #ts = env.step(action)
        episode.append(ts)
        plt_img.set_data(ts.observation['images']['angle'])
        plt.pause(0.02)

xml_path = os.path.join(XML_DIR, f'single_viperx_pickup_cube.xml')
physics = mujoco.Physics.from_xml_path(xml_path)
task = PickupTask()
env = control.Environment(physics, task, time_limit=60, control_timestep=DT,
                          n_sub_steps=None, flat_observation=False)

teleop_handler = TeleOpHandler()
teleop_handler.start()
launch(env, policy=lambda _: task.action)
teleop_handler.stop()
