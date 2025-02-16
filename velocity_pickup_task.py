import collections
import numpy as np
from dm_control.suite import base
from dm_env import specs
from constants import SIM_TASK_CONFIGS


class PickupTask(base.Task):
    def initialize_episode(self, physics):
        super().initialize_episode(physics)
        random_pos = np.array([np.random.uniform(-.5, .5) + 6, np.random.uniform(-1.5, 1.5), .4])
        physics.named.data.qpos['red_box_joint'][:3] = random_pos
        self.action_len = SIM_TASK_CONFIGS['sim_pickup_task']['action_len']
        self.action = np.zeros(self.action_len)  # Initialize action array
        self.action_impulse = np.zeros(self.action_len)
        self.max_reward = 5

    def before_step(self, action, physics):
        self.action += action
        self.action_impulse = action
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
        # Red box joint is defined last as a free joint so should always be the last 7 DoF
        env_state = physics.data.qpos.copy()[-7:]
        return env_state

    def get_observation(self, physics):
        obs = collections.OrderedDict()
        # Red box joint is defined last as a free joint so last 7 qpos DoF (3 position + 4 quaternion)
        #   and 6 qvel DoF (linear xyz and angular xyz)
        obs['qpos'] = self.get_qpos(physics)[:-7]
        obs['qvel'] = self.get_qvel(physics)[:-6]
        obs['action'] = self.action_impulse.copy()
        obs['env_state'] = self.get_env_state(physics)
        obs['images'] = dict()
        for camera in SIM_TASK_CONFIGS['sim_pickup_task']['camera_names']:
            obs['images'][camera] = physics.render(height=480, width=640, camera_id=camera)
        return obs

    def get_reward(self, physics):
        all_contact_pairs = []
        for i_contact in range(physics.data.ncon):
            id_geom_1 = physics.data.contact[i_contact].geom1
            id_geom_2 = physics.data.contact[i_contact].geom2
            name_geom_1 = physics.model.id2name(id_geom_1, 'geom')
            name_geom_2 = physics.model.id2name(id_geom_2, 'geom')
            contact_pair = (name_geom_1, name_geom_2)
            all_contact_pairs.append(contact_pair)
        touch_gripper = ("red_box", "bucket_bottom") in all_contact_pairs or \
                        ("red_box", "bucket_top") in all_contact_pairs
        touch_table = ("red_box", "table") in all_contact_pairs
        reward = 0
        if touch_gripper:
            reward = 1
        if touch_gripper and touch_table:
            reward = 2
        if touch_gripper and not touch_table:
            reward = 4
        table_z = physics.named.data.xpos['table'][2]
        red_box_size = 0.2  # The height of the box from the MJCF definition
        red_box_current_z = physics.named.data.xpos['box'][2]
        if touch_gripper and (not touch_table) and red_box_current_z > table_z + 5 * red_box_size:
            reward = 5
        #print(f"{touch_gripper}  {touch_table}  {red_box_current_z > table_z + 5 * red_box_size}")
        return reward

    def action_spec(self, physics):
        # Define the action specification
        return specs.BoundedArray(
            shape=(self.action_len,), dtype=np.float32, minimum=-1.0, maximum=1.0, name='action')

    @staticmethod
    def control_input_to_action(teleop_handler, action):
        teleop_actions = teleop_handler.get_actions()
        action = [
            teleop_actions['waist_rotation'],
            teleop_actions['shoulder_elevation'],
            teleop_actions['wrist_elevation'],
            teleop_actions['gripper_rotation'],
            teleop_actions['gripper_rotation'],
            # Claw has two hinges so claw action must be duplicated.  Model handles inversion of second hinge.
            # Also clamp claw ctrlrange to match model limitations
            #max(-.4, min(teleop_actions['gripper_rotation'], .4))*100,
            #-max(-.4, min(teleop_actions['gripper_rotation'], .4))*100
        ]
        return action
