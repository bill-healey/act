import os
import h5py
import numpy as np
import time
import cv2
from dm_control import mujoco, viewer
from dm_control.rl import control
from teleop import TeleOpHandler
from constants import DT, XML_DIR, TASK_CONFIGS
from velocity_pickup_task import PickupTask
from tracker import MobileIOARTracker
from display_thread import DisplayThread
from roarm import RoArm


def record_sim_teleop():
    task_config = TASK_CONFIGS['sim_pickup_task']
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
    display_thread = DisplayThread()
    display_thread.start()
    success = []
    env = control.Environment(physics, task, time_limit=episode_len, control_timestep=DT,
                              n_sub_steps=None, flat_observation=False)
    for episode_idx in range(num_episodes):
        ts = env.reset()
        episode = [ts]
        action = np.zeros(task_config['action_len'])
        for t in range(episode_len):
            action = task.control_input_to_action(teleop_handler, action)
            ts = env.step(action)
            episode.append(ts)
            display_thread.update_frames(ts.observation['images'])
        episode_return = np.sum([ts.reward for ts in episode[1:]])
        episode_max_reward = np.max([ts.reward for ts in episode[1:]])
        success_status = episode_max_reward == env.task.max_reward
        success.append(int(success_status))

        print(f"{episode_idx=} {'Successful' if success_status else 'Failed'} "
              f"with reward {env.task.max_reward}/{episode_max_reward} sum {episode_return}")

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
            data_dict['/action'].append(ts.observation['action'])
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
    display_thread.stop()
    print(f'Saved to {dataset_dir}')
    print(f'Success: {np.sum(success)} / {len(success)}')


def clamp(val, mn, mx): return max(mn, min(val, mx))


def record_real_teleop():
    cfg = TASK_CONFIGS['roarm_pickup_task']
    if not os.path.isdir(cfg['dataset_dir']):
        os.makedirs(cfg['dataset_dir'], exist_ok=True)

    arm = RoArm()
    arm.set_led(180)
    tracker = MobileIOARTracker()
    display_thread = DisplayThread()
    display_thread.start()
    episode_started = False

    caps = [cv2.VideoCapture(idx, cv2.CAP_DSHOW) for idx in cfg['camera_indexes']]

    success_flags = []

    def tracking_update(feedback):
        axes, pos, btn = feedback['axes'], feedback['relative_pos'], feedback['buttons']
        if episode_started:
            scale_factor = 800
            ee_pos['x'] = max(210, initial_pos['x'] - pos[0]*scale_factor)
            ee_pos['y'] = initial_pos['y'] - pos[1] * scale_factor
            ee_pos['z'] = max(-100, initial_pos['z'] + pos[2]*scale_factor)
            ee_pos['t'] = clamp(ee_pos['t'] + (axes[1]+axes[7])*.05 - .03*btn[1] + .03*btn[5], 1.5, 3.3)
            ee_pos['spd'] = 2.0
            arm.move_to_position(**ee_pos)

    def wait_for_button_press():
        while True:
            fb = tracker.get_last_feedback()
            if fb and fb['buttons'][4] == 1:
                return 'fail'
            if fb and fb['buttons'][8] == 1:
                return 'success'
            time.sleep(0.05)

    try:
        for ep in range(cfg['num_episodes']):
            ep_data = []
            print("Returning arm to home position")
            arm.home()
            print("Press B4 or B8 to start episode")
            tracker.continuous_tracking(callback=tracking_update, duration=6000, blocking=False)
            wait_for_button_press()
            tracker.reset_initial_position()
            initial_pos = arm.get_position()
            ee_pos = dict(x=initial_pos.get('x', 0), y=initial_pos.get('y', 0),
                          z=initial_pos.get('z', 0), t=initial_pos.get('t', 0))
            arm.set_led(50)
            episode_started = True
            for step in range(cfg['episode_len']):
                t0 = time.time()
                arm_pos = arm.get_position()
                ee_action = [ee_pos['x'], ee_pos['y'], ee_pos['z'], ee_pos['t']]
                action = [arm_pos['b'], arm_pos['s'], arm_pos['e'], arm_pos['t']]
                frames = {}
                for i, cam_name in enumerate(cfg['camera_names']):
                    ret, frame = caps[i].read()
                    if ret:
                        frames[cam_name] = frame
                    else:
                        frames[cam_name] = None
                ep_data.append({'arm_pos': arm_pos, 'action': action, 'ee_action': ee_action, 'images': frames})
                display_thread.update_frames(frames)
                #display_thread.plot_action(ep_data)
                elapsed = time.time() - t0
                if elapsed < DT:
                    time.sleep(DT - elapsed)
                print(f'Step time elapsed: {time.time() - t0} target: {DT}')

            # Wait for user to decide success/fail
            print("Press B4 for FAIL, B8 for SUCCESS...")
            success_flag = wait_for_button_press()
            success_flags.append(1 if success_flag == 'success' else 0)
            print(f"Episode {ep} complete -> Marked as {success_flag.upper()}")
            tracker.stop_tracking()
            arm.set_led(0)
            episode_started = False

            data_dict = {'/observations/qpos': [], '/observations/qvel': [],
                         '/action': [], '/ee_action': [], **{f'/observations/images/{c}': [] for c in cfg['camera_names']}}
            prev_qpos = None
            for step in ep_data:
                pos = step['arm_pos']
                qpos = [pos['b'], pos['s'], pos['e'], pos['t']]
                data_dict['/observations/qpos'].append(qpos)
                qvel = np.subtract(qpos, prev_qpos).tolist() if prev_qpos else [0] * 4
                data_dict['/observations/qvel'].append(qvel)
                prev_qpos = qpos
                data_dict['/action'].append(step['action'])
                data_dict['/ee_action'].append(step['ee_action'])
                for c in cfg['camera_names']:
                    data_dict[f'/observations/images/{c}'].append(step['images'][c])

            files = [f for f in os.listdir(cfg['dataset_dir'])
                     if f.endswith('.hdf5') and f'episode_{success_flag}' in f]
            idx = max([int(f.split(f'_{success_flag}_')[1].split('.')[0]) for f in files], default=-1) + 1
            p = os.path.join(cfg['dataset_dir'], f'episode_{success_flag}_{idx}.hdf5')
            with h5py.File(p, 'w', rdcc_nbytes=2*(1024**2)) as f:
                f.attrs['sim'] = False
                obs = f.create_group('observations')
                img = obs.create_group('images')
                for c in cfg['camera_names']:
                    img.create_dataset(c, (cfg['episode_len'], 480, 640, 3),
                                       dtype='uint8', chunks=(1, 480, 640, 3))
                obs.create_dataset('qpos', (cfg['episode_len'], 4))
                obs.create_dataset('qvel', (cfg['episode_len'], 4))
                f.create_dataset('action', (cfg['episode_len'], cfg['action_len']))
                f.create_dataset('ee_action', (cfg['episode_len'], cfg['action_len']))
                for k, arr in data_dict.items():
                    if 'images' in k:
                        ds = f[k]
                        for i, im in enumerate(arr):
                            ds[i] = cv2.resize(im, (640, 480))
                    else:
                        f[k][...] = arr
            print(f"Saved episode {ep} -> {p}")
    finally:
        tracker.stop_tracking()
        display_thread.stop()
        arm.set_led(0)
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()

    print(f"\nSaved to {cfg['dataset_dir']}, success: {sum(success_flags)} / {len(success_flags)}")


if __name__ == '__main__':
    #record_sim_teleop()
    record_real_teleop()
