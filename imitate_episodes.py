import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange
from dm_control import mujoco
from dm_control.rl import control

from utils import load_data
from utils import sample_box_pose
from utils import compute_dict_mean, set_seed, detach_dict
from policy import ACTPolicy
from visualize_episodes import save_videos
from sim_record import PickupTask
from constants import DT, XML_DIR
from plot_handler import PlotHandler

from constants import SIM_TASK_CONFIGS
from torch.profiler import profile, ProfilerActivity

import IPython
e = IPython.embed

BOX_POSE = [None] # to be changed from outside

def main(args):
    task_name = args['task_name']
    task_config = SIM_TASK_CONFIGS[task_name]
    str_train_params = '_'.join([str(task_config[k]) for k in [
        'episode_len',
        'batch_size',
        'action_len',
        'kl_weight',
        'hidden_dim',
        'dim_feedforward',
        'num_epochs',
        'lr',
        'seed',
    ]])
    config = dict(task_config, **{
                     'task_name': task_name,
                     'policy_class': 'ACT',
                     'onscreen_render': args.get('onscreen_render', False),
                     'is_eval': args.get('eval', False),
                     'ckpt_dir': os.path.join(task_config['dataset_dir'], f'ckpt_{str_train_params}'),
                     'lr_backbone': task_config['lr'],
                     'backbone': 'resnet18',
                     'enc_layers': 4,
                     'dec_layers': 7,
                     'nheads': 8,
                     'num_queries': task_config['episode_len'],
                     'state_dim': task_config['action_len']
                     })
    set_seed(task_config['seed'])
    torch.backends.cudnn.benchmark = True
    if config['is_eval']:
        ckpt_names = [f'policy_best.ckpt']
        results = []
        for ckpt_name in ckpt_names:
            success_rate, avg_return = eval_bc(config, ckpt_name, save_episode=True)
            results.append([ckpt_name, success_rate, avg_return])

        for ckpt_name, success_rate, avg_return in results:
            print(f'{ckpt_name}: {success_rate=} {avg_return=}')
        print()
        exit()

    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    train_dataloader, val_dataloader, stats, _ = load_data(config['dataset_dir'],
                                                           config['num_episodes'],
                                                           config['camera_names'],
                                                           config['batch_size'],
                                                           config['batch_size'])

    # save dataset stats
    if not os.path.isdir(config['ckpt_dir']):
        os.makedirs(config['ckpt_dir'])
    stats_path = os.path.join(config['ckpt_dir'], f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(config['ckpt_dir'], f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')
    #print(prof.key_averages().table(sort_by="cuda_time_total"))
    #prof.export_chrome_trace("trace.json")


def get_image(ts, camera_names):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(ts.observation['images'][cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image


def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1001)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    onscreen_render = config['onscreen_render']
    policy_config = config
    camera_names = config['camera_names']
    max_timesteps = config['episode_len']
    task_name = config['task_name']
    temporal_agg = config['temporal_agg']
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)

    policy = ACTPolicy(policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)
    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)
    pre_process = lambda s_qpos: (s_qpos - stats['qpos_mean']) / stats['qpos_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']
    xml_path = os.path.join(XML_DIR, config['mujoco_xml'])
    physics = mujoco.Physics.from_xml_path(xml_path)
    task = PickupTask()
    env = control.Environment(physics, task, time_limit=config['episode_len'], control_timestep=DT,
                              n_sub_steps=None, flat_observation=False)

    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']
    else:
        query_frequency = policy_config['num_queries']
        num_queries = 0

    num_rollouts = 50
    episode_returns = []
    highest_rewards = []
    plot_handler = PlotHandler(policy_config['camera_names'])
    for rollout_id in range(num_rollouts):
        rollout_id += 0
        BOX_POSE[0] = sample_box_pose()
        ts = env.reset()
        if temporal_agg:
            all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()
        qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
        image_list = []
        rewards = []
        with torch.inference_mode():
            for t in range(max_timesteps):
                obs = ts.observation
                if onscreen_render:
                    plot_handler.render_images(obs['images'])
                if 'images' in obs:
                    image_list.append(obs['images'])
                else:
                    image_list.append({'main': obs['image']})
                qpos_numpy = np.array(obs['qpos']) # seems to have no effect
                qpos = pre_process(qpos_numpy)
                qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
                qpos_history[:, t] = qpos
                curr_image = get_image(ts, camera_names)

                if t % query_frequency == 0:
                    all_actions = policy(qpos, curr_image)
                if temporal_agg:
                    all_time_actions[[t], t:t+num_queries] = all_actions
                    actions_for_curr_step = all_time_actions[:, t]
                    actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                    actions_for_curr_step = actions_for_curr_step[actions_populated]
                    k = 0.01
                    exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                    exp_weights = exp_weights / exp_weights.sum()
                    exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                    raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                else:
                    raw_action = all_actions[:, t % query_frequency]

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                target_qpos = action

                ### step the environment
                #ts = env.step(target_qpos[:5])
                ts = env.step(target_qpos[-7:])

                ### for visualization
                rewards.append(ts.reward)

        rewards = np.array(rewards)
        episode_return = np.sum(rewards[rewards!=None])
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env.task.max_reward=}, Success: {episode_highest_reward==env.task.max_reward}')

        if save_episode:
            save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}-reward-{episode_highest_reward}.mp4'))

    success_rate = np.mean(np.array(highest_rewards) == env.task.max_reward)
    avg_return = np.mean(episode_returns)
    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    for r in range(env.task.max_reward+1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / num_rollouts
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

    print(summary_str)

    result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
    with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
        f.write(summary_str)
        f.write(repr(episode_returns))
        f.write('\n\n')
        f.write(repr(highest_rewards))

    return success_rate, avg_return


def forward_pass(data, policy):
    image_data, qpos_data, action_data, is_pad = data
    image_data, qpos_data, action_data, is_pad = image_data.cuda(), qpos_data.cuda(), action_data.cuda(), is_pad.cuda()
    return policy(qpos_data, image_data, action_data, is_pad)


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_config = config

    set_seed(seed)

    policy = ACTPolicy(policy_config)
    policy.cuda()
    optimizer = policy.configure_optimizers()
    device = torch.device("cuda:0")
    policy = policy.to(device)

    # Check if there is an existing checkpoint to resume from
    resume_ckpt_path = os.path.join(ckpt_dir, 'policy_last.ckpt')
    if os.path.exists(resume_ckpt_path):
        checkpoint = torch.load(resume_ckpt_path)
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        best_ckpt_info = checkpoint
        policy.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        train_history = checkpoint.get('train_history', [])
        validation_history = checkpoint.get('validation_history', [])
        print(f'Resuming training from epoch {start_epoch} with best val loss {best_val_loss:.6f}')
    else:
        start_epoch = 0
        best_val_loss = np.inf
        best_ckpt_info = None
        train_history = []
        validation_history = []

    min_next_ckpt_epoch = start_epoch+25
    for epoch in tqdm(range(start_epoch, start_epoch+num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < best_val_loss:
                print(f"New Best Validation Loss {epoch_val_loss}")
                best_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, best_val_loss, deepcopy(policy.state_dict()))
                if epoch > min_next_ckpt_epoch:
                    min_next_ckpt_epoch = epoch + 50
                    ckpt_path = os.path.join(ckpt_dir, f'policy_seed_{seed}_temp_best.ckpt')
                    torch.save(policy.state_dict(), ckpt_path)
                    plot_history(train_history, validation_history, epoch, ckpt_dir, seed)
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save({
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'model_state_dict': policy.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_history': train_history,
        'validation_history': validation_history
    }, ckpt_path)
    best_epoch, best_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {best_val_loss:.6f} at epoch {best_epoch}')
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)
    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.close()
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
        plt.pause(0.001)  # Pause to update figures
        plt.show(block=False)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    main(vars(parser.parse_args()))
