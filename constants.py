import pathlib

### Task parameters
DATA_DIR = 'data'
SIM_TASK_CONFIGS = {
    'sim_transfer_cube_scripted':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_transfer_cube_human':{
        'dataset_dir': DATA_DIR + '/sim_transfer_cube_human',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_scripted': {
        'dataset_dir': DATA_DIR + '/sim_insertion_scripted',
        'num_episodes': 50,
        'episode_len': 400,
        'camera_names': ['top']
    },

    'sim_insertion_human': {
        'dataset_dir': DATA_DIR + '/sim_insertion_human',
        'num_episodes': 50,
        'episode_len': 500,
        'camera_names': ['top']
    },
    'sim_pickup_task': {
        'dataset_dir': DATA_DIR + '\\excavator_qpos_oct2',
        'num_episodes': 36,
        'episode_len': 230,
        'camera_names': ['top', 'angle'],
        'action_len': 5,
        'kl_weight': 10,
        'hidden_dim': 256,
        'batch_size': 8,#4096,
        'dim_feedforward': 32,
        'num_epochs': 24000,
        'lr': 1e-5,
        'seed': 42,
        'temporal_agg': False,
        'mujoco_xml': 'excavator_scene.xml',
    },
}

DT = 0.02
XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path