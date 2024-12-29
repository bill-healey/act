import pathlib

DATA_DIR = 'data'
TASK_CONFIGS = {
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
        'batch_size': 10,#4096,
        'dim_feedforward': 32,
        'num_epochs': 24000,
        'lr': 1e-5,
        'seed': 42,
        'temporal_agg': False,
        'mujoco_xml': 'excavator_scene.xml',
    },
    'roarm_pickup_task': {
        'dataset_dir': DATA_DIR + '\\roam_dec29',
        'num_episodes': 25,
        'episode_len': 290,
        'camera_names': ['top', 'angle'],
        'camera_indexes': [1, 2],
        'action_len': 4,
        'kl_weight': 10,
        'hidden_dim': 256,
        'batch_size': 4,
        'dim_feedforward': 32,
        'num_epochs': 2400,
        'num_queries': 1,
        'lr': 1e-5,
        'seed': 42,
        'temporal_agg': False,
        'mujoco_xml': 'excavator_scene.xml',
    },
}

DT = 0.05
XML_DIR = str(pathlib.Path(__file__).parent.resolve()) + '/assets/' # note: absolute path