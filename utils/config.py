import yaml
from easydict import EasyDict as edict


def _get_default_config():
    c = edict()

    # dataset
    c.data = edict()
    c.data.name = 'DefaultDataset'
    c.data.sample_submission_path = 'sub/to/path'
    c.data.test_dir = 'train/to/path'
    c.data.df_path = 'train/to/path'
    c.data.train_dir = '../input/understanding_cloud_organization/train_images'
    c.data.params = edict()
    c.data.image_size = 128

    # model
    c.model = edict()
    c.model.name = 'effcientnet'
    c.model.version = 'effcientnet-b0'
    c.model.pretrained = None
    c.model.params = edict()

    # train
    c.train = edict()
    c.train.batch_size = 32
    c.train.num_epochs = 50
    c.train.main_metric = 'loss'
    c.train.minimize_metric = True
    c.train.pseudo_label_path = 'dummy.csv'
    c.train.early_stop_patience = 0
    c.train.accumulation_size = 0
    c.train.mixup=False


    # optimizer
    c.optimizer = edict()
    c.optimizer.name = 'Adam'
    c.optimizer.params = edict()
    c.optimizer.lookahead = edict()

    # scheduler
    c.scheduler = edict()
    c.scheduler.name = 'plateau'
    c.scheduler.params = edict()

    # transforms
    c.transforms = edict()
    # c.transforms.params = edict()


    # losses
    c.loss = edict()
    c.loss.name = 'CrossEntropy'
    c.loss.params = edict()

    c.device = 'cuda'
    c.num_workers = 2
    c.work_dir = './work_dir'
    c.checkpoint_path = None
    c.debug = False

    return c


def _merge_config(src, dst):
    if not isinstance(src, edict):
        return

    for k, v in src.items():
        if isinstance(v, edict):
            _merge_config(src[k], dst[k])
        else:
            dst[k] = v


def load_config(config_path):
    with open(config_path, 'r') as fid:
        yaml_config = edict(yaml.load(fid, Loader=yaml.SafeLoader))

    config = _get_default_config()
    _merge_config(yaml_config, config)

    return config


def save_config(config, file_name):
    with open(file_name, "w") as wf:
        yaml.dump(config, wf)