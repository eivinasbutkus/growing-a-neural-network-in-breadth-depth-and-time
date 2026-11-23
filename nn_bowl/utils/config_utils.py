
from functools import wraps
from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf
from hydra import main
import copy

from nn_bowl.paths import CONFIG_PATH

"""
CONFIG
"""
def get_config_entry(cfg, key_string):
    keys = key_string.split('.')
    value = cfg
    for key in keys:
        value = getattr(value, key, None)
        if value is None:
            return ''
    
    if key_string.endswith("dataset"):
        return value.name

    return value

def pretty_print_cfg(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

def load_config(config_name: str):
    with initialize(config_path="../../config", version_base=None):
        cfg = compose(config_name=config_name)
    return cfg

def checkpoint_elements_to_str(cfg: DictConfig):
    s = ""

    # combining experiment variables + explicitly specified checkpoint elements
    experiment_variable_elements = [variable["key"] for variable in cfg.experiment.variables]
    for key_string in experiment_variable_elements:
        s += "-" + key_string + "=" + str(get_config_entry(cfg, key_string))

    for key_string in list(cfg.experiment.checkpoint_elements):
        if key_string not in experiment_variable_elements:  # avoid duplicates
            s += "-" + key_string + "=" + str(get_config_entry(cfg, key_string))

    return s


CONFIG_NAME = "config"
def hydra_config(func):
    @wraps(func)
    @main(version_base=None, config_path=CONFIG_PATH, config_name=CONFIG_NAME)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    return wrapper



def get_cfg(cfg_base, combo):
    """Build a config from a combination dict as returned by get_combinations.

    combo keys are dot-notation config paths (e.g. 'costs.space.lam_breadth')
    plus the special key 'instance' which maps to train.instance.
    """
    import os

    cfg = copy.deepcopy(cfg_base)

    for key, value in combo.items():
        if key == "instance":
            cfg.train.instance = value
        elif key == "dataset":
            dataset_cfg_path = os.path.join(CONFIG_PATH, "dataset", f"{value}.yaml")
            dataset_cfg = OmegaConf.load(dataset_cfg_path)
            OmegaConf.update(cfg, "dataset", dataset_cfg, merge=True)
        else:
            OmegaConf.update(cfg, key, value)

    return cfg