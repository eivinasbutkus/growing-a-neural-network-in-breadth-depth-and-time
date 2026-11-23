


import hydra
import nn_bowl


@hydra.main(version_base=None, config_name='config', config_path=nn_bowl.paths.CONFIG_PATH)
def evaluate(cfg):
    nn_bowl.evaluation.evaluate(cfg)



if __name__ == '__main__':
    evaluate()