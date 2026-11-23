


import hydra
import nn_bowl


@hydra.main(version_base=None, config_name='config', config_path=nn_bowl.paths.CONFIG_PATH)
def evaluate_occlusion(cfg):
    nn_bowl.occlusion_evaluation.evaluate_occlusion(cfg)


if __name__ == '__main__':
    evaluate_occlusion()