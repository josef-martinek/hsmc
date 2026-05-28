import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="run_experiment_config")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg, resolve=True))

if __name__ == "__main__":
    main()
