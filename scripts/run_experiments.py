import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
from manager.experiment import ExperimentManager

@hydra.main(version_base=None, config_path="../conf", config_name="run_experiment_config")
def main(cfg: DictConfig):
    # Dynamically instantiate the runner based on the config.
    # Instantiates either Test2StageMCMCExperiment or GenericSweepExperiment
    print("Dynamically instantiating experiment runner...")
    runner = instantiate(cfg.experiment.runner)

        
    # Initialize and execute the ExperimentManager
    print("Initializing Experiment Manager...")
    manager = ExperimentManager(runner, cfg)
    manager.run()

if __name__ == "__main__":
    main()
