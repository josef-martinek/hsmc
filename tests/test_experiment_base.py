import os
import numpy as np
import pytest
import ray
from manager.experiment import ExperimentRunner, ExperimentManager

class DummyRunner(ExperimentRunner):
    def setup(self, cfg):
        self.setup_called = True
        self.results = []
        self.save_called = False

    def generate_tasks(self, cfg):
        return [{"idx": i, "val": i * 1.5} for i in range(5)]

    @staticmethod
    def run_task(task_params):
        # A simple dummy task
        idx = task_params["idx"]
        val = task_params["val"]
        # Return dummy particle-like arrays: (particles=10, dim=2)
        return np.ones((10, 2)) * (idx + val)

    def process_result(self, task_params, result, storage_mode):
        self.results.append((task_params, result, storage_mode))

    def save_results(self, output_dir):
        self.save_called = True
        self.saved_output_dir = output_dir


def test_experiment_manager():
    # Construct a dummy Hydra-like configuration dict
    cfg = {
        "num_cpus": 1,
        "results_base_dir": "data",
        "problem": {
            "n_particles": 10,
            "param_dim": 2
        }
    }
    
    runner = DummyRunner()
    manager = ExperimentManager(runner, cfg)
    
    # Run the experiment manager
    output_dir = manager.run()
    
    assert runner.setup_called is True
    assert len(runner.results) == 5
    assert runner.save_called is True
    assert os.path.exists(output_dir)
    
    # Clean up output dir
    if os.path.exists(output_dir):
        import shutil
        shutil.rmtree(output_dir)
        
    # Test threshold (storage_mode detection)
    # The dummy experiment has 5 tasks, each with 10 particles and 2 dimensions.
    # total_floats = 5 * 10 * 2 = 100.
    # If we set threshold to 50, it should use "predefined_stats"
    manager.threshold = 50
    runner.results = []
    output_dir = manager.run()
    assert runner.results[0][2] == "predefined_stats"
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    # If we set threshold to 200, it should use "raw_particles"
    manager.threshold = 200
    runner.results = []
    output_dir = manager.run()
    assert runner.results[0][2] == "raw_particles"
    
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
        
    # Shutdown ray at the end of the test
    ray.shutdown()
