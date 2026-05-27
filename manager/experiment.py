import abc
import os
import ray
import utils.export_info as info

class ExperimentRunner(abc.ABC):
    @abc.abstractmethod
    def setup(self, cfg):
        """Prepare references/prerequisites once before task submission."""
        pass

    @abc.abstractmethod
    def generate_tasks(self, cfg):
        """Generate a list or iterable of task configurations (dicts) for the sweep."""
        pass

    @staticmethod
    @abc.abstractmethod
    def run_task(task_params):
        """
        The worker function executed on a Ray worker.
        Must be a static method to be cleanly pickled/wrapped by Ray.
        """
        pass

    @abc.abstractmethod
    def process_result(self, task_params, result, storage_mode):
        """
        Callback executed in the main process when a Ray task completes.
        Applies on-the-fly aggregation or statistics calculation depending on storage_mode.
        """
        pass

    @abc.abstractmethod
    def save_results(self, output_dir):
        """Save the aggregated results to disk inside output_dir."""
        pass


class ExperimentManager:
    def __init__(self, runner: ExperimentRunner, cfg):
        self.runner = runner
        self.cfg = cfg
        self.threshold = 5 * 10**6

    def run(self):
        # 1. Housekeeping and Ray Init
        num_cpus = self.cfg.get("num_cpus", None)
        if num_cpus is not None and num_cpus > 0:
            print(f"Initializing Ray with num_cpus={num_cpus}...")
            ray.init(ignore_reinit_error=True, num_cpus=num_cpus)
        else:
            print("Initializing Ray (default settings)...")
            ray.init(ignore_reinit_error=True)

        # 2. Output Directory Setup
        base_dir = self.cfg.get("results_base_dir", "data")
        output_dir = info.create_results_folder(base_dir=base_dir)
        print(f"Created results folder: {output_dir}")

        # 3. Runner Setup
        print("Executing runner setup...")
        self.runner.setup(self.cfg)

        # 4. Generate Tasks
        print("Generating tasks...")
        tasks = self.runner.generate_tasks(self.cfg)
        num_tasks = len(tasks)
        print(f"Generated {num_tasks} tasks.")

        # 5. Dual-Regime Storage Check
        problem_cfg = self.cfg.get("problem", {})
        num_particles = problem_cfg.get("n_particles", 100)
        param_dim = problem_cfg.get("param_dim", 1)
        total_floats = num_tasks * num_particles * param_dim

        if total_floats <= self.threshold:
            storage_mode = "raw_particles"
            print(f"Total floats: {total_floats} <= {self.threshold}. Using Regime A: Raw Particles.")
        else:
            storage_mode = "predefined_stats"
            print(f"Total floats: {total_floats} > {self.threshold}. Using Regime B: Predefined Stats (Wasserstein/etc.).")

        # 6. Dispatch Tasks
        # Dynamic Ray wrapping of the static run_task method
        print("Dispatching tasks to Ray cluster...")
        remote_task = ray.remote(self.runner.run_task)
        futures = {remote_task.remote(task): task for task in tasks}

        # 7. Asynchronous Collection
        pending = list(futures.keys())
        total_tasks = len(pending)
        done_tasks = 0

        print(f"Starting asynchronous collection of {total_tasks} tasks...")
        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            ref = ready[0]
            task_params = futures[ref]
            try:
                result = ray.get(ref)
                self.runner.process_result(task_params, result, storage_mode)
            except Exception as e:
                print(f"Error processing task with params {task_params}: {e}")
                raise e

            done_tasks += 1
            if done_tasks % max(1, total_tasks // 10) == 0 or done_tasks == total_tasks:
                print(f"[{done_tasks}/{total_tasks}] tasks completed.")

        # 8. Save results
        print(f"Saving final aggregated results to {output_dir}...")
        self.runner.save_results(output_dir)
        print("Experiment run completed successfully.")
        return output_dir
