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
    def process_result(self, task_params, result):
        """
        Callback executed in the main process when a Ray task completes.
        Applies on-the-fly aggregation or statistics calculation.
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

    def _export_files(self, output_dir):
        import inspect
        import shutil
        import importlib
        
        # 1. Export runner simulation file next to setup.yaml
        try:
            runner_file = inspect.getfile(self.runner.__class__)
            shutil.copy(runner_file, output_dir)
            print(f"Exported runner simulation file: {runner_file} to {output_dir}")
        except Exception as e:
            print(f"Could not export runner source file: {e}")

        # 2. Export likelihood model if problem path is provided
        try:
            problem_cfg = self.cfg.get("problem", None)
            if problem_cfg is not None:
                problem_path = problem_cfg.get("path", None)
                if problem_path:
                    module_name = f"{problem_path}.likelihood_model"
                    likelihood_module = importlib.import_module(module_name)
                    info.copy_problem_setup(likelihood_module, output_dir)
                    print(f"Exported likelihood model: {likelihood_module.__file__} to {output_dir}")
        except Exception as e:
            print(f"Could not export likelihood model: {e}")

    def run(self):
        # If we only want to export the setup configuration
        if self.cfg.get("export_setup_only", False):
            base_dir = self.cfg.results_base_dir
            output_dir = info.create_results_folder(base_dir=base_dir)
            print(f"Created results folder: {output_dir}")
            from omegaconf import OmegaConf
            
            keys_to_export = ["problem", "experiment"]
            export_dict = {k: self.cfg[k] for k in keys_to_export if k in self.cfg}
            export_cfg = OmegaConf.create(export_dict)
            
            setup_file = os.path.join(output_dir, "setup.yaml")
            with open(setup_file, "w") as f:
                f.write("# @package _global_\n\n")
                f.write(OmegaConf.to_yaml(export_cfg, resolve=True))
            
            self._export_files(output_dir)

            print("export_setup_only is True. Exiting early.")
            return output_dir

        # 1. Housekeeping and Ray Init
        num_cpus = self.cfg.num_cpus
        import multiprocessing
        max_cpus = multiprocessing.cpu_count()
        if num_cpus is None or num_cpus <= 0:
            num_cpus = max_cpus
            print(f"num_cpus not specified or <= 0. Defaulting to maximum available CPUs: {num_cpus}...")
            ray.init(ignore_reinit_error=True, num_cpus=num_cpus)
        else:
            print(f"Initializing Ray with num_cpus={num_cpus} (max available: {max_cpus})...")
            ray.init(ignore_reinit_error=True, num_cpus=num_cpus)

        # 2. Output Directory Setup
        base_dir = self.cfg.results_base_dir
        output_dir = info.create_results_folder(base_dir=base_dir)
        print(f"Created results folder: {output_dir}")

        # Save setup configuration
        from omegaconf import OmegaConf
        
        keys_to_export = ["problem", "experiment"]
        export_dict = {k: self.cfg[k] for k in keys_to_export if k in self.cfg}
        export_cfg = OmegaConf.create(export_dict)
        
        setup_file = os.path.join(output_dir, "setup.yaml")
        with open(setup_file, "w") as f:
            f.write("# @package _global_\n\n")
            f.write(OmegaConf.to_yaml(export_cfg, resolve=True))

        self._export_files(output_dir)


        # 3. Runner Setup
        print("Executing runner setup...")
        self.runner.setup(self.cfg)

        # 4. Generate Tasks
        print("Generating tasks...")
        tasks = self.runner.generate_tasks(self.cfg)
        num_tasks = len(tasks)
        print(f"Generated {num_tasks} tasks.")

        # 5. Dispatch Tasks
        # Dynamic Ray wrapping of the static run_task method
        print("Dispatching tasks to Ray cluster...")
        import time
        start_time = time.time()
        remote_task = ray.remote(self.runner.run_task)
        
        WAVE_SIZE = 5000
        BATCH_SIZE = 500
        
        total_tasks = len(tasks)
        done_tasks = 0
        last_printed_tasks = 0
        print_interval = max(1, total_tasks // 40)

        print(f"Starting asynchronous collection of {total_tasks} tasks...")
        
        for i in range(0, total_tasks, WAVE_SIZE):
            wave_tasks = tasks[i:i + WAVE_SIZE]
            
            # Submit current wave
            futures = {remote_task.remote(task): task for task in wave_tasks}
            pending = list(futures.keys())
            
            # Collect current wave
            while pending:
                num_to_wait = min(BATCH_SIZE, len(pending))
                ready, pending = ray.wait(pending, num_returns=num_to_wait)
                
                for ref in ready:
                    task_params = futures.pop(ref)
                    try:
                        result = ray.get(ref)
                        self.runner.process_result(task_params, result)
                    except Exception as e:
                        print(f"Error processing task with params {task_params}: {e}")
                        raise e
                    
                    done_tasks += 1
                
                if done_tasks - last_printed_tasks >= print_interval or done_tasks == total_tasks:
                    elapsed_time = time.time() - start_time
                    print(f"[{done_tasks}/{total_tasks}] tasks completed in {elapsed_time:.2f}s.")
                    last_printed_tasks = done_tasks

        # 7. Save results
        print(f"Saving final aggregated results to {output_dir}...")
        self.runner.save_results(output_dir)
        print("Experiment run completed successfully.")
        return output_dir
