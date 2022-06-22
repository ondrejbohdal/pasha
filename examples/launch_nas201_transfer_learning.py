from typing import Dict

from benchmarking.blackbox_repository import load
from benchmarking.blackbox_repository.tabulated_benchmark import BlackboxRepositoryBackend
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.experiments import load_experiment
from syne_tune.optimizer.schedulers.fifo import FIFOScheduler
from syne_tune.optimizer.schedulers.transfer_learning import TransferLearningTaskEvaluations
from syne_tune.optimizer.schedulers.transfer_learning.bounding_box import BoundingBox
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner


def load_transfer_learning_evaluations(blackbox_name: str, test_task: str, metric: str) -> Dict[str, TransferLearningTaskEvaluations]:
    bb_dict = load(blackbox_name)
    metric_index = [i for i, name in enumerate(bb_dict[test_task].objectives_names) if name == metric][0]
    transfer_learning_evaluations = {
        task: TransferLearningTaskEvaluations(
            hyperparameters=bb.hyperparameters,
            configuration_space=bb.configuration_space,
            objectives_evaluations=bb.objectives_evaluations[..., metric_index:metric_index + 1],
            objectives_names=[metric],

        )
        for task, bb in bb_dict.items()
        if task != test_task
    }
    return transfer_learning_evaluations


if __name__ == '__main__':
    blackbox_name = "nasbench201"
    test_task = "cifar100"
    elapsed_time_attr = "metric_elapsed_time"
    time_this_resource_attr = 'metric_runtime'
    metric = "metric_valid_error"

    bb_dict = load(blackbox_name)
    transfer_learning_evaluations = load_transfer_learning_evaluations(blackbox_name, test_task, metric)

    scheduler = BoundingBox(
        scheduler_fun=lambda new_config_space, mode, metric: FIFOScheduler(
            new_config_space,
            points_to_evaluate=[],
            searcher='random',
            metric=metric,
            mode=mode,
        ),
        mode="min",
        config_space=bb_dict[test_task].configuration_space,
        metric=metric,
        num_hyperparameters_per_task=10,
        transfer_learning_evaluations=transfer_learning_evaluations,
    )

    stop_criterion = StoppingCriterion(max_wallclock_time=7200)

    backend = BlackboxRepositoryBackend(
        blackbox_name=blackbox_name,
        elapsed_time_attr=elapsed_time_attr,
        time_this_resource_attr=time_this_resource_attr,
        dataset=test_task,
    )

    # It is important to set `sleep_time` to 0 here (mandatory for simulator backend)
    tuner = Tuner(
        backend=backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=4,
        sleep_time=0,
        # This callback is required in order to make things work with the
        # simulator callback. It makes sure that results are stored with
        # simulated time (rather than real time), and that the time_keeper
        # is advanced properly whenever the tuner loop sleeps
        callbacks=[SimulatorCallback()],
    )
    tuner.run()

    tuning_experiment = load_experiment(tuner.name)
    print(tuning_experiment)

    print(f"best result found: {tuning_experiment.best_config()}")

    tuning_experiment.plot()