from benchmarking.definitions.definition_nasbench201 import \
    nasbench201_benchmark, nasbench201_default_params
from syne_tune.experiments import load_experiment
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.tuner import Tuner
from syne_tune.optimizer.baselines import baselines
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from benchmarking.blackbox_repository.tabulated_benchmark import BlackboxRepositoryBackend
from benchmarking.blackbox_repository import load
import logging
import pandas as pd
import random
import numpy as np
from argparse import ArgumentParser
import json

def run_experiment(dataset_name, random_seed, nb201_random_seed, hpo_approach, reduction_factor=None, rung_system_kwargs={'ranking_criterion': 'soft_ranking', 'epsilon': 0.025}):
    logging.getLogger().setLevel(logging.WARNING)

    n_workers = 4
    default_params = nasbench201_default_params({'backend': 'simulated'})
    benchmark = nasbench201_benchmark(default_params)
    # Benchmark must be tabulated to support simulation:
    assert benchmark.get('supports_simulated', False)
    mode = benchmark['mode']
    metric = benchmark['metric']
    blackbox_name = benchmark.get('blackbox_name')
    # NASBench201 is a blackbox from the repository
    assert blackbox_name is not None

    # If you don't like the default config_space, change it here. But let
    # us use the default
    config_space = benchmark['config_space']

    # Simulator back-end specialized to tabulated blackboxes
    backend = BlackboxRepositoryBackend(
        blackbox_name=blackbox_name,
        elapsed_time_attr=benchmark['elapsed_time_attr'],
        time_this_resource_attr=benchmark.get('time_this_resource_attr'),
        dataset=dataset_name,
        seed=nb201_random_seed)

    # set logging of the simulator backend to WARNING level
    logging.getLogger(
        'syne_tune.backend.simulator_backend.simulator_backend').setLevel(logging.WARNING)

    if not reduction_factor:
        reduction_factor = default_params['reduction_factor']

    if hpo_approach == 'pasha':
        scheduler = baselines['PASHA'](
            config_space,
            max_t=default_params['max_resource_level'],
            grace_period=default_params['grace_period'],
            reduction_factor=reduction_factor,
            resource_attr=benchmark['resource_attr'],
            mode=mode,
            metric=metric,
            random_seed=random_seed,
            rung_system_kwargs=rung_system_kwargs)
    elif hpo_approach == 'asha':
        scheduler = baselines['ASHA'](
            config_space,
            max_t=default_params['max_resource_level'],
            grace_period=default_params['grace_period'],
            reduction_factor=reduction_factor,
            resource_attr=benchmark['resource_attr'],
            mode=mode,
            type='promotion',
            metric=metric,
            random_seed=random_seed)
    elif hpo_approach == 'pasha-bo':
        scheduler = HyperbandScheduler(
            config_space,
            max_t=default_params['max_resource_level'],
            grace_period=default_params['grace_period'],
            reduction_factor=reduction_factor,
            resource_attr=benchmark['resource_attr'],
            mode=mode,
            searcher='bayesopt',
            type='pasha',
            metric=metric,
            random_seed=random_seed,
            rung_system_kwargs=rung_system_kwargs)
    elif hpo_approach == 'asha-bo':
        scheduler = HyperbandScheduler(
            config_space,
            max_t=default_params['max_resource_level'],
            grace_period=default_params['grace_period'],
            reduction_factor=reduction_factor,
            resource_attr=benchmark['resource_attr'],
            mode=mode,
            searcher='bayesopt',
            type='promotion',
            metric=metric,
            random_seed=random_seed)
    else:
        raise ValueError('HPO approach not implemented')

    stop_criterion = StoppingCriterion(max_num_trials_started=256)
    # Printing the status during tuning takes a lot of time, and so does
    # storing results.
    print_update_interval = 700
    results_update_interval = 300
    # It is important to set `sleep_time` to 0 here (mandatory for simulator
    # backend)

    tuner = Tuner(
        backend=backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
        sleep_time=0,
        results_update_interval=results_update_interval,
        print_update_interval=print_update_interval,
        # This callback is required in order to make things work with the
        # simulator callback. It makes sure that results are stored with
        # simulated time (rather than real time), and that the time_keeper
        # is advanced properly whenever the tuner loop sleeps
        callbacks=[SimulatorCallback()],
    )

    tuner.run()

    return tuner.name


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--dataset_name", type=str,
                        required=True)
    parser.add_argument("--nb201_random_seed", type=int,
                        required=True)
    parser.add_argument("--random_seed", type=int,
                        required=True)
    parser.add_argument("--scheduler", type=str,
                        required=True)
    # scheduler is pasha-bo or asha-bo
    args, _ = parser.parse_known_args()


    # store the information
    metric_valid_error_dim = 0
    metric_runtime_dim = 2
    dataset_names = ['cifar10', 'cifar100', 'ImageNet16-120']
    epoch_names = ['val_acc_epoch_' + str(e) for e in range(200)]
    random_seeds = [31415927, 0, 1234, 3458, 7685]
    nb201_random_seeds = [0, 1, 2]
    n_workers = 4

    bb_dict = load('nasbench201')
    df_dict = {}

    for seed in nb201_random_seeds:
        df_dict[seed] = {}
        for dataset in dataset_names:
            df_val_acc = pd.DataFrame((1.0-bb_dict[dataset].objectives_evaluations[:, seed, :, metric_valid_error_dim])
                                    * 100, columns=['val_acc_epoch_' + str(e) for e in range(200)])
            df_val_acc['val_acc_best'] = df_val_acc[epoch_names].max(axis=1)
            df_hp = bb_dict[dataset].hyperparameters
            df_time = pd.DataFrame(bb_dict[dataset].objectives_evaluations[:,
                                                                        seed, :, metric_runtime_dim][:, -1], columns=['eval_time_epoch'])
            df_dict[seed][dataset] = pd.concat(
                [df_hp, df_val_acc, df_time], axis=1)

    experiment_name = run_experiment(
        args.dataset_name, args.random_seed, args.nb201_random_seed, args.scheduler)

    experiment_dict = {'experiment_name': experiment_name,
                       'dataset_name': args.dataset_name,
                       'random_seed': args.random_seed,
                       'nb201_random_seed': args.nb201_random_seed,
                       'scheduler': args.scheduler}

    # now we need to store the experiment_name and also the values of the arguments
    with open('bo_experiment_details.json', 'r') as f:
        current_list = json.load(f)

    current_list.append(experiment_dict)
    with open('bo_experiment_details.json', 'w') as f:
        json.dump(current_list, f)
