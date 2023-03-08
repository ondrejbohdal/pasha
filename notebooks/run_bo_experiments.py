import json
import logging
import numpy as np
import pandas as pd
import random
import string
from argparse import ArgumentParser

from syne_tune.blackbox_repository import load_blackbox
from syne_tune.blackbox_repository.simulated_tabular_backend import BlackboxRepositoryBackend
from benchmarking.definitions.definition_nasbench201 import (
    nasbench201_benchmark,
    nasbench201_default_params
)
from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune.optimizer.schedulers.hyperband import HyperbandScheduler
from syne_tune.optimizer.baselines import baselines_dict
from syne_tune.tuner import Tuner
from syne_tune.stopping_criterion import StoppingCriterion
from syne_tune.experiments import load_experiment


def run_experiment(dataset_name, random_seed, benchmark_random_seed, hpo_approach, reduction_factor=None, rung_system_kwargs={'ranking_criterion': 'soft_ranking', 'epsilon': 0.025}, benchmark_metric='metric_valid_error', benchmark='nasbench201', benchmark_mode=None):
    """
    Function to run an experiment. It is similar to the NASBench201 example script
    in syne-tune but extended to make it simple to run our experiments.
    
    When describing the following parameters we say what values we use, but feel free to also use other values.
    
    :param dataset_name: one of 'cifar10', 'cifar100', 'ImageNet16-120'
    :param random_seed: one of 31415927, 0, 1234, 3458, 7685
    :param benchmark_random_seed: one of 0, 1, 2 for nasbench201
    :param hpo_approach: one of 'pasha', 'asha', 'pasha-bo', 'asha-bo'
    :param reduction_factor: by default None (resulting in using the default value 3) or 2, 4
    :param rung_system_kwargs: dictionary of ranking criterion (str) and epsilon or epsilon scaling (both float)
    :return: tuner.name
    
    """

    # this function is similar to the NASBench201 example script
    logging.getLogger().setLevel(logging.WARNING)

    default_params = nasbench201_default_params({'backend': 'simulated'})
    benchmark = nasbench201_benchmark(default_params)
    resource_attr = benchmark['resource_attr']
    max_t = default_params['max_resource_level']
    blackbox_name = dataset_name
    # NASBench201 is a blackbox from the repository
    assert blackbox_name is not None
    elapsed_time_attr = 'elapsed_time'
    surrogate = None

    # benchmark must be tabulated to support simulation
    assert benchmark.get('supports_simulated', False)
    if benchmark_mode:
        mode = benchmark_mode
    else:
        mode = benchmark['mode']

    metric = benchmark_metric  # benchmark['metric']

    config_space = benchmark['config_space']
    config_space['dataset_name'] = dataset_name

    # simulator back-end specialized to tabulated blackboxes
    trial_backend = BlackboxRepositoryBackend(
        blackbox_name=blackbox_name,
        elapsed_time_attr=elapsed_time_attr,
        dataset=dataset_name,
        seed=benchmark_random_seed,
        surrogate=surrogate)

    # set logging of the simulator backend to WARNING level
    logging.getLogger(
        'syne_tune.backend.simulator_backend.simulator_backend').setLevel(logging.WARNING)

    if not reduction_factor:
        reduction_factor = default_params['reduction_factor']

    # we support various schedulers within the function
    # NOTE: previously we used resource_attr instead of max_resource_attr
    if hpo_approach == 'pasha':
        scheduler = baselines_dict['PASHA'](
            config_space,
            max_t=max_t,
            grace_period=default_params['grace_period'],
            reduction_factor=reduction_factor,
            resource_attr=resource_attr,
            mode=mode,
            metric=metric,
            random_seed=random_seed,
            rung_system_kwargs=rung_system_kwargs)
    elif hpo_approach == 'asha':
        scheduler = baselines_dict['ASHA'](
            config_space,
            max_t=max_t,
            grace_period=default_params['grace_period'],
            reduction_factor=reduction_factor,
            resource_attr=resource_attr,
            mode=mode,
            type='promotion',
            metric=metric,
            random_seed=random_seed)
    elif hpo_approach == 'pasha-bo':
        scheduler = HyperbandScheduler(
            config_space,
            max_t=max_t,
            grace_period=default_params['grace_period'],
            reduction_factor=reduction_factor,
            resource_attr=resource_attr,
            mode=mode,
            searcher='bayesopt',
            type='pasha',
            metric=metric,
            random_seed=random_seed,
            rung_system_kwargs=rung_system_kwargs)
    elif hpo_approach == 'asha-bo':
        scheduler = HyperbandScheduler(
            config_space,
            max_t=max_t,
            grace_period=default_params['grace_period'],
            reduction_factor=reduction_factor,
            resource_attr=resource_attr,
            mode=mode,
            searcher='bayesopt',
            type='promotion',
            metric=metric,
            random_seed=random_seed)
    else:
        raise ValueError('The selected scheduler is not implemented')

    stop_criterion = StoppingCriterion(max_num_trials_started=256)
    # printing the status during tuning takes a lot of time, and so does
    # storing results
    print_update_interval = 7000
    results_update_interval = 3000
    # it is important to set `sleep_time` to 0 here (mandatory for simulator
    # backend)

    tuner_name = 'nb201bo-' + ''.join(random.choice(string.ascii_lowercase) for _ in range(5))

    tuner = Tuner(
        trial_backend=trial_backend,
        scheduler=scheduler,
        stop_criterion=stop_criterion,
        n_workers=n_workers,
        sleep_time=0,
        results_update_interval=results_update_interval,
        print_update_interval=print_update_interval,
        # this callback is required in order to make things work with the
        # simulator callback. It makes sure that results are stored with
        # simulated time (rather than real time), and that the time_keeper
        # is advanced properly whenever the tuner loop sleeps
        callbacks=[SimulatorCallback()],
        tuner_name=tuner_name
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
    parser.add_argument("--experiment_type", type=str,
                        required=True)
    # scheduler is pasha-bo or asha-bo
    args, _ = parser.parse_known_args()


    # store the information
    metric_valid_error_dim = 0
    metric_runtime_dim = 8
    dataset_names = ['cifar10', 'cifar100', 'ImageNet16-120']
    epoch_names = ['val_acc_epoch_' + str(e) for e in range(200)]
    random_seeds = [31415927, 0, 1234, 3458, 7685]
    nb201_random_seeds = [0, 1, 2]
    n_workers = 4

    bb_dict = {}
    for dataset in dataset_names:
        bb_dict[dataset] = load_blackbox(dataset)[dataset]

    df_dict = {}

    for seed in nb201_random_seeds:
        df_dict[seed] = {}
        for dataset in dataset_names:
            # create a dataframe with the validation accuracies for various epochs
            df_val_acc = pd.DataFrame((1.0-bb_dict[dataset].objectives_evaluations[:, seed, :, metric_valid_error_dim])
                                    * 100, columns=['val_acc_epoch_' + str(e) for e in range(200)])

            # add a new column with the best validation accuracy
            df_val_acc['val_acc_best'] = df_val_acc[epoch_names].max(axis=1)
            # create a dataframe with the hyperparameter values
            df_hp = bb_dict[dataset].hyperparameters
            # create a dataframe with the times it takes to run an epoch
            df_time = pd.DataFrame(bb_dict[dataset].objectives_evaluations[:, seed, :, metric_runtime_dim][:, -1], columns=['eval_time_epoch'])    
            # combine all smaller dataframes into one dataframe for each NASBench201 random seed and dataset
            df_dict[seed][dataset] = pd.concat([df_hp, df_val_acc, df_tcal, df_time], axis=1)

    # we need to specify the following:

    benchmark_metric = 'metric_valid_error'
    rung_system_kwargs = {'ranking_criterion': 'soft_ranking_auto', 'epsilon': 0.0}

    experiment_name = run_experiment(
        args.dataset_name, args.random_seed, args.nb201_random_seed, args.scheduler, benchmark_metric=benchmark_metric, rung_system_kwargs=rung_system_kwargs)

    experiment_dict = {'experiment_name': experiment_name,
                       'dataset_name': args.dataset_name,
                       'random_seed': args.random_seed,
                       'nb201_random_seed': args.nb201_random_seed,
                       'scheduler': args.scheduler,
                       'experiment_type': args.experiment_type}

    # now we need to store the experiment_name and also the values of the arguments
    with open('bo_experiment_details_' + args.experiment_type + '.json', 'r') as f:
        current_list = json.load(f)

    current_list.append(experiment_dict)
    with open('bo_experiment_details_' + args.experiment_type + '.json', 'w') as f:
        json.dump(current_list, f)
