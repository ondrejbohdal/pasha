# Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.
import numpy as np
import itertools
import logging
from argparse import ArgumentParser
from tqdm import tqdm

from benchmarking.benchmark_loop.baselines import methods
from benchmarking.benchmark_loop.benchmark_definitions import benchmark_definitions
from syne_tune.blackbox_repository import BlackboxRepositoryBackend

from syne_tune.backend.simulator_backend.simulator_callback import SimulatorCallback
from syne_tune import StoppingCriterion, Tuner
from coolname import generate_slug


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_tag", type=str, required=False, default=generate_slug(2)
    )
    parser.add_argument("--num_seeds", type=int, required=False, default=2)
    parser.add_argument("--method", type=str, required=False)
    parser.add_argument("--benchmark", type=str, required=False)
    args, _ = parser.parse_known_args()
    experiment_tag = args.experiment_tag
    num_seeds = args.num_seeds
    method_names = [args.method] if args.method is not None else list(methods.keys())
    benchmark_names = (
        [args.benchmark]
        if args.benchmark is not None
        else list(benchmark_definitions.keys())
    )

    logging.getLogger("syne_tune.optimizer.schedulers").setLevel(logging.WARNING)
    logging.getLogger("syne_tune.backend").setLevel(logging.WARNING)
    logging.getLogger("syne_tune.backend.simulator_backend.simulator_backend").setLevel(
        logging.WARNING
    )

    combinations = list(
        itertools.product(method_names, range(num_seeds), benchmark_names)
    )

    print(combinations)
    for method, seed, benchmark_name in tqdm(combinations):
        np.random.seed(seed)
        benchmark = benchmark_definitions[benchmark_name]

        print(
            f"Starting experiment ({method}/{benchmark_name}/{seed}) of {experiment_tag}"
        )

        trial_backend = BlackboxRepositoryBackend(
            elapsed_time_attr=benchmark.elapsed_time_attr,
            blackbox_name=benchmark.blackbox_name,
            dataset=benchmark.dataset_name,
        )

        max_t = max(trial_backend.blackbox.fidelity_values)
        resource_attr = next(iter(trial_backend.blackbox.fidelity_space.keys()))
        scheduler = methods[method](
            config_space=trial_backend.blackbox.configuration_space,
            metric=benchmark.metric,
            mode=benchmark.mode,
            random_seed=seed,
            max_t=max_t,
            resource_attr=resource_attr,
        )

        stop_criterion = StoppingCriterion(
            max_wallclock_time=benchmark.max_wallclock_time
        )

        tuner = Tuner(
            trial_backend=trial_backend,
            scheduler=scheduler,
            stop_criterion=stop_criterion,
            n_workers=benchmark.n_workers,
            sleep_time=0,
            callbacks=[SimulatorCallback()],
            results_update_interval=600,
            print_update_interval=600,
            tuner_name=f"{experiment_tag}-{method}-{seed}-{benchmark_name}".replace(
                "_", "-"
            ),
            metadata={
                "seed": seed,
                "algorithm": method,
                "tag": experiment_tag,
                "benchmark": benchmark_name,
            },
        )
        tuner.run()
