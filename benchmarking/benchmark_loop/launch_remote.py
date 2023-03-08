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
from argparse import ArgumentParser
from pathlib import Path

from coolname import generate_slug
from sagemaker.pytorch import PyTorch

from syne_tune.backend.sagemaker_backend.sagemaker_utils import get_execution_role
import syne_tune
import benchmarking
from syne_tune.util import s3_experiment_path

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--experiment_tag", type=str, required=False, default=generate_slug(2)
    )
    args, _ = parser.parse_known_args()
    experiment_tag = args.experiment_tag
    print(experiment_tag)
    est = PyTorch(
        entry_point="benchmark_main.py",
        source_dir=str(Path(__file__).parent),
        # instance_type="local",
        checkpoint_s3_uri=s3_experiment_path(tuner_name=experiment_tag),
        instance_type="ml.c5.4xlarge",
        instance_count=1,
        py_version="py38",
        framework_version="1.10.0",
        role=get_execution_role(),
        dependencies=syne_tune.__path__ + benchmarking.__path__,
        disable_profiler=True,
        hyperparameters={"experiment_tag": experiment_tag, "num_seeds": 30},
    )
    est.fit(job_name=experiment_tag)
