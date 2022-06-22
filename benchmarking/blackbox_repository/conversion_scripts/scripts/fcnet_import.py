"""
Convert tabular data from
 Tabular Benchmarks for Joint Architecture and Hyperparameter Optimization
 Aaron Klein Frank Hutter
 https://arxiv.org/pdf/1905.04970.pdf.
"""
import urllib
import tarfile

from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np
import ast
import h5py

from benchmarking.blackbox_repository.blackbox_tabular import serialize, BlackboxTabular
from benchmarking.blackbox_repository.conversion_scripts.utils import repository_path

from syne_tune.util import catchtime
import syne_tune.search_space as sp


BLACKBOX_NAME = 'fcnet'

METRIC_VALID_LOSS = 'metric_valid_loss'

METRIC_ELAPSED_TIME = 'metric_elapsed_time'

RESOURCE_ATTR = 'hp_epoch'


def convert_dataset(dataset_path: Path, max_rows: int = None):
    data = h5py.File(dataset_path, "r")
    keys = data.keys()
    if max_rows is not None:
        keys = list(keys)[:max_rows]

    hyperparameters = pd.DataFrame(ast.literal_eval(key) for key in keys)
    hyperparameters.rename(columns={col: "hp_" + col for col in hyperparameters.columns}, inplace=True)

    objective_names = [
        'valid_loss',
        'train_loss',
        'final_test_error',
        'n_params',
        'elapsed_time',
    ]

    # todo for now only full metrics
    fidelity_values = np.arange(1, 101)
    n_fidelities = len(fidelity_values)
    n_objectives = len(objective_names)
    n_seeds = 4
    n_hps = len(keys)

    objective_evaluations = np.empty((n_hps, n_seeds, n_fidelities, n_objectives)).astype('float32')

    def save_objective_values_helper(name, values):
        assert values.shape == (n_hps, n_seeds, n_fidelities)

        name_index = dict(zip(
            objective_names,
            range(len(objective_names)))
        )
        objective_evaluations[..., name_index[name]] = values

    # (n_hps, n_seeds,)
    final_test_error = np.stack([data[key]['final_test_error'][:].astype('float32') for key in keys])

    # (n_hps, n_seeds, n_fidelities)
    final_test_error = np.repeat(np.expand_dims(final_test_error, axis=-1), n_fidelities, axis=-1)
    save_objective_values_helper('final_test_error', final_test_error)

    # (n_hps, n_seeds,)
    n_params = np.stack([data[key]['n_params'][:].astype('float32') for key in keys])

    # (n_hps, n_seeds, n_fidelities)
    n_params = np.repeat(np.expand_dims(n_params, axis=-1), n_fidelities, axis=-1)
    save_objective_values_helper('n_params', n_params)

    # (n_hps, n_seeds,)
    runtime = np.stack([data[key]['runtime'][:].astype('float32') for key in keys])

    # linear interpolation to go from total training time to training time per epoch as in fcnet code
    # (n_hps, n_seeds, n_epochs)
    # todo utilize expand dim instead of reshape
    epochs = np.repeat(np.arange(1, 101).reshape(1, -1), n_hps * n_seeds, axis=0).reshape(n_hps, n_seeds, -1)
    elapsed_time = (epochs / 100) * runtime.reshape((n_hps, n_seeds, 1))
    
    save_objective_values_helper('elapsed_time', elapsed_time)

    # metrics that are fully observed, only use train/valid loss as mse are the same numbers
    # for m in ['train_loss', 'train_mse', 'valid_loss', 'valid_mse']:
    for m in ['train_loss', 'valid_loss']:
        save_objective_values_helper(
            m,
            np.stack([data[key][m][:].astype('float32') for key in keys])
        )

    configuration_space = {
        "hp_activation_fn_1": sp.choice(["tanh", "relu"]),
        "hp_activation_fn_2": sp.choice(["tanh", "relu"]),
        "hp_batch_size": sp.choice([8, 16, 32, 64]),
        "hp_dropout_1": sp.choice([0.0, 0.3, 0.6]),
        "hp_dropout_2": sp.choice([0.0, 0.3, 0.6]),
        "hp_init_lr": sp.choice([0.0005, 0.001, 0.005, 0.01, 0.05, 0.1]),
        'hp_lr_schedule': sp.choice(["cosine", "const"]),
        'hp_n_units_1': sp.choice([16, 32, 64, 128, 256, 512]),
        'hp_n_units_2': sp.choice([16, 32, 64, 128, 256, 512]),
    }
    fidelity_space = {
        RESOURCE_ATTR: sp.randint(lower=1, upper=100)
    }

    objective_names = [f"metric_{m}" for m in objective_names]
    # Sanity checks:
    assert objective_names[0] == METRIC_VALID_LOSS
    assert objective_names[4] == METRIC_ELAPSED_TIME
    return BlackboxTabular(
        hyperparameters=hyperparameters,
        configuration_space=configuration_space,
        fidelity_space=fidelity_space,
        objectives_evaluations=objective_evaluations,
        fidelity_values=fidelity_values,
        objectives_names=objective_names,
    )


def generate_fcnet(s3_root: Optional[str] = None):
    blackbox_name = BLACKBOX_NAME
    fcnet_file = repository_path / "fcnet_tabular_benchmarks.tar.gz"
    if not (repository_path / "fcnet_tabular_benchmarks.tar.gz").exists():
        src = "http://ml4aad.org/wp-content/uploads/2019/01/fcnet_tabular_benchmarks.tar.gz"
        print(f"did not find {fcnet_file}, downloading {src}")
        urllib.request.urlretrieve(src, fcnet_file)

    with tarfile.open(repository_path / "fcnet_tabular_benchmarks.tar.gz") as f:
        f.extractall(path=repository_path)

    with catchtime("converting"):
        bb_dict = {}
        for dataset in ['protein_structure', 'naval_propulsion', 'parkinsons_telemonitoring', 'slice_localization']:
            print(f"converting {dataset}")
            dataset_path = repository_path / "fcnet_tabular_benchmarks" / f"fcnet_{dataset}_data.hdf5"
            bb_dict[dataset] = convert_dataset(dataset_path=dataset_path)

    with catchtime("saving to disk"):
        serialize(bb_dict=bb_dict, path=repository_path / blackbox_name)

    with catchtime("uploading to s3"):
        from benchmarking.blackbox_repository.conversion_scripts.utils import upload
        upload(blackbox_name, s3_root=s3_root)


def plot_learning_curves():
    import matplotlib.pyplot as plt
    from benchmarking.blackbox_repository.repository import load
    # plot one learning-curve for sanity-check
    bb_dict = load(BLACKBOX_NAME)

    b = bb_dict['naval_propulsion']
    configuration = {k: v.sample() for k, v in b.configuration_space.items()}
    print(configuration)
    errors = []
    for i in range(1, 101):
        res = b.objective_function(configuration=configuration, fidelity={'epochs': i})
        errors.append(res[METRIC_VALID_LOSS])
    plt.plot(errors)


if __name__ == '__main__':
    generate_fcnet()

    # plot_learning_curves()
