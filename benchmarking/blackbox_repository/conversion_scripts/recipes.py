from benchmarking.blackbox_repository.conversion_scripts.scripts.nasbench201_import import generate_nasbench201
from benchmarking.blackbox_repository.conversion_scripts.scripts.fcnet_import import generate_fcnet
from benchmarking.blackbox_repository.conversion_scripts.scripts.icml2020_import import generate_deepar, generate_xgboost

generate_blackbox_recipe = {
    "icml-deepar": generate_deepar,
    "icml-xgboost": generate_xgboost,
    "nasbench201": generate_nasbench201,
    "fcnet": generate_fcnet,
}
