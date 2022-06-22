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
import logging

from syne_tune.optimizer.schedulers.searchers.gp_fifo_searcher import \
    GPFIFOSearcher
from syne_tune.optimizer.schedulers.searchers.constrained_gp_fifo_searcher \
    import ConstrainedGPFIFOSearcher
from syne_tune.optimizer.schedulers.searchers.cost_aware_gp_fifo_searcher \
    import CostAwareGPFIFOSearcher
from syne_tune.optimizer.schedulers.searchers.gp_multifidelity_searcher \
    import GPMultiFidelitySearcher
from syne_tune.optimizer.schedulers.searchers.gp_sync_multifidelity_searcher \
    import GPSyncMultiFidelitySearcher
from syne_tune.optimizer.schedulers.searchers.cost_aware_gp_multifidelity_searcher \
    import CostAwareGPMultiFidelitySearcher
from syne_tune.optimizer.schedulers.searchers.searcher import \
    RandomSearcher

__all__ = ['searcher_factory']

logger = logging.getLogger(__name__)


_OUR_ASYNC_MULTIFIDELITY_SCHEDULERS = {
    'hyperband_stopping', 'hyperband_promotion', 'hyperband_cost_promotion',
    'hyperband_pasha'}


_OUR_MULTIFIDELITY_SCHEDULERS = _OUR_ASYNC_MULTIFIDELITY_SCHEDULERS.union(
    {'hyperband_synchronous'})


def searcher_factory(searcher_name, **kwargs):
    """Factory for searcher objects

    This function creates searcher objects from string argument name and
    additional kwargs. It is typically called in the constructor of a
    scheduler (see :class:`FIFOScheduler`), which provides most of the required
    kwargs.

    """
    supported_schedulers = None
    scheduler = kwargs.get('scheduler')
    model = kwargs.get('model', 'gp_multitask')
    if searcher_name == 'random':
        searcher_cls = RandomSearcher
    elif searcher_name == 'kde':
        from syne_tune.optimizer.schedulers.searchers.kde_searcher import \
            KernelDensityEstimator
        from syne_tune.optimizer.schedulers.searchers.multi_fidelity_kde_searcher \
            import MultiFidelityKernelDensityEstimator

        if scheduler == 'fifo':
            searcher_cls = KernelDensityEstimator
        else:
            supported_schedulers = _OUR_MULTIFIDELITY_SCHEDULERS
            searcher_cls = MultiFidelityKernelDensityEstimator
    elif searcher_name == 'bayesopt':
        if scheduler == 'fifo':
            searcher_cls = GPFIFOSearcher
        elif scheduler == 'hyperband_synchronous':
            searcher_cls = GPSyncMultiFidelitySearcher
        else:
            supported_schedulers = _OUR_ASYNC_MULTIFIDELITY_SCHEDULERS
            if model == 'gp_multitask' and \
                    kwargs.get('gp_resource_kernel') == 'freeze-thaw':
                logger.warning(
                    "You are combining model = gp_multitask with "
                    "gp_resource_kernel = freeze-thaw. This is mainly "
                    "for debug purposes. The same surrogate model is "
                    "obtained with model = gp_expdecay, but computations "
                    "are faster then.")
            searcher_cls = GPMultiFidelitySearcher
    elif searcher_name == 'bayesopt_constrained':
        supported_schedulers = {'fifo'}
        searcher_cls = ConstrainedGPFIFOSearcher
    elif searcher_name == 'bayesopt_cost':
        if scheduler == 'fifo':
            searcher_cls = CostAwareGPFIFOSearcher
        else:
            supported_schedulers = _OUR_ASYNC_MULTIFIDELITY_SCHEDULERS
            searcher_cls = CostAwareGPMultiFidelitySearcher
    else:
        raise AssertionError(f"searcher '{searcher_name}' is not supported")

    if supported_schedulers is not None:
        assert scheduler is not None, \
            "Scheduler must set search_options['scheduler']"
        assert scheduler in supported_schedulers, \
            f"Searcher '{searcher_name}' only works with schedulers " + \
            f"{supported_schedulers} (not with '{scheduler}')"
    searcher = searcher_cls(**kwargs)
    return searcher
