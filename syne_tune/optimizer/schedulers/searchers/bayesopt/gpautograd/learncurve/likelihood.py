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
from typing import Union, Optional, Dict
import autograd.numpy as anp

from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.model_params \
    import ISSModelParameters
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.posterior_state \
    import GaussProcISSMPosteriorState, GaussProcExpDecayPosteriorState, \
    IncrementalUpdateGPAdditivePosteriorState
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.learncurve.freeze_thaw \
    import ExponentialDecayBaseKernelFunction
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.constants \
    import INITIAL_NOISE_VARIANCE, NOISE_VARIANCE_LOWER_BOUND, \
    NOISE_VARIANCE_UPPER_BOUND, DEFAULT_ENCODING
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.distribution \
    import Gamma
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gluon \
    import Block
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.gluon_blocks_helpers \
    import encode_unwrap_parameter, register_parameter, create_encoding
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.kernel \
    import KernelFunction
from syne_tune.optimizer.schedulers.searchers.bayesopt.gpautograd.mean \
    import ScalarMeanFunction, MeanFunction
from syne_tune.optimizer.schedulers.utils.simple_profiler \
    import SimpleProfiler


LCModel = Union[ISSModelParameters, ExponentialDecayBaseKernelFunction]


class MarginalLikelihood(Block):
    """
    Marginal likelihood of joint learning curve model, where each curve is
    modelled as sum of a Gaussian process over x (for the value at r_max)
    and a Gaussian model over r.

    The latter `res_model` is either an ISSM or another Gaussian process with
    exponential decay covariance function.

    :param kernel: Kernel function k(x, x')
    :param res_model: Gaussian model over r
    :param mean: Mean function mu(x)
    :param initial_noise_variance: A scalar to initialize the value of the
        residual noise variance
    """
    def __init__(
            self, kernel: KernelFunction, res_model: LCModel,
            mean: MeanFunction = None, initial_noise_variance=None,
            encoding_type=None, **kwargs):
        super(MarginalLikelihood, self).__init__(**kwargs)
        assert isinstance(res_model, (ISSModelParameters,
                                      ExponentialDecayBaseKernelFunction)), \
            "res_model must be ISSModelParameters or ExponentialDecayBaseKernelFunction"
        if mean is None:
            mean = ScalarMeanFunction()
        if initial_noise_variance is None:
            initial_noise_variance = INITIAL_NOISE_VARIANCE
        if encoding_type is None:
            encoding_type = DEFAULT_ENCODING
        self.encoding = create_encoding(
             encoding_type, initial_noise_variance, NOISE_VARIANCE_LOWER_BOUND,
             NOISE_VARIANCE_UPPER_BOUND, 1, Gamma(mean=0.1, alpha=0.1))
        self.mean = mean
        self.kernel = kernel
        self.res_model = res_model
        if isinstance(res_model, ISSModelParameters):
            tag = 'issm_'
            self._type = GaussProcISSMPosteriorState
            self._posterstate_kwargs = {
                'mean': self.mean,
                'kernel': self.kernel,
                'iss_model': self.res_model}
        else:
            tag = 'expdecay_'
            self._type = GaussProcExpDecayPosteriorState
            self._posterstate_kwargs = {
                'mean': self.mean,
                'kernel': self.kernel,
                'res_kernel': self.res_model}
        self._components = [('kernel_', self.kernel), ('mean_', self.mean),
                            (tag, self.res_model)]
        self._profiler = None
        with self.name_scope():
            self.noise_variance_internal = register_parameter(
                self.params, 'noise_variance', self.encoding)

    def set_profiler(self, profiler: Optional[SimpleProfiler]):
        self._profiler = profiler

    def get_posterior_state(self, data: Dict, crit_only: bool = False) \
            -> IncrementalUpdateGPAdditivePosteriorState:
        return self._type(
            data, **self._posterstate_kwargs,
            noise_variance=self.get_noise_variance(),
            profiler=self._profiler)

    def forward(self, data: Dict):
        """
        The criterion is the negative log marginal likelihood of `data`, which
        is obtained from `issm.prepare_data`.
        Depending on `self._type`, `data` may also have to contain precomputed
        values.

        :param data: Input points (features, configs), targets
        """
        assert not data['do_fantasizing'], \
            "data must not be for fantasizing. Call prepare_data with " +\
            "do_fantasizing=False"
        return self.get_posterior_state(
            data, crit_only=True).neg_log_likelihood()

    def param_encoding_pairs(self):
        """
        Return a list of tuples with the Gluon parameters of the likelihood and
        their respective encodings
        """
        own_param_encoding_pairs = [
            (self.noise_variance_internal, self.encoding)]
        return own_param_encoding_pairs + self.mean.param_encoding_pairs() + \
               self.kernel.param_encoding_pairs() + \
               self.res_model.param_encoding_pairs()

    def box_constraints_internal(self):
        """
        Collect the box constraints for all the underlying parameters
        """
        all_box_constraints = {}
        for param, encoding in self.param_encoding_pairs():
            assert encoding is not None,\
                "encoding of param {} should not be None".format(param.name)
            all_box_constraints.update(encoding.box_constraints_internal(param))
        return all_box_constraints

    def get_noise_variance(self, as_ndarray=False):
        noise_variance = encode_unwrap_parameter(
            self.noise_variance_internal, self.encoding)
        return noise_variance if as_ndarray else anp.reshape(noise_variance, (1,))[0]

    def set_noise_variance(self, val):
        self.encoding.set(self.noise_variance_internal, val)

    def get_params(self):
        result = {'noise_variance': self.get_noise_variance()}
        for pref, func in self._components:
            result.update({
                (pref + k): v for k, v in func.get_params().items()})
        return result
    
    def set_params(self, param_dict):
        for pref, func in self._components:
            len_pref = len(pref)
            stripped_dict = {
                k[len_pref:]: v for k, v in param_dict.items()
                if k.startswith(pref)}
            func.set_params(stripped_dict)
        self.set_noise_variance(param_dict['noise_variance'])

    def data_precomputations(self, data: Dict):
        """
        For some `res_model` types, precomputations on top of `data` are
        needed. This is done here, and the precomputed variables are appended
        to `data` as extra entries.
        """
        self._type.data_precomputations(data)
