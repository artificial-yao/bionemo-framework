# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
import torch
from torch import nn

from bionemo.model.molecule.moco.interpolant.continuous_diffusion import ContinuousDiffusionInterpolant
from bionemo.model.molecule.moco.interpolant.continuous_euclidean_fm import ContinuousFlowMatchingInterpolant
from bionemo.model.molecule.moco.interpolant.discrete_diffusion import DiscreteDiffusionInterpolant
from bionemo.model.molecule.moco.interpolant.discrete_fm import DiscreteFlowMatchingInterpolant
from bionemo.model.molecule.moco.interpolant.interpolant_utils import float_time_to_index


def build_interpolant(
    interpolant_type: str,
    prior_type: str = "uniform",
    vector_field_type: str = "standard",
    diffusion_type: str = "d3pm",
    solver_type: str = "sde",
    time_type: str = 'continuous',
    scheduler_type='cosine_adaptive',
    scheduler_cut: bool = False,
    s: float = 0.008,
    sqrt: bool = False,
    nu: float = 1.0,
    clip: bool = True,
    timesteps: int = 500,
    num_classes: int = 10,
    min_t: float = 1e-2,
    custom_prior: torch.Tensor = None,
    com_free: bool = True,
    variable_name: str = None,
    concat: str = None,
    offset: int = 0,
    noise_sigma: float = 0.0,
    optimal_transport: str = None
    # TODO: here is where we add all the possible things that could go into any interpolant class
):
    """
     Builds an interpolant for the specified configuration.

    The interpolant is constructed based on various parameters that define the type of interpolation,
    prior distribution, update mechanism, diffusion process, solver method, and other configurations.

    Parameters:
    -----------
    interpolant_type : str
        The type of interpolant to build.
    prior_type : str, optional
        The type of prior distribution. Default is "uniform".
    vector_field_type : str, optional
        The type of vector field to use for update. Default is "standard".
    diffusion_type : str, optional
        The type of diffusion process. Default is "d3pm".
    solver_type : str, optional
        The type of solver to use. Default is "sde".
    time_type : str, optional
        The type of time representation. Default is 'continuous'.
    scheduler_type : str, optional
        The type of scheduler to use. Default is 'cosine_adaptive'.
    scheduler_cut : bool, optional
        Whether to apply a scheduler cut. Default is False.
    s : float, optional
        A parameter for the scheduler. Default is 0.008.
    sqrt : bool, optional
        Whether to apply a square root transformation. Default is False.
    nu : float, optional
        A parameter for the scheduler. Default is 1.0.
    clip : bool, optional
        Whether to clip the values. Default is True.
    timesteps : int, optional
        The number of timesteps. Default is 500.
    num_classes : int, optional
        The number of classes. Default is 10.
    min_t : float, optional
        The minimum time value. Default is 1e-2.
    custom_prior : torch.Tensor, optional
        A custom prior distribution. Default is None.
    com_free : bool, optional
        Whether to use a center-of-mass-free configuration. Default is True.
    variable_name : str, optional
        The name of the variable to use. Default is None.
    concat : str, optional
        Concatenation target variable. Default is None.

    Returns:
    --------
    Interpolant
        The constructed interpolant object.

    Notes:
    ------
    The setup for uniform and absorbing priors is assumed to be the same, and the +1 mask state is controlled
    to ensure that the number of classes remains constant in the configuration, representing the desired number
    of classes to model.
    """
    if interpolant_type == "continuous_diffusion":
        return ContinuousDiffusionInterpolant(
            prior_type,
            diffusion_type,
            solver_type,
            timesteps,
            time_type,
            num_classes,
            scheduler_type,
            s,
            sqrt,
            nu,
            clip,
            com_free,
            scheduler_cut,
        )
    elif interpolant_type == "continuous_flow_matching":
        return ContinuousFlowMatchingInterpolant(
            prior_type,
            vector_field_type,
            "ode",
            timesteps,
            min_t,
            time_type,
            num_classes,
            scheduler_type,
            s,
            sqrt,
            nu,
            clip,
            com_free,
            noise_sigma,
            optimal_transport,
        )
    elif interpolant_type == "discrete_diffusion":
        if prior_type in ["absorb", "mask"]:
            num_classes = num_classes + 1
        return DiscreteDiffusionInterpolant(
            prior_type,
            diffusion_type,
            solver_type,
            timesteps,
            time_type,
            num_classes,
            custom_prior,
            scheduler_type,
            s,
            sqrt,
            nu,
            clip,
            scheduler_cut,
        )
    elif interpolant_type == "discrete_flow_matching":
        if prior_type in ["absorb", "mask"]:
            num_classes = num_classes + 1
        return DiscreteFlowMatchingInterpolant(
            prior_type,
            vector_field_type,
            "ode",
            timesteps,
            min_t,
            time_type,
            num_classes,
            custom_prior,
            scheduler_type,
            s,
            sqrt,
            nu,
            clip,
        )
    elif interpolant_type is None:
        #! This is to allow variables we just want to pass in and not noise/denoise
        return None
    else:
        raise NotImplementedError('Interpolant not supported: %s' % interpolant_type)


class Interpolant(nn.Module):
    def __init__(
        self,
        prior_type: str,
        solver_type: str = "sde",
        timesteps: int = 500,
        time_type: str = 'discrete',
    ):
        super(Interpolant, self).__init__()
        self.prior_type = prior_type
        self.timesteps = timesteps
        self.solver_type = solver_type
        self.time_type = time_type

    def sample_time(self, num_samples, method='uniform', device='cpu', mean=0, scale=0.81, min_t=0):
        if self.time_type == "continuous":
            return self.sample_time_continuous(num_samples, method, device, mean, scale, min_t)
        else:
            return self.sample_time_idx(num_samples, method, device, mean, scale)

    def sample_time_idx(self, num_samples, method, device='cpu', mean=0, scale=0.81):
        if method == 'symmetric':
            time_step = torch.randint(0, self.timesteps, size=(num_samples // 2 + 1,))
            time_step = torch.cat([time_step, self.timesteps - time_step - 1], dim=0)[:num_samples]

        elif method == 'uniform':
            time_step = torch.randint(0, self.timesteps, size=(num_samples,))

        elif method == "stab_mode":  #! converts uniform to Stability AI mode distribution

            def fmode(u: torch.Tensor, s: float) -> torch.Tensor:
                return 1 - u - s * (torch.cos((torch.pi / 2) * u) ** 2 - 1 + u)

            time_step = float_time_to_index(fmode(torch.rand(num_samples), scale), self.timesteps)
        elif (
            method == 'logit_normal'
        ):  # see Figure 11 https://stabilityai-public-packages.s3.us-west-2.amazonaws.com/Stable+Diffusion+3+Paper.pdf
            time_step = float_time_to_index(
                torch.sigmoid(torch.normal(mean=mean, std=scale, size=(num_samples,))), self.timesteps
            )
        elif method == "beta":
            dist = torch.distributions.Beta(2.0, 1.0)
            time_step = float_time_to_index(dist.sample([num_samples]), self.timesteps)
        else:
            raise ValueError
        return time_step.to(device)

    def sample_time_continuous(self, num_samples, method, device='cpu', mean=0, scale=0.81, min_t=0):
        if method == 'symmetric':
            time_step = torch.rand(num_samples // 2 + 1)
            time_step = torch.cat([time_step, 1 - time_step], dim=0)[:num_samples]

        elif method == 'uniform':
            time_step = torch.rand(num_samples)
        elif method == "stab_mode":  #! converts uniform to Stability AI mode distribution

            def fmode(u: torch.Tensor, s: float) -> torch.Tensor:
                return 1 - u - s * (torch.cos((torch.pi / 2) * u) ** 2 - 1 + u)

            time_step = fmode(torch.rand(num_samples), scale)
        elif (
            method == 'logit_normal'
        ):  # see Figure 11 https://stabilityai-public-packages.s3.us-west-2.amazonaws.com/Stable+Diffusion+3+Paper.pdf
            time_step = torch.sigmoid(torch.normal(mean=mean, std=scale, size=(num_samples,)))
        elif method == "beta":  # From MolFlow
            dist = torch.distributions.Beta(2.0, 1.0)
            time_step = dist.sample([num_samples])
        else:
            raise ValueError
        if min_t > 0:
            time_step = time_step * (1 - 2 * min_t) + min_t
        return time_step.to(device)

    def snr_loss_weight(self, time):
        return torch.clamp(self.snr(time), min=0.05, max=1.5)
