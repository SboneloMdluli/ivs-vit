"""Joint return / IV-surface scenario generation with arbitrage-aware VolGAN weights."""

from implied_volatility_diffusion.scenarios.generators import (
    CallableJointScenarioGenerator,
    FilteredHistoricalSettings,
    FilteredHistoricalSimulation,
    JointScenarioGenerator,
)
from implied_volatility_diffusion.scenarios.penalty import SurfaceArbitragePenalty, SurfaceArbitrageWeights
from implied_volatility_diffusion.scenarios.pipeline import (
    generate_weighted_joint_scenarios,
    penalize_and_weight_iv_surfaces,
    penalize_and_weight_iv_surfaces_torch,
    penalize_iv_surfaces,
    weight_scenarios_from_penalties,
)
from implied_volatility_diffusion.scenarios.types import (
    JointHistoricalState,
    JointScenarioBatch,
    PenaltyWeightingResult,
)
from implied_volatility_diffusion.scenarios.weighting import (
    volgan_exponential_weights,
    volgan_exponential_weights_torch,
)

__all__ = [
    "CallableJointScenarioGenerator",
    "FilteredHistoricalSettings",
    "FilteredHistoricalSimulation",
    "JointHistoricalState",
    "JointScenarioBatch",
    "JointScenarioGenerator",
    "PenaltyWeightingResult",
    "SurfaceArbitragePenalty",
    "SurfaceArbitrageWeights",
    "generate_weighted_joint_scenarios",
    "penalize_and_weight_iv_surfaces",
    "penalize_and_weight_iv_surfaces_torch",
    "penalize_iv_surfaces",
    "volgan_exponential_weights",
    "volgan_exponential_weights_torch",
    "weight_scenarios_from_penalties",
]
