"""
Safety-Critical Control Systems for Robotics

This project implements advanced safety-critical control systems for robotics applications,
including Control Barrier Functions (CBF), Model Predictive Control (MPC) with safety constraints,
emergency stop systems, and collision avoidance algorithms.

DISCLAIMER: This software is for research and educational purposes only. 
DO NOT use on real hardware without proper safety review and testing.
"""

import numpy as np
import random
import torch
from typing import Dict, Any, Optional, Tuple, List
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_deterministic_seeds(seed: int = 42) -> None:
    """Set deterministic seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> str:
    """Get the best available device with fallback."""
    if torch.cuda.is_available():
        return "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


# Set seeds for reproducibility
set_deterministic_seeds(42)
DEVICE = get_device()

logger.info(f"Using device: {DEVICE}")
logger.info("Safety-Critical Control Systems initialized")
