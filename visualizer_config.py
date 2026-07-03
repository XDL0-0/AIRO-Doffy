from dataclasses import dataclass


@dataclass
class VisualizerConfig:
    ENABLED: bool = True
    HZ: float = 30.0
    WINDOW_S: float = 8.0
    FORCE_PANEL_RANGE: float = 30.0  # N, +/- range for force plots and Fx/Fy panel.

    def __post_init__(self) -> None:
        if self.HZ <= 0:
            raise ValueError("VisualizerConfig.HZ must be positive.")
        if self.WINDOW_S <= 0:
            raise ValueError("VisualizerConfig.WINDOW_S must be positive.")
        if self.FORCE_PANEL_RANGE <= 0:
            raise ValueError("VisualizerConfig.FORCE_PANEL_RANGE must be positive.")
