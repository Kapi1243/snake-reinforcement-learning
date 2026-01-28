"""
Experiment tracking and logging utilities.

Supports multiple backends:
- TensorBoard for visualization
- Weights & Biases (wandb) for cloud-based experiment tracking
- JSON logging for lightweight tracking

Author: Kacper Kowalski
Date: January 2026
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import json
from datetime import datetime
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class ExperimentLogger:
    """
    Unified interface for experiment tracking across multiple backends.
    
    Automatically logs metrics to all available backends and maintains
    a JSON log for reproducibility.
    """
    
    def __init__(
        self,
        experiment_name: str,
        project_name: str = "snake-rl",
        log_dir: str = "runs",
        config: Optional[Dict[str, Any]] = None,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_entity: Optional[str] = None
    ):
        """
        Initialize experiment logger.
        
        Args:
            experiment_name: Name of this experiment
            project_name: Name of the overall project
            log_dir: Directory for logs
            config: Configuration dictionary to log
            use_tensorboard: Whether to use TensorBoard
            use_wandb: Whether to use Weights & Biases
            wandb_entity: W&B entity (username/team)
        """
        self.experiment_name = experiment_name
        self.project_name = project_name
        self.start_time = datetime.now()
        
        # Create log directory
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")
        self.log_dir = Path(log_dir) / f"{experiment_name}_{timestamp}"
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize backends
        self.tensorboard_writer = None
        self.wandb_run = None
        self.json_log: List[Dict[str, Any]] = []
        
        # TensorBoard
        if use_tensorboard and TENSORBOARD_AVAILABLE:
            self.tensorboard_writer = SummaryWriter(log_dir=str(self.log_dir))
            print(f"TensorBoard logging to: {self.log_dir}")
        elif use_tensorboard and not TENSORBOARD_AVAILABLE:
            print("Warning: TensorBoard requested but not installed. Install with: pip install tensorboard")
        
        # Weights & Biases
        if use_wandb and WANDB_AVAILABLE:
            self.wandb_run = wandb.init(
                project=project_name,
                name=experiment_name,
                entity=wandb_entity,
                config=config,
                dir=str(self.log_dir)
            )
            print(f"W&B run: {self.wandb_run.url}")
        elif use_wandb and not WANDB_AVAILABLE:
            print("Warning: W&B requested but not installed. Install with: pip install wandb")
        
        # Save config
        if config:
            config_path = self.log_dir / "config.json"
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=2)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        commit: bool = True
    ) -> None:
        """
        Log metrics to all backends.
        
        Args:
            metrics: Dictionary of metric_name -> value
            step: Global step number
            commit: Whether to commit to W&B (allows batching)
        """
        # JSON log
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'step': step,
            **metrics
        }
        self.json_log.append(log_entry)
        
        # TensorBoard
        if self.tensorboard_writer is not None:
            for key, value in metrics.items():
                self.tensorboard_writer.add_scalar(key, value, step)
        
        # W&B
        if self.wandb_run is not None:
            wandb.log(metrics, step=step, commit=commit)
    
    def log_histogram(
        self,
        name: str,
        values: np.ndarray,
        step: Optional[int] = None
    ) -> None:
        """Log histogram of values."""
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_histogram(name, values, step)
        
        if self.wandb_run is not None:
            wandb.log({name: wandb.Histogram(values)}, step=step)
    
    def log_image(
        self,
        name: str,
        image: np.ndarray,
        step: Optional[int] = None
    ) -> None:
        """
        Log image.
        
        Args:
            name: Name/tag for the image
            image: Image array (H, W, C) or (C, H, W)
            step: Global step number
        """
        if self.tensorboard_writer is not None:
            # TensorBoard expects (C, H, W)
            if image.ndim == 3 and image.shape[-1] in [1, 3, 4]:
                image = np.transpose(image, (2, 0, 1))
            self.tensorboard_writer.add_image(name, image, step)
        
        if self.wandb_run is not None:
            wandb.log({name: wandb.Image(image)}, step=step)
    
    def log_text(self, name: str, text: str, step: Optional[int] = None) -> None:
        """Log text."""
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_text(name, text, step)
        
        if self.wandb_run is not None:
            wandb.log({name: text}, step=step)
    
    def log_hyperparameters(self, hparams: Dict[str, Any], metrics: Dict[str, float]) -> None:
        """Log hyperparameters and final metrics."""
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_hparams(hparams, metrics)
    
    def save_checkpoint(self, filepath: str, metadata: Optional[Dict] = None) -> None:
        """
        Save checkpoint metadata.
        
        Args:
            filepath: Path where model was saved
            metadata: Additional metadata to log
        """
        checkpoint_info = {
            'filepath': filepath,
            'timestamp': datetime.now().isoformat(),
            **(metadata or {})
        }
        
        checkpoints_log = self.log_dir / "checkpoints.json"
        
        if checkpoints_log.exists():
            with open(checkpoints_log, 'r') as f:
                checkpoints = json.load(f)
        else:
            checkpoints = []
        
        checkpoints.append(checkpoint_info)
        
        with open(checkpoints_log, 'w') as f:
            json.dump(checkpoints, f, indent=2)
    
    def finish(self) -> None:
        """Close all logging backends and save final logs."""
        # Save JSON log
        log_path = self.log_dir / "metrics.json"
        with open(log_path, 'w') as f:
            json.dump(self.json_log, f, indent=2)
        
        # Close TensorBoard
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
        
        # Finish W&B
        if self.wandb_run is not None:
            wandb.finish()
        
        duration = datetime.now() - self.start_time
        print(f"\nExperiment finished. Duration: {duration}")
        print(f"Logs saved to: {self.log_dir}")


class MetricsAggregator:
    """
    Aggregate metrics over multiple episodes/steps.
    
    Useful for computing running averages, min/max, etc.
    """
    
    def __init__(self, window_size: int = 100):
        """
        Initialize metrics aggregator.
        
        Args:
            window_size: Number of recent values to track
        """
        from collections import defaultdict, deque
        
        self.window_size = window_size
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
    
    def update(self, **kwargs) -> None:
        """Add new metric values."""
        for key, value in kwargs.items():
            self.metrics[key].append(value)
    
    def get_stats(self, metric_name: str) -> Dict[str, float]:
        """
        Get statistics for a metric.
        
        Returns:
            Dictionary with mean, std, min, max
        """
        if metric_name not in self.metrics or len(self.metrics[metric_name]) == 0:
            return {'mean': 0.0, 'std': 0.0, 'min': 0.0, 'max': 0.0}
        
        values = list(self.metrics[metric_name])
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values)
        }
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all metrics."""
        return {name: self.get_stats(name) for name in self.metrics.keys()}


class PerformanceMonitor:
    """
    Monitor training performance and detect issues.
    
    Tracks metrics like:
    - Training speed (episodes/sec, steps/sec)
    - Convergence indicators
    - Performance plateaus
    """
    
    def __init__(self):
        """Initialize performance monitor."""
        self.start_time = datetime.now()
        self.episode_count = 0
        self.step_count = 0
        self.last_report_time = self.start_time
        self.last_report_episode = 0
        self.last_report_step = 0
        
        self.recent_scores = []
        self.max_score = float('-inf')
        self.plateau_counter = 0
    
    def update(self, episode: int, step: int, score: float) -> Dict[str, float]:
        """
        Update performance metrics.
        
        Args:
            episode: Current episode number
            step: Current step count
            score: Episode score
            
        Returns:
            Dictionary of performance metrics
        """
        self.episode_count = episode
        self.step_count = step
        self.recent_scores.append(score)
        
        if len(self.recent_scores) > 100:
            self.recent_scores.pop(0)
        
        # Calculate speeds
        now = datetime.now()
        time_elapsed = (now - self.last_report_time).total_seconds()
        
        if time_elapsed > 0:
            episodes_per_sec = (episode - self.last_report_episode) / time_elapsed
            steps_per_sec = (step - self.last_report_step) / time_elapsed
        else:
            episodes_per_sec = 0
            steps_per_sec = 0
        
        # Track improvement
        avg_recent_score = np.mean(self.recent_scores) if self.recent_scores else 0
        if avg_recent_score > self.max_score:
            self.max_score = avg_recent_score
            self.plateau_counter = 0
        else:
            self.plateau_counter += 1
        
        # Update report time
        self.last_report_time = now
        self.last_report_episode = episode
        self.last_report_step = step
        
        return {
            'episodes_per_sec': episodes_per_sec,
            'steps_per_sec': steps_per_sec,
            'avg_recent_score': avg_recent_score,
            'max_avg_score': self.max_score,
            'plateau_counter': self.plateau_counter
        }
    
    def is_plateaued(self, patience: int = 1000) -> bool:
        """Check if training has plateaued."""
        return self.plateau_counter > patience


if __name__ == "__main__":
    # Example usage
    print(f"TensorBoard available: {TENSORBOARD_AVAILABLE}")
    print(f"W&B available: {WANDB_AVAILABLE}")
    
    # Demo logger
    logger = ExperimentLogger(
        experiment_name="demo",
        config={'learning_rate': 0.001, 'gamma': 0.99}
    )
    
    for step in range(10):
        logger.log_metrics({
            'score': np.random.randint(0, 100),
            'loss': np.random.random()
        }, step=step)
    
    logger.finish()
    print("Demo complete!")
