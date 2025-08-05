"""
Base Pipeline Classes
Provides base functionality for all ML pipelines
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class BasePipeline(ABC):
    """Base class for all ML pipelines"""
    
    def __init__(self, 
                 runpod_manager=None,
                 wandb_tracker=None,
                 config: Dict[str, Any] = None):
        """Initialize base pipeline
        
        Args:
            runpod_manager: RunPodManager instance
            wandb_tracker: WandbTracker instance
            config: Pipeline configuration
        """
        self.runpod_manager = runpod_manager
        self.wandb_tracker = wandb_tracker
        self.config = config or {}
        
        # Pipeline state
        self.is_initialized = False
        self.current_step = 0
        self.results = {}
        
    def setup(self) -> bool:
        """Setup pipeline prerequisites
        
        Returns:
            True if setup successful, False otherwise
        """
        try:
            # Initialize tracking if available
            if self.wandb_tracker and not self.wandb_tracker.is_initialized:
                self.wandb_tracker.initialize()
            
            self.is_initialized = True
            logger.info(f"Pipeline {self.__class__.__name__} setup completed")
            return True
            
        except Exception as e:
            logger.error(f"Pipeline setup failed: {e}")
            return False
    
    @abstractmethod
    def run(self, **kwargs) -> Dict[str, Any]:
        """Run the pipeline
        
        Returns:
            Dictionary with results
        """
        pass
    
    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: int = None):
        """Log metrics if tracker is available
        
        Args:
            metrics: Metrics to log
            step: Step number
        """
        if self.wandb_tracker:
            self.wandb_tracker.log_metrics(metrics, step)
    
    def save_results(self, results: Dict[str, Any], filepath: str = None):
        """Save pipeline results
        
        Args:
            results: Results to save
            filepath: Path to save results
        """
        import json
        
        if filepath is None:
            filepath = f"results_{self.__class__.__name__.lower()}.json"
        
        # Ensure results directory exists
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {filepath}")
    
    def cleanup(self):
        """Cleanup pipeline resources"""
        if self.wandb_tracker:
            self.wandb_tracker.finish()
        
        if self.runpod_manager:
            self.runpod_manager.close_connection()
        
        logger.info(f"Pipeline {self.__class__.__name__} cleanup completed")
    
    def __enter__(self):
        """Context manager entry"""
        self.setup()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()


# TrainingPipeline removed - this project is inference-only


class InferencePipeline(BasePipeline):
    """Base class for inference pipelines"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = None
        
    @abstractmethod
    def load_model(self, model_path: str) -> Any:
        """Load and return the model
        
        Args:
            model_path: Path to model file
            
        Returns:
            Loaded model
        """
        pass
    
    @abstractmethod
    def preprocess_input(self, input_data: Any) -> Any:
        """Preprocess input data
        
        Args:
            input_data: Raw input data
            
        Returns:
            Preprocessed data
        """
        pass
    
    @abstractmethod
    def predict(self, input_data: Any) -> Any:
        """Run prediction
        
        Args:
            input_data: Preprocessed input data
            
        Returns:
            Prediction results
        """
        pass
    
    @abstractmethod
    def postprocess_output(self, predictions: Any) -> Any:
        """Postprocess predictions
        
        Args:
            predictions: Raw predictions
            
        Returns:
            Processed predictions
        """
        pass