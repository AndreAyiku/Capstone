"""
Model Manager - Handles loading and switching between different LLM models
"""
import gc
import os

# Force offline mode BEFORE importing transformers — skips HuggingFace Hub validation
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from typing import Tuple, Optional


class ModelManager:
    """
    Manages loading, unloading, and switching between different language models.
    """
    
    # Model configurations
    MODELS = {
        "gemma-2-9b": {
            "path": "/home/compute.ashesi.lan/andre.ayiku/Andre_Ayiku/models/gemma-2-9b-it/",
            "size": "9B parameters",
            "recommended_for": "Complex reasoning, detailed responses"
        },
        "gemma-3-4b": {
            "path": "/home/compute.ashesi.lan/andre.ayiku/Andre_Ayiku/models/gemma-3-4b-it/",
            "size": "4B parameters",
            "recommended_for": "Fast inference, general conversation"
        },
        "llama-3.1-8b": {
            "path": "/home/compute.ashesi.lan/andre.ayiku/Andre_Ayiku/models/Llama-3.1-8B-Instruct/",
            "size": "8B parameters",
            "recommended_for": "Balanced performance"
        },
        "llama-3.2-3b": {
            "path": "/home/compute.ashesi.lan/andre.ayiku/Andre_Ayiku/models/Llama-3.2-3B-Instruct/",
            "size": "3B parameters",
            "recommended_for": "Fast responses, lower memory"
        },
        "mistral-7b": {
            "path": "/home/compute.ashesi.lan/andre.ayiku/Andre_Ayiku/models/Mistral-7B-Instruct-v0.3/",
            "size": "7B parameters",
            "recommended_for": "Instruction following"
        }
    }
    
    def __init__(self, default_model: str = "gemma-3-4b"):
        """
        Initialize the model manager.
        
        Args:
            default_model: Name of the model to load initially
        """
        self.current_model_name = None
        self.model = None
        self.tokenizer = None
        
        # Load the default model
        self.load_model(default_model)
    
    def load_model(self, model_name: str) -> Tuple[object, object]:
        """
        Load a specific model.
        
        Args:
            model_name: Name of the model (e.g., "gemma-3-4b")
            
        Returns:
            Tuple of (model, tokenizer)
            
        Raises:
            ValueError: If model name not found
        """
        if model_name not in self.MODELS:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available models: {list(self.MODELS.keys())}"
            )
        
        # Unload current model if exists
        if self.model is not None:
            self.unload_current_model()
        
        model_config = self.MODELS[model_name]
        model_path = model_config["path"]
        
        model_path_obj = Path(model_path)
        print(f"Loading {model_name} from {model_path}...")

        # Load config explicitly first — prevents AutoTokenizer from calling
        # AutoConfig internally which triggers HuggingFace Hub path validation
        config = AutoConfig.from_pretrained(
            str(model_path_obj),
            local_files_only=True,
            trust_remote_code=True
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(model_path_obj),
            config=config,
            local_files_only=True,
            trust_remote_code=True
        )
        
        # Set pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path_obj),
            config=config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            local_files_only=True
        )
        
        self.current_model_name = model_name
        
        print(f"✅ Model loaded successfully!")
        print(f"   Device: {self.model.device}")
        print(f"   Dtype: {self.model.dtype}")
        
        return self.model, self.tokenizer
    
    def unload_current_model(self):
        """
        Unload the current model to free up memory.
        """
        if self.model is not None:
            print(f"Unloading {self.current_model_name}...")
            del self.model
            del self.tokenizer
            gc.collect()
            torch.cuda.empty_cache()
            print("✅ Model unloaded")
    
    def switch_model(self, new_model_name: str) -> Tuple[object, object]:
        """
        Switch to a different model.
        
        Args:
            new_model_name: Name of the model to switch to
            
        Returns:
            Tuple of (model, tokenizer)
        """
        if new_model_name == self.current_model_name:
            print(f"Already using {new_model_name}")
            return self.model, self.tokenizer
        
        return self.load_model(new_model_name)
    
    def get_current_model_info(self) -> dict:
        """
        Get information about the currently loaded model.
        
        Returns:
            Dictionary with model information
        """
        if self.current_model_name is None:
            return {"error": "No model loaded"}
        
        config = self.MODELS[self.current_model_name]
        return {
            "name": self.current_model_name,
            "size": config["size"],
            "recommended_for": config["recommended_for"],
            "path": config["path"],
            "device": str(self.model.device) if self.model else "N/A"
        }
    
    def list_available_models(self) -> dict:
        """
        Get list of all available models.
        
        Returns:
            Dictionary of available models and their info
        """
        return {
            name: {
                "size": config["size"],
                "recommended_for": config["recommended_for"]
            }
            for name, config in self.MODELS.items()
        }
    
    def __del__(self):
        """Cleanup when object is destroyed."""
        if hasattr(self, 'model'):
            self.unload_current_model()


# Example usage
if __name__ == "__main__":
    # Initialize with default model
    manager = ModelManager(default_model="gemma-3-4b")
    
    # Get current model info
    print("\nCurrent model:")
    print(manager.get_current_model_info())
    
    # List all available models
    print("\nAvailable models:")
    for name, info in manager.list_available_models().items():
        print(f"  {name}: {info['size']} - {info['recommended_for']}")
    
    # Switch to a different model
    # manager.switch_model("llama-3.1-8b")
