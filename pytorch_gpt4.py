#!/usr/bin/env python3
"""
PyTorch-Enhanced GPT-4 Local Client
Extended client with PyTorch integration for tensor operations and model utilities.
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, Optional, List, Union
import json
from main import GPT4LocalClient

class PyTorchGPT4Client(GPT4LocalClient):
    """Extended GPT-4 client with PyTorch integration."""
    
    def __init__(self, base_url: str = "http://localhost:8080", api_key: Optional[str] = None, device: str = "auto"):
        """
        Initialize the PyTorch-enhanced GPT-4 client.
        
        Args:
            base_url: The base URL of your local GPT-4 API
            api_key: Optional API key if your local setup requires authentication
            device: Device to use for PyTorch operations ('cpu', 'cuda', 'mps', or 'auto')
        """
        super().__init__(base_url, api_key)
        
        # Set up PyTorch device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
                print(f"🔥 Using CUDA GPU: {torch.cuda.get_device_name()}")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
                print("🍎 Using Apple Metal Performance Shaders (MPS)")
            else:
                self.device = torch.device("cpu")
                print("💻 Using CPU")
        else:
            self.device = torch.device(device)
            print(f"⚡ Using device: {device}")
    
    def analyze_tensor_with_gpt4(self, tensor: torch.Tensor, description: str = "") -> str:
        """
        Analyze a PyTorch tensor using GPT-4.
        
        Args:
            tensor: PyTorch tensor to analyze
            description: Optional description of what the tensor represents
            
        Returns:
            GPT-4's analysis of the tensor
        """
        # Get tensor statistics
        stats = {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "device": str(tensor.device),
            "min": float(tensor.min().item()) if tensor.numel() > 0 else None,
            "max": float(tensor.max().item()) if tensor.numel() > 0 else None,
            "mean": float(tensor.mean().item()) if tensor.numel() > 0 else None,
            "std": float(tensor.std().item()) if tensor.numel() > 1 else None,
            "num_elements": tensor.numel(),
            "requires_grad": tensor.requires_grad
        }
        
        prompt = f"""
        Please analyze this PyTorch tensor:
        
        Description: {description or "No description provided"}
        
        Tensor Statistics:
        - Shape: {stats['shape']}
        - Data type: {stats['dtype']}
        - Device: {stats['device']}
        - Number of elements: {stats['num_elements']}
        - Requires gradient: {stats['requires_grad']}
        - Min value: {stats['min']}
        - Max value: {stats['max']}
        - Mean: {stats['mean']}
        - Standard deviation: {stats['std']}
        
        Please provide insights about:
        1. What this tensor might represent based on its shape and statistics
        2. Any potential issues or observations
        3. Suggestions for operations or transformations
        """
        
        return self.simple_chat(prompt, "You are an expert in PyTorch and machine learning. Analyze tensors and provide practical insights.")
    
    def suggest_model_architecture(self, input_shape: tuple, output_shape: tuple, task_type: str = "classification") -> str:
        """
        Get GPT-4 suggestions for a PyTorch model architecture.
        
        Args:
            input_shape: Shape of input data (excluding batch dimension)
            output_shape: Shape of output data (excluding batch dimension)
            task_type: Type of ML task ('classification', 'regression', 'generation', etc.)
            
        Returns:
            GPT-4's suggested architecture
        """
        prompt = f"""
        I need a PyTorch model architecture suggestion for the following specifications:
        
        - Input shape: {input_shape}
        - Output shape: {output_shape}
        - Task type: {task_type}
        - Framework: PyTorch
        
        Please provide:
        1. A complete PyTorch model class definition
        2. Explanation of the architecture choices
        3. Suggested loss function and optimizer
        4. Training loop structure
        5. Any preprocessing recommendations
        
        Make the code practical and ready to use.
        """
        
        return self.simple_chat(prompt, "You are an expert PyTorch developer. Provide complete, working code examples.")
    
    def debug_training_issue(self, loss_history: List[float], model_info: Dict[str, Any]) -> str:
        """
        Get GPT-4 help with training issues.
        
        Args:
            loss_history: List of loss values over training steps
            model_info: Dictionary with model information (architecture, params, etc.)
            
        Returns:
            GPT-4's debugging suggestions
        """
        # Analyze loss trend
        if len(loss_history) > 1:
            trend = "increasing" if loss_history[-1] > loss_history[0] else "decreasing"
            volatility = np.std(loss_history) if len(loss_history) > 2 else 0
        else:
            trend = "insufficient data"
            volatility = 0
        
        prompt = f"""
        I'm having issues with my PyTorch model training. Here's the information:
        
        Loss History (last {len(loss_history)} steps): {loss_history}
        Loss Trend: {trend}
        Loss Volatility (std): {volatility:.6f}
        
        Model Information:
        {json.dumps(model_info, indent=2)}
        
        Please help me debug this training issue by analyzing:
        1. Loss behavior patterns
        2. Potential causes of the current trend
        3. Specific recommendations to fix the issue
        4. Code snippets for implementing solutions
        5. Hyperparameter suggestions
        
        Focus on actionable PyTorch-specific solutions.
        """
        
        return self.simple_chat(prompt, "You are an expert in debugging PyTorch training issues. Provide specific, actionable solutions.")
    
    def optimize_model_performance(self, model_code: str, performance_metrics: Dict[str, float]) -> str:
        """
        Get GPT-4 suggestions for optimizing model performance.
        
        Args:
            model_code: String containing the model code
            performance_metrics: Dictionary with current performance metrics
            
        Returns:
            GPT-4's optimization suggestions
        """
        prompt = f"""
        Please help optimize this PyTorch model for better performance:
        
        Current Model Code:
        ```python
        {model_code}
        ```
        
        Current Performance Metrics:
        {json.dumps(performance_metrics, indent=2)}
        
        Please provide optimization suggestions for:
        1. Model architecture improvements
        2. Training efficiency optimizations
        3. Memory usage reduction
        4. Inference speed improvements
        5. Hardware-specific optimizations (GPU/CPU)
        
        Include specific PyTorch code examples for each suggestion.
        """
        
        return self.simple_chat(prompt, "You are a PyTorch optimization expert. Focus on practical performance improvements.")

def create_sample_tensor(shape: tuple, tensor_type: str = "random", device: str = "cpu") -> torch.Tensor:
    """
    Create sample tensors for testing.
    
    Args:
        shape: Shape of the tensor
        tensor_type: Type of tensor ('random', 'zeros', 'ones', 'randn')
        device: Device to create tensor on
        
    Returns:
        PyTorch tensor
    """
    device = torch.device(device)
    
    if tensor_type == "random":
        return torch.rand(shape, device=device)
    elif tensor_type == "zeros":
        return torch.zeros(shape, device=device)
    elif tensor_type == "ones":
        return torch.ones(shape, device=device)
    elif tensor_type == "randn":
        return torch.randn(shape, device=device)
    else:
        raise ValueError(f"Unknown tensor type: {tensor_type}")

def main():
    """Demo the PyTorch-enhanced GPT-4 client."""
    print("🔥 PyTorch + GPT-4 Local Client")
    print("=" * 40)
    
    # Initialize the enhanced client
    client = PyTorchGPT4Client()
    
    # Create a sample tensor for demonstration
    print("\n📊 Creating sample tensor...")
    sample_tensor = create_sample_tensor((3, 224, 224), "randn", str(client.device))
    print(f"Created tensor with shape: {sample_tensor.shape}")
    
    print("\n🤖 Analyzing tensor with GPT-4...")
    analysis = client.analyze_tensor_with_gpt4(
        sample_tensor, 
        "Image-like tensor, possibly for computer vision model input"
    )
    print(f"\n📝 Analysis:\n{analysis}")
    
    # Interactive mode
    print("\n" + "="*40)
    print("Interactive PyTorch + GPT-4 Mode")
    print("Commands:")
    print("- 'tensor': Analyze a new random tensor")
    print("- 'architecture': Get model architecture suggestions")
    print("- 'debug': Get training debugging help")
    print("- 'optimize': Get performance optimization tips")
    print("- 'quit': Exit")
    
    while True:
        try:
            command = input("\n> ").strip().lower()
            
            if command in ['quit', 'exit', 'q']:
                print("Goodbye! 🔥👋")
                break
            elif command == 'tensor':
                shape_input = input("Enter tensor shape (e.g., '3,224,224'): ")
                try:
                    shape = tuple(map(int, shape_input.split(',')))
                    tensor = create_sample_tensor(shape, "randn", str(client.device))
                    analysis = client.analyze_tensor_with_gpt4(tensor, "User-specified tensor")
                    print(f"\n📝 Analysis:\n{analysis}")
                except Exception as e:
                    print(f"Error creating tensor: {e}")
            elif command == 'architecture':
                try:
                    input_shape = input("Input shape (e.g., '3,224,224'): ")
                    output_shape = input("Output shape (e.g., '1000'): ")
                    task = input("Task type (classification/regression/generation): ") or "classification"
                    
                    input_shape = tuple(map(int, input_shape.split(',')))
                    output_shape = tuple(map(int, output_shape.split(',')))
                    
                    suggestion = client.suggest_model_architecture(input_shape, output_shape, task)
                    print(f"\n🏗️ Architecture Suggestion:\n{suggestion}")
                except Exception as e:
                    print(f"Error: {e}")
            elif command == 'debug':
                losses_input = input("Enter recent loss values (comma-separated): ")
                try:
                    losses = list(map(float, losses_input.split(',')))
                    model_info = {"architecture": "User model", "parameters": "Unknown"}
                    debug_help = client.debug_training_issue(losses, model_info)
                    print(f"\n🐛 Debug Help:\n{debug_help}")
                except Exception as e:
                    print(f"Error: {e}")
            elif command == 'optimize':
                print("For optimization help, please provide your model code and metrics.")
                print("This is a simplified demo - in practice, you'd provide actual code.")
                metrics = {"accuracy": 0.85, "inference_time": 50, "memory_usage": "2GB"}
                help_text = client.optimize_model_performance("# Your model code here", metrics)
                print(f"\n⚡ Optimization Help:\n{help_text}")
            else:
                print("Unknown command. Type 'quit' to exit or use the commands listed above.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye! 🔥👋")
            break
        except Exception as e:
            print(f"\nError: {e}")

if __name__ == "__main__":
    main()
