#!/usr/bin/env python3
"""
PyTorch Examples with GPT-4 Integration
Practical examples showing how to use PyTorch with your local GPT-4 for ML development.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pytorch_gpt4 import PyTorchGPT4Client

class SimpleNN(nn.Module):
    """A simple neural network for demonstration."""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

def example_1_tensor_analysis():
    """Example 1: Analyze tensors with GPT-4."""
    print("🔍 Example 1: Tensor Analysis with GPT-4")
    print("-" * 40)
    
    client = PyTorchGPT4Client()
    
    # Create various tensors
    tensors = {
        "Image-like": torch.randn(3, 224, 224),
        "Batch of embeddings": torch.randn(32, 512),
        "Weights matrix": torch.randn(256, 128),
        "1D signal": torch.randn(1000),
    }
    
    for name, tensor in tensors.items():
        print(f"\n📊 Analyzing: {name}")
        analysis = client.analyze_tensor_with_gpt4(tensor, name)
        print(f"Shape: {tensor.shape}")
        print(f"GPT-4 says: {analysis[:200]}...")  # Truncated for demo

def example_2_model_architecture():
    """Example 2: Get model architecture suggestions."""
    print("\n🏗️ Example 2: Model Architecture Suggestions")
    print("-" * 40)
    
    client = PyTorchGPT4Client()
    
    # Example scenarios
    scenarios = [
        {
            "input_shape": (3, 32, 32),
            "output_shape": (10,),
            "task": "classification",
            "description": "CIFAR-10 image classification"
        },
        {
            "input_shape": (784,),
            "output_shape": (1,),
            "task": "regression",
            "description": "MNIST digit recognition as regression"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📝 Scenario: {scenario['description']}")
        suggestion = client.suggest_model_architecture(
            scenario["input_shape"],
            scenario["output_shape"],
            scenario["task"]
        )
        print(f"GPT-4 suggestion: {suggestion[:300]}...")  # Truncated for demo

def example_3_training_debug():
    """Example 3: Debug training issues with GPT-4."""
    print("\n🐛 Example 3: Training Debug Assistance")
    print("-" * 40)
    
    client = PyTorchGPT4Client()
    
    # Simulate training scenarios
    scenarios = [
        {
            "loss_history": [2.3, 2.1, 2.0, 1.9, 1.8, 1.7, 1.6],
            "description": "Normal decreasing loss"
        },
        {
            "loss_history": [0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01],
            "description": "Potentially overfitting"
        },
        {
            "loss_history": [1.5, 1.6, 1.4, 1.7, 1.3, 1.8, 1.2],
            "description": "Oscillating loss"
        }
    ]
    
    for scenario in scenarios:
        print(f"\n📊 Scenario: {scenario['description']}")
        model_info = {
            "architecture": "SimpleNN",
            "parameters": 10000,
            "optimizer": "Adam",
            "learning_rate": 0.001
        }
        
        debug_help = client.debug_training_issue(scenario["loss_history"], model_info)
        print(f"Loss trend: {scenario['loss_history']}")
        print(f"GPT-4 debug help: {debug_help[:250]}...")  # Truncated for demo

def example_4_real_training():
    """Example 4: Train a model and get GPT-4 feedback."""
    print("\n🎯 Example 4: Real Training with GPT-4 Feedback")
    print("-" * 40)
    
    client = PyTorchGPT4Client()
    
    # Create a simple dataset
    X = torch.randn(1000, 10)
    y = torch.sum(X[:, :5], dim=1) + torch.randn(1000) * 0.1  # Simple linear relationship
    
    # Create model
    model = SimpleNN(10, 20, 1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    
    print("🏃‍♂️ Training model...")
    losses = []
    
    # Train for a few epochs
    for epoch in range(20):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs.squeeze(), y)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item():.4f}")
    
    # Get GPT-4 analysis of training
    print("\n🤖 Getting GPT-4 analysis of training...")
    model_info = {
        "architecture": str(model),
        "parameters": sum(p.numel() for p in model.parameters()),
        "dataset_size": len(X),
        "epochs": 20,
        "final_loss": losses[-1]
    }
    
    analysis = client.debug_training_issue(losses[-10:], model_info)  # Last 10 losses
    print(f"Training completed. Final loss: {losses[-1]:.4f}")
    print(f"GPT-4 analysis: {analysis[:300]}...")

def main():
    """Run all PyTorch + GPT-4 examples."""
    print("🔥 PyTorch + GPT-4 Integration Examples")
    print("=" * 50)
    
    try:
        # Check if GPT-4 client can connect
        client = PyTorchGPT4Client()
        test_response = client.simple_chat("Hello, are you working?")
        if "error" in test_response.lower():
            print("⚠️  Warning: Could not connect to local GPT-4. Examples will show structure only.")
            print("Make sure your local GPT-4 server is running!")
            return
        
        print("✅ Connected to local GPT-4! Running examples...\n")
        
        # Run examples
        example_1_tensor_analysis()
        example_2_model_architecture()
        example_3_training_debug()
        example_4_real_training()
        
        print("\n🎉 All examples completed!")
        print("\nNext steps:")
        print("1. Modify the examples for your specific use case")
        print("2. Try the interactive mode: python pytorch_gpt4.py")
        print("3. Integrate GPT-4 assistance into your own ML projects")
        
    except Exception as e:
        print(f"❌ Error running examples: {e}")
        print("Make sure your local GPT-4 server is running and accessible.")

if __name__ == "__main__":
    main()
