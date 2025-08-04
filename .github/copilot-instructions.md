<!-- Use this file to provide workspace-specific custom instructions to Copilot. For more details, visit https://code.visualstudio.com/docs/copilot/copilot-customization#_use-a-githubcopilotinstructionsmd-file -->

# GPT-4 Local Client Project Instructions

This is a Python project for interacting with a local GPT-4 installation. When generating code for this project:

## Project Context
- This is a simple Python script that acts as a client for a local GPT-4 API
- The main functionality is in `main.py` which provides both a programmatic API and interactive chat interface
- The project uses the requests library for HTTP communication with the local GPT-4 server

## Code Style Guidelines
- Use type hints for function parameters and return values
- Include docstrings for all functions and classes
- Handle errors gracefully with try-catch blocks
- Use meaningful variable names and comments
- Follow PEP 8 style guidelines

## API Integration
- The default assumption is that the local GPT-4 API follows OpenAI-compatible endpoints
- Support both authenticated and non-authenticated setups
- Include proper error handling for network requests and API responses
- Make the base URL and API key configurable

## User Experience
- Provide clear feedback to users during API calls
- Include helpful error messages when things go wrong
- Support interactive chat mode for easy testing
- Include command-line help and usage instructions
