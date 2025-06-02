# Language Model Fine-Tuner GUI

A graphical user interface for managing the fine-tuning process of language models. This tool allows users to upload datasets, configure training parameters, and evaluate fine-tuned models.

## Features

- Dataset management (JSONL format)
- Training parameter configuration
- Real-time training progress monitoring
- Model evaluation with test prompts
- Error handling and user feedback

## Requirements

- Python 3.8+
- OpenAI API key (for OpenAI models)
- Required packages listed in requirements.txt

## Installation

1. Clone this repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY='your-api-key'
   ```

## Usage

Run the application:
```python
python lm_fine_tuner.py
```
