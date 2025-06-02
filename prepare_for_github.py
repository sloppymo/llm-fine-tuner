import os
import shutil

# 1. Remove .env files with secrets
env_path = os.path.join(os.getcwd(), ".env")
if os.path.exists(env_path):
    with open(env_path, "r") as f:
        lines = f.readlines()
    # Optionally, print a warning if a key was found
    for line in lines:
        if "OPENAI_API_KEY" in line and "your_openai_api_key_here" not in line:
            print("WARNING: Removing real OPENAI_API_KEY from .env!")
    os.remove(env_path)
    print("Deleted .env file containing secrets.")

# 2. Create .env.example
env_example_path = os.path.join(os.getcwd(), ".env.example")
if not os.path.exists(env_example_path):
    with open(env_example_path, "w") as f:
        f.write("OPENAI_API_KEY=your_openai_api_key_here\n")
    print("Created .env.example.")

# 3. Create or update .gitignore
gitignore_lines = [
    ".env",
    ".env.*",
    "__pycache__/",
    "*.pyc",
    "*.pyo",
    "*.pyd",
    "venv/",
    ".envrc",
    ".DS_Store"
]
gitignore_path = os.path.join(os.getcwd(), ".gitignore")
existing = []
if os.path.exists(gitignore_path):
    with open(gitignore_path, "r") as f:
        existing = [line.strip() for line in f.readlines()]
with open(gitignore_path, "a") as f:
    for line in gitignore_lines:
        if line not in existing:
            f.write(line + "\n")
print("Ensured .gitignore is up to date.")

# 4. Optionally, create a starter README.md
readme_path = os.path.join(os.getcwd(), "README.md")
if not os.path.exists(readme_path):
    with open(readme_path, "w") as f:
        f.write(
"""# LLM Fine Tuner

A Python toolkit for fine-tuning large language models (LLMs) on custom datasets.

## Setup

pip install -r requirements.txt
cp .env.example .env

## Usage

python train.py --data data/my_dataset.jsonl

## License

Apache 2.0
""")
    print("Created starter README.md.")

print("\nCleanup complete! Your folder is safe to upload to GitHub.")