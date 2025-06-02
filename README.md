# Language Model Fine-Tuner GUI

A graphical user interface for managing the fine-tuning process of language models. This tool allows users to upload datasets, configure training parameters, and evaluate fine-tuned models.

## Features

- Dataset management (JSONL format)
- Training parameter configuration
- Real-time training progress monitoring
- Model evaluation with test prompts
- Error handling and user feedback

## LLM Fine Tuner

A Python toolkit for fine-tuning large language models (LLMs) on your own datasets. Supports OpenAI API and local models, with a focus on usability, reproducibility, and extensibility.

---

## 🚀 Features

- **Custom Dataset Training:** Fine-tune LLMs on your own JSONL or CSV datasets.
- **OpenAI API & Local Model Support:** Easily switch between cloud and local backends.
- **CLI & Python API:** Use as a command-line tool or import as a library.
- **Configurable Training:** Control epochs, batch size, learning rate, and more.
- **Evaluation & Export:** Evaluate model performance and export results.
- **Jupyter Notebook Demos:** Example notebooks to get you started quickly.
- **Secure by Default:** No secrets in repo; use `.env` for API keys.

---

## 📦 Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/sloppymo/llm-fine-tuner.git
cd llm-fine-tuner
pip install -r requirements.txt
```

---

## ⚙️ Configuration

Copy the example environment file and add your own API key:

```bash
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=your_key_here
```

---

## 🏃 Quickstart

### Command Line

Fine-tune a model with your dataset:

```bash
python train.py --data data/my_dataset.jsonl --epochs 3 --batch-size 8
```

### Python API

```python
from lm_fine_tuner import LMFineTuner

tuner = LMFineTuner(api_key="your_openai_api_key")
tuner.train("data/my_dataset.jsonl", epochs=3, batch_size=8)
```

---

## 📊 Example

See [notebooks/demo.ipynb](notebooks/demo.ipynb) for a full walkthrough, including data prep, training, and evaluation.

---

## 📝 Data Format

Input data should be in JSONL format, e.g.:

```json
{"prompt": "Translate to French: Hello, world!", "completion": "Bonjour, le monde!"}
```

---

## 🛡️ Security

- **Never commit your real `.env` file or API keys.**
- Use `.env.example` to show required variables.

---

## 🤝 Contributing

Contributions are welcome! Please open issues or submit pull requests for bugs, features, or improvements.

---

## 📄 License

Apache 2.0

---

## 🙏 Acknowledgements

- [OpenAI](https://openai.com/)
- [Hugging Face](https://huggingface.co/) (if you add local model support)
- All contributors and testers!

---

## 💡 Roadmap

- [ ] Add support for Hugging Face Transformers
- [ ] Add experiment tracking and logging
- [ ] Add more evaluation metrics
- [ ] Improve CLI UX

---

## 📬 Contact

Questions or suggestions? Open an issue or reach out via [GitHub Discussions](https://github.com/sloppymo/llm-fine-tuner/discussions).

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
