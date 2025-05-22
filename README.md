# MARK: Memory Augmented Refinement of Knowledge

This repository contains the implementation of **MARK: Memory Augmented Refinement of Knowledge**, by Prabal Deb and Anish Ganguli, which has been accepted at Microsoft Journal of Applied Research (MSJAR) 2025. The public version of the paper is available at **[arXiv:2505.05177](https://arxiv.org/abs/2505.05177)**.


## Overview

MARK is a framework for memory-augmented conversational agents, designed to enhance knowledge retrieval and response quality using Azure OpenAI, Azure AI Search, and custom memory-building logic. It supports batch experimentation and evaluation on datasets such as MedMCQA, with modular agent and memory components.

## Features

- **Memory-Augmented Agents:** Integrates Azure AI Search and OpenAI embeddings for context-aware responses.
- **Batch Experimentation:** Run large-scale experiments with different datasets and agent configurations.
- **Automated Evaluation:** Evaluate generated answers using custom information capture metrics.
- **Extensible Architecture:** Modular design for agents, memory, evaluation, and data handling.

## Setup

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd MARK
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Environment Variables:**
   - Create a `.env` file in the project root with the following keys (see code for required variables):
     ```
        AZURE_OPENAI_BASE_URL=
        AZURE_OPENAI_API_KEY=
        AZURE_OPENAI_DEPLOYMENT_NAME=
        AZURE_OPENAI_MODEL_NAME=
        AZURE_OPENAI_API_VERSION=
        AZURE_OPENAI_EMBEDDING_MODEL=
        AZURE_OPENAI_EMBEDDING_API_VERSION=
        AZURE_SEARCH_ENDPOINT=
        AZURE_SEARCH_API_KEY=
        AZURE_SEARCH_INDEX_NAME=
        CHAINLIT_USERNAME=
        CHAINLIT_PASSWORD=
        CHAINLIT_ROLE=
        CHAINLIT_AUTH_SECRET=
        AZURE_OPENAI_EVALUATION_DEPLOYMENT_NAME=
        AZURE_OPENAI_EVALUATION_API_VERSION=
     ```

## Usage

### 1. Run Memory Builder Agents

Start an interactive memory builder agent session:
```bash
chainlit run ./experiment_mem_builder.py
```

### 2. Batch Experimentation

Run batch experiments with memory-augmented agents:
```bash
python run_batch_experiment.py --file <input_data.csv> --limit 10 --type med_mcqa
```
- `--file`: Path to the input data file (CSV or MedMCQA format).
- `--limit`: Number of records to process (default: 10).
- `--type`: Dataset type (`med_mcqa` or `exp_2`).

Results are saved in the `.evaluation_input_data` directory.

### 3. Batch Evaluation

Evaluate generated answers using information capture metrics:
```bash
python run_batch_evaluation.py --file <experiment_results.jsonl>
```
- `--file`: Path to the `.jsonl` file with generated answers.

Evaluation results and summary are saved in the `.evaluation_output_data` directory.

## Project Structure

- `src/agents/`: Agent implementations (e.g., ChatbotAgent).
- `src/data/`: Data loaders and models (e.g., MedMCQADataSet, EvaluationData).
- `src/memory/`: Memory and Azure AI Search integration.
- `src/service/`: Memory builder logic.
- `src/evaluation/`: Evaluation metrics and scoring.
- `run_batch_experiment.py`: Batch experiment runner.
- `run_batch_evaluation.py`: Batch evaluation runner.

## Customization

- Add new agents in `src/agents/`.
- Extend memory or evaluation logic in `src/memory/` and `src/evaluation/`.
- Update `.env` for new Azure/OpenAI endpoints or models.

## Notes

- Ensure all Azure resources (OpenAI, AI Search) are provisioned and accessible.
- For large-scale experiments, adjust batch sizes and thresholds as needed in the scripts.

## License
MARK is licensed under the MIT License.

## Citation
If you use this code in your research or use our research paper for your own research work, please provide proper citation:

```
@misc{ganguli2025markmemoryaugmentedrefinement,
      title={MARK: Memory Augmented Refinement of Knowledge}, 
      author={Anish Ganguli and Prabal Deb and Debleena Banerjee},
      year={2025},
      eprint={2505.05177},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2505.05177}, 
}
```