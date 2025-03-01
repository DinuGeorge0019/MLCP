## Replicating the Paper's Main Experiments

To replicate the experiments, we recommend using the instructions provided in the notebook located in the `__ColabEnvironment` directory.

If you prefer to use a local machine, you need to install the required dependencies listed in `requirements.txt` (standard requirements from Google Colab plus the `scikit-learn` library).

## Replicating the Experiments using Gpt4o, Gpt4o-mini, and o1-mini

To replicate the experiments with Gpt4o, Gpt4o-mini, and o1-mini, use the notebooks from the `app_src/gpt_models_experiments` directory.

- `00_GPT_4_Experiments.ipynb` contains the experiments with Gpt4o and Gpt4o-mini.
- `00_GPT_O1Mini_Experiments.ipynb` contains the experiments with o1-mini.

Install the requirements from `app_src/gpt_models_experiments/requirements.txt`.

Results of the experiments from the paper are in `GPT_MODELS_RESULTS.xlsx`.

## Hardware Requirements

The experiments were conducted using a Colab environment with an A100 GPU (40GB) and a Linux distribution.

