
# Directory Structure

### 00_CODEFORCES_DATA
  List of all Codeforces contests that were scraped.

### 01_CODEFORCES_DATASET
  JSON format of the proposed dataset.

### 01_TASK_DATASETS

The `01_TASK_DATASETS` directory has the following structure:

- `00_Datasets_Info`: Contains information about the raw Codeforces fetched dataset.
- `01_Raw_Datasets`: Contains the raw and preprocessed Codeforces datasets.
- `02_Base_Datasets`: Contains the base datasets of Kim et al. and our proposed dataset.
  - `OUR_DATASET`: Contains the base train/test/validation datasets derived from the preprocessed dataset from `01_Raw_Datasets`.
  - `PSG_PREDICTING_ALGO`: Contains the base train/test/validation datasets from Kim et. all
- `03_Task_Datasets`: Contains the train/test datasets used for the experiments presented in the paper.
  - `00_DATASET_INFO`: Contains information about the tag distribution of our dataset and the dataset from Kim et al.
  - `01_DATASETS_W_TAG_ENCODING`: Contains the datasets used for experiments from our dataset and Kim et al. dataset with tag encoding.
  - `02_DATASETS_WO_TAG_ENCODING`: Contains the datasets used for experiments from our dataset and Kim et al. dataset without tag encoding.
- `04_NLI_Datasets`: Contains the basic and augmented datasets used for domain adaptation experiments presented in the paper.
  - `AUGMENTED_NLI_DATASETS`: Contains the datasets used for domain adaptation using our augmentation method.
  - `BASIC_NLI_DATASETS`: Contains the datasets used for domain adaptation using no augmentation.

### 02_DatasetBuilder
Library is designed to facilitate the construction of the proposed dataset and the processing of other datasets referenced in the paper. It includes scripts for data processing, dataset construction, and exploratory data analysis.

For more information, please refer to the README inside the `02_DatasetBuilder` directory.

### 03_Classifier
This framework is used for training and evaluating the classifiers proposed in the paper. It includes all necessary scripts and configurations to replicate the experiments.

Follow the instructions in the README inside the `03_Classifier` directory to replicate the experiments.

### 04_BENCHMARK
This directory contains the benchmark results obtained from executing the experiments described in the `03_Classifier` directory.

### 05_MODELS
This directory contains the models saved after executing domain adaptation.

