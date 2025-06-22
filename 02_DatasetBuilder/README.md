# DatasetBuilder

The `DatasetBuilder` library is designed to facilitate the construction of the proposed dataset and the processing of other datasets referenced in the paper.

### Fetching the Dataset Locally
To download the dataset directly from Kaggle, use the command `--fetch_dataset_from_kaggle` available in main.py script.

## Usage

The main functionalities of the `DatasetBuilder` library can be accessed through the `main.py` script. 

The most important commands are the following:
- `--fetch_dataset_from_kaggle`: Download the dataset from Kaggle and extract it locally.
- `--build_base_train_test_dataset`: Build base train/test datasets.
- `--build_train_test_dataset TOP_N`: Build datasets (train/test/val) for the top N tags.
- `--build_train_test_dataset_without_tag_encoding TOP_N`: Build train/test dataset without tag encoding for the top N tags.
- `--build_basic_nli_dataset TOP_N`: Build basic NLI dataset for the top N tags.
- `--build_nli_dataset TOP_N`: Build NLI dataset for the top N tags.
- `--build_outside_train_test_dataset TOP_N`: Build train/test datasets from Kim et. all proposed dataset.
- `--build_outside_train_test_dataset_without_tag_encoding TOP_N`: Build train/test dataset from Kim et. all proposed dataset without tag encoding for the top N tags.
- `--build_outside_nli_dataset TOP_N`: Build NLI dataset from Kim et. all proposed dataset for the top N tags.
- `--build_outside_basic_nli_dataset TOP_N`: Build basic NLI dataset from Kim et. all proposed dataset for the top N tags.

## Note

The augmentation method proposed in the paper is present in the `build_nli_dataset` and `build_outside_nli_dataset` methods.

## Output datasets

The output datasets can be found in `{root_folder}\01_TASK_DATASETS`.

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

## Example

To build the augmented dataset for domain adaptation for the top 5 tags on our dataset, you can run: 

```bash
python main.py --build_nli_dataset 5
```

## Installation

To use the `02_DatasetBuilder` library, you need to install the required dependencies. You can do this by running:

```bash
pip install -r requirements.txt
```

