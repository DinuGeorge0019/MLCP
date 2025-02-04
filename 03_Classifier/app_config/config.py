

import datetime
import os

class Config(object):
    """
    Config with all the paths and flags needed.
    """

    def __init__(self):
        
        WORKING_DIR = os.path.dirname(os.getcwd())
        BASE_DIR = os.path.dirname(os.path.abspath(WORKING_DIR))
        
        # Generate the current date and time
        current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.config = {
            
            "WORKING_DIR": f"{WORKING_DIR}",
            "BASE_DIR": f"{BASE_DIR}",

            ############################################################################################################          
            
            "TOP_5_TRAINING_DATASET_PATH": os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', 'top_5_training_dataset.csv'),
            "TOP_10_TRAINING_DATASET_PATH": os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', 'top_10_training_dataset.csv'),
            "TOP_15_TRAINING_DATASET_PATH": os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', 'top_15_training_dataset.csv'),
            "TOP_20_TRAINING_DATASET_PATH": os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', 'top_20_training_dataset.csv'),
            
            "TOP_5_TESTING_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', 'top_5_testing_dataset.csv'),
            "TOP_10_TESTING_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', 'top_10_testing_dataset.csv'),
            "TOP_15_TESTING_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', 'top_15_testing_dataset.csv'),
            "TOP_20_TESTING_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', 'top_20_testing_dataset.csv'),

            "TOP_5_VALIDATION_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', 'top_5_validation_dataset.csv'), 
            "TOP_10_VALIDATION_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', 'top_10_validation_dataset.csv'),
            "TOP_15_VALIDATION_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', 'top_15_validation_dataset.csv'),
            "TOP_20_VALIDATION_DATASET_PATH":os.path.join(BASE_DIR, '01_TASK_DATASETS', '03_Task_Datasets', 'top_20_validation_dataset.csv'),
            
            ############################################################################################################          

            "TOP_5_BENCHMARK_BASELINE_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', '01_Baseline_Models', 'top_5_baseline_models.csv'),
            "TOP_10_BENCHMARK_BASELINE_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', '01_Baseline_Models', 'top_10_baseline_models.csv'),
            "TOP_15_BENCHMARK_BASELINE_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', '01_Baseline_Models', 'top_15_baseline_models.csv'),
            "TOP_20_BENCHMARK_BASELINE_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', '01_Baseline_Models', 'top_20_baseline_models.csv'),

            ############################################################################################################          

            "TOP_5_BENCHMARK_ONEVSALL_BASELINE_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', '01_Baseline_Models', 'top_5_onevsall_baseline_models.csv'),
            "TOP_10_BENCHMARK_ONEVSALL_BASELINE_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', '01_Baseline_Models', 'top_10_onevsall_baseline_models.csv'),
            "TOP_15_BENCHMARK_ONEVSALL_BASELINE_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', '01_Baseline_Models', 'top_15_onevsall_baseline_models.csv'),
            "TOP_20_BENCHMARK_ONEVSALL_BASELINE_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', '01_Baseline_Models', 'top_20_onevsall_baseline_models.csv'), 
            
            ############################################################################################################
            "TOP_5_BENCHMARK_TRANSFORMER_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', '01_Baseline_Models', 'top_5_baseline_transformer_models.csv'),
            "TOP_10_BENCHMARK_TRANSFORMER_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', '01_Baseline_Models', 'top_10_baseline_transformer_models.csv'),
            "TOP_15_BENCHMARK_TRANSFORMER_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', '01_Baseline_Models', 'top_15_baseline_transformer_models.csv'),
            "TOP_20_BENCHMARK_TRANSFORMER_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', '01_Baseline_Models', 'top_20_baseline_transformer_models.csv'),
            
            ############################################################################################################
            "TOP_5_BENCHMARK_ONEVSALL_TRANSFORMER_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', '01_Baseline_Models', 'top_5_onevsall_baseline_transformer_models.csv'),
            "TOP_10_BENCHMARK_ONEVSALL_TRANSFORMER_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', '01_Baseline_Models', 'top_10_onevsall_baseline_transformer_models.csv'),
            "TOP_15_BENCHMARK_ONEVSALL_TRANSFORMER_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', '01_Baseline_Models', 'top_15_onevsall_baseline_transformer_models.csv'),
            "TOP_20_BENCHMARK_ONEVSALL_TRANSFORMER_MODELS_PATH": os.path.join(BASE_DIR, '04_BENCHMARKS', '01_Baseline_Models', 'top_20_onevsall_baseline_transformer_models.csv'),
            
            ############################################################################################################          

            "MODEL_SAVE_PATH": os.path.join(BASE_DIR, '05_MODELS', '01_Custom_Models', f'custom_model_{current_time}.weights.h5'),
            "TRANSFORMER_SAVE_PATH": os.path.join(BASE_DIR, '05_MODELS', '02_Transformer_Models', f'transformer_model_{current_time}'),
            "MODEL_SAVE_PATH_ROOT": os.path.join(BASE_DIR, '05_MODELS', '01_Custom_Models'),
            "TRANSFORMER_SAVE_PATH_ROOT": os.path.join(BASE_DIR, '05_MODELS', '02_Transformer_Models')
        }   

        self.global_constants = {
            "RANDOM_SEED": 42
        }
        
    def return_global_constants(self):
        """
        Return global constants
        Returns:
            None
        """
        return self.global_constants
    
    def return_config(self):
        """
        Return entire config dictionary
        Returns:
            None
        """
        return self.config
