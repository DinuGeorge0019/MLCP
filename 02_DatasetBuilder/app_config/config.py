


class Config(object):
    """
    Config with all the paths and flags needed.
    """

    def __init__(self, WORKING_DIR):
        self.request_headers = {
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET',
            'Access-Control-Allow-Headers': 'Content-Type',
            'Access-Control-Max-Age': '3600',
        }

        self.config = {
            "CODEFORCES_GET_CONTEST_LIST_REQUEST_LINK": "http://codeforces.com/api/contest.list",
            "CODEFORCES_BASE_CONSTEST_LINK": "http://codeforces.com/contest/",
            "CHROME_DRIVER_PATH": "C:\\Program Files\\Google\\Chrome\\Application\\chromedriver.exe",
            "CODEFORCES_LINK": "https://codeforces.com/",

            ############################################################################################################
            
            "WORKING_DIR": f"{WORKING_DIR}",
            "DATASET_DESTINATION": f"{WORKING_DIR}\\01_CODEFORCES_DATASET",
            "CODEFORCES_EDUCATIONAL_FILE": f"{WORKING_DIR}\\00_CODEFORCES_DATA\\EDUCATIONAL_CONTESTS.in",
            "CODEFORCES_DIV1_FILE": f"{WORKING_DIR}\\00_CODEFORCES_DATA\\DIV1_CONTESTS.in",
            "CODEFORCES_DIV2_FILE": f"{WORKING_DIR}\\00_CODEFORCES_DATA\\DIV2_CONTESTS.in",
            "CODEFORCES_DIV1&2_FILE": f"{WORKING_DIR}\\00_CODEFORCES_DATA\\DIV1&2_CONTESTS.in",
            "CODEFORCES_DIV3_FILE": f"{WORKING_DIR}\\00_CODEFORCES_DATA\\DIV3_CONTESTS.in",
            "CODEFORCES_DIV4_FILE": f"{WORKING_DIR}\\00_CODEFORCES_DATA\\DIV4_CONTESTS.in",
            
            ############################################################################################################
            
            "RAW_DATASET_PATH": f"{WORKING_DIR}\\01_TASK_DATASETS\\01_Raw_Datasets\\raw_dataset.csv",
            "TOP_5_FILTERED_DATASET_PATH":f"{WORKING_DIR}\\01_TASK_DATASETS\\01_Raw_Datasets\\top_5_dataset.csv",
            "TOP_10_FILTERED_DATASET_PATH":f"{WORKING_DIR}\\01_TASK_DATASETS\\01_Raw_Datasets\\top_10_dataset.csv",
            "TOP_15_FILTERED_DATASET_PATH":f"{WORKING_DIR}\\01_TASK_DATASETS\\01_Raw_Datasets\\top_15_dataset.csv",
            "TOP_20_FILTERED_DATASET_PATH":f"{WORKING_DIR}\\01_TASK_DATASETS\\01_Raw_Datasets\\top_20_dataset.csv",
            
            ############################################################################################################
            
            "TOP_5_BASE_TRAINING_DATASET_PATH":f"{WORKING_DIR}\\01_TASK_DATASETS\\02_Base_Datasets\\top_5_base_training_dataset.csv",
            "TOP_10_BASE_TRAINING_DATASET_PATH":f"{WORKING_DIR}\\01_TASK_DATASETS\\02_Base_Datasets\\top_10_base_training_dataset.csv",
            "TOP_15_BASE_TRAINING_DATASET_PATH":f"{WORKING_DIR}\\01_TASK_DATASETS\\02_Base_Datasets\\top_15_base_training_dataset.csv",
            "TOP_20_BASE_TRAINING_DATASET_PATH":f"{WORKING_DIR}\\01_TASK_DATASETS\\02_Base_Datasets\\top_20_base_training_dataset.csv",
            
            "TOP_5_BASE_TESTING_DATASET_PATH":f"{WORKING_DIR}\\01_TASK_DATASETS\\02_Base_Datasets\\top_5_base_testing_dataset.csv",
            "TOP_10_BASE_TESTING_DATASET_PATH":f"{WORKING_DIR}\\01_TASK_DATASETS\\02_Base_Datasets\\top_10_base_testing_dataset.csv",
            "TOP_15_BASE_TESTING_DATASET_PATH":f"{WORKING_DIR}\\01_TASK_DATASETS\\02_Base_Datasets\\top_15_base_testing_dataset.csv",
            "TOP_20_BASE_TESTING_DATASET_PATH":f"{WORKING_DIR}\\01_TASK_DATASETS\\02_Base_Datasets\\top_20_base_testing_dataset.csv",

            "TOP_5_BASE_VALIDATION_DATASET_PATH":f"{WORKING_DIR}\\01_TASK_DATASETS\\02_Base_Datasets\\top_5_base_validation_dataset.csv",
            "TOP_10_BASE_VALIDATION_DATASET_PATH":f"{WORKING_DIR}\\01_TASK_DATASETS\\02_Base_Datasets\\top_10_base_validation_dataset.csv",
            "TOP_15_BASE_VALIDATION_DATASET_PATH":f"{WORKING_DIR}\\01_TASK_DATASETS\\02_Base_Datasets\\top_15_base_validation_dataset.csv",
            "TOP_20_BASE_VALIDATION_DATASET_PATH":f"{WORKING_DIR}\\01_TASK_DATASETS\\02_Base_Datasets\\top_20_base_validation_dataset.csv",

            ############################################################################################################          
            
            "TOP_5_TRAINING_DATASET_PATH":f"{WORKING_DIR}\\01_TASK_DATASETS\\03_Task_Datasets\\top_5_training_dataset.csv",
            "TOP_10_TRAINING_DATASET_PATH":f"{WORKING_DIR}\\01_TASK_DATASETS\\03_Task_Datasets\\top_10_training_dataset.csv",
            "TOP_15_TRAINING_DATASET_PATH":f"{WORKING_DIR}\\01_TASK_DATASETS\\03_Task_Datasets\\top_15_training_dataset.csv",
            "TOP_20_TRAINING_DATASET_PATH":f"{WORKING_DIR}\\01_TASK_DATASETS\\03_Task_Datasets\\top_20_training_dataset.csv",
            
            "TOP_5_TESTING_DATASET_PATH":f"{WORKING_DIR}\\01_TASK_DATASETS\\03_Task_Datasets\\top_5_testing_dataset.csv",
            "TOP_10_TESTING_DATASET_PATH":f"{WORKING_DIR}\\01_TASK_DATASETS\\03_Task_Datasets\\top_10_testing_dataset.csv",
            "TOP_15_TESTING_DATASET_PATH":f"{WORKING_DIR}\\01_TASK_DATASETS\\03_Task_Datasets\\top_15_testing_dataset.csv",
            "TOP_20_TESTING_DATASET_PATH":f"{WORKING_DIR}\\01_TASK_DATASETS\\03_Task_Datasets\\top_20_testing_dataset.csv",

            "TOP_5_VALIDATION_DATASET_PATH":f"{WORKING_DIR}\\01_TASK_DATASETS\\03_Task_Datasets\\top_5_validation_dataset.csv",
            "TOP_10_VALIDATION_DATASET_PATH":f"{WORKING_DIR}\\01_TASK_DATASETS\\03_Task_Datasets\\top_10_validation_dataset.csv",
            "TOP_15_VALIDATION_DATASET_PATH":f"{WORKING_DIR}\\01_TASK_DATASETS\\03_Task_Datasets\\top_15_validation_dataset.csv",
            "TOP_20_VALIDATION_DATASET_PATH":f"{WORKING_DIR}\\01_TASK_DATASETS\\03_Task_Datasets\\top_20_validation_dataset.csv",
            
            ############################################################################################################          

            "DIFICULTY_DISTRIBUTION_PLOT_PATH": f"{WORKING_DIR}\\01_TASK_DATASETS\\00_Datasets_Info\\raw_dataset_dificulty_distribution_plot.png",
            "TAG_DISTRIBUTION_PLOT_PATH": f"{WORKING_DIR}\\01_TASK_DATASETS\\00_Datasets_Info\\raw_dataset_tag_distribution_plot.png",
            "DATASET_INFO_PATH": f"{WORKING_DIR}\\01_TASK_DATASETS\\00_Datasets_Info\\raw_dataset_info.txt"            
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

    def return_request_headers(self):
        """
        Return the headers used to request the GET operation
        Returns:
            None
        """
        return self.request_headers
