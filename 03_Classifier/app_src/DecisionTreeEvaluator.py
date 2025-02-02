
import os
import pandas as pd
import numpy as np
import ast

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer 

# local application/library specific imports
from app_config import AppConfig
from app_src import ClassifierChainWrapper
from app_src import CustomEncoder

# define configuration proxy
configProxy = AppConfig()
CONFIG = configProxy.return_config()

# get global constants configuration
GLOBAL_CONSTANTS = configProxy.return_global_constants()
RANDOM_STATE = GLOBAL_CONSTANTS['RANDOM_SEED']

class DecisionTreeEvaluator():
    
    def __init__(self) -> None:        
        self.estimator_collection = {}
        self.encoder_collection = []
        self.test_dataset = None
        self.train_dataset = None
        self.validation_dataset = None
        
        self.__define_models()
        self.tfidf_vectorizer = TfidfVectorizer(max_features=512)

    def __define_models(self):
        logistic_regression_classifier = LogisticRegression(solver='newton-cg', random_state=RANDOM_STATE)
        kn_classifier = KNeighborsClassifier()
        decision_tree_classifier = DecisionTreeClassifier(random_state=RANDOM_STATE)
        gaussian_nb_classifier = GaussianNB()
        random_forest_classifier = RandomForestClassifier(random_state=RANDOM_STATE)
        gradient_boosting_classifier = GradientBoostingClassifier(random_state=RANDOM_STATE)
        xgb_classifier = XGBClassifier(random_state=RANDOM_STATE)
        lgb_classifier = LGBMClassifier(random_state=RANDOM_STATE)
        svc_classifier = SVC(decision_function_shape='ovo')

        self.estimator_collection = {
            # 'CatBoostClassifier': catb_classifier,
            'LogisticRegression': logistic_regression_classifier,
            'KNeighborsClassifier': kn_classifier,
            'DecisionTreeClassifier': decision_tree_classifier,
            'GaussianNB': gaussian_nb_classifier,
            'RandomForestClassifier': random_forest_classifier,
            'GradientBoostingClassifier': gradient_boosting_classifier,
            'XGBClassifier': xgb_classifier,
            'LGBMClassifier': lgb_classifier,
            'SVC': svc_classifier
        }
        
        self.encoder_collection = [
            'sentence-transformers/all-mpnet-base-v2',
            'sentence-transformers/multi-qa-mpnet-base-dot-v1',
            'sentence-transformers/multi-qa-distilbert-cos-v1',
            'sentence-transformers/multi-qa-MiniLM-L6-cos-v1',
            'sentence-transformers/all-distilroberta-v1',
            'sentence-transformers/all-MiniLM-L12-v2',
            'sentence-transformers/all-MiniLM-L6-v2',
            'microsoft/mpnet-base',
            'roberta-base',
            'bert-base-uncased',
            'tfidf'
        ]


    def __read_test_data(self, number_of_tags):
        self.test_dataset = pd.read_csv(CONFIG[f'TOP_{number_of_tags}_TESTING_DATASET_PATH'])
        
    def __read_train_data(self, number_of_tags):
        self.train_dataset = pd.read_csv(CONFIG[f'TOP_{number_of_tags}_TRAINING_DATASET_PATH'])
    
    def __read_validation_data(self, number_of_tags):
        self.validation_dataset = pd.read_csv(CONFIG[f'TOP_{number_of_tags}_VALIDATION_DATASET_PATH'])
    
    def __encode_with_tfidf(self, problem_statements):
        return self.tfidf_vectorizer.fit_transform(problem_statements).toarray()

    def __encode_problem_statements(self, encoder_model, problem_statements, batch_size):
        if encoder_model == 'tfidf':
            return self.__encode_with_tfidf(problem_statements)
        else:
            encoder = CustomEncoder(encoder_model)
            return encoder.encode_problem_statement(problem_statements, batch_size)
    
    def __encode_tags(self, tags):        
        for idx, string_tag_list in enumerate(tags):
            tags[idx] = ast.literal_eval(string_tag_list)
        return np.array(tags)
    
    def __save_metrics(self, encoder, estimator_name, metrics_results, number_of_tags):
        
        scores = {key: str(value) for key, value in metrics_results.items()}
        csv_headers = ['Encoder Name', 'Estimator Name'] + list(scores.keys())
        output_data = [f'{encoder}', f'{estimator_name}'] + list(scores.values())

        # Create a DataFrame
        df = pd.DataFrame([output_data], columns=csv_headers)

        if not os.path.isfile(CONFIG[f'TOP_{number_of_tags}_BENCHMARK_BASELINE_MODELS_PATH']):
            # Write the DataFrame to the csv file
            df.to_csv(CONFIG[f'TOP_{number_of_tags}_BENCHMARK_BASELINE_MODELS_PATH'], index=False)
        else:
            # Append the DataFrame to an existing CSV file
            df.to_csv(CONFIG[f'TOP_{number_of_tags}_BENCHMARK_BASELINE_MODELS_PATH'], index=False, mode='a', header=False)


    def benchmark_models(self, encoder_batch_size, number_of_tags, validation=False):
        
        self.__read_train_data(number_of_tags)
        if validation:
            self.__read_validation_data(number_of_tags)
        self.__read_test_data(number_of_tags)
        
        for encoder in self.encoder_collection:
            print('Benchmarking encoder model:', encoder)
            
            train_problem_statements = self.__encode_problem_statements(encoder, self.train_dataset['problem_statement'].tolist(), batch_size=encoder_batch_size)
            train_tags = self.__encode_tags(self.train_dataset['problem_tags'].tolist())   
                                  
            # print('Train problem statements shape:', train_problem_statements.shape)
            # print(train_problem_statements)

            # print('Train tags shape:', train_tags.shape)   
            # print(train_tags)
            
            if validation:
                validation_problem_statements = self.__encode_problem_statements(encoder, self.validation_dataset['problem_statement'].tolist(), batch_size=encoder_batch_size)
                validation_tags = self.__encode_tags(self.validation_dataset['problem_tags'].tolist())
            
            test_problem_statements = self.__encode_problem_statements(encoder, self.test_dataset['problem_statement'].tolist(), batch_size=encoder_batch_size)
            test_tags = self.__encode_tags(self.test_dataset['problem_tags'].tolist())
            
            # print('Test problem statements shape:', test_problem_statements.shape)
            # print('Test problem statements type:', test_problem_statements.dtype)
            # print(test_problem_statements)

            # print('Test tags shape:', test_tags.shape)   
            # print('Test tags type:',test_tags.dtype)
            # print(test_tags)
            
            for estimator_name, estimator in self.estimator_collection.items():
                print('Benchmarking estimator model:', estimator_name)
                
                classifierChainWrapper = ClassifierChainWrapper(estimator)
                if validation:
                    classifierChainWrapper.fit(train_problem_statements, train_tags, validation_problem_statements, validation_tags)
                else:
                    classifierChainWrapper.fit(train_problem_statements, train_tags)

                metrics_results = classifierChainWrapper.predict(test_problem_statements, test_tags)
                
                self.__save_metrics(encoder, estimator_name, metrics_results, number_of_tags)
                