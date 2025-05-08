# standard library imports
import os
import random

# related third-party
import pandas as pd
import numpy as np
import json
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from collections import defaultdict
import ast
import string
import re
import tensorflow as tf
from sklearn.model_selection import train_test_split
from time import sleep
# local application/library specific imports
from app_config import AppConfig
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import networkx as nx
from scipy.spatial.distance import cosine
from sklearn.utils import resample
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from transformers import AutoTokenizer, TFAutoModel, AutoModel
from tqdm.auto import tqdm
import networkx as nx
import ast, json, pathlib

# define configuration proxy
working_dir = os.path.dirname(os.getcwd())
configProxy = AppConfig(working_dir)

# get configuration
CONFIG = configProxy.return_config()

# get global constants configuration
GLOBAL_CONSTANTS = configProxy.return_global_constants()

RANDOM_STATE = GLOBAL_CONSTANTS['RANDOM_SEED']

GENERATE_VALIDATION_DATASET = True


class DatasetFactory:
    def __init__(self):
        pass

    def build_raw_dataset(self):
        """
        Reads input files from the dataset destination and creates a dataframe of the full raw competitive programming dataset.

        Returns:
            None
        """

        ignored_links = [
        ]
        
        print("\nReading corpus")
        raw_dataset = []
        for folder_name in os.listdir(CONFIG['DATASET_DESTINATION']):
            folder_path = os.path.join(CONFIG['DATASET_DESTINATION'], folder_name)
            for file in tqdm(os.listdir(folder_path)):
                file_path = os.path.join(folder_path, file)
                with open(file_path, "r", encoding="utf-8") as input_file:
                    json_data = json.load(input_file)

                if 'file_name' in json_data:
                    if json_data['link'] not in ignored_links:                
                        raw_dataset.append([json_data['link'],
                                            json_data['problemId'],
                                            json_data['problem_idx'],
                                            json_data['shortId'],
                                            json_data['contest_number'],
                                            json_data['name'],
                                            json_data['statement'],
                                            json_data['solutions'],
                                            json_data['input'],
                                            json_data['output'],
                                            json_data['tags'],
                                            json_data['dificulty'],
                                            json_data['file_name'],
                                            json_data['editorial_link'],
                                            json_data['editorial']])

        # Create the raw dataset df of the
        raw_dataset_df = pd.DataFrame(
            raw_dataset,
            columns=['problem_link',
                     'problem_id',
                     'problem_idx',
                     'short_id',
                     'contest_number',
                     'problem_name',
                     'problem_statement',
                     'problem_solution',
                     'problem_input',
                     'problem_output',   
                     'problem_tags',
                     'problem_dificulty',
                     'file_name',
                     'editorial_link',
                     'problem_editorial'
                     ])
        
        # Remove instances where tags are empty
        raw_dataset_df = raw_dataset_df[(raw_dataset_df['problem_statement'].str.strip().str.len() > 100) &
                                        (raw_dataset_df['problem_solution'].str.len() > 0) &
                                        (raw_dataset_df['problem_tags'].str.len() > 0) &
                                        (raw_dataset_df['problem_dificulty'].str.len() > 0) &
                                        (raw_dataset_df['problem_editorial'].str.strip().str.len() > 100)
                                        ]

        # Check for duplicate problem statements
        duplicate_problem_statements = raw_dataset_df[raw_dataset_df['problem_statement'].str.strip().duplicated(keep=False)]

        if not duplicate_problem_statements.empty:
            # Drop duplicates, keeping the first occurrence
            raw_dataset_df = raw_dataset_df.drop_duplicates(subset='problem_statement', keep='first')
            print("Duplicates removed. Remaining dataset:")
            print(raw_dataset_df)
        else:
            print("All problem statements are unique.")
        
        # Save the raw dataset to a CSV file
        raw_dataset_df.to_csv(CONFIG['RAW_DATASET_PATH'],  index=False)

    def backward_update_json_files(self):
        def remove_non_utf8_characters(text):
            """
            Removes invalid UTF-8 characters from the text.
            """
            if isinstance(text, str):
                return text.encode('utf-8', 'replace').decode('utf-8')  # Replace invalid characters
            return text  # Return as is if it's not a string
        
        raw_dataset_df = pd.read_csv(CONFIG['RAW_DATASET_PATH'], encoding="ISO-8859-1")        
        for index, row in raw_dataset_df.iterrows():
            file_path = row['file_name']
            if pd.isna(row['problem_editorial']) or row['problem_editorial'].strip() == '':
                # If the file exists, remove it
                if os.path.exists(file_path):
                    os.remove(file_path)
            else:
                # pass
                problem_editorial_utf8 = remove_non_utf8_characters(row['problem_editorial'])
                
                if os.path.exists(file_path):
    
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as input_file:
                        try:
                            json_data = json.load(input_file)
                        except json.JSONDecodeError:
                            print(f"Skipping file due to JSON error: {file_path}")
                            continue  # Skip files with JSON issues
                            
                    json_data['editorial'] = problem_editorial_utf8

                            
                    with open(file_path, 'w', encoding='utf-8') as json_file:
                        json.dump(json_data, json_file, indent=4)  # Ensure non-ASCII characters are preserved
                else:
                    print(index)
                    print(f"File not found: {file_path}")

    def generate_dataset_overview(self):
        # read the raw dataset
        df = pd.read_csv(CONFIG['RAW_DATASET_PATH'])

        # get the number of problems
        number_of_problems = df.shape[0]
            
        # count the occurrences of each difficulty class
        difficulty_classes = df['problem_dificulty'].unique()
        difficulty_counts = df['problem_dificulty'].value_counts()
        
        # plot difficulty distribution information
        self.__plot_difficulty_distribution(difficulty_counts)
        
        # count the occurrences of each tag
        tags = df['problem_tags'].apply(lambda x: x.lstrip('[').rstrip(']').split(','))

        # Initialize the counter for all tags
        tag_counts = defaultdict(int)

        # Iterate over all problems and count each tag
        for tag_list in tags:
            for tag in tag_list:
                tag_counts[tag.strip().strip("'")] += 1
                
        sorted_tag_counts = dict(sorted(tag_counts.items(), key=lambda item: item[1], reverse=True))

        # plot the tag distribution information
        self.__plot_tag_distribution(sorted_tag_counts)
                
        # print dataset informations
        self.__print_dataset_info(df, number_of_problems, difficulty_counts, sorted_tag_counts)
        
    def __print_dataset_info(self, df, number_of_problems, difficulty_counts, tag_counts):
        
        output_file = open(CONFIG['DATASET_INFO_PATH'], "w")
        
        # Print the number of problems
        output_file.write(f"Number of problems: {number_of_problems} \n\n")
        
        # Get the minimum and maximum problem statement lengths
        min_statement_length = df['problem_statement'].apply(len).min()
        max_statement_length = df['problem_statement'].apply(len).max()
        mean_statement_length = df['problem_statement'].apply(len).mean()
        std_statement_length = df['problem_statement'].apply(len).std()

        output_file.write(f"Minimum problem statement length: {min_statement_length}\n")
        output_file.write(f"Maximum problem statement length: {max_statement_length}\n")
        output_file.write(f"Mean problem statement length: {mean_statement_length}\n\n")
        output_file.write(f"Standard deviation of problem statement length: {std_statement_length}\n\n")

        # Get the minimum and maximum problem editorials lengths
        min_editorial_length = df['problem_editorial'].apply(len).min()
        max_editorial_length = df['problem_editorial'].apply(len).max()
        mean_editorial_length = df['problem_editorial'].apply(len).mean()
        std_editorial_length = df['problem_editorial'].apply(len).std()

        output_file.write(f"Minimum problem editorial length: {min_editorial_length}\n")
        output_file.write(f"Maximum problem editorial length: {max_editorial_length}\n")
        output_file.write(f"Mean problem editorial length: {mean_editorial_length}\n\n")
        output_file.write(f"Standard deviation of problem editorial length: {std_editorial_length}\n\n")

        # Get the minimum and maximum problem solution lengths
        min_solution_length = df['problem_solution'].apply(len).min()
        max_solution_length = df['problem_solution'].apply(len).max()
        mean_statement_length = df['problem_solution'].apply(len).mean()
        std_solution_length = df['problem_solution'].apply(len).std()

        output_file.write(f"Minimum problem solution length: {min_solution_length}\n")
        output_file.write(f"Maximum problem solution length: {max_solution_length}\n")
        output_file.write(f"Mean problem solution length: {mean_statement_length}\n\n")
        output_file.write(f"Standard deviation of problem solution length: {std_solution_length}\n\n")

        # Get the number of difficulty classes
        number_of_difficulty_classes = len(difficulty_counts)
        output_file.write(f"Number of difficulty classes: {number_of_difficulty_classes}\n\n")        
        
        # Print the counts for all difficulty classes
        output_file.write("Difficulty distribution:\n")
        for difficulty, count in difficulty_counts.items():
            output_file.write(f"{difficulty}: {count}\n")
        
        output_file.write("\n")
        
        # Get the number of tags
        number_of_tags = len(tag_counts)
        output_file.write(f"Number of tags: {number_of_tags}\n\n")
        
        #Print the counts for all tags
        output_file.write("Tag distribution:\n")
        for tag, count in tag_counts.items():
            output_file.write(f"{tag}: {count}\n")
            
        output_file.write("\n")
        
    def __plot_tag_distribution(self, tag_counts):
        # Convert the tag_counts dictionary to two lists
        tags = list(tag_counts.keys())
        counts = list(tag_counts.values())
        
        plt.figure(figsize=(20, 10))
        plt.bar(tags, counts, color='skyblue')
        plt.xlabel('Tags', fontsize=24)
        plt.ylabel('Number of Problems', fontsize=24)
        plt.xticks(rotation=90, fontsize=24)
        plt.yticks(fontsize=24)
        plt.tight_layout()
        
        plt.savefig(CONFIG['TAG_DISTRIBUTION_PLOT_PATH'])
        plt.close()  # Close the plot to free up memory

    def __plot_difficulty_distribution(self, difficulty_counts):

        # Plot the bar graph
        plt.figure(figsize=(20, 6))
        difficulty_counts.plot(kind='bar')
        plt.title('Distribution of Difficulty Classes')
        plt.xlabel('Difficulty Class')
        plt.ylabel('Number of Problems')
        plt.xticks(rotation=0)
        
        # Save the plot to the specified path
        plt.savefig(CONFIG['DIFICULTY_DISTRIBUTION_PLOT_PATH'])
        plt.close()  # Close the plot to free up memory
    
    def __search_unknown_symbols(self, df, collumn_name, print_symbols=False):
        
        unknown_symbols_set = set()
        # search for unknown symbols inside the problem statements
        for text in df[collumn_name].values:
            for character in text:
                if character not in string.printable:
                    unknown_symbols_set.add(character)
        # get the full unknown symbols set
        found_unknown_symbols = "".join(unknown_symbols_set)

        if print_symbols:
            print(found_unknown_symbols)

        # return the full unknown symbols set
        return found_unknown_symbols

    def __preprocess_problem_statements(self, df):

        print("\nPreprocessing problem statements")

        # remove unknown symbols
        unknown_symbols = self.__search_unknown_symbols(df, collumn_name='problem_statement')
        if unknown_symbols:
            df.loc[:, 'problem_statement'] = df['problem_statement'].apply(lambda text: re.sub('[%s]' % re.escape(unknown_symbols), ' ', text))
        else:
            pass

        # removing punctuations
        df.loc[:, 'problem_statement'] = df['problem_statement'].apply(lambda text: re.sub('[%s]' % re.escape(string.punctuation), ' ', text))

        # removing all unwanted spaces
        df.loc[:, 'problem_statement'] = df['problem_statement'].apply(lambda text: re.sub('\s+', ' ', text))
        
        # Remove all duplicated sequences
        duplicated_rows_bool = df['problem_statement'].duplicated()
        df = df[~duplicated_rows_bool]
        
        return df

    def __preprocess_problem_editorials(self, df):
        print("\nPreprocessing problem editorials")

        # remove unknown symbols
        unknown_symbols = self.__search_unknown_symbols(df, collumn_name='problem_editorial')
        if unknown_symbols:
            df.loc[:, 'problem_editorial'] = df['problem_editorial'].apply(lambda text: re.sub('[%s]' % re.escape(unknown_symbols), ' ', text))
        else:
            pass

        # removing punctuations
        df.loc[:, 'problem_editorial'] = df['problem_editorial'].apply(lambda text: re.sub('[%s]' % re.escape(string.punctuation), ' ', text))

        # removing all unwanted spaces
        df.loc[:, 'problem_editorial'] = df['problem_editorial'].apply(lambda text: re.sub('\s+', ' ', text))
        
        # Remove all duplicated sequences
        duplicated_rows_bool = df['problem_editorial'].duplicated()
        df = df[~duplicated_rows_bool]

        return df
        
    def __preprocess_problem_solutions(self, df):
        return df
    
    def build_preprocessed_dataset(self):
        # read the raw dataset
        self.preprocessed_df = pd.read_csv(CONFIG['RAW_DATASET_PATH'])
        
        # Preprocess the problem statements
        self.preprocessed_df = self.__preprocess_problem_statements(self.preprocessed_df)
        
        # Preprocess the problem editorials
        self.preprocessed_df = self.__preprocess_problem_editorials(self.preprocessed_df)
        
        # Preprocess the problem solutions
        self.preprocessed_df = self.__preprocess_problem_solutions(self.preprocessed_df)
        
        # Save the filtered dataset to a CSV file
        self.preprocessed_df.to_csv(CONFIG[f'PREPROCESSED_DATASET_PATH'],  index=False)
        
    def build_filtered_dataset(self, df, unique_tags):
                
        # Filter the DataFrame and create a copy
        df = df[df['problem_tags'].apply(
            lambda x: bool(set(ast.literal_eval(x) if isinstance(x, str) else x) & set(unique_tags))
        )].copy()
        
        # Modify the 'problem_tags' column on the copied DataFrame
        df['problem_tags'] = df['problem_tags'].apply(
            lambda x: list(set(ast.literal_eval(x) if isinstance(x, str) else x) & set(unique_tags))
        )
        
        return df
    
    # Function to create binary vector for tags
    def __create_binary_vector(self, tag_list, unique_tags):
        
        binary_vector = [0]*len(unique_tags)
                
        if 'nan' != str(tag_list):
            tag_list = ast.literal_eval(tag_list) if isinstance(tag_list, str) else tag_list  # Convert string representation of list to actual list
            for tag in tag_list:
                if tag in unique_tags:
                    binary_vector[unique_tags.index(tag)] = 1
        
        return binary_vector

    def count_tag_occurrences(self, df, unique_tags, tag_column='problem_tags'):
        # Ensure the tags are in list format
        df[tag_column] = df[tag_column].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        # Initialize a dictionary to store the counts
        tag_counts = {tag: {'has_tag': 0, 'does_not_have_tag': 0} for tag in unique_tags}
        
        # Iterate over each row in the DataFrame
        for tags in df[tag_column]:
            for tag in unique_tags:
                if tag in tags:
                    tag_counts[tag]['has_tag'] += 1
                else:
                    tag_counts[tag]['does_not_have_tag'] += 1
        
        return tag_counts
    
    def get_unique_tags(self, top_n_tags=5, outside_dataset=False):
        if outside_dataset:
            df = pd.read_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_BASE_TESTING_DATASET_PATH'])
            
            # count the occurrences of each tag
            tags = df['tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        else:
            # read the raw dataset
            df = pd.read_csv(CONFIG['RAW_DATASET_PATH'])
        
            # count the occurrences of each tag
            tags = df['problem_tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            
        # Initialize the counter for all tags
        tag_counts = defaultdict(int)

        # Iterate over all problems and count each tag
        for tag_list in tags:
            for tag in tag_list:
                tag_counts[tag] += 1
                
        sorted_tag_counts = dict(sorted(tag_counts.items(), key=lambda item: item[1], reverse=True))
                
        # get only the top n tags
        top_tags = list(sorted_tag_counts.keys())[:top_n_tags]
        
        return top_tags
    
    ############################################################################################################
    
    def build_base_train_test_dataset(self):
        
        # Preprocess the dataset
        self.build_preprocessed_dataset()
        
        # Load the preprocessed dataset
        self.df = pd.read_csv(CONFIG[f'PREPROCESSED_DATASET_PATH'])

        # Split dataset between train and test
        self.df_train, self.df_test = train_test_split(self.df, test_size=0.2, random_state=RANDOM_STATE)
        
        if GENERATE_VALIDATION_DATASET:
            # Split the train set into train and validation sets
            self.df_train, self.df_val = train_test_split(self.df_train, test_size=0.1, random_state=RANDOM_STATE)
            self.df_val.to_csv(CONFIG[f'BASE_VALIDATION_DATASET_PATH'], index=False)
            
        self.df_train.to_csv(CONFIG[f'BASE_TRAINING_DATASET_PATH'], index=False)
        self.df_test.to_csv(CONFIG[f'BASE_TESTING_DATASET_PATH'], index=False)
    
    def build_train_test_dataset(self, top_n_tags=5):
        
        self.train_df = pd.read_csv(CONFIG[f'BASE_TRAINING_DATASET_PATH'])
        self.test_df = pd.read_csv(CONFIG[f'BASE_TESTING_DATASET_PATH'])
        self.val_df = pd.read_csv(CONFIG[f'BASE_VALIDATION_DATASET_PATH'])
        
        unique_tags = self.get_unique_tags(top_n_tags)
        
        # Filter the dataset
        self.train_df = self.build_filtered_dataset(self.train_df, unique_tags)
        self.test_df = self.build_filtered_dataset(self.test_df, unique_tags)
        self.val_df = self.build_filtered_dataset(self.val_df, unique_tags)
        
        # # clean the datasets of unnecessary collumns
        self.train_df = self.train_df.drop(columns=['problem_link', 'problem_id', 'problem_idx', 'short_id', 'contest_number', 'problem_name', 'problem_solution', 'problem_input', 'problem_output', 'problem_dificulty', 'file_name', 'editorial_link', 'problem_editorial']) # 'problem_statement_embeddings', 'problem_statement_cluster'
        self.test_df = self.test_df.drop(columns=['problem_link', 'problem_id', 'problem_idx', 'short_id', 'contest_number', 'problem_name', 'problem_solution', 'problem_input', 'problem_output', 'problem_dificulty', 'file_name', 'editorial_link', 'problem_editorial'])
        self.val_df = self.val_df.drop(columns=['problem_link', 'problem_id', 'problem_idx', 'short_id', 'contest_number', 'problem_name', 'problem_solution', 'problem_input', 'problem_output', 'problem_dificulty', 'file_name', 'editorial_link', 'problem_editorial'])
                
        # for 5 classes ->  ['greedy', 'math', 'implementation', 'dp', 'data structures']
        # for 10 classes -> ['greedy', 'math', 'implementation', 'dp', 'data structures', 'brute force', 'constructive algorithms', 'sortings', 'binary search', 'sortings', 'graphs']
        # for 15 classes -> ['greedy', 'math', 'implementation', 'dp', 'data structures', 'brute force', 'constructive algorithms', 'binary search', 'sortings', 'graphs', 'dfs and similar', 'trees', 'number theory', 'strings', 'combinatorics']
                
        # encode the tags to one hot encoding
        self.train_df['problem_tags'] = self.train_df['problem_tags'].apply(lambda x: self.__create_binary_vector(x, unique_tags))
        self.test_df['problem_tags'] = self.test_df['problem_tags'].apply(lambda x: self.__create_binary_vector(x, unique_tags))
        self.val_df['problem_tags'] = self.val_df['problem_tags'].apply(lambda x: self.__create_binary_vector(x, unique_tags))
        
        # Shuffle the datasets
        self.train_df = self.train_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        self.test_df = self.test_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        self.val_df = self.val_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        
        #save the datasets
        self.train_df.to_csv(CONFIG[f'TOP_{top_n_tags}_TRAINING_DATASET_PATH'], index=False)
        self.test_df.to_csv(CONFIG[f'TOP_{top_n_tags}_TESTING_DATASET_PATH'], index=False)
        self.val_df.to_csv(CONFIG[f'TOP_{top_n_tags}_VALIDATION_DATASET_PATH'], index=False)

    def build_train_test_dataset_without_tag_encoding(self, top_n_tags=5):
        
        self.train_df = pd.read_csv(CONFIG[f'BASE_TRAINING_DATASET_PATH'])
        self.test_df = pd.read_csv(CONFIG[f'BASE_TESTING_DATASET_PATH'])
        self.val_df = pd.read_csv(CONFIG[f'BASE_VALIDATION_DATASET_PATH'])
        
        unique_tags = self.get_unique_tags(top_n_tags)
        
        # Filter the dataset
        self.train_df = self.build_filtered_dataset(self.train_df, unique_tags)
        self.test_df = self.build_filtered_dataset(self.test_df, unique_tags)
        self.val_df = self.build_filtered_dataset(self.val_df, unique_tags)
        
        # # clean the datasets of unnecessary collumns
        self.train_df = self.train_df.drop(columns=['problem_link', 'problem_id', 'problem_idx', 'short_id', 'contest_number', 'problem_name', 'problem_solution', 'problem_input', 'problem_output', 'problem_dificulty', 'file_name', 'editorial_link', 'problem_editorial'])
        self.test_df = self.test_df.drop(columns=['problem_link', 'problem_id', 'problem_idx', 'short_id', 'contest_number', 'problem_name', 'problem_solution', 'problem_input', 'problem_output', 'problem_dificulty', 'file_name', 'editorial_link', 'problem_editorial'])
        self.val_df = self.val_df.drop(columns=['problem_link', 'problem_id', 'problem_idx', 'short_id', 'contest_number', 'problem_name', 'problem_solution', 'problem_input', 'problem_output', 'problem_dificulty', 'file_name', 'editorial_link', 'problem_editorial'])
        
        # Shuffle the datasets
        self.train_df = self.train_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        self.test_df = self.test_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        self.val_df = self.val_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        
        #save the datasets
        self.train_df.to_csv(CONFIG[f'TOP_{top_n_tags}_TRAINING_WO_TAG_ENCODING_DATASET_PATH'], index=False)
        self.test_df.to_csv(CONFIG[f'TOP_{top_n_tags}_TESTING_WO_TAG_ENCODING_DATASET_PATH'], index=False)
        self.val_df.to_csv(CONFIG[f'TOP_{top_n_tags}_VALIDATION_WO_TAG_ENCODING_DATASET_PATH'], index=False)

    def build_nli_dataset(self, top_n_tags=5):
        random.seed(RANDOM_STATE)
        
        #BEST ALPHA 2 BETA 6
        
        ALPHA = 2 # AUGUMENTATION FACTOR / MULTIPLY THE BALANCED DATASET BY THIS FACTOR
        BETA = 6 # MAXIMUM NUMBER OF PAIRS PER STATEMENT
        
        unique_tags = self.get_unique_tags(top_n_tags)
                
        self.df_train = pd.read_csv(CONFIG[f'BASE_TRAINING_DATASET_PATH'])
                
        self.df_train = self.build_filtered_dataset(self.df_train, unique_tags)
        
        self.df_train = self.df_train.drop(columns=['problem_link', 'problem_id', 'problem_idx', 'short_id', 'contest_number', 'problem_name', 'problem_solution', 'problem_input', 'problem_output', 'problem_dificulty', 'file_name', 'editorial_link', 'problem_editorial'])

        self.check_dataset_distribution(self.df_train)

        # Ensure the tags are in list format
        self.df_train['problem_tags'] = self.df_train['problem_tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        # Initialize a list to store the NLI training data
        nli_data = []        
        
        # Assume unique_tags and df_train are already defined
        max_pairs_per_statement = BETA  # For example, allow at most 3 pairs per problem statement

        entailment_counts = {tag: 0 for tag in unique_tags}
        
        # Iterate over each row in the DataFrame
        for index, row in self.df_train.iterrows():
            problem_statement = row['problem_statement']
            problem_tags = row['problem_tags']

            # Check if the problem has all the tags from unique_tags
            if set(problem_tags) == set(unique_tags):
                continue
            
            # Create pairs of (problem statement, actual tag)
            actual_tags = list(set(problem_tags) & set(unique_tags))
            if not actual_tags:
                continue  # Skip if there are no common tags
            
            # Find all non-actual tags
            non_actual_tags = set(unique_tags) - set(actual_tags)
            if not non_actual_tags:
                continue  # Skip if there are no common tags
            
            # Generate all possible pairs for this statement
            all_pairs = []
            for actual_tag in actual_tags:
                for non_tag in non_actual_tags:
                    pair = {
                        'problem_statement': problem_statement,
                        'entailment': actual_tag,
                        'contradiction': non_tag
                    }
                    all_pairs.append(pair)
            
            # Limit pairs per problem statement to avoid over-representation
            if len(all_pairs) > max_pairs_per_statement:
                sampled_pairs = random.sample(all_pairs, max_pairs_per_statement)
            else:
                sampled_pairs = all_pairs
                    
            # Optionally update global counters if you're tracking usage per tag
            for pair in sampled_pairs:
                entailment_counts[pair['entailment']] += 1
                nli_data.append(pair)
                
        # Filter out entries where contradiction is None
        nli_data = [entry for entry in nli_data if entry['entailment'] is not None]
        
        # Filter out entries where contradiction is None
        nli_data = [entry for entry in nli_data if entry['contradiction'] is not None]
        
        # Convert the NLI data to a DataFrame
        nli_df = pd.DataFrame(nli_data)
        
        target = max(entailment_counts.values()) * ALPHA
        
        oversampled_subsets = []
        for tag in unique_tags:
            subset = nli_df[nli_df['entailment'] == tag]
            current_count = len(subset)
            if current_count < target:
                n_needed = target - current_count
                additional_samples = resample(
                    subset,
                    replace=True,
                    n_samples=n_needed,
                    random_state=RANDOM_STATE
                )
                balanced_subset = pd.concat([subset, additional_samples])
            else:
                balanced_subset = subset
            oversampled_subsets.append(balanced_subset)
    
        # Combine the oversampled subsets into one DataFrame
        nli_df = pd.concat(oversampled_subsets).reset_index(drop=True)        
        
        # Shuffle the dataset
        nli_df = nli_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        print("Dataset size:", len(nli_df))
                
        # Save the balanced NLI training dataset to a CSV file
        nli_df.to_csv(CONFIG[f'TOP_{top_n_tags}_NLI_TRAINING_DATASET_PATH'], index=False)

        self.check_nli_dataset(top_n_tags, outside_dataset=False)
        
        return nli_df

    def build_nli_dataset_dynamic_sampling(self, top_n_tags=5):
        random.seed(RANDOM_STATE)
        
        #BEST ALPHA 2 BETA 6
        
        ALPHA = 2 # AUGUMENTATION FACTOR / MULTIPLY THE BALANCED DATASET BY THIS FACTOR
        BETA = 6 # MAXIMUM NUMBER OF PAIRS PER STATEMENT
        
        unique_tags = self.get_unique_tags(top_n_tags)
                
        self.df_train = pd.read_csv(CONFIG[f'BASE_TRAINING_DATASET_PATH'])
                
        self.df_train = self.build_filtered_dataset(self.df_train, unique_tags)
        
        self.df_train = self.df_train.drop(columns=['problem_link', 'problem_id', 'problem_idx', 'short_id', 'contest_number', 'problem_name', 'problem_solution', 'problem_input', 'problem_output', 'problem_dificulty', 'file_name', 'editorial_link', 'problem_editorial'])

        self.check_dataset_distribution(self.df_train)

        # Ensure the tags are in list format
        self.df_train['problem_tags'] = self.df_train['problem_tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        # Compute tag embeddings for semantic similarity
        model = TFAutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        tag_embeddings = self.encode_problem_statements(model, unique_tags)
        tag_similarity_matrix = cosine_similarity(tag_embeddings)
    
        # Initialize a list to store the NLI training data
        nli_data = []        
        
        # Assume unique_tags and df_train are already defined
        max_pairs_per_statement = BETA  # For example, allow at most 3 pairs per problem statement

        entailment_counts = {tag: 0 for tag in unique_tags}
        
        # Iterate over each row in the DataFrame
        for index, row in self.df_train.iterrows():
            problem_statement = row['problem_statement']
            problem_tags = row['problem_tags']

            # Check if the problem has all the tags from unique_tags
            if set(problem_tags) == set(unique_tags):
                continue
            
            # Create pairs of (problem statement, actual tag)
            actual_tags = list(set(problem_tags) & set(unique_tags))
            if not actual_tags:
                continue  # Skip if there are no common tags
            
            # Find all non-actual tags
            non_actual_tags = set(unique_tags) - set(actual_tags)
            if not non_actual_tags:
                continue  # Skip if there are no common tags
            
            # Dynamic sampling based on semantic similarity
            hard_negatives = []
            for actual_tag in actual_tags:
                actual_idx = unique_tags.index(actual_tag)
                non_actual_similarities = [(non_tag, tag_similarity_matrix[actual_idx][unique_tags.index(non_tag)]) for non_tag in non_actual_tags]
                non_actual_similarities = sorted(non_actual_similarities, key=lambda x: x[1], reverse=True)

                # Select top-k hard negatives based on similarity
                hard_negatives.extend([non_tag for non_tag, sim in non_actual_similarities[:max_pairs_per_statement]])
            
            # Generate pairs
            all_pairs = []
            for actual_tag in actual_tags:
                for non_tag in hard_negatives:
                    pair = {
                        'problem_statement': problem_statement,
                        'entailment': actual_tag,
                        'contradiction': non_tag
                    }
                    all_pairs.append(pair)
            
            # Limit pairs per problem statement to avoid over-representation
            if len(all_pairs) > max_pairs_per_statement:
                sampled_pairs = random.sample(all_pairs, max_pairs_per_statement)
            else:
                sampled_pairs = all_pairs
                    
            # Optionally update global counters if you're tracking usage per tag
            for pair in sampled_pairs:
                entailment_counts[pair['entailment']] += 1
                nli_data.append(pair)
                
        # Filter out entries where contradiction is None
        nli_data = [entry for entry in nli_data if entry['entailment'] is not None]
        
        # Filter out entries where contradiction is None
        nli_data = [entry for entry in nli_data if entry['contradiction'] is not None]
        
        # Convert the NLI data to a DataFrame
        nli_df = pd.DataFrame(nli_data)
        
        target = max(entailment_counts.values()) * ALPHA
        
        oversampled_subsets = []
        for tag in unique_tags:
            subset = nli_df[nli_df['entailment'] == tag]
            current_count = len(subset)
            if current_count < target:
                n_needed = target - current_count
                additional_samples = resample(
                    subset,
                    replace=True,
                    n_samples=n_needed,
                    random_state=RANDOM_STATE
                )
                balanced_subset = pd.concat([subset, additional_samples])
            else:
                balanced_subset = subset
            oversampled_subsets.append(balanced_subset)
    
        # Combine the oversampled subsets into one DataFrame
        nli_df = pd.concat(oversampled_subsets).reset_index(drop=True)        
        
        # Shuffle the dataset
        nli_df = nli_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        print("Dataset size:", len(nli_df))
                
        # Save the balanced NLI training dataset to a CSV file
        nli_df.to_csv(CONFIG[f'TOP_{top_n_tags}_NLI_DYNAMIC_SAMPLING_TRAINING_DATASET_PATH'], index=False)

        self.check_nli_dataset(top_n_tags, outside_dataset=False)
        
        return nli_df

    def update_tags_to_descriptions(self, top_n_tags=5):
        tags_with_descriptions = {
            "greedy":"A greedy algorithm iteratively makes locally optimal choices at each step, aiming to reach a global optimum without reconsidering previous decisions.",
            "math":"Math problems involve numerical computations, algebraic manipulations, or geometric reasoning, often requiring mathematical formulas and insights.",
            "implementation":"Implementation problems test coding proficiency, emphasizing careful attention to detail, correctness, and the ability to translate problem specifications directly into efficient code.",
            "dp":"Dynamic programming is a method for solving complex problems by breaking them down into simpler overlapping subproblems, storing solutions to these subproblems to avoid redundant computations.",
            "data structures":"Data structures refer to specialized formats for organizing, processing, storing, and retrieving data efficiently, such as arrays, stacks, queues, trees, or hash tables.",
            "brute force":"Brute force approaches systematically enumerate all possible solutions or combinations, often simple but computationally expensive, used when more optimized methods are impractical.",
            "constructive algorithms":"Constructive algorithms involve explicitly constructing or designing a solution through incremental building or logical reasoning, rather than simply verifying existence or correctness.",
            "binary search":"Binary search is an efficient algorithm for finding a target value's position within a sorted array or list by repeatedly dividing the search interval in half.",
            "sortings":"Sorting algorithms systematically rearrange items in a particular order (e.g., ascending or descending), often fundamental in preprocessing data for more complex problems.",
            "graphs":"Graph problems involve nodes (vertices) connected by edges, used to represent relationships or networks, requiring traversal, connectivity checks, or pathfinding.",
            "dfs and similar":"Depth-First Search (DFS) and related techniques systematically explore graph structures by traversing as far as possible along each branch before backtracking.",
            "trees":"Tree problems involve hierarchical graph structures with nodes and edges but no cycles, commonly requiring traversal, modification, or querying of structured data.",
            "number theory":"Number theory problems focus on the properties and relationships of integers, including prime numbers, divisibility, modular arithmetic, and greatest common divisors.",
            "strings":"String problems involve manipulation, analysis, and transformation of text sequences or character arrays, often including tasks like pattern matching, substring searching, or text editing.",
            "combinatorics":"Combinatorial problems deal with counting, arranging, or selecting objects from finite sets, often involving permutations, combinations, and probability principles."
        }
        
        nli_df = pd.read_csv(CONFIG[f'TOP_{top_n_tags}_NLI_TRAINING_DATASET_PATH'])
        
        # update each tag to its description
        nli_df['entailment'] = nli_df['entailment'].apply(lambda x: tags_with_descriptions[x])
        nli_df['contradiction'] = nli_df['contradiction'].apply(lambda x: tags_with_descriptions[x])
        
        # save the updated dataset
        nli_df.to_csv(CONFIG[f'TOP_{top_n_tags}_NLI_TRAINING_DATASET_PATH'], index=False)
        
    def build_basic_nli_dataset(self, top_n_tags=5):
        random.seed(RANDOM_STATE)
                        
        unique_tags = self.get_unique_tags(top_n_tags)
                
        self.df_train = pd.read_csv(CONFIG[f'BASE_TRAINING_DATASET_PATH'])
                
        self.df_train = self.build_filtered_dataset(self.df_train, unique_tags)
        
        self.df_train = self.df_train.drop(columns=['problem_link', 'problem_id', 'problem_idx', 'short_id', 'contest_number', 'problem_name', 'problem_solution', 'problem_input', 'problem_output', 'problem_dificulty', 'file_name', 'editorial_link', 'problem_editorial'])

        self.check_dataset_distribution(self.df_train)

        # Ensure the tags are in list format
        self.df_train['problem_tags'] = self.df_train['problem_tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        # Initialize a list to store the NLI training data
        nli_data = []        
                
        # Iterate over each row in the DataFrame
        for index, row in self.df_train.iterrows():
            problem_statement = row['problem_statement']
            problem_tags = row['problem_tags']

            # Check if the problem has all the tags from unique_tags
            if set(problem_tags) == set(unique_tags):
                continue
            
            # Create pairs of (problem statement, actual tag)
            actual_tags = list(set(problem_tags) & set(unique_tags))
            if not actual_tags:
                continue  # Skip if there are no common tags
            
            # Find all non-actual tags
            non_actual_tags = set(unique_tags) - set(actual_tags)
            if not non_actual_tags:
                continue  # Skip if there are no common tags
            
            # Select a random entailment and contradiction
            entailment_tag = random.choice(list(actual_tags))
            contradiction_tag = random.choice(list(non_actual_tags))

            pair = {
                'problem_statement': problem_statement,
                'entailment': entailment_tag,
                'contradiction': contradiction_tag
            }
            
            nli_data.append(pair)
                
        # Filter out entries where contradiction is None
        nli_data = [entry for entry in nli_data if entry['entailment'] is not None]
        
        # Filter out entries where contradiction is None
        nli_data = [entry for entry in nli_data if entry['contradiction'] is not None]
        
        # Convert the NLI data to a DataFrame
        nli_df = pd.DataFrame(nli_data)    
        
        # Shuffle the dataset
        nli_df = nli_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        print("Dataset size:", len(nli_df))
                
        # Save the balanced NLI training dataset to a CSV file
        nli_df.to_csv(CONFIG[f'TOP_{top_n_tags}_BASIC_NLI_TRAINING_DATASET_PATH'], index=False)
        
        return nli_df

    ############################################################################################################        
    
    def build_outside_train_test_dataset(self, top_n_tags=5):
        
        self.train_df = pd.read_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_BASE_TRAINING_DATASET_PATH'])
        self.test_df = pd.read_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_BASE_TESTING_DATASET_PATH'])
        self.val_df = pd.read_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_BASE_VALIDATION_DATASET_PATH'])
        
        unique_tags = self.get_unique_tags(top_n_tags, outside_dataset=True)
                        
        # # clean the datasets of unnecessary collumns
        self.train_df = self.train_df.drop(columns=[self.train_df.columns[0], 'rating'])
        self.test_df = self.test_df.drop(columns=[self.test_df.columns[0], 'rating'])
        self.val_df = self.val_df.drop(columns=[self.val_df.columns[0], 'rating'])
                                        
        # encode the tags to one hot encoding
        self.train_df['tags'] = self.train_df['tags'].apply(lambda x: self.__create_binary_vector(x, unique_tags))
        self.test_df['tags'] = self.test_df['tags'].apply(lambda x: self.__create_binary_vector(x, unique_tags))
        self.val_df['tags'] = self.val_df['tags'].apply(lambda x: self.__create_binary_vector(x, unique_tags))
        
        # Shuffle the datasets
        self.train_df = self.train_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        self.test_df = self.test_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        self.val_df = self.val_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        
        # Rename columns if needed
        self.train_df = self.train_df.rename(columns={'description': 'problem_statement', 'tags': 'problem_tags'})
        self.test_df = self.test_df.rename(columns={'description': 'problem_statement', 'tags': 'problem_tags'})
        self.val_df = self.val_df.rename(columns={'description': 'problem_statement', 'tags': 'problem_tags'})
        
        self.train_df = self.__preprocess_problem_statements(self.train_df)
        self.test_df = self.__preprocess_problem_statements(self.test_df)
        self.val_df = self.__preprocess_problem_statements(self.val_df)
        
        #save the datasets
        self.train_df.to_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_TRAINING_DATASET_PATH'], index=False)
        self.test_df.to_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_TESTING_DATASET_PATH'], index=False)
        self.val_df.to_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_VALIDATION_DATASET_PATH'], index=False)

    def build_outside_train_test_dataset_without_tag_encoding(self, top_n_tags=5):
        
        self.train_df = pd.read_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_BASE_TRAINING_DATASET_PATH'])
        self.test_df = pd.read_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_BASE_TESTING_DATASET_PATH'])
        self.val_df = pd.read_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_BASE_VALIDATION_DATASET_PATH'])
                                
        # # clean the datasets of unnecessary collumns
        self.train_df = self.train_df.drop(columns=[self.train_df.columns[0], 'rating'])
        self.test_df = self.test_df.drop(columns=[self.test_df.columns[0], 'rating'])
        self.val_df = self.val_df.drop(columns=[self.val_df.columns[0], 'rating'])
                                        
        # Shuffle the datasets
        self.train_df = self.train_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        self.test_df = self.test_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        self.val_df = self.val_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
        
        # Rename columns if needed
        self.train_df = self.train_df.rename(columns={'description': 'problem_statement', 'tags': 'problem_tags'})
        self.test_df = self.test_df.rename(columns={'description': 'problem_statement', 'tags': 'problem_tags'})
        self.val_df = self.val_df.rename(columns={'description': 'problem_statement', 'tags': 'problem_tags'})
        
        self.train_df = self.__preprocess_problem_statements(self.train_df)
        self.test_df = self.__preprocess_problem_statements(self.test_df)
        self.val_df = self.__preprocess_problem_statements(self.val_df)
        
        #save the datasets
        self.train_df.to_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_TRAINING_WO_TAG_ENCODING_DATASET_PATH'], index=False)
        self.test_df.to_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_TESTING_WO_TAG_ENCODING_DATASET_PATH'], index=False)
        self.val_df.to_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_VALIDATION_WO_TAG_ENCODING_DATASET_PATH'], index=False)

    def build_outside_nli_dataset(self, top_n_tags=5):
        
        random.seed(RANDOM_STATE)
        
        #BEST ALPHA 2 BETA 6
        
        ALPHA = 2 # AUGUMENTATION FACTOR / MULTIPLY THE BALANCED DATASET BY THIS FACTOR
        BETA = 6 # MAXIMUM NUMBER OF PAIRS PER STATEMENT
        
        unique_tags = self.get_unique_tags(top_n_tags, outside_dataset=True)

        self.df_train = pd.read_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_BASE_TRAINING_DATASET_PATH'])
        
        self.df_train = self.df_train.drop(columns=[self.df_train.columns[0], 'rating'])
        self.df_train = self.df_train.rename(columns={'description': 'problem_statement', 'tags': 'problem_tags'})

        self.check_dataset_distribution(self.df_train)
        
        # Preprocess the problem statements
        self.df_train = self.__preprocess_problem_statements(self.df_train)

        # Ensure the tags are in list format
        self.df_train['problem_tags'] = self.df_train['problem_tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        # Initialize a list to store the NLI training data
        nli_data = []        
        
        # Assume unique_tags and df_train are already defined
        max_pairs_per_statement = BETA  # For example, allow at most 3 pairs per problem statement

        entailment_counts = {tag: 0 for tag in unique_tags}
        
        # Iterate over each row in the DataFrame
        for index, row in self.df_train.iterrows():
            problem_statement = row['problem_statement']
            problem_tags = row['problem_tags']

            # Check if the problem has all the tags from unique_tags
            if set(problem_tags) == set(unique_tags):
                continue
            
            # Create pairs of (problem statement, actual tag)
            actual_tags = list(set(problem_tags) & set(unique_tags))
            if not actual_tags:
                continue  # Skip if there are no common tags
            
            # Find all non-actual tags
            non_actual_tags = set(unique_tags) - set(actual_tags)
            if not non_actual_tags:
                continue  # Skip if there are no common tags
            
            # Generate all possible pairs for this statement
            all_pairs = []
            for actual_tag in actual_tags:
                for non_tag in non_actual_tags:
                    pair = {
                        'problem_statement': problem_statement,
                        'entailment': actual_tag,
                        'contradiction': non_tag
                    }
                    all_pairs.append(pair)
            
            # Limit pairs per problem statement to avoid over-representation
            if len(all_pairs) > max_pairs_per_statement:
                sampled_pairs = random.sample(all_pairs, max_pairs_per_statement)
            else:
                sampled_pairs = all_pairs
                    
            # Optionally update global counters if you're tracking usage per tag
            for pair in sampled_pairs:
                entailment_counts[pair['entailment']] += 1
                nli_data.append(pair)
                
        # Filter out entries where contradiction is None
        nli_data = [entry for entry in nli_data if entry['entailment'] is not None]
        
        # Filter out entries where contradiction is None
        nli_data = [entry for entry in nli_data if entry['contradiction'] is not None]
        
        # Convert the NLI data to a DataFrame
        nli_df = pd.DataFrame(nli_data)
        
        target = max(entailment_counts.values()) * ALPHA
        
        oversampled_subsets = []
        for tag in unique_tags:
            subset = nli_df[nli_df['entailment'] == tag]
            current_count = len(subset)
            if current_count < target:
                n_needed = target - current_count
                additional_samples = resample(
                    subset,
                    replace=True,
                    n_samples=n_needed,
                    random_state=RANDOM_STATE
                )
                balanced_subset = pd.concat([subset, additional_samples])
            else:
                balanced_subset = subset
            oversampled_subsets.append(balanced_subset)
    
        # Combine the oversampled subsets into one DataFrame
        nli_df = pd.concat(oversampled_subsets).reset_index(drop=True)        
        
        # Shuffle the dataset
        nli_df = nli_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        print("Dataset size:", len(nli_df))
                
        # Save the balanced NLI training dataset to a CSV file
        nli_df.to_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_NLI_TRAINING_DATASET_PATH'], index=False)

        self.check_nli_dataset(top_n_tags, outside_dataset=True)
        
        return nli_df

    def build_outside_nli_dataset_dynamic_sampling(self, top_n_tags=5):
        random.seed(RANDOM_STATE)
        
        #BEST ALPHA 2 BETA 6
        
        ALPHA = 2 # AUGUMENTATION FACTOR / MULTIPLY THE BALANCED DATASET BY THIS FACTOR
        BETA = 6 # MAXIMUM NUMBER OF PAIRS PER STATEMENT
        
        unique_tags = self.get_unique_tags(top_n_tags, outside_dataset=True)

        self.df_train = pd.read_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_BASE_TRAINING_DATASET_PATH'])
        
        self.df_train = self.df_train.drop(columns=[self.df_train.columns[0], 'rating'])
        self.df_train = self.df_train.rename(columns={'description': 'problem_statement', 'tags': 'problem_tags'})

        self.check_dataset_distribution(self.df_train)

        # Preprocess the problem statements
        self.df_train = self.__preprocess_problem_statements(self.df_train)

        # Ensure the tags are in list format
        self.df_train['problem_tags'] = self.df_train['problem_tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        # Compute tag embeddings for semantic similarity
        model = TFAutoModel.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        tag_embeddings = self.encode_problem_statements(model, unique_tags)
        tag_similarity_matrix = cosine_similarity(tag_embeddings)
    
        # Initialize a list to store the NLI training data
        nli_data = []        
        
        # Assume unique_tags and df_train are already defined
        max_pairs_per_statement = BETA  # For example, allow at most 3 pairs per problem statement

        entailment_counts = {tag: 0 for tag in unique_tags}
        
        # Iterate over each row in the DataFrame
        for index, row in self.df_train.iterrows():
            problem_statement = row['problem_statement']
            problem_tags = row['problem_tags']

            # Check if the problem has all the tags from unique_tags
            if set(problem_tags) == set(unique_tags):
                continue
            
            # Create pairs of (problem statement, actual tag)
            actual_tags = list(set(problem_tags) & set(unique_tags))
            if not actual_tags:
                continue  # Skip if there are no common tags
            
            # Find all non-actual tags
            non_actual_tags = set(unique_tags) - set(actual_tags)
            if not non_actual_tags:
                continue  # Skip if there are no common tags
            
            # Dynamic sampling based on semantic similarity
            hard_negatives = []
            for actual_tag in actual_tags:
                actual_idx = unique_tags.index(actual_tag)
                non_actual_similarities = [(non_tag, tag_similarity_matrix[actual_idx][unique_tags.index(non_tag)]) for non_tag in non_actual_tags]
                non_actual_similarities = sorted(non_actual_similarities, key=lambda x: x[1], reverse=True)

                # Select top-k hard negatives based on similarity
                hard_negatives.extend([non_tag for non_tag, sim in non_actual_similarities[:max_pairs_per_statement]])
            
            # Generate pairs
            all_pairs = []
            for actual_tag in actual_tags:
                for non_tag in hard_negatives:
                    pair = {
                        'problem_statement': problem_statement,
                        'entailment': actual_tag,
                        'contradiction': non_tag
                    }
                    all_pairs.append(pair)
            
            # Limit pairs per problem statement to avoid over-representation
            if len(all_pairs) > max_pairs_per_statement:
                sampled_pairs = random.sample(all_pairs, max_pairs_per_statement)
            else:
                sampled_pairs = all_pairs
                    
            # Optionally update global counters if you're tracking usage per tag
            for pair in sampled_pairs:
                entailment_counts[pair['entailment']] += 1
                nli_data.append(pair)
                
        # Filter out entries where contradiction is None
        nli_data = [entry for entry in nli_data if entry['entailment'] is not None]
        
        # Filter out entries where contradiction is None
        nli_data = [entry for entry in nli_data if entry['contradiction'] is not None]
        
        # Convert the NLI data to a DataFrame
        nli_df = pd.DataFrame(nli_data)
        
        target = max(entailment_counts.values()) * ALPHA
        
        oversampled_subsets = []
        for tag in unique_tags:
            subset = nli_df[nli_df['entailment'] == tag]
            current_count = len(subset)
            if current_count < target:
                n_needed = target - current_count
                additional_samples = resample(
                    subset,
                    replace=True,
                    n_samples=n_needed,
                    random_state=RANDOM_STATE
                )
                balanced_subset = pd.concat([subset, additional_samples])
            else:
                balanced_subset = subset
            oversampled_subsets.append(balanced_subset)
    
        # Combine the oversampled subsets into one DataFrame
        nli_df = pd.concat(oversampled_subsets).reset_index(drop=True)        
        
        # Shuffle the dataset
        nli_df = nli_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        print("Dataset size:", len(nli_df))
                
        # Save the balanced NLI training dataset to a CSV file
        nli_df.to_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_NLI_DYNAMIC_SAMPLING_TRAINING_DATASET_PATH'], index=False)

        self.check_nli_dataset(top_n_tags, outside_dataset=True)
        
        return nli_df

    def build_outside_basic_nli_dataset(self, top_n_tags=5):
        random.seed(RANDOM_STATE)
                        
        unique_tags = self.get_unique_tags(top_n_tags)
                
        self.df_train = pd.read_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_BASE_TRAINING_DATASET_PATH'])
                
        self.df_train = self.df_train.drop(columns=[self.df_train.columns[0], 'rating'])
        self.df_train = self.df_train.rename(columns={'description': 'problem_statement', 'tags': 'problem_tags'})

        self.check_dataset_distribution(self.df_train)
        
        # Preprocess the problem statements
        self.df_train = self.__preprocess_problem_statements(self.df_train)

        # Ensure the tags are in list format
        self.df_train['problem_tags'] = self.df_train['problem_tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        # Initialize a list to store the NLI training data
        nli_data = []        
                
        # Iterate over each row in the DataFrame
        for index, row in self.df_train.iterrows():
            problem_statement = row['problem_statement']
            problem_tags = row['problem_tags']

            # Check if the problem has all the tags from unique_tags
            if set(problem_tags) == set(unique_tags):
                continue
            
            # Create pairs of (problem statement, actual tag)
            actual_tags = list(set(problem_tags) & set(unique_tags))
            if not actual_tags:
                continue  # Skip if there are no common tags
            
            # Find all non-actual tags
            non_actual_tags = set(unique_tags) - set(actual_tags)
            if not non_actual_tags:
                continue  # Skip if there are no common tags
            
            # Select a random entailment and contradiction
            entailment_tag = random.choice(list(actual_tags))
            contradiction_tag = random.choice(list(non_actual_tags))

            pair = {
                'problem_statement': problem_statement,
                'entailment': entailment_tag,
                'contradiction': contradiction_tag
            }
            
            nli_data.append(pair)
                
        # Filter out entries where contradiction is None
        nli_data = [entry for entry in nli_data if entry['entailment'] is not None]
        
        # Filter out entries where contradiction is None
        nli_data = [entry for entry in nli_data if entry['contradiction'] is not None]
        
        # Convert the NLI data to a DataFrame
        nli_df = pd.DataFrame(nli_data)    
        
        # Shuffle the dataset
        nli_df = nli_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

        print("Dataset size:", len(nli_df))
                
        # Save the balanced NLI training dataset to a CSV file
        nli_df.to_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_BASIC_NLI_TRAINING_DATASET_PATH'], index=False)
        
        return nli_df

    ############################################################################################################

    def check_dataset_distribution(self, df):
        # count the occurrences of each tag
        tags = df['problem_tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

        # Initialize the counter for all tags
        tag_counts = defaultdict(int)

        # Iterate over all problems and count each tag
        for tag_list in tags:
            for tag in tag_list:
                tag_counts[tag.strip().strip("'")] += 1
                
        sorted_tag_counts = dict(sorted(tag_counts.items(), key=lambda item: item[1], reverse=True))
        
        print(sorted_tag_counts.keys())
        
        print('Tags distribution :')
        for tag, count in sorted_tag_counts.items():
            print(f"{tag}: {count}")
                    
    def check_nli_dataset(self, top_n_tags=5, outside_dataset=False):
        if outside_dataset:
            self.df_train = pd.read_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_NLI_TRAINING_DATASET_PATH'])
        else:
            self.df_train = pd.read_csv(CONFIG[f'TOP_{top_n_tags}_NLI_TRAINING_DATASET_PATH'])
        
        # Initialize dictionaries to store the counts
        entailment_counts = defaultdict(int)
        contradiction_counts = defaultdict(int)
        
        # Iterate over each row in the DataFrame
        for index, row in self.df_train.iterrows():
            entailment_tag = row['entailment']
            contradiction_tag = row['contradiction']
            
            # Update the counts
            entailment_counts[entailment_tag] += 1
            contradiction_counts[contradiction_tag] += 1
            
        # Print the counts for entailment tags
        print("Entailment Tag Counts:")
        for tag, count in entailment_counts.items():
            print(f"{tag}: {count}")
        
        # Print the counts for contradiction tags
        print("Contradiction Tag Counts:")
        for tag, count in contradiction_counts.items():
            print(f"{tag}: {count}")

        return entailment_counts, contradiction_counts
    
        # Create an adjacency matrix based on problem similarity

    def get_dataset_tags_and_distribution(self, top_n_tags=5):
        # Load the dataset
        df = pd.read_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_TRAINING_WO_TAG_ENCODING_DATASET_PATH'])
        
        # Ensure the tags are in list format
        df['problem_tags'] = df['problem_tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        # Initialize a dictionary to store the counts
        tag_counts = defaultdict(int)
        
        # Iterate over each row in the DataFrame
        for tags in df['problem_tags']:
            for tag in tags:
                tag_counts[tag] += 1
        
        print(tag_counts.keys())

        # Print the counts for each tag
        print("Tag Counts:")
        for tag, count in tag_counts.items():
            print(f"{tag}: {count}")
            
    def get_dataset_top_tags(self, top_n_tags=5):
        # Load the dataset
        unique_tags = self.get_unique_tags(top_n_tags, outside_dataset=True)
        
        print(unique_tags)
        


    ############################################################################################################

    def __create_similarity_graph(self, embeddings, threshold=0.8):
        """Creates a similarity graph where edges are added if cosine similarity > threshold."""
        num_nodes = len(embeddings)
        graph = nx.Graph()

        # Add nodes
        for i in range(num_nodes):
            graph.add_node(i)

        # Add edges based on cosine similarity
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                similarity = 1 - cosine(embeddings[i], embeddings[j])
                if similarity > threshold:
                    graph.add_edge(i, j, weight=similarity)

        return graph
    
    def __propagate_labels_with_graph(self, df, graph):
        """
        Uses a similarity graph to propagate missing labels.
        Here, each label is represented as a binary vector (e.g. [0, 0, ..., 1, 0]).
        For each node, we update its label vector using an elementwise maximum (logical OR)
        across all of its neighbors.
        """
        # Create a dictionary mapping each node to its label vector as a numpy array
        label_dict = {i: np.array(df.iloc[i]['problem_tags']) for i in graph.nodes()}

        for node in graph.nodes():
            neighbors = list(graph.neighbors(node))
            if not neighbors:
                continue  # Skip isolated nodes

            # Start with the current node's label vector
            aggregated = label_dict[node].copy()
            # Propagate labels from neighbors (logical OR via elementwise maximum)
            for neighbor in neighbors:
                aggregated = np.maximum(aggregated, label_dict[neighbor])
            label_dict[node] = aggregated

        # Write the updated labels back to the DataFrame (convert numpy arrays to lists)
        df['problem_tags'] = [label_dict[i].tolist() for i in range(len(df))]
        return df

    def encode_problem_statements(self, model, problem_statements, batch_size=32):
        """
        Encodes a list of problem statements into embeddings using mean pooling.

        """
        tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-mpnet-base-v2')
        
        # Tokenize the list of problem statements
        inputs = tokenizer(
            problem_statements,
            add_special_tokens=True,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='tf'
        )

        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Create a tf.data.Dataset from the tokenized inputs
        dataset = tf.data.Dataset.from_tensor_slices((input_ids, attention_mask))
        dataset = dataset.batch(batch_size)
        
        all_embeddings = []
        
        for batch in tqdm(dataset, desc="Encoding problem statements"):
            input_ids_batch, attention_mask_batch = batch
            outputs = model(input_ids_batch, attention_mask=attention_mask_batch)  # outputs.last_hidden_state has shape (batch_size, seq_length, hidden_size)
            
            # Create a mask for the attention tokens (0 for padding tokens)
            attention_mask_batch = tf.cast(tf.expand_dims(attention_mask_batch, -1), dtype=tf.float32)
            # Sum the token embeddings
            sum_embeddings = tf.reduce_sum(outputs.last_hidden_state * attention_mask_batch, axis=1)
            # Sum the mask to get counts of valid tokens, and then compute the mean
            sum_mask = tf.reduce_sum(attention_mask_batch, axis=1)
            sentence_embeddings_mean = sum_embeddings / sum_mask
            
            all_embeddings.append(sentence_embeddings_mean)

        # Concatenate all embeddings
        all_embeddings = tf.concat(all_embeddings, axis=0)
        
        return all_embeddings.numpy()

    def print_binary_dataset_distribution(self, df):
        """
        Prints the distribution of 1s for each index in the problem_tags column.
        """
        # Ensure the tags are in list format
        df['problem_tags'] = df['problem_tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        # Initialize a counter for the number of 1s at each index
        tag_counts = Counter()
        
        # Iterate over each row in the DataFrame
        for tags in df['problem_tags']:
            for index, value in enumerate(tags):
                if value == 1:
                    tag_counts[index] += 1
        
        # Print the counts for each index
        print("Distribution of 1s for each index in problem_tags:")
        for index, count in sorted(tag_counts.items()):
            print(f"Index {index}: {count}")

    def augument_train_data(self, top_n_tags):
        
        target='problem_statement'
        
        # Load the base training dataset
        df = pd.read_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_TRAINING_DATASET_PATH'])
        
        # Convert problem_tags to list format.
        # For example, if stored as a string "[0, 0, ..., 1, 0]", convert it to a list.
        df['problem_tags'] = df['problem_tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        
        self.print_binary_dataset_distribution(df)
        
        # Load the encoder model
        # model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        model_path = os.path.join(CONFIG["TRANSFORMER_SAVE_PATH_ROOT"], 'transformer_model_20250218_205740_best_6_to_1')
        model = TFAutoModel.from_pretrained(model_path)
        
        # Generate embeddings for problem statements
        problem_statements = df[target].tolist()
        problem_statement_embeddings = self.encode_problem_statements(model, problem_statements, batch_size=32)

        # Create the similarity graph. 
        # A threshold of 0.70 is chosen based on prior experiments.
        graph = self.__create_similarity_graph(problem_statement_embeddings, threshold=0.99)
        
        # Apply label propagation using the similarity graph
        balanced_df = self.__propagate_labels_with_graph(df, graph)
        
        self.print_binary_dataset_distribution(balanced_df)
        
        balanced_df.to_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_TRAINING_DATASET_PATH'], index=False)

    def get_label_binary_df(self, df, tag_column="problem_tags"):
        """
        Converts a DataFrame column containing lists of tags into a binary DataFrame.
        Each column in the new DataFrame corresponds to one unique tag, with 1 if the tag is present, 0 otherwise.
        """
        unique_tags = set()
        for tags in df[tag_column]:
            unique_tags.update(tags)
        unique_tags = sorted(list(unique_tags))
        
        binary_rows = []
        for tags in df[tag_column]:
            row = {tag: 1 if tag in tags else 0 for tag in unique_tags}
            binary_rows.append(row)
        return pd.DataFrame(binary_rows)
    
    def analyze_label_relationships(self, df_labels):
        """
        Analyzes the relationships between labels by computing the Pearson correlation
        between each pair of binary label columns.
        
        Args:
            df_labels (pd.DataFrame): A DataFrame where each column is a binary indicator for a label.
            threshold (float): Only label pairs with correlation above this value are returned.
            
        Returns:
            similar_pairs (list): A list of tuples (label1, label2, correlation)
                                  for label pairs with correlation above the threshold.
        """
        # Compute the correlation matrix for the binary labels.
        # (For binary variables, Pearson correlation is a rough measure of co-occurrence.)
        corr_matrix = df_labels.corr()
        
        labels = corr_matrix.columns.tolist()
        
        # Iterate over the upper-triangle of the correlation matrix.
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                corr_value = corr_matrix.iloc[i, j]
                print(f"Correlation between {labels[i]} and {labels[j]}: {corr_value:.2f}")
            
        #         if corr_value > threshold:
        #             similar_pairs.append((labels[i], labels[j], corr_value))
        
        # # Sort pairs by correlation in descending order.
        # similar_pairs = sorted(similar_pairs, key=lambda x: x[2], reverse=True)
        # return similar_pairs

    def analyze_tag_distribution(self, top_n_tags):
        
        print(f"Analyzing tag distribution for top {top_n_tags} tags")
        
        df_training = pd.read_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_TESTING_WO_TAG_ENCODING_DATASET_PATH'])
        df_testing = pd.read_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_TESTING_WO_TAG_ENCODING_DATASET_PATH'])
        df_validation = pd.read_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_VALIDATION_WO_TAG_ENCODING_DATASET_PATH'])
        
        # Concatenate all dataframes
        df = pd.concat([df_training, df_testing, df_validation], ignore_index=True)
                
        # Ensure the tags are in list format
        df['problem_tags'] = df['problem_tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            
        # Ensure the tags are in list format
        df['problem_tags'] = df['problem_tags'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
        # Initialize the counter for all tags
        tag_counts = defaultdict(int)

        # Iterate over all problems and count each tag
        for tag_list in df['problem_tags']:
            for tag in tag_list:
                tag_counts[tag] += 1
                
        sorted_tag_counts = dict(sorted(tag_counts.items(), key=lambda item: item[1], reverse=True))

        print(sorted_tag_counts)
    
        # Convert the tag lists into a binary DataFrame.
        df_labels = self.get_label_binary_df(df, tag_column="problem_tags")
        
        # Analyze label relationships using a correlation threshold, e.g. 0.5.
        self.analyze_label_relationships(df_labels)
        
    ###########################################################################################################
    
    def create_alpaca_datasets(self, top_n_tags, outside=False):
        if outside:
            df = pd.read_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_TRAINING_WO_TAG_ENCODING_DATASET_PATH'])
        else:
            df = pd.read_csv(CONFIG[f'TOP_{top_n_tags}_TRAINING_WO_TAG_ENCODING_DATASET_PATH'])

        records = self.create_alpaca_dataset(df, top_n_tags, outside=outside)
        
        if outside:
            pathlib.Path(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_ALPACA_TRAINING_DATASET_PATH']).write_text(
                json.dumps(records, ensure_ascii=False, indent=2)
            )
        else:
            pathlib.Path(CONFIG[f'TOP_{top_n_tags}_ALPACA_TRAINING_DATASET_PATH']).write_text(
                json.dumps(records, ensure_ascii=False, indent=2)
            )

        if outside:
            df = pd.read_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_TESTING_WO_TAG_ENCODING_DATASET_PATH'])
        else:
            df = pd.read_csv(CONFIG[f'TOP_{top_n_tags}_TESTING_WO_TAG_ENCODING_DATASET_PATH'])

        records = self.create_alpaca_dataset(df, top_n_tags, outside=outside)
        
        if outside:
            pathlib.Path(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_ALPACA_TESTING_DATASET_PATH']).write_text(
                json.dumps(records, ensure_ascii=False, indent=2)
            )
        else:
            pathlib.Path(CONFIG[f'TOP_{top_n_tags}_ALPACA_TESTING_DATASET_PATH']).write_text(
                json.dumps(records, ensure_ascii=False, indent=2)
            )

        if outside:
            df = pd.read_csv(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_VALIDATION_WO_TAG_ENCODING_DATASET_PATH'])
        else:
            df = pd.read_csv(CONFIG[f'TOP_{top_n_tags}_VALIDATION_WO_TAG_ENCODING_DATASET_PATH'])

        records = self.create_alpaca_dataset(df, top_n_tags, outside=outside)
        
        if outside:
            pathlib.Path(CONFIG[f'OUTSIDE_TOP_{top_n_tags}_ALPACA_VALIDATION_DATASET_PATH']).write_text(
                json.dumps(records, ensure_ascii=False, indent=2)
            )
        else:
            pathlib.Path(CONFIG[f'TOP_{top_n_tags}_ALPACA_VALIDATION_DATASET_PATH']).write_text(
                json.dumps(records, ensure_ascii=False, indent=2)
            )

    def create_alpaca_dataset(self, df, top_n_tags, outside=False):
        def tags_to_string(cell):
            """Turn the Python-repr list in each CSV row into a plain text string."""
            return ", ".join(ast.literal_eval(cell))
        
        top_tags = self.get_unique_tags(top_n_tags, outside_dataset=outside)
                        
        # Initialize a list to store the Alpaca dataset entries
        records = []

        # Iterate over each row in the DataFrame
        for _, row in df.iterrows():
                    
            records.append({
                "instruction": (
                    "You are a strict tag classifier. Return ONLY the applicable tags from this fixed list:\n"
                    f"{top_tags}\n"
                    "Output exactly the tags, lowercase, separated by comma-no-space.\nNo other words.\n\nExample - \nProblem: All piles have equal size...\nAnswer: dp\n\nNow classify:"
                ),
                "input": row["problem_statement"],
                "output": tags_to_string(row["problem_tags"])
            })

        return records
