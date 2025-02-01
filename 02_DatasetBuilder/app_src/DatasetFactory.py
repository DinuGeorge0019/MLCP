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
from sklearn.model_selection import train_test_split
from time import sleep
# local application/library specific imports
from app_config import AppConfig

# define configuration proxy
working_dir = os.path.dirname(os.getcwd())
configProxy = AppConfig(working_dir)

# get configuration
CONFIG = configProxy.return_config()

# get global constants configuration
GLOBAL_CONSTANTS = configProxy.return_global_constants()

RANDOM_STATE = GLOBAL_CONSTANTS['RANDOM_SEED']

GENERATE_VALIDATION_DATASET = True

random.seed(RANDOM_STATE)

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
        
        output_file.write(f"Minimum problem statement length: {min_statement_length}\n")
        output_file.write(f"Maximum problem statement length: {max_statement_length}\n")
        output_file.write(f"Mean problem statement length: {mean_statement_length}\n\n")

        # Get the minimum and maximum problem editorials lengths
        min_editorial_length = df['problem_editorial'].apply(len).min()
        max_editorial_length = df['problem_editorial'].apply(len).max()
        mean_editorial_length = df['problem_editorial'].apply(len).mean()
        
        output_file.write(f"Minimum problem editorial length: {min_editorial_length}\n")
        output_file.write(f"Maximum problem editorial length: {max_editorial_length}\n")
        output_file.write(f"Mean problem editorial length: {mean_editorial_length}\n\n")

        # Get the minimum and maximum problem solution lengths
        min_solution_length = df['problem_solution'].apply(len).min()
        max_solution_length = df['problem_solution'].apply(len).max()
        mean_statement_length = df['problem_solution'].apply(len).mean()
        
        output_file.write(f"Minimum problem solution length: {min_solution_length}\n")
        output_file.write(f"Maximum problem solution length: {max_solution_length}\n")
        output_file.write(f"Mean problem solution length: {mean_statement_length}\n\n")
        
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
        
        plt.figure(figsize=(20, 6))
        plt.bar(tags, counts, color='skyblue')
        plt.xlabel('Tags')
        plt.ylabel('Number of Problems')
        plt.title('Distribution of Tag Classes')
        plt.xticks(rotation=90)
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

    def __preprocess_problem_statements(self):

        print("\nPreprocessing problem statements")

        # remove unknown symbols
        unknown_symbols = self.__search_unknown_symbols(self.filtered_df, collumn_name='problem_statement')
        if unknown_symbols:
            self.filtered_df.loc[:, 'problem_statement'] = self.filtered_df['problem_statement'].apply(lambda text: re.sub('[%s]' % re.escape(unknown_symbols), ' ', text))
        else:
            pass

        # removing punctuations
        self.filtered_df.loc[:, 'problem_statement'] = self.filtered_df['problem_statement'].apply(lambda text: re.sub('[%s]' % re.escape(string.punctuation), ' ', text))

        # removing all unwanted spaces
        self.filtered_df.loc[:, 'problem_statement'] = self.filtered_df['problem_statement'].apply(lambda text: re.sub('\s+', ' ', text))
        
        # Remove all duplicated sequences
        duplicated_rows_bool = self.filtered_df['problem_statement'].duplicated()
        self.filtered_df = self.filtered_df[~duplicated_rows_bool]

        # Remove non relevant editorials / statements
        # df = df.drop(index=[325, 506, 722, 730, 736, 911])

    def __preprocess_problem_editorials(self):
        print("\nPreprocessing problem editorials")

        # remove unknown symbols
        unknown_symbols = self.__search_unknown_symbols(self.filtered_df, collumn_name='problem_editorial')
        if unknown_symbols:
            self.filtered_df.loc[:, 'problem_editorial'] = self.filtered_df['problem_editorial'].apply(lambda text: re.sub('[%s]' % re.escape(unknown_symbols), ' ', text))
        else:
            pass

        # removing punctuations
        self.filtered_df.loc[:, 'problem_editorial'] = self.filtered_df['problem_editorial'].apply(lambda text: re.sub('[%s]' % re.escape(string.punctuation), ' ', text))

        # removing all unwanted spaces
        self.filtered_df.loc[:, 'problem_editorial'] = self.filtered_df['problem_editorial'].apply(lambda text: re.sub('\s+', ' ', text))
        
        # Remove all duplicated sequences
        duplicated_rows_bool = self.filtered_df['problem_editorial'].duplicated()
        self.filtered_df = self.filtered_df[~duplicated_rows_bool]
    
    def __preprocess_problem_solutions(self):
        pass

    def __replace_tags(self, tags, top_tags):
        tags_set = set(ast.literal_eval(tags))
        if tags_set.isdisjoint(set(top_tags)):                  #  isdisjoint
            return 'nan'
        return tags

    def build_filtered_dataset(self, top_n_tags=5):
        # read the raw dataset
        df = pd.read_csv(CONFIG['RAW_DATASET_PATH'])
        
        # count the occurrences of each tag
        tags = df['problem_tags'].apply(lambda x: x.lstrip('[').rstrip(']').split(','))

        # Initialize the counter for all tags
        tag_counts = defaultdict(int)

        # Iterate over all problems and count each tag
        for tag_list in tags:
            for tag in tag_list:
                tag_counts[tag.strip().strip("'")] += 1
                
        sorted_tag_counts = dict(sorted(tag_counts.items(), key=lambda item: item[1], reverse=True))
        
        # get only the top n tags
        top_tags = list(sorted_tag_counts.keys())[:top_n_tags]
                
        # filter the dataset
        self.filtered_df = df[df['problem_tags'].apply(lambda x: bool(set(ast.literal_eval(x)) & set(top_tags)))]
        
        # Preprocess the problem statements
        self.__preprocess_problem_statements()
        
        # Preprocess the problem editorials
        self.__preprocess_problem_editorials()
        
        # Preprocess the problem solutions
        self.__preprocess_problem_solutions()
        
        # Save the filtered dataset to a CSV file
        self.filtered_df.to_csv(CONFIG[f'TOP_{top_n_tags}_FILTERED_DATASET_PATH'],  index=False)
        
        return top_tags

    def build_base_train_test_dataset(self, top_n_tags=5):
        # Load the preprocessed dataset
        self.df = pd.read_csv(CONFIG[f'TOP_{top_n_tags}_FILTERED_DATASET_PATH'])

        # Split dataset between train and test
        self.df_train, self.df_test = train_test_split(self.df, test_size=0.2, random_state=RANDOM_STATE)
        
        if GENERATE_VALIDATION_DATASET:
            # Split the train set into train and validation sets
            self.df_train, self.df_val = train_test_split(self.df_train, test_size=0.1, random_state=RANDOM_STATE)
            self.df_val.to_csv(CONFIG[f'TOP_{top_n_tags}_BASE_VALIDATION_DATASET_PATH'], index=False)
            
        self.df_train.to_csv(CONFIG[f'TOP_{top_n_tags}_BASE_TRAINING_DATASET_PATH'], index=False)
        self.df_test.to_csv(CONFIG[f'TOP_{top_n_tags}_BASE_TESTING_DATASET_PATH'], index=False)
    
    # Function to create binary vector for tags
    def __create_binary_vector(self, tag_list, unique_tags):
        
        binary_vector = [0]*len(unique_tags)
        
        if 'nan' != str(tag_list):
            tag_list = ast.literal_eval(tag_list)  # Convert string representation of list to actual list
            for tag in tag_list:
                if tag in unique_tags:
                    binary_vector[unique_tags.index(tag)] = 1
        
        return binary_vector
    
    def build_train_test_dataset(self, top_n_tags=5):
        
        # Filter the dataset
        unique_tags = self.build_filtered_dataset(top_n_tags)
                
        # Build the base train test dataset
        self.build_base_train_test_dataset(top_n_tags)
        
        # Load the base dataset
        self.df_train = pd.read_csv(CONFIG[f'TOP_{top_n_tags}_BASE_TRAINING_DATASET_PATH'])
        self.df_test = pd.read_csv(CONFIG[f'TOP_{top_n_tags}_BASE_TESTING_DATASET_PATH'])
        self.df_val = pd.read_csv(CONFIG[f'TOP_{top_n_tags}_BASE_VALIDATION_DATASET_PATH'])
        
        # clean the datasets of unnecessary collumns
        self.df_train = self.df_train.drop(columns=['problem_link', 'problem_id', 'problem_idx', 'short_id', 'contest_number', 'problem_name', 'problem_input', 'problem_output', 'file_name', 'editorial_link'])
        self.df_test = self.df_test.drop(columns=['problem_link', 'problem_id', 'problem_idx', 'short_id', 'contest_number', 'problem_name', 'problem_input', 'problem_output', 'file_name', 'editorial_link'])
        self.df_val = self.df_val.drop(columns=['problem_link', 'problem_id', 'problem_idx', 'short_id', 'contest_number', 'problem_name', 'problem_input', 'problem_output', 'file_name', 'editorial_link'])
                
        # for 5 classes ->  ['greedy', 'math', 'implementation', 'dp', 'data structures']
        # for 10 classes -> ['greedy', 'math', 'implementation', 'dp', 'data structures', 'constructive algorithms', 'brute force', 'graphs', 'binary search', 'sortings']
        # for 15 classes -> ['greedy', 'math', 'implementation', 'dp', 'data structures', 'constructive algorithms', 'brute force', 'graphs', 'binary search', 'sortings', 'dfs and similar', 'trees', 'number theory', 'strings', 'combinatorics']
        # for 20 classes -> ['greedy', 'math', 'implementation', 'dp', 'data structures', 'constructive algorithms', 'brute force', 'graphs', 'binary search', 'sortings', 'dfs and similar', 'trees', 'number theory', 'strings', 'combinatorics', 'two pointers', 'bitmasks', 'geometry', 'dsu', 'shortest paths']
        
        # encode the tags to one hot encoding
        # self.df_train['tags'] = self.df_train['tags'].apply(lambda x: self.__create_binary_vector(x, unique_tags))
        # self.df_test['tags'] = self.df_test['tags'].apply(lambda x: self.__create_binary_vector(x, unique_tags))
        # self.df_val['tags'] = self.df_val['tags'].apply(lambda x: self.__create_binary_vector(x, unique_tags))
        
        #save the datasets
        self.df_train.to_csv(CONFIG[f'TOP_{top_n_tags}_TRAINING_DATASET_PATH'], index=False)
        self.df_test.to_csv(CONFIG[f'TOP_{top_n_tags}_TESTING_DATASET_PATH'], index=False)
        self.df_val.to_csv(CONFIG[f'TOP_{top_n_tags}_VALIDATION_DATASET_PATH'], index=False)
    