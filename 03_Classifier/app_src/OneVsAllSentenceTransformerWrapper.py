import os
import pandas as pd
import numpy as np
import ast
import random
import tensorflow as tf
from transformers import AutoTokenizer
from math import ceil
import keras as K
from tqdm import tqdm


# local application/library specific imports
from app_src.SentenceTransformerEncoderModel import SentenceTransformerEncoderModel
from app_src.CustomMetrics import PrintScoresCallback, PrintValidationScoresCallback
from app_config import AppConfig
from app_src.common import set_random_seed
from transformers import AutoTokenizer, TFAutoModel

from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, average_precision_score, f1_score
from app_src.CustomMetrics import subset_precision, subset_recall, subset_f1, label_wise_macro_accuracy, label_wise_accuracy, label_wise_f1_score

# define configuration proxy
configProxy = AppConfig()
CONFIG = configProxy.return_config()

# get global constants configuration
GLOBAL_CONSTANTS = configProxy.return_global_constants()
RANDOM_STATE = GLOBAL_CONSTANTS['RANDOM_SEED']

random.seed(RANDOM_STATE)

class OneVsAllSentenceTransformerWrapper():
    def __init__(self, model_name, number_of_tags):
        
        self.model_name = model_name
        self.number_of_tags = number_of_tags

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.test_dataset = None
        self.train_dataset = None
        self.validation_dataset = None
        
        self.models = []

        set_random_seed(RANDOM_STATE)
        
    def __read_test_data(self, test_dataset_path):
        self.test_dataset = pd.read_csv(test_dataset_path)
        
    def __read_train_data(self, train_dataset_path):
        self.train_dataset = pd.read_csv(train_dataset_path)
    
    def __read_validation_data(self, val_dataset_path):
        self.validation_dataset = pd.read_csv(val_dataset_path)
    
    def __encode_tags(self, tags):
        for idx, string_tag_list in enumerate(tags):
            tags[idx] = ast.literal_eval(string_tag_list)
        return np.array(tags)
    
    def __tokenize_data(self, data):
        tokens = self.tokenizer(
            data,
            add_special_tokens=True,
            truncation=True,
            max_length=512,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='tf'
        )
        return tokens['input_ids'], tokens['attention_mask']
    
    def __build_tf_dataset(self, data_df, batch_size = 32, shuffle_buffer_size=10000, testing=False):
        problem_statements = data_df['problem_statement'].tolist()
        tags = self.__encode_tags(data_df['problem_tags'].tolist())
        # Ensure tags are in a consistent format (e.g., a NumPy array)
        if not isinstance(tags, (np.ndarray, tf.Tensor)):
            tags = np.array(tags)
            
        # Tokenize all problem statements
        input_ids, attention_mask = self.__tokenize_data(problem_statements)

        # print(input_ids.shape)
        # print(attention_mask.shape)
        # print(tags.shape)        
        
        tf_dataset = tf.data.Dataset.from_tensor_slices((
            {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            },
            tags
        ))
        
        # Apply optimizations: shuffle, cache, batch, prefetch
        if not testing:
            tf_dataset = tf_dataset.shuffle(buffer_size=shuffle_buffer_size)
        tf_dataset = tf_dataset.cache()  # Use caching if your dataset fits in memory; otherwise, consider file-based caching.
        if not testing:
            tf_dataset = tf_dataset.batch(batch_size, drop_remainder=True)
        else:
            tf_dataset = tf_dataset.batch(batch_size, drop_remainder=False)
        tf_dataset = tf_dataset.prefetch(tf.data.AUTOTUNE)
        
        # Print the shapes of the batches to verify
        # for batch in dataset.take(1):
        #     input_batch, tag_batch = batch
        #     print(f"Input batch shape: {input_batch['input_ids'].shape}")
        #     print(f"Attention mask batch shape: {input_batch['attention_mask'].shape}")
        #     print(f"Tags batch shape: {tag_batch.shape}")
        
        return tf_dataset
    
    def train_model(self, train_dataset_path, val_dataset_path, epochs=5, batch_size=32, train_model=True, threshold=0.5):
        
        self.__read_train_data(train_dataset_path)
        self.__read_validation_data(val_dataset_path)
        
        self.train_dataset = self.__build_tf_dataset(self.train_dataset, batch_size)
        self.validation_dataset = self.__build_tf_dataset(self.validation_dataset, batch_size)
        
        for label_idx in range(self.number_of_tags):
            
            print('Starting training for label:', label_idx)
            
            # single-label training dataset
            single_label_train_ds = self.train_dataset.map(
                lambda x, y: (x, y[:, label_idx])
            )
            # single-label validation dataset
            single_label_val_ds = self.validation_dataset.map(
                lambda x, y: (x, y[:, label_idx])
            )
            
            transformer_model = TFAutoModel.from_pretrained(self.model_name)
            encoder_model = SentenceTransformerEncoderModel(transformer_model, 1)
            
            # Unfreeze the transformer layers
            encoder_model.unfreeze_transformer()

            # Compile the model
            encoder_model.compile_model(run_eagerly=False, threshold=threshold)
            
            # Define callbacks
            callbacks = [
                # Custom callback for printing validation scores
                PrintValidationScoresCallback()
            ]
            
            # Start training
            history = encoder_model.fit(
                single_label_train_ds,
                validation_data=single_label_val_ds,
                epochs=epochs,
                callbacks=callbacks
            )
            
            self.models.append(encoder_model)
    
    def benchmark_model(self, test_dataset_path, batch_size=32, model_path=None, transformer_model_path=None, threshold=0.5):
        
        self.__read_test_data(test_dataset_path)
        test_tags_np = self.__encode_tags(self.test_dataset['problem_tags'].tolist())
        # Ensure tags are in a consistent format (e.g., a NumPy array)
        if not isinstance(test_tags_np, (np.ndarray, tf.Tensor)):
            test_tags_np = np.array(test_tags_np)
        
        # Convert test data to tf.data.Dataset
        self.test_dataset = self.__build_tf_dataset(self.test_dataset, batch_size, testing=True)
        
        # We'll create an array to hold predictions for all 5 labels
        all_preds = []
        
        for label_idx, estimator in enumerate(self.models):
            
            # single-label testing dataset
            single_label_test_ds = self.test_dataset.map(
                lambda x, y: (x, y[:, label_idx])
            )
            
            estimator.freeze_transformer()
            
            pred_probs = estimator.predict(single_label_test_ds)
            all_preds.append(pred_probs)
        
        all_preds = np.column_stack(all_preds)

        predictions = (all_preds > threshold).astype(int)
        
        print(predictions)
        
        print(test_tags_np)
        
        print(f"test_tags_np shape: {test_tags_np.shape}")
        print(f"predictions shape: {predictions.shape}")    
        
        # Label Wise Metrics
        f1_scores = label_wise_f1_score(test_tags_np, predictions)
        f1_scores = [float(t.numpy()) for t in f1_scores]
        accuracies = label_wise_accuracy(test_tags_np, predictions)
        accuracies = [float(t.numpy()) for t in accuracies]
        accuracy = label_wise_macro_accuracy(test_tags_np, predictions).numpy()
        precision = precision_score(test_tags_np, predictions, average='macro')
        recall = recall_score(test_tags_np, predictions, average='macro')
        f1 = f1_score(test_tags_np, predictions, average='macro')
        
        # Subset Metrics
        sub_accuracy = accuracy_score(test_tags_np, predictions)
        sub_precision = subset_precision(test_tags_np, predictions).numpy()
        sub_recall = subset_recall(test_tags_np, predictions).numpy()
        sub_f1 = subset_f1(test_tags_np, predictions).numpy()
        
        # Area Metrics
        auc = roc_auc_score(test_tags_np, predictions, average='macro', multi_class='ovr')
        prc_auc = average_precision_score(test_tags_np, predictions, average='macro')
        
        # Store the results
        results = {
            # Label Wise Metrics
            'Label F1 Scores': f1_scores,
            'Label Accuracies': accuracies,
            # Macro Label Metrics
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            # Subset Metrics
            'Subset Accuracy': sub_accuracy,
            'Subset Precision': sub_precision,
            'Subset Recall': sub_recall,
            'Subset F1': sub_f1,
            # Area Metrics
            'AUC': auc,
            'PRC AUC': prc_auc
        }

        return results
    