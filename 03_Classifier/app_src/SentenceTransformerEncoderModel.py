import tensorflow as tf

import keras as K
from keras.layers import Dense
from keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from keras.losses import BinaryCrossentropy
from keras.optimizers import AdamW
from transformers import AutoTokenizer, TFAutoModel

import random

# local application/library specific imports
from app_config import AppConfig
from app_src.CustomMetrics import subset_accuracy, subset_precision, subset_recall, subset_f1, label_wise_macro_accuracy, label_wise_macro_f1, LabelWiseF1Score, LabelWiseAccuracy

# define configuration proxy
configProxy = AppConfig()
CONFIG = configProxy.return_config()

# get global constants configuration
GLOBAL_CONSTANTS = configProxy.return_global_constants()
RANDOM_STATE = GLOBAL_CONSTANTS['RANDOM_SEED']

random.seed(RANDOM_STATE)

class SentenceTransformerEncoderModel(K.Model):
    def __init__(self, model_name, num_classes=5, **kwargs):
        super().__init__(**kwargs)
        self.model_name = model_name
        self.num_classes = num_classes
        self.problem_statement_tokenizer = AutoTokenizer.from_pretrained(model_name)        
        self.problem_statement_model = TFAutoModel.from_pretrained(model_name)
        self.classifier = Dense(num_classes, activation='sigmoid')

    def get_config(self):
        config = super().get_config()
        config.update({
            'model_name': self.model_name,
            'num_classes': self.num_classes,
        })
        return config

    def call(self, inputs, **kwargs):
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        
        # Print shapes for debugging
        # print(f"Input IDs shape: {input_ids.shape}")
        # print(f"Attention mask shape: {attention_mask.shape}")
        
        # Sentence transformer outputs embeddings
        transformer_output = self.problem_statement_model(input_ids, attention_mask=attention_mask)[0]
        # print(f"Transformer embeddings shape: {transformer_output.shape}")
        
        # Use [CLS] token for classification
        sentence_embeddings = transformer_output[:, 0, :]  

        # Add dense layers for classification
        output = self.classifier(sentence_embeddings)
        
        return output
    
    def compile_model(self, run_eagerly=False, learning_rate=2e-5, weight_decay=1e-4):
        # Define the optimizer, loss function and metrics
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=weight_decay)
        loss = BinaryCrossentropy()
        metrics = [
            # Label Wise Metrics
            LabelWiseF1Score(name='label_wise_f1_score'),
            LabelWiseAccuracy(name='label_wise_accuracy'),
            # Macro Label Metrics
            BinaryAccuracy(name='binary_accuracy'), 
            Precision(name='precision'), 
            Recall(name='recall'), 
            label_wise_macro_f1,
            # Subset Metrics
            subset_accuracy,
            subset_precision,
            subset_recall,
            subset_f1,
            # Area Metrics
            AUC(name='auc'),
            AUC(name='prc_auc', curve='PR')
            ]
        
        # Compile the model
        self.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=run_eagerly)
