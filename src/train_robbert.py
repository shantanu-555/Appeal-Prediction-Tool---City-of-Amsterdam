import torch
import pandas as pd
import numpy as np
from transformers import (
    RobertaTokenizer, 
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    DataCollatorWithPadding
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from datasets import Dataset
import torch.nn.functional as F
import logging
import nltk
from nltk.corpus import stopwords
import re

logging.basicConfig(level=logging.INFO)
nltk.download('stopwords')

class CustomTrainer(Trainer):
    def __init__(self, class_weights=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights.to(self.args.device) if class_weights is not None else None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Weighted cross entropy loss
        if self.class_weights is not None:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            loss_fct = torch.nn.CrossEntropyLoss()
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        # Add confidence penalty
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * torch.log(probs + 1e-6)).sum(dim=-1).mean()
        confidence_penalty = 0.1 * (1.0 - entropy)
        
        total_loss = loss + confidence_penalty
        
        return (total_loss, outputs) if return_outputs else total_loss

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Get prediction probabilities
    probs = F.softmax(torch.tensor(pred.predictions), dim=-1).numpy()
    confidence = probs.max(axis=-1).mean()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    acc = accuracy_score(labels, preds)
    
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'confidence': confidence
    }

def clean_dutch_text(text):
    if text is None:
        return ""
    
    text = str(text)
    # Remove special characters and digits
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.lower().strip()
    
    # Remove Dutch stopwords
    stop_words = set(stopwords.words('dutch'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    
    return ' '.join(words)

def main():
    # Load data
    df = pd.read_csv('merged_data.csv', index_col=0)
    df['cleaned_text'] = df['geanonimiseerd_doc_inhoud'].apply(clean_dutch_text)
    
    # Create train/test split
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['target'])
    
    # Prepare datasets with proper column names
    train_df = train_df.rename(columns={'cleaned_text': 'text', 'target': 'labels'})
    test_df = test_df.rename(columns={'cleaned_text': 'text', 'target': 'labels'})
    
    # Convert to HuggingFace datasets
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained("pdelobelle/robbert-v2-dutch-base")
    
    # Modified tokenize function to include labels
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=256
        )
        # Ensure labels are included in the dataset
        tokenized['labels'] = examples['labels']
        return tokenized
    
    # Tokenize datasets
    tokenized_train = train_dataset.map(tokenize_function, batched=True)
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    
    # Set the format for PyTorch
    tokenized_train.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Calculate class weights
    labels = train_df['labels'].values
    class_counts = np.bincount(labels)
    class_weights = torch.FloatTensor(1.0 / class_counts)
    class_weights = class_weights / class_weights.sum()
    
    # Initialize model
    model = RobertaForSequenceClassification.from_pretrained(
        "pdelobelle/robbert-v2-dutch-base",
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./robbert_results",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=10,
        weight_decay=0.01,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=3,
        warmup_ratio=0.1,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=2,
    )
    
    # Initialize trainer
    trainer = CustomTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    
    # Train the model
    trainer.train()
    
    # Evaluate
    eval_results = trainer.evaluate()
    print("\nEvaluation Results:", eval_results)
    
    # Save model and tokenizer
    model.save_pretrained("./robbert_model")
    tokenizer.save_pretrained("./robbert_tokenizer")
    print("\nModel and tokenizer saved successfully!")

if __name__ == "__main__":
    main() 