import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn.functional as F
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer
)
from datasets import Dataset
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_curve,
    auc
)
from sklearn.model_selection import train_test_split
import re
from nltk.corpus import stopwords
import nltk
import logging

logging.basicConfig(level=logging.INFO)
nltk.download('stopwords')

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

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    
    # Get prediction probabilities
    probs = F.softmax(torch.tensor(pred.predictions), dim=-1).numpy()
    confidence = probs.max(axis=-1).mean()
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average='weighted'
    )
    
    return {
        'accuracy': (preds == labels).mean(),
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'avg_confidence': confidence
    }

def main():
    # Load the saved model and tokenizer
    model_path = "../models/robbert_model"
    tokenizer_path = "../models/robbert_tokenizer"
    
    logging.info("Loading model and tokenizer...")
    model = RobertaForSequenceClassification.from_pretrained(model_path)
    tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
    
    # Load and preprocess test data
    logging.info("Loading and preprocessing test data...")
    df = pd.read_csv('../data/merged_data.csv', index_col=0)
    df['text'] = df['geanonimiseerd_doc_inhoud'].apply(clean_dutch_text)
    
    # Split data
    _, test_df = train_test_split(
        df, 
        test_size=0.2, 
        stratify=df['target'], 
        random_state=55
    )
    
    # Prepare test dataset
    test_df = test_df.rename(columns={'text': 'text', 'target': 'labels'})
    test_dataset = Dataset.from_pandas(test_df)
    
    # Tokenize test dataset
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples['text'],
            padding='max_length',
            truncation=True,
            max_length=256
        )
        tokenized['labels'] = examples['labels']
        return tokenized
    
    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    tokenized_test.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    # Initialize trainer for evaluation
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics
    )
    
    # Evaluate model
    logging.info("Evaluating model...")
    results = trainer.predict(tokenized_test)
    
    # Get predictions and probabilities
    predictions = results.predictions
    predicted_classes = np.argmax(predictions, axis=1)
    probabilities = F.softmax(torch.tensor(predictions), dim=-1).numpy()
    
    # Print evaluation metrics
    logging.info("\nEvaluation Metrics:")
    for metric, value in results.metrics.items():
        logging.info(f"{metric}: {value:.4f}")
    
    # Print detailed classification report
    logging.info("\nClassification Report:")
    print(classification_report(
        results.label_ids,
        predicted_classes,
        target_names=['Class 0', 'Class 1']
    ))
    
    # Generate confusion matrices (raw and normalized)
    cm = confusion_matrix(results.label_ids, predicted_classes)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Raw counts matrix
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['Class 0', 'Class 1'],
        yticklabels=['Class 0', 'Class 1'],
        ax=ax1
    )
    ax1.set_xlabel('Predicted Class')
    ax1.set_ylabel('True Class')
    ax1.set_title('RobBERT Confusion Matrix (Raw Counts)')
    
    # Normalized matrix
    sns.heatmap(
        cm_normalized, annot=True, fmt='.2%', cmap='Blues',
        xticklabels=['Class 0', 'Class 1'],
        yticklabels=['Class 0', 'Class 1'],
        ax=ax2
    )
    ax2.set_xlabel('Predicted Class')
    ax2.set_ylabel('True Class')
    ax2.set_title('RobBERT Confusion Matrix (Normalized)')
    
    plt.tight_layout()
    plt.savefig('../eval_results/robbert_confusion_matrix.png')
    plt.close()
    
    # Generate ROC curve
    fpr, tpr, _ = roc_curve(results.label_ids, probabilities[:, 1])
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(
        fpr, tpr, color='darkorange', lw=2,
        label=f'ROC curve (area = {roc_auc:.2f})'
    )
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('RobBERT ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig('../eval_results/robbert_roc_curve.png')
    plt.close()
    
    # Save detailed metrics to file
    metrics_df = pd.DataFrame({
        'Metric': ['Accuracy', 'F1 Score', 'Precision', 'Recall', 'ROC AUC', 'Avg Confidence'],
        'Value': [
            results.metrics['test_accuracy'],
            results.metrics['test_f1'],
            results.metrics['test_precision'],
            results.metrics['test_recall'],
            roc_auc,
            results.metrics['test_avg_confidence']
        ]
    })
    metrics_df.to_csv('../eval_results/robbert_evaluation_metrics.csv', index=False)
    logging.info("\nEvaluation completed. Results saved to files.")

if __name__ == "__main__":
    main() 