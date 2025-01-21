import streamlit as st
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import torch.nn.functional as F
import plotly.graph_objects as go
from typing import Dict, List, Tuple
import re
import pandas as pd
import base64
from fpdf import FPDF
import PyPDF2
from io import BytesIO
import lime
import lime.lime_text
import matplotlib.pyplot as plt
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib
import os
import datetime

class RobBERTClassificationPipeline:
    def __init__(self, model_path: str, tokenizer_path: str):
        """Initialize the pipeline with trained RobBERT model and tokenizer."""
        try:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Update model and tokenizer paths to use os.path.join
            model_path = os.path.join('models', 'robbert_model')
            tokenizer_path = os.path.join('models', 'robbert_tokenizer')
            
            # Check if model and tokenizer files exist
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model not found at {model_path}")
            if not os.path.exists(tokenizer_path):
                raise FileNotFoundError(f"Tokenizer not found at {tokenizer_path}")
            
            self.tokenizer = RobertaTokenizer.from_pretrained(tokenizer_path)
            self.model = RobertaForSequenceClassification.from_pretrained(model_path)
            self.model.to(self.device)
            self.model.eval()
            
            # Initialize sentence transformer
            try:
                self.sentence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
            except Exception as e:
                st.warning("Could not initialize similarity search. Some features may be unavailable.")
                self.sentence_model = None
            
            # Update embeddings and training texts paths
            self.embeddings_file = os.path.join('models', 'training_embeddings.joblib')
            self.training_texts_file = os.path.join('models', 'training_texts.joblib')
            
            if os.path.exists(self.embeddings_file) and os.path.exists(self.training_texts_file):
                try:
                    self.training_embeddings = joblib.load(self.embeddings_file)
                    self.training_texts = joblib.load(self.training_texts_file)
                except Exception as e:
                    st.warning("Could not load similarity data. Similar cases feature will be disabled.")
                    self.training_embeddings = None
                    self.training_texts = None
            else:
                try:
                    self._create_training_embeddings()
                except Exception as e:
                    st.warning("Could not create similarity data. Similar cases feature will be disabled.")
                    self.training_embeddings = None
                    self.training_texts = None
                    
        except Exception as e:
            raise Exception(f"Error initializing pipeline: {str(e)}")

    def _create_training_embeddings(self):
        """Create and save embeddings for training data."""
        # Load training data
        df = pd.read_csv('merged_data.csv', index_col=0)
        
        # Clean texts
        self.training_texts = []
        for idx, row in df.iterrows():
            text = self.clean_dutch_text(row['geanonimiseerd_doc_inhoud'])
            label = row['target']
            self.training_texts.append({
                'text': text,
                'label': label
            })
        
        # Create embeddings
        texts_only = [item['text'] for item in self.training_texts]
        self.training_embeddings = self.sentence_model.encode(texts_only)
        
        # Save embeddings and texts
        joblib.dump(self.training_embeddings, self.embeddings_file)
        joblib.dump(self.training_texts, self.training_texts_file)

    def clean_dutch_text(self, text, stopword_list=None):
        if stopword_list is None:
            # Load Dutch stopwords from NLTK
            try:
                from nltk.corpus import stopwords
                stopword_list = set(stopwords.words('dutch'))
            except LookupError:
                import nltk
                nltk.download('stopwords')
                stopword_list = set(stopwords.words('dutch'))

        # Replace placeholders with a generic indicator
        text = re.sub(r'\[\w+\]', '[placeholder]', text)
        
        # Remove line breaks and tabs
        text = re.sub(r'\n+', ' ', text)
        text = re.sub(r'\t+', ' ', text)
        
        # Normalize multiple spaces into a single space
        text = re.sub(r'\s+', ' ', text)
        
        # Convert text to lowercase
        text = text.lower()
        
        # Remove stopwords
        words = text.split()
        words = [word for word in words if word not in stopword_list]
        
        # Rejoin words into a cleaned string
        cleaned_text = ' '.join(words)
        
        return cleaned_text

    def get_prediction(self, text: str) -> Dict:
        """Get prediction, probability, and important words for input text."""
        # Clean the text
        cleaned_text = self.clean_dutch_text(text)
        
        # Tokenize - updated for RobBERT
        inputs = self.tokenizer(
            cleaned_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=256,  # RobBERT typically uses 256
            return_attention_mask=True,
        ).to(self.device)

        # Get model outputs and attention weights
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
            
        # Get prediction and probability
        logits = outputs.logits
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        prediction_probability = probabilities[0][predicted_class].item()

        # Get attention weights and important words
        attention_weights = outputs.attentions[-1]
        attention_weights = attention_weights.mean(dim=(0, 1))
        
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        word_importances = [(token, weight.item()) for token, weight in zip(tokens, attention_weights[0])]
        word_importances = sorted(word_importances, key=lambda x: x[1], reverse=True)
        
        # Filter out special tokens specific to RobBERT
        important_words = [
            word for word, _ in word_importances 
            if not word.startswith('<') and not word.startswith('Ġ')  # RobBERT specific tokens
        ][:5]

        return {
            'prediction': predicted_class,
            'probability': prediction_probability,
            'important_words': important_words
        }

    def predict_proba(self, texts):
        """Get probability predictions for LIME/SHAP compatibility"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Clean and tokenize texts
        cleaned_texts = [self.clean_dutch_text(text) for text in texts]
        inputs = self.tokenizer(
            cleaned_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        ).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)
            
        return probabilities.cpu().numpy()

    def predict_proba_for_shap(self, texts: List[str]) -> np.ndarray:
        """Modified prediction function for SHAP compatibility"""
        if isinstance(texts, str):
            texts = [texts]
        
        # Process all texts
        outputs = []
        for text in texts:
            cleaned_text = self.clean_dutch_text(text)
            inputs = self.tokenizer(
                cleaned_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                output = self.model(**inputs)
                probs = torch.nn.functional.softmax(output.logits, dim=1)
                outputs.append(probs.cpu().numpy())
        
        return np.vstack(outputs)

    def find_similar_cases(self, text: str, n_cases: int = 2) -> List[Dict]:
        """Find the most similar cases from the training data."""
        if self.sentence_model is None or self.training_embeddings is None:
            return []
            
        try:
            # Get embedding for input text
            text_embedding = self.sentence_model.encode([text])
            
            # Calculate similarities
            similarities = cosine_similarity(text_embedding, self.training_embeddings)[0]
            
            # Get top N similar cases
            top_indices = similarities.argsort()[-n_cases:][::-1]
            
            similar_cases = []
            for idx in top_indices:
                similar_cases.append({
                    'text': self.training_texts[idx]['text'],
                    'label': self.training_texts[idx]['label'],
                    'similarity': similarities[idx]
                })
            
            return similar_cases
        except Exception as e:
            st.warning("Error finding similar cases. Feature temporarily unavailable.")
            return []

def create_confidence_gauge(probability: float) -> go.Figure:
    """Create a gauge chart showing prediction confidence."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "lightgray"},
                {'range': [33, 66], 'color': "gray"},
                {'range': [66, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    fig.update_layout(height=250)
    return fig

def highlight_important_words(text: str, important_words: list) -> str:
    """Highlight important words in the text using HTML."""
    # Escape any existing HTML characters first
    text = text.replace('<', '&lt;').replace('>', '&gt;')
    
    # Create a copy of text for highlighting
    highlighted_text = text
    
    # Sort words by length (longest first) to avoid partial word matches
    sorted_words = sorted(important_words, key=len, reverse=True)
    
    for word in sorted_words:
        if not word.strip():  # Skip empty words
            continue
            
        # Clean the word of any special characters
        word = re.escape(word.strip())
        
        # Case-insensitive word boundary match
        pattern = re.compile(r'\b(' + word + r')\b', re.IGNORECASE)
        highlighted_text = pattern.sub(
            r'<span style="background-color: yellow; font-weight: bold">\1</span>',
            highlighted_text
        )
    
    # Wrap in a div with word-wrap
    highlighted_text = f'<div style="word-wrap: break-word;">{highlighted_text}</div>'
    
    return highlighted_text

def export_to_pdf(results: List[Dict], texts: List[str], lime_explanations=None, similar_cases=None) -> bytes:
    """Create a comprehensive PDF report of the analysis results."""
    pdf = FPDF()
    # Add unicode font support
    pdf.add_page()
    
    # Title and Header
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Appeal Prediction Analysis Report", ln=True, align="C")
    pdf.set_font("Arial", "", 11)
    pdf.cell(0, 10, "City of Amsterdam - Appeal Prediction Tool", ln=True, align="C")
    pdf.ln(10)

    for idx, (text, result) in enumerate(zip(texts, results), 1):
        # Main Analysis Section
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 10, "Analysis Results", ln=True)
        pdf.ln(5)
        
        # Input Text
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Input Text:", ln=True)
        pdf.set_font("Arial", "", 10)
        # Clean text of any problematic characters
        cleaned_text = text.encode('latin-1', errors='replace').decode('latin-1')
        pdf.multi_cell(0, 5, cleaned_text)
        pdf.ln(5)
        
        # Prediction Results
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Prediction Results:", ln=True)
        pdf.set_font("Arial", "", 10)
        prediction_label = "Positive (Likely to Appeal)" if result['prediction'] == 1 else "Negative (Unlikely to Appeal)"
        pdf.cell(0, 5, f"Classification: {prediction_label}", ln=True)
        pdf.cell(0, 5, f"Confidence: {result['probability']:.2%}", ln=True)
        pdf.ln(5)

        # LIME Analysis
        if lime_explanations and len(lime_explanations) > idx-1:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Detailed Word Impact Analysis:", ln=True)
            pdf.set_font("Arial", "", 10)
            lime_exp = lime_explanations[idx-1]
            
            # Add explanation of colors
            pdf.multi_cell(0, 5, "The following words had significant impact on the prediction. Positive values (contributing to appeal) are marked with (+), negative values (contributing to no appeal) are marked with (-)")
            pdf.ln(5)
            
            for word, weight in zip(lime_exp['words'], lime_exp['weights']):
                sign = '+' if weight > 0 else '-'
                # Clean word of any problematic characters
                cleaned_word = word.encode('latin-1', errors='replace').decode('latin-1')
                pdf.cell(0, 5, f"{sign} {cleaned_word}: {abs(weight):.3f}", ln=True)
            pdf.ln(5)

        # Similar Cases
        if similar_cases and len(similar_cases) > idx-1:
            pdf.set_font("Arial", "B", 12)
            pdf.cell(0, 10, "Similar Cases Analysis:", ln=True)
            pdf.set_font("Arial", "", 10)
            
            for i, case in enumerate(similar_cases[idx-1], 1):
                pdf.add_page()  # Add new page for each similar case
                pdf.set_font("Arial", "B", 12)
                appeal_status = "Appealed" if case['label'] == 1 else "No Appeal"
                pdf.cell(0, 10, f"Similar Case {i}", ln=True)
                pdf.set_font("Arial", "", 10)
                pdf.cell(0, 5, f"Status: {appeal_status}", ln=True)
                pdf.cell(0, 5, f"Similarity Score: {case['similarity']:.2%}", ln=True)
                pdf.ln(5)
                
                pdf.set_font("Arial", "B", 10)
                pdf.cell(0, 5, "Full Text:", ln=True)
                pdf.set_font("Arial", "", 10)
                # Clean case text of any problematic characters
                cleaned_case_text = case['text'].encode('latin-1', errors='replace').decode('latin-1')
                pdf.multi_cell(0, 5, cleaned_case_text)
                pdf.ln(5)

    # Footer
    pdf.set_y(-30)
    pdf.set_font("Arial", "I", 8)
    pdf.cell(0, 10, "Generated by Appeal Prediction Tool - City of Amsterdam", ln=True, align="C")
    pdf.cell(0, 10, f"Report Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align="C")

    return pdf.output(dest='S').encode('latin1')

def explain_prediction_lime(text: str, pipeline: RobBERTClassificationPipeline) -> dict:
    """Generate LIME explanation for the prediction"""
    # First clean the text using the same pipeline cleaning method
    cleaned_text = pipeline.clean_dutch_text(text)
    
    explainer = lime.lime_text.LimeTextExplainer(
        class_names=['Negative', 'Positive'],
        split_expression=lambda x: x.split(),
        bow=False
    )
    
    exp = explainer.explain_instance(
        cleaned_text,  # Use cleaned text instead of raw
        pipeline.predict_proba,
        num_features=10,
        num_samples=100
    )
    
    # Extract explanation data
    explanation_data = {
        'words': [],
        'weights': [],
        'top_words': []
    }
    
    for word, weight in exp.as_list():
        if abs(weight) > 0.01:  # Only include significant impacts
            explanation_data['words'].append(word)
            explanation_data['weights'].append(weight)
            explanation_data['top_words'].append((word, abs(weight)))
    
    explanation_data['top_words'].sort(key=lambda x: x[1], reverse=True)
    explanation_data['top_words'] = [word for word, _ in explanation_data['top_words'][:5]]
    
    return explanation_data

def plot_word_importance(words: list, scores: list, title: str) -> plt.Figure:
    """Create a horizontal bar plot for word importance"""
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(words))
    
    # Create bars
    bars = ax.barh(y_pos, scores)
    
    # Color bars based on positive/negative values
    for i, bar in enumerate(bars):
        bar.set_color('lightcoral' if scores[i] > 0 else 'lightblue')
    
    # Customize plot
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words)
    ax.invert_yaxis()
    ax.set_xlabel('Impact on Prediction')
    ax.set_title(title)
    
    plt.tight_layout()
    return fig

def display_similar_cases(similar_cases: List[Dict]):
    """Display similar cases in a more streamlined way."""
    st.markdown("### 📚 Similar Cases from Training Data")
    st.markdown("""
    The following cases from the training data are most similar to the input text:
    """)
    
    for i, case in enumerate(similar_cases, 1):
        with st.expander(f"Similar Case {i} - {'Appealed' if case['label'] == 1 else 'No Appeal'} (Similarity: {case['similarity']:.2%})"):
            st.markdown("**Text Content:**")
            st.markdown(f"```\n{case['text'][:500]}{'...' if len(case['text']) > 500 else ''}\n```")

def create_triple_gauge_chart(model_prob: float, similar_prob: float, composite_prob: float, explanation: str) -> go.Figure:
    """Create a figure with three gauge charts - composite on top, model and similarity below."""
    fig = go.Figure()
    
    # Composite Confidence Gauge (larger, on top)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=composite_prob * 100,
        domain={'x': [0.2, 0.8], 'y': [0.52, 0.95]},  # Adjusted y domain to prevent cutoff
        title={
            'text': "Composite Confidence",
            'font': {'size': 18}
        },
        number={'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkgreen"},
            'steps': [
                {'range': [0, 33], 'color': "lightgray"},
                {'range': [33, 66], 'color': "gray"},
                {'range': [66, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    # Model Confidence Gauge (bottom left)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=model_prob * 100,
        domain={'x': [0, 0.45], 'y': [0.1, 0.4]},
        title={'text': "Model Confidence", 'font': {'size': 14}},
        number={'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 33], 'color': "lightgray"},
                {'range': [33, 66], 'color': "gray"},
                {'range': [66, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    # Similarity Confidence Gauge (bottom right)
    fig.add_trace(go.Indicator(
        mode="gauge+number",
        value=similar_prob * 100,
        domain={'x': [0.55, 1], 'y': [0.1, 0.4]},
        title={'text': "Similarity Confidence", 'font': {'size': 14}},
        number={'font': {'size': 16}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkorange"},
            'steps': [
                {'range': [0, 33], 'color': "lightgray"},
                {'range': [33, 66], 'color': "gray"},
                {'range': [66, 100], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=500,
        margin=dict(t=70, b=150),  # Increased bottom margin
        annotations=[
            dict(
                x=0.5,
                y=-0.15,  # Moved text further down
                showarrow=False,
                text=explanation,
                xref="paper",
                yref="paper",
                font=dict(size=15, color='rgba(0,0,0,0.9)'),
                align='center',
                bgcolor='rgba(240,240,240,0.9)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1,
                borderpad=10
            )
        ]
    )
    return fig

def calculate_composite_confidence(prediction: int, model_probability: float, similar_cases: List[Dict]) -> Tuple[float, float, float, str, int]:
    """Calculate all confidence scores and return model, similarity, composite confidences, explanation and final prediction."""
    if not similar_cases:
        composite_conf = model_probability
        # Only return the model's prediction if confidence > 0.5
        final_pred = prediction if composite_conf > 0.5 else None
        return model_probability, 0.0, composite_conf, "Based on model prediction only", final_pred
        
    # Calculate similarity-weighted vote from similar cases
    total_similarity = sum(case['similarity'] for case in similar_cases)
    if total_similarity == 0:
        composite_conf = model_probability
        final_pred = prediction if composite_conf > 0.5 else None
        return model_probability, 0.0, composite_conf, "Based on model prediction only", final_pred
        
    similar_vote = sum(
        case['similarity'] * (1 if case['label'] == prediction else 0)
        for case in similar_cases
    ) / total_similarity
    
    # Use optimized weights
    model_weight = 0.6
    similar_weight = 1 - model_weight
    
    # Calculate composite confidence
    composite_conf = (model_weight * model_probability) + (similar_weight * similar_vote)
    
    # Generate explanation
    agreement = sum(1 for case in similar_cases if case['label'] == prediction)
    explanation = f"Based on model prediction ({model_probability:.1%}, weight: {model_weight:.1%}) and "
    explanation += f"{agreement}/{len(similar_cases)} similar cases agreeing (weight: {similar_weight:.1%})"
    
    # Only return a prediction if composite confidence is high enough
    final_pred = prediction if composite_conf > 0.5 else None
    
    return model_probability, similar_vote, composite_conf, explanation, final_pred

def main():
    st.set_page_config(
        page_title="Appeal Prediction Tool - City of Amsterdam",
        page_icon="🏛️",
        layout="wide"
    )
    
    # Add URL helper message
    st.markdown("""
    <div style='background-color: #f0f2f6; padding: 10px; border-radius: 5px; margin-bottom: 20px'>
    📌 If you see '0.0.0.0:8501' in your browser, please use <a href='http://localhost:8501'>http://localhost:8501</a> instead
    </div>
    """, unsafe_allow_html=True)
    
    st.title("🏛️ Appeal Prediction Analysis - City of Amsterdam")
    st.markdown("""
    This application uses RobBERT, a state-of-the-art Dutch language model, to predict whether a citizen is likely 
    to appeal against a decision on their objection in court. The analysis is based on the text content of their 
    initial objection letter.

    **Prediction meaning:**
    - **Positive**: High likelihood that the citizen will appeal in court
    - **Negative**: Low likelihood that the citizen will appeal in court

    The tool also identifies key words and phrases that influenced this prediction, helping to understand 
    the factors that might lead to an appeal.
    """)

    # Initialize the pipeline with RobBERT
    @st.cache_resource
    def load_model():
        return RobBERTClassificationPipeline(
            model_path=os.path.join('models', 'robbert_model'),
            tokenizer_path=os.path.join('models', 'robbert_tokenizer')
        )

    try:
        pipeline = load_model()

        # Modify tabs to remove batch processing
        tab1, tab2 = st.tabs(["Single Text", "PDF Upload"])

        with tab1:
            # Single text processing
            col1, col2 = st.columns([4, 1])
            with col1:
                text_input = st.text_area(
                    "Enter the objection letter text:",
                    height=150,
                    placeholder="Paste the content of the objection letter here..."
                )
            with col2:
                if st.button("Clear Results"):
                    st.session_state['text_input'] = ""
                    st.rerun()

            if st.button("Analyze Text", key="single_analyze"):
                if text_input.strip():
                    with st.spinner("Analyzing text..."):
                        try:
                            result = pipeline.get_prediction(text_input)
                            lime_explanation = explain_prediction_lime(text_input, pipeline)
                            similar_cases = pipeline.find_similar_cases(text_input)
                            
                            # Calculate composite confidence
                            model_conf, similar_conf, composite_conf, explanation, final_prediction = calculate_composite_confidence(
                                result['prediction'],
                                result['probability'],
                                similar_cases
                            )
                            
                            # Main results section
                            col1, col2 = st.columns([0.8, 0.2])
                            
                            with col1:
                                st.subheader("Classification Results")
                                if final_prediction is not None:
                                    prediction_label = "Positive (Likely to Appeal)" if final_prediction == 1 else "Negative (Unlikely to Appeal)"
                                    confidence_color = "red" if final_prediction == 1 else "green"
                                    st.markdown(f"**Prediction:** <span style='color:{confidence_color}'>{prediction_label}</span>", unsafe_allow_html=True)
                                else:
                                    st.markdown("**Prediction:** <span style='color:orange'>Insufficient confidence to make a prediction</span>", unsafe_allow_html=True)
                                
                                # Create and display triple gauge chart
                                fig = create_triple_gauge_chart(
                                    model_conf,
                                    similar_conf,
                                    composite_conf,
                                    explanation
                                )
                                st.plotly_chart(fig, use_container_width=True)

                            with col2:
                                st.markdown("##### Influential Words")  # Smaller header
                                if lime_explanation['top_words']:
                                    for word in lime_explanation['top_words']:
                                        st.markdown(f"• {word}")  # Simplified bullet points
                                else:
                                    st.info("No significant words found.")
                            
                            # Text Analysis with LIME visualization
                            st.subheader("Detailed Word Impact Analysis")
                            if lime_explanation['words']:
                                fig_lime = plot_word_importance(
                                    lime_explanation['words'],
                                    lime_explanation['weights'],
                                    'Word Impact on Prediction'
                                )
                                st.pyplot(fig_lime)
                                
                                st.markdown("""
                                **Understanding the Analysis:**
                                * Red bars show words that contribute to a **positive** prediction (likely to appeal)
                                * Blue bars show words that contribute to a **negative** prediction (unlikely to appeal)
                                * The length of each bar shows how strongly the word influences the prediction
                                """)
                            else:
                                st.info("No significant word impacts found in the analysis.")
                            
                            # Show highlighted text
                            st.subheader("Text Analysis")
                            highlighted_text = highlight_important_words(
                                text_input,
                                lime_explanation['top_words']  # Use LIME's top words for highlighting
                            )
                            st.markdown(highlighted_text, unsafe_allow_html=True)

                            # Add similar cases section after other visualizations
                            if similar_cases:  # Only show if we have similar cases
                                st.markdown("---")
                                display_similar_cases(similar_cases)
                            
                            # Add PDF download button
                            st.markdown("---")
                            pdf_bytes = export_to_pdf(
                                [result], 
                                [text_input],
                                lime_explanations=[lime_explanation],
                                similar_cases=[similar_cases] if similar_cases else None
                            )
                            st.download_button(
                                label="📥 Download Analysis Report",
                                data=pdf_bytes,
                                file_name="text_analysis_report.pdf",
                                mime="application/pdf"
                            )

                        except Exception as e:
                            st.error(f"An error occurred during analysis: {str(e)}")

        with tab2:
            # PDF file upload
            uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
            if uploaded_file is not None:
                try:
                    # Read PDF content
                    pdf_reader = PyPDF2.PdfReader(BytesIO(uploaded_file.read()))
                    text = ""
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
                    
                    # Show extracted text
                    with st.expander("Show extracted text"):
                        st.text(text)
                    
                    if st.button("Analyze PDF", key="pdf_analyze"):
                        with st.spinner("Analyzing PDF..."):
                            result = pipeline.get_prediction(text)
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.subheader("Classification Results")
                                prediction_label = "Positive" if result['prediction'] == 1 else "Negative"
                                st.markdown(f"**Prediction:** {prediction_label}")
                                
                                fig = create_confidence_gauge(result['probability'])
                                st.plotly_chart(fig, use_container_width=True)

                            with col2:
                                st.subheader("Important Words")
                                for idx, word in enumerate(result['important_words'], 1):
                                    st.markdown(f"{idx}. **{word}**")
                            
                            st.subheader("Text Analysis")
                            highlighted_text = highlight_important_words(
                                text[:1000] + "..." if len(text) > 1000 else text,  # Limit displayed text
                                result['important_words']
                            )
                            st.markdown(highlighted_text, unsafe_allow_html=True)
                            
                            # Download PDF report
                            pdf_bytes = export_to_pdf(
                                [result], 
                                [text],
                                lime_explanations=[lime_explanation],
                                similar_cases=[similar_cases] if similar_cases else None
                            )
                            st.download_button(
                                label="📥 Download Analysis Report",
                                data=pdf_bytes,
                                file_name="pdf_analysis_report.pdf",
                                mime="application/pdf"
                            )
                            
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    st.error("Please make sure the uploaded file is a valid PDF.")

        # Update footer
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center'>
            <p>Developed for the City of Amsterdam</p>
        </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please make sure the model and tokenizer files are in the correct location.")

if __name__ == "__main__":
    main()