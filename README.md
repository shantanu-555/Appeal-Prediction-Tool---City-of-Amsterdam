# 🏛️ Appeal Prediction Tool - City of Amsterdam

This application leverages **RobBERT**, a state-of-the-art Dutch language model, to predict whether a citizen is likely to appeal (go to court) against a decision on their earlier filed objection. Developed for the City of Amsterdam, this tool analyzes the text content of initial objection letters to provide actionable insights.

## 📋 Table of Contents

- [Features](#features)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
  - [1. Clone the Repository](#1-clone-the-repository)
  - [2. Download Model Files](#2-download-model-files)
  - [3. Build and Run with Docker](#3-build-and-run-with-docker)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)  
- [License](#license)
- [Contact](#contact)

## 🛠️ Features

- **Single Text Analysis**: Enter text directly to receive predictions.
- **PDF Upload**: Upload a PDF document for analysis.
- **Comprehensive PDF Reports**: Download detailed analysis reports.
- **Word Impact Analysis**: Understand which words influence predictions using LIME.
- **Similar Cases Retrieval**: Find similar cases from training data to contextualize predictions.

## 🤖 Model Training and Evaluation

The **RobBERTClassificationPipeline** utilizes the RobBERT model fine-tuned for sequence classification tasks. Here's a brief overview of the training and evaluation process:

1. **Data Preparation**:
   - **Dataset**: The model is trained on a dataset comprising objection letters labeled as either "Appeal" or "No Appeal."
   - **Preprocessing**: Text data undergoes cleaning, including the removal of stopwords, normalization of whitespace, and handling of placeholders.

2. **Model Architecture**:
   - **RobBERT**: A robust state-of-the-art Dutch BERT-based model optimized for sequence classification.
   - **Tokenizer**: Uses `RobertaTokenizer` for tokenizing input text.

3. **Training**:
   - **Optimization**: The model is fine-tuned using cross-entropy loss with Adam optimizer.
   - **Parameters**: Training parameters such as learning rate, batch size, and epochs are optimized based on validation performance.

4. **Evaluation**:
   - **Metrics**: The model is evaluated using metrics like accuracy, precision, recall, and F1-score.
   - **Results**: Evaluation metrics are stored in `eval_results/robbert_evaluation_metrics.csv` for reference.

## 🚀 Prerequisites

Before setting up the application, ensure you have the following installed on your system:

- [Docker](https://www.docker.com/get-started) (version 20.10.0 or higher)
- [Docker Compose](https://docs.docker.com/compose/install/) (version 1.29.0 or higher)
- Git

## 📥 Installation

Follow these steps to clone the repository, set up the necessary models, and run the application using Docker.

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/appeal-prediction-tool.git
cd appeal-prediction-tool
```

### 2. Download Model Files

The application requires specific model files to function correctly. Follow these steps to obtain and set up the models:

1. **Create Models Directory**:
   
   Ensure a `models` directory exists in the project root:

   ```bash
   mkdir -p models
   ```

2. **Download Models**:
   
   - **RobBERT Model and Tokenizer**:
     - Download the RobBERT model files (`robbert_model/`) and tokenizer files (`robbert_tokenizer/`) from [here](https://your-model-storage-link).
     - Extract and place them into the `models` directory:

       ```
       models/
       ├── robbert_model/
       └── robbert_tokenizer/
       ```

   - **Training Embeddings and Texts**:
     - Download `training_embeddings.joblib` and `training_texts.joblib` from [here](https://your-model-storage-link).
     - Place these files directly into the `models` directory:

       ```
       models/
       ├── training_embeddings.joblib
       └── training_texts.joblib
       ```

> **Note**: Replace `https://your-model-storage-link` with the actual URLs where your model files are hosted. You may use services like Google Drive, Dropbox, or an internal company storage solution.

### 3. Build and Run with Docker

Using Docker ensures a consistent environment for running the application.

1. **Build the Docker Image**:

   ```bash
   docker-compose build
   ```

2. **Run the Docker Container**:

   ```bash
   docker-compose up
   ```

   The application will start and be accessible at [http://localhost:8501](http://localhost:8501).

3. **Stopping the Application**:

   To stop the application, press `CTRL+C` in the terminal where Docker is running, then execute:

   ```bash
   docker-compose down
   ```

## 📈 Usage

Once the application is running, you can access it via your web browser at [http://localhost:8501](http://localhost:8501). The interface provides two main functionalities:

### 1. Single Text Analysis

- **Input**: Paste the content of an objection letter into the text area.
- **Analyze**: Click on "Analyze Text" to receive a prediction.
- **Results**:
  - **Prediction**: Indicates whether an appeal is likely.
  - **Confidence Score**: Shows the probability of the prediction.
  - **Word Impact Analysis**: Displays which words influenced the prediction using LIME.
  - **Similar Cases**: Lists similar cases from the training data.
- **Download Report**: Click the "📥 Download Analysis Report" button to obtain a comprehensive PDF report.

### 2. PDF Upload

- **Upload**: Choose a PDF file containing the objection letter.
- **Analyze**: Click on "Analyze PDF" to process the document.
- **Results**:
  - **Prediction**: Similar to single text analysis.
  - **Important Words**: Lists key influential words.
  - **Text Analysis**: Highlights important words in the text.
- **Download Report**: Click the "📥 Download Analysis Report" button to download the PDF report.

## 🛠️ Troubleshooting

If you encounter issues while running the application, consider the following steps:

1. **Check Docker Containers**:
   
   Ensure that Docker containers are running:

   ```bash
   docker ps
   ```

2. **Review Logs**:
   
   Inspect Docker logs for any errors:

   ```bash
   docker-compose logs
   ```

3. **Model Files**:
   
   - Verify that all required model files are correctly placed in the `models` directory.
   - Ensure that the directory structure inside `models/` matches the expected layout.

4. **Port Conflicts**:
   
   If port `8501` is already in use, stop the conflicting service or modify the `docker-compose.yml` to use a different port.

5. **Rebuild Docker Image**:
   
   Sometimes, rebuilding the Docker image can resolve issues:

   ```bash
   docker-compose up --build
   ```

6. **Network Issues**:
   
   Ensure that Docker has the necessary network access and that no firewalls are blocking the port.

## 📄 License

[MIT License](LICENSE)

## 📬 Contact

For any questions or support, please contact:

- **Name**: Shantanu Motiani
- **Email**: shantanumotiani@gmail.com
- **LinkedIn**: [linkedin.com/in/shantanu-motiani](https://www.linkedin.com/in/shantanu-motiani/)

---

Developed for the **City of Amsterdam** 🏛️