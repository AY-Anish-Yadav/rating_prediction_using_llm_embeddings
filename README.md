# Review Prediction Using LLM Embedding Model

![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)

## Overview

Using embeddings from large language models, we can effectively predict ratings from text reviews by converting reviews into dense vector representations and then training a regression model on these embeddings.

## Features

- Input text through a web interface
- Display the predicted rating on the web page.

## Project-Flow
<div align="center">
<img src="./project_flow.png">
</div>

### Data Preparation

1. Data Downloading

a. Downloads a file from Google Drive.
b. Creates the output directory if it doesn’t exist.
c. Measures and prints download time.

2. Data Conversion to embeddings

a. Reads a CSV file in chunks.
b. Processes each chunk to:
c. Drop unnecessary columns.
d. Convert ratings and reviews to appropriate formats.
e. Encode or embed reviews based on the specified model.
f. Save processed data as numpy arrays.
g. Combines all processed chunks into one numpy array.
h. Saves the combined data to ratings_embeddings.npy.
I. Cleans up temporary files and memory.

### Model Training
1. Load dataset and split into training and test sets.
2. Model Training
a. Linear Regression: Train without hyperparameter tuning.
b. Lasso Regression: Train with hyperparameter tuning (alpha).
c. Ridge Regression: Train with hyperparameter tuning (alpha).
d. Random Forest: Train with hyperparameter tuning (n_estimators, max_depth).
e. Support Vector Machine (SVM): Train with hyperparameter tuning (C, epsilon and kernel).
f. Artificial Neural Network (ANN): Train with hyperparameter tuning (hidden_layer_sizes, activation, solver).

### Model Evaluation
1. MSE
2. R2 Squared
3. Adusted R2 Squared

### Model Inference
1. Initialize Flask App: Create and configure a basic Flask application.
2. Load Model: Load the pre-trained machine learning model and Embedding model.
3. Define Endpoint: Create a route to handle prediction requests.
4. Handle Requests: Parse input data, perform inference, and return results.
5. Run Server: Start the Flask server to handle incoming requests.

## Technologies Used

- Python
- Flask
- Transformers
- Embedding Models 

## Setup and Installation

### Prerequisites

- Python 3.9+
- pip
- pip install -r requirements.txt

### Installing Dependencies

1. Clone the repository:

```bash
git clone https://github.com/AY-Anish-Yadav/review_prediction_using_llm_embeddings.git
cd review_prediction_using_llm_embeddings
   python app.py
   ```

## Contributing

Contributions are welcome! If you would like to contribute to the project.

## License

This project is licensed under the Apache 2.0 License - see the LICENSE file for details.

## Contact

For any inquiries or feedback, feel free to contact:

Anish Yadav - reach.anish.yadav@gmail.com

