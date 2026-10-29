# Energy Consumption Prediction Using Machine Learning

A machine learning-based web application that predicts energy consumption using a trained LightGBM regression model. Built with Python and Flask, the application provides a user interface and a JSON API for generating predictions from input features.

## Overview

Energy consumption prediction can help analyze usage patterns and support energy planning. This project integrates a trained machine learning model into a web application, allowing users to submit input data and obtain predicted energy consumption.

The application uses Flask to handle web requests, LightGBM for prediction, and Pandas and NumPy for input data preparation. A dedicated utility module handles model loading, numeric conversion, and feature alignment before inference.

## Key Features

- **Machine Learning Prediction:** Uses a saved LightGBM model to generate energy consumption predictions.
- **Web Interface:** Provides a Flask-based interface for submitting prediction inputs.
- **REST-Style JSON API:** Accepts JSON requests and returns prediction results.
- **Input Preprocessing:** Converts numeric-like inputs and handles common missing-value representations.
- **Feature Alignment:** Aligns input columns with the feature names expected by the trained model when available.
- **Model Loading:** Loads the trained model when the application starts.
- **Error Handling and Logging:** Records application errors and provides feedback when prediction requests fail.

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Application and prediction logic |
| LightGBM | Regression model for energy consumption prediction |
| Flask | Web application and HTTP endpoints |
| Pandas | Input data preparation and DataFrame handling |
| NumPy | Numerical data processing |
| scikit-learn | Machine learning ecosystem dependency |
| Joblib | Model-related ecosystem dependency |
| HTML and CSS | Web interface templates and static assets |
| Gunicorn | WSGI server dependency for deployment |

## System Architecture

```text
        User Input
            |
            v
      Flask Web App
            |
            v
    Input Preprocessing
       model_utils.py
            |
            v
     Feature Alignment
            |
            v
   Trained LightGBM Model
            |
            v
   Energy Consumption
        Prediction
            |
       +----+----+
       |         |
       v         v
   Web Result  JSON Response
```

## Project Structure

```text
energy-consumption-prediction-using-ml/
│
├── models/
│   └── lgb_model_final97.pkl
│
├── static/
│   └── CSS and other static assets
│
├── templates/
│   └── index.html
│
├── app.py
├── model_utils.py
├── requirements.txt
└── README.md
```

The structure above describes the main application files. Keep the existing filenames and additional assets from your repository when updating the documentation.

## How It Works

1. **Model Initialization:** The Flask application loads the saved model from the `models` directory.
2. **Input Collection:** Users provide input through the web form or submit a JSON request to the prediction API.
3. **Preprocessing:** Input values are parsed, converted where possible, and organized into a Pandas DataFrame.
4. **Feature Alignment:** The input columns are aligned with the expected model features when those names are available.
5. **Prediction:** The trained LightGBM model generates a prediction.
6. **Output:** The web interface displays the result, while the JSON API returns the prediction in a structured response.

## Installation and Setup

### Prerequisites

- Python
- pip
- Git

### 1. Clone the repository

```bash
git clone https://github.com/amal1879314/energy-consumption-prediction-using-ml.git
cd energy-consumption-prediction-using-ml
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the environment on macOS or Linux:

```bash
source venv/bin/activate
```

On Windows:

```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Verify the model file

Ensure that the trained model file is available at:

```text
models/lgb_model_final97.pkl
```

The application expects this file to be present when loading the model.

### 5. Start the application

```bash
python app.py
```

Open the following address in your browser:

http://127.0.0.1:5000

The application runs locally using Flask's development server.

## Prediction API

The application provides a JSON endpoint for submitting prediction requests.

**Endpoint**

```text
POST /api/predict
```

**Request format**

The endpoint accepts a JSON object containing the model's expected input features. The features can be provided directly or inside a `features` object.

```json
{
  "features": {
    "feature_name_1": 10,
    "feature_name_2": 20
  }
}
```

Replace the example feature names and values with the actual features expected by the saved model.

**Example response format**

```json
{
  "prediction": 123.45,
  "prep_report": {}
}
```

The prediction value above is illustrative, not an actual model result. The preprocessing report contains details about input preparation.

**Error handling**

The endpoint returns structured JSON error responses when JSON parsing, model loading, or prediction processing fails.

## Web Application Routes

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Display the prediction interface |
| POST | `/predict` | Process form input and display the prediction |
| POST | `/api/predict` | Generate a prediction from JSON input |
| GET | `/_debug_model` | Inspect model loading status and expected features |

The debug endpoint is intended for development. It should be removed or protected before public deployment.

## Future Improvements

- Add model evaluation metrics such as MAE, RMSE, and R² from verified experiments.
- Include visualizations comparing actual and predicted consumption.
- Add stronger input validation and clearer API error responses.
- Write automated tests for preprocessing and prediction endpoints.
- Improve production configuration and application security.
- Deploy the application using a production WSGI server.
- Add historical energy usage analysis and forecasting capabilities.

## Author

**Amal Joe**

- GitHub: [amal1879314](https://github.com/amal1879314)
- LinkedIn: [Amal Joe](https://www.linkedin.com/in/amal-joe-93a485301)

---

This project demonstrates the integration of a trained machine learning model with a Python web application, including input preprocessing, model inference, and API-based prediction.
