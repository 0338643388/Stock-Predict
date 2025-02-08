# Stock Price Prediction

## Overview
The **Stock Price Prediction** project is a web-based application built with Streamlit, leveraging machine learning models to predict stock prices based on historical data. The application fetches stock data from Yahoo Finance, processes technical indicators, and provides visualizations to assist in investment decisions.

## Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Data Sources](#data-sources)
- [Model Training](#model-training)
- [Deployment](#Deployment)
- [Contributing](#contributing)
- [License](#license)

## Features
- **Data Collection**: Fetches historical stock data from **investing.com**.
- **Data Preprocessing**: Cleans and prepares the data for analysis.
- **Technical Indicators**: Calculate indicators such as PSAR, MACD, RSI, OBV, SMA, EMA and more.
- **Model Selection**: Uses three models to predict future stocks prices: Linear Regression, ARIMA, and LSTM (Long Short-Term Memory) networks.
- **Data Visualization**: Displays stock price trends and rolling average charts.
- **Prediction**: Generates future stock price predictions based on the trained models.
- **Interactive UI**: Allows users to select stocks and view predictions in realtime.

## Technologies Used
- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- TensorFlow
- Streamlit
- yFinance
- Joblib

## Installation
To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/0338643388/Stock-Predict.git
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. run Streamlit application:
```bash
streamlit run app.py
```
2. Open the browser and go to
```bash
http://localhost:8501
```
3. Select a stock from the dropdown menu to view historical charts and predictions.

## Data Sources
The project uses historical stock price data from sources like:
- Yahoo Finance API
- Investing.com

## Model Training
The project includes scripts for training different models. The primary steps are:
1. **Data Preprocessing**: Cleaned and prepared historical stock data.
2. **Feature Engineering**: Computed technical indicators like MACD, RSI, OBV, etc.
3. **Model Selection**: Used Linear Regression, ARIMA, LSTM for predictions.
4. **Training & Evaluation**: The model was trained and evaluated using metrics like MSE, MAE, and R² score.
6. **Model Serialization**: Saved using joblib for deployment.

## Deployment
The project is deployed on **Streamlit Cloud**. You can access it here:
🔗 [Stock Price Prediction App](https://stock-predict-dr9faqumhea7uthcgoq7cy.streamlit.app/)

## Contributing
Contributions are welcome! If you'd like to contribute to this project, please follow these steps:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature/YourFeature`).
3. Make your changes and commit them (`git commit -m 'Add some feature'`).
4. Push to your branch (`git push origin feature/YourFeature`).
5. Open a pull request.

