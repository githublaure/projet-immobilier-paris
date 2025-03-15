
# Machine Learning Real Estate Price Estimator

## Overview
This application uses machine learning to estimate real estate prices in France based on various features including location, property characteristics, and nearby amenities.

## Data Sources
The model is trained on the following data:
- DVF (Demande de Valeurs Foncières) database for real estate transactions
- RATP and SNCF station locations
- Population statistics from INSEE (2015)
- Schools and educational institutions data
- Green areas and parks information
- Average income data by postal code (2018)

## Features Used for Prediction
- Surface area (m²)
- Number of rooms
- Distance to nearest metro/RER station
- Year of transaction
- Property type (apartment/house)
- Postal code
- Population density
- Proximity to schools and green spaces

## Technical Stack
- Python Flask for the web application
- Scikit-learn for the machine learning model
- Ridge Regression with cross-validation
- StandardScaler for feature normalization
- pandas for data processing

## Model Performance
- RMSE: 51,367.86 €
- R²: 0.783

## How to Use
1. Enter the property address
2. Select the city
3. Choose property type (apartment/house)
4. The model will return an estimated price per square meter

## Running the Application
```bash
python ml-final.py
```
The application will be available at port 8080.
