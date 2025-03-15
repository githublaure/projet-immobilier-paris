from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load and prepare model
try:
    data = pd.read_csv('data/data-final-clean3.csv')
    features = ['surface', 'nombre_pieces_principales', 'gare_proche', 'year']
    X = data[features].copy()
    X['revenu_cat'] = pd.cut(data.med_revenu, bins=[0., 20000., 25000., 30000., 35000., np.inf], labels=[1, 2, 3, 4, 5])
    y = data['prix_mcarre']

    # Train model
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=7)
    for train_index, test_index in split.split(X, X['revenu_cat']):
        X_train = X.loc[train_index].drop('revenu_cat', axis=1)
        X_test = X.loc[test_index].drop('revenu_cat', axis=1)
        y_train = y.loc[train_index]
        y_test = y.loc[test_index]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RidgeCV(alphas=np.logspace(-10, 10, 21))
    model.fit(X_train_scaled, y_train)

except FileNotFoundError:
    print("Error: 'data/data-final-clean3.csv' not found. Please ensure the data file exists in the specified location.")
    exit(1)
except Exception as e:
    print(f"An error occurred during model loading and preparation: {e}")
    exit(1)



@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        try:
            # Get form data
            address = request.form['adresse']
            city = request.form['ville']
            property_type = request.form['local']

            # Make prediction using sample data.  Error handling added.
            sample_house = pd.DataFrame({
                'surface': [float(request.form.get('surface', 75))],  # Default to 75 if not provided
                'nombre_pieces_principales': [int(request.form.get('nombre_pieces_principales', 3))], # Default to 3
                'gare_proche': [float(request.form.get('gare_proche', 0.5))], # Default to 0.5
                'year': [int(request.form.get('year', 2020))] # Default to 2020
            })

            sample_scaled = scaler.transform(sample_house)
            predicted_price = model.predict(sample_scaled)[0]

            return render_template('arrive.html', message=f"Prix estimé: {predicted_price:,.2f}€/m²", address=address, city=city, property_type=property_type)
        except (ValueError, KeyError) as e:
            return render_template('error.html', error=f"Erreur de saisie: {e}")
        except Exception as e:
            return render_template('error.html', error=f"Une erreur s'est produite: {e}")

    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)