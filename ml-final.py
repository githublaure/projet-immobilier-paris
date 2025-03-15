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
    # Map features from the data
    features = ['surface', 'nombre_pieces_principales', 'dist_ratp', 'year']
    X = data[features].copy()
    X['revenu_cat'] = pd.cut(data.med_revenu, bins=[0., 20000., 25000., 30000., 35000., np.inf], labels=[1, 2, 3, 4, 5])
    y = data['prix_mcarre']

    # Train model using simple train test split
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X.drop('revenu_cat', axis=1), 
        y, 
        test_size=0.15, 
        random_state=7
    )

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
            address = request.form.get('adresse', '')
            city = request.form.get('ville', '')
            property_type = request.form.get('local', '')

            from geopy.geocoders import GoogleV3
            from geopy.distance import geodesic
            import datetime

            # Initialize the geocoder
            geolocator = GoogleV3(api_key='YOUR_GOOGLE_API_KEY')

            # Get coordinates from address
            if address:
                location = geolocator.geocode(address + ", " + city)
                if location:
                    # Load RATP stations data
                    stations_ratp = pd.read_csv('static/ratp-geocoordinates.csv')
                    
                    # Calculate distance to nearest RATP station
                    distances = stations_ratp.apply(
                        lambda row: geodesic(
                            (location.latitude, location.longitude),
                            (row['lat'], row['lng'])
                        ).kilometers,
                        axis=1
                    )
                    min_dist_ratp = distances.min()

                    # Create sample with real location data
                    sample_house = pd.DataFrame({
                        'surface': [float(request.form.get('surface', 75.0))],
                        'nombre_pieces_principales': [int(request.form.get('pieces', 3))],
                        'dist_ratp': [min_dist_ratp],
                        'year': [datetime.datetime.now().year]
                    })
                else:
                    raise ValueError("Adresse non trouvée")
            else:
                raise ValueError("Adresse requise")

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