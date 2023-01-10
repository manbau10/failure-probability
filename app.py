import numpy as np
from flask import Flask, request, render_template
import pickle

# Create flask app
flask_app = Flask(__name__, static_folder='static')

model = pickle.load(open("model.pkl", "rb"))

# List of variables for the model
variables = [
    'CP',
    'Length',
    'Diameter',
    'Pressure',
    'Age',
    'AADT',
    'Temperature',
    'Humidity',
    'Precipitation',
    'Material',
    'Corrosivity',
    'Road_type',
    'Landuse'
]

# Map of categorical variable names to dummy variable names
dummy_variables = {
    'Material': ['Material_AC', 'Material_CI', 'Material_DI', 'Material_GI', 'Material_GIL', 'Material_PE', 'Material_SS', 'Material_Steel', 'Material_UPVC'],
    'Corrosivity': ['Corrosivity_Highly-corrosive', 'Corrosivity_Mildly Corrosive', 'Corrosivity_Non-corrosive'],
    'Road_type': ['Road_type_CARRIAGEWAY', 'Road_type_FOOTWAY', 'Road_type_Other Location'],
    'Landuse': ['Landuse_RURAL', 'Landuse_SEA', 'Landuse_URBAN', 'Landuse_WATERBODY']
}

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    # Convert form data to dictionary
    form_data = {key: request.form.get(key) for key in variables}

    # Convert numerical variables to floats
    for key in ['CP', 'Length', 'Diameter', 'Pressure', 'Age', 'AADT', 'Temperature', 'Humidity', 'Precipitation']:
        form_data[key] = float(form_data[key])

    # Convert categorical variables to dummy variables
    for key in dummy_variables:
        form_data[key] = 1 if request.form.get(key) == '1' else 0
        for dummy_var in dummy_variables[key]:
            form_data[dummy_var] = 1 if request.form.get(dummy_var) == '1' else 0

    # Make sure the input data has the correct number of features
    input_data = []
    for var in variables:
        if var in dummy_variables:
            input_data += [form_data[dummy_var] for dummy_var in dummy_variables[var]]
        else:
            input_data.append(form_data[var])

    # Predict using the model
    prediction = model.predict_proba([input_data])
    output = np.round(prediction[0], 2)

    return render_template("index.html",
                           prediction_text="The reliablity and failure probability of the pipe are {}".format(output))
                            # prediction_text = "".format(output))





if __name__ == "__main__":
    flask_app.run(debug=True)
