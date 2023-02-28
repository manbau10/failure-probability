import numpy as np
from flask import Flask, request, render_template
import pickle
import  copy

# Create flask app
app = Flask(__name__, static_folder='static')


model = pickle.load(open("model.pkl", "rb"))


gui_variables = [
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
    'Material': ['Material_AC', 'Material_CI', 'Material_DI', 'Material_GI', 'Material_GIL', 'Material_PE', 'Material_S', 'Material_SS'],
    'Corrosivity': ['Corrosivity_Highly corrosive','Corrosivity_Mildly corrosive', 'Corrosivity_Non corrosive'],
    'Road_type': ['Road_type_Carriageway', 'Road_type_Footway', 'Road_type_Other location'],
    'Landuse': ['Landuse_Rural', 'Landuse_Sea','Landuse_Urban', 'Landuse_Waterbody']
}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    input_data = []
    for key in gui_variables:
        val = request.form.get(key)
        print(key, val)
        if key == "CP":
            if val == "Yes":
                input_data.append(1)
            else:
                input_data.append(0)
        elif key in dummy_variables.keys():
            tmp = dummy_variables.get(key)

            input_data.extend(
                (np.array(tmp) == val).astype(np.uint8)
            )

        else:
            input_data.append(
                float(val)
            )
    print("input_data:\n", input_data)
    #print("test: ", request.form.get("test1"))
    print(list(request.form.keys()))

    prediction = model.predict_proba([input_data])
    output = np.round(prediction[0], 2)

    # retained_dic = copy.deepcopy(request.form)
    # prediction_text="The reliability and failure probability of the pipe are {}".format(output),


    return render_template("index.html",
                           # context=[prediction_text],
                           # context=retained_dic,
                           # form=form
                           prediction_text="The reliability and failure probability of the pipe are {}".format(output),
                           pressure=request.form.get("Pressure"),
                           cp=request.form.get("Cathodic Protection"),
                           length=request.form.get("Length"),
                           diameter=request.form.get("Diameter"),
                           age=request.form.get("Age"),
                           traffic=request.form.get("AADT"),
                           temperature=request.form.get("Temperature"),
                           humidity=request.form.get("Humidity"),
                           precipitation=request.form.get("Precipitation"),

                           )



if __name__ == "__main__":
    app.run(debug=True)
