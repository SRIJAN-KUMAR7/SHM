from flask import Flask, render_template, request
import joblib
import os
import numpy as np
import os
print("Current working directory:", os.getcwd())


app = Flask(
    __name__,
    template_folder='../frontend/templates',
    static_folder='../frontend/static'
)

healing_model = joblib.load('./efficiency_model.pkl')
stress_model = joblib.load('./peak_stress_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    if request.method == 'POST':
        crack_length = float(request.form['crack_length'])
        temperature = float(request.form['temperature'])
        input_features = np.array([[crack_length, temperature]])
        efficiency = healing_model.predict(input_features)[0]
        stress = stress_model.predict(input_features)[0]

        result = {
            'crack_length': crack_length,
            'temperature': temperature,
            'efficiency': efficiency,
            'stress': stress
        }
        with open('../last_prediction.csv', 'w') as f:
            f.write("Crack Length,Temperature,Healing Efficiency,Peak Stress\n")
            f.write(f"{crack_length},{temperature},{efficiency},{stress}\n")

    return render_template("index.html", result=result)

if __name__ == '__main__':
    app.run(debug=True)
