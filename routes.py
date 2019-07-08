from flask import Flask, render_template
import joblib
import pandas as pd
import numpy as np
app=Flask(__name__)

@app.route("/")
def index():
	model = joblib.load('regr.pkl')
	predictpass=[4, 2.5, 3005, 15, 17903.0,0,0,1,0,0,1,0,0]
	prediction=np.squeeze(model.predict([predictpass]).round(1))
	return render_template("index.html",prediction=prediction)

if __name__ == '__main__':
    model = joblib.load('regr.pkl')
    app.run(debug=True)