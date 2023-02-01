import pickle
from flask import Flask,render_template,jsonify,request,Response
import pandas as pd
import numpy as np


with open('Logistic_model.pkl','rb') as f:
    log_clf = pickle.load(f)
app=Flask(__name__)


@app.route('/')


@app.route('/Homepage')
def homepage():
    print('Welcome to pumpkin Seed Model')
    return render_template('index.html')


@app.route('/predict_class',methods=['POST'])
def abc():
    
    data = request.form
    area              = eval(data['area'])
    perimeter         = eval(data['perimeter'])
    major_axis_length = eval(data['major_axis_length'])
    minor_axis_length = eval(data['minor_axis_length'])
    convex_area       = eval(data['convex_area'])
    equiv_diameter    = eval(data['equiv_diameter'])
    eccentricity      = eval(data['eccentricity'])
    solidity          = eval(data['solidity'])
    extent            = eval(data['extent'])
    roundness         = eval(data['roundness'])
    aspect_ration     = eval(data['aspect_ration'])
    compactness       = eval(data['compactness'])
    
    
    test_array=np.zeros(12)
    test_array[0]=area
    test_array[1]=perimeter
    test_array[2]=major_axis_length
    test_array[3]=minor_axis_length
    test_array[4]=convex_area
    test_array[5]=equiv_diameter
    test_array[6]=eccentricity
    test_array[7]=solidity
    test_array[8]=extent
    test_array[9]=roundness
    test_array[10]=aspect_ration
    test_array[11]=compactness
        
    print('Test Array is',test_array)
    
    classes = log_clf.predict([test_array])
    return render_template ('after.html',data=classes)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=9000)
    
