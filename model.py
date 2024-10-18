import pandas as pd
import pickle
from flask import Flask, render_template, request
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split as tt

# Load and preprocess the data
d=pd.read_csv(r'C:\Users\rjeev\Document\data.csv')#dataset path
d=d.iloc[:,:12]
target=d.diagnosis  
l=['diagnosis','id']
d=d.drop(l,axis=1)
target=np.where(target.values=='M',0,1)
Scaler=MinMaxScaler()
d=Scaler.fit_transform(d)
x_train,x_test,y_train,y_test=tt(d,target,test_size=0.2)

# Train the models
lr_model = LogisticRegression()
lr_model.fit(x_train, y_train)

svm_model = SVC()
svm_model.fit(x_train, y_train)

knn_model = KNeighborsClassifier()
knn_model.fit(x_train, y_train)

# Train the novel model
lr_predictions = lr_model.predict(x_test)
svm_predictions = svm_model.predict(x_test)
knn_predictions = knn_model.predict(x_test)
combined_predictions = np.array([lr_predictions, svm_predictions, knn_predictions])
novel_model = LogisticRegression()
novel_model.fit(combined_predictions.T, y_test)

# Create the Flask app
app = Flask(__name__)

# Define the home page route
@app.route('/')
def home():
    return render_template('home.html')

# Define the predict route
@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    features = np.array(features).reshape(1,-1)
    features = Scaler.transform(features)
    lr_pred = lr_model.predict(features)[0]
    svm_pred = svm_model.predict(features)[0]
    knn_pred = knn_model.predict(features)[0]
    combined_pred = np.array([lr_pred, svm_pred, knn_pred]).reshape(1,-1)
    novel_pred = novel_model.predict(combined_pred)[0]
    if novel_pred == 0:
        prediction = 'Malignant'
    else:
        prediction = 'Benign'
    return render_template('result.html', prediction=prediction)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
