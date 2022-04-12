import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pickle

datatrain=pd.read_csv(r'data train numerik no box 20 k5 91.csv')

datatrain["DERMAGA"]=datatrain["DERMAGA"].astype('category')
datatrain["JENIS_KAPAL"]=datatrain["JENIS_KAPAL"].astype('category')
datatrain["DELAY"]=datatrain["DELAY"].astype('category')
datatrain["PALKA"]=datatrain["PALKA"].astype('int')
datatrain["BD"]=datatrain["BD"].astype('int')
datatrain["SHIFTING"]=datatrain["SHIFTING"].astype('int')
datatrain["WAG"]=datatrain["WAG"].astype('int')
datatrain["BAD_WEATHER"]=datatrain["BAD_WEATHER"].astype('int')
datatrain["JUMLAH_CC"]=datatrain["JUMLAH_CC"].astype('int')
datatrain["DISCHARGE"]=datatrain["DISCHARGE"].astype('int')
datatrain["LOADING"]=datatrain["LOADING"].astype('int')
  
arr = datatrain.values
X_train = arr[:, 0:10]
Y_train = arr[:, 10]

datatest=pd.read_csv(r'data test numerik no box 20 k5 91.csv')

#tambahin ubah data integer
datatest["DERMAGA"]=datatest["DERMAGA"].astype('category')
datatest["JENIS_KAPAL"]=datatest["JENIS_KAPAL"].astype('category')
datatest["DELAY"]=datatest["DELAY"].astype('category')
datatest["PALKA"]=datatest["PALKA"].astype('int')
datatest["BD"]=datatest["BD"].astype('int')
datatest["SHIFTING"]=datatest["SHIFTING"].astype('int')
datatest["WAG"]=datatest["WAG"].astype('int')
datatest["BAD_WEATHER"]=datatest["BAD_WEATHER"].astype('int')
datatest["JUMLAH_CC"]=datatest["JUMLAH_CC"].astype('int')
datatest["DISCHARGE"]=datatest["DISCHARGE"].astype('int')
datatest["LOADING"]=datatest["LOADING"].astype('int')

arr = datatest.values
X_test = arr[:, 0:10]
Y_test = arr[:, 10]

knn=KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train,Y_train)
Y_PredKNN=knn.predict(X_test)

#Saving the Model
pickle_out = open("knn.pkl", "wb") 
pickle.dump(knn, pickle_out) 
pickle_out.close()

def predict(arr):
    # Load the model
    with open('knn.pkl', 'rb') as f:
        model = pickle.load(f)
    classes = {0:'tidak delay',1:'delay kurang dari sama dengan 4 jam',2:'delay > 4 jam'}
    # return prediction as well as class probabilities
    preds = model.predict_proba([arr])[0]
    return (classes[np.argmax(preds)], preds)