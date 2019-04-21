from flask import Flask
import os
import json
from mnist_model import readImage
import Predict_image as pre
import numpy as np
import keras.backend as K
# Don't forget to run `memcached' before running this code
app = Flask(__name__)
@app.route('/predict/<image_name>')
def handle(image_name):
    model = readImage.loadmodel()
    dirPath = os.getcwd().replace('\\','/')
    imagePath = dirPath + '/server/upload/'+image_name
    predict_iden,predict_date = pre.predict(imagePath,model)
    list_obj = []
    for i in range(len(predict_iden)):
        iden = [np.asscalar(x) for x in predict_iden[i]]
        date = [np.asscalar(x) for x in predict_date[i]]
        temp = {'iden':iden,'date':date}
        list_obj.append(temp)
    tempjson = json.dumps(list_obj)
    return tempjson

if __name__ == "__main__":
    app.run()