from flask import Flask
import os
import json
from mnist_model import readImage
import Predict_image as pre

validatagen = readImage.getvaliddatagen()

# Don't forget to run `memcached' before running this code
app = Flask(__name__)
@app.route('/predict/<image_name>')
def handle(image_name):
    model = readImage.loadmodel()
    server_dir = os.path.join(os.getcwd(), 'server')
    upload_dir = os.path.join(server_dir, 'upload')
    imagePath = os.path.join(upload_dir, image_name)
    #print(imagePath)
    predict_iden,predict_date = pre.predict(imagePath,model,validatagen)
    list_obj = []
    for i in range(len(predict_iden)):
        iden = [int(x) for x in predict_iden[i]]
        date = [int(x) for x in predict_date[i]]
        temp = {'iden':iden,'date':date}
        list_obj.append(temp)
    tempjson = json.dumps(list_obj)
    return tempjson

if __name__ == "__main__":
    app.run()
