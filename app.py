# coding=utf-8
from flask import Flask, url_for, request, render_template
# from flask_cors import *

import json
import numpy as np
import tensorflow as tf

from flask import jsonify
from model_train import token_dict, OurTokenizer
from keras.models import load_model
from albert import get_custom_objects

app = Flask(__name__)

maxlen = 256

# to load the model and save it for the entire environment use graph
global graph
graph = tf.get_default_graph()

# 加载训练好的模型
model = load_model("albert_base_multi_label_ee.h5", custom_objects=get_custom_objects())
tokenizer = OurTokenizer(token_dict)
with open("label.json", "r", encoding="utf-8") as f:
    label_dict = json.loads(f.read())


@app.route('/classify/tc', methods=['POST'])
def sa_classify():
    params = json.loads(request.get_data(as_text=True))
    params['text'] = params['text'].replace('\t', ' ').replace('\n', ' ').replace(' ', '')
    text = params['text']

    # 利用BERT进行tokenize
    text = text[:maxlen]
    x1, x2 = tokenizer.encode(first=text)

    X1 = x1 + [0] * (maxlen - len(x1)) if len(x1) < maxlen else x1
    X2 = x2 + [0] * (maxlen - len(x2)) if len(x2) < maxlen else x2

    # You need to use the following line
    with graph.as_default():
        # 模型预测并输出预测结果
        # prediction = model._make_predict_function([[X1], [X2]])
        prediction = model.predict([[X1], [X2]])
        one_hot = np.where(prediction > 0.5, 1, 0)[0]

        res = {
            'label': [label_dict[str(i)] for i in range(len(one_hot)) if one_hot[i]]
        }

    return jsonify(res)


if __name__ == "__main__":
    # port: -》 8897
    """ TODO 更换port """
    app.run(host='127.0.0.1', port=5001, debug=True)
