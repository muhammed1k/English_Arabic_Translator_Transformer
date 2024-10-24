import tensorflow as tf
import utils
import data
from config import config
from schedulers import *
from models import *
from loss import *
from layers import *
from accuracy import *
from flask import Flask,jsonify,request
from flask_cors import CORS
import numpy as np
import tensorflow_text as tf_text
import warnings
import os
import argparse

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppresses INFO, WARNING, and ERROR logs
warnings.filterwarnings("ignore", category=UserWarning, module='keras')

app = Flask(__name__)
CORS(app)

def main(train_model=False,epochs=20,output_path=None,input_path=None):
    
    transformer = utils.load_model(input_path,output_path,epochs,train_model)

    translator = Translator(data.tokenizers, transformer)

    @app.route('/translate',methods=['POST'])
    def translate():
        english_sentence = request.json.get('sentence')
        translated_text, translated_tokens, attention_weights = translator(
                                                        tf.constant(english_sentence))
        return jsonify({'translation':translated_text.numpy().decode('utf-8')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Translation model')
    parser.add_argument('--train',action='store_true',help='Train the transformer model for specific number of epochs')
    parser.add_argument('--epochs',type=int,default=20,help='number of epochs to train')
    parser.add_argument('--output_weights',type=str,help='output weights save location')
    parser.add_argument('--train_data',type=str,help='train data path')

    args = parser.parse_args()

    main(train_model=args.train,epochs=args.epochs,output_path=args.output_weights,input_path=args.train_data)
    app.run(host='localhost',port=9897,debug=True)
else:
    main()