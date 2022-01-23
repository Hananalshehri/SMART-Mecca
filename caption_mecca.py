from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import image
from keras.models import load_model, Model
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
import numpy as np
from deep_translator import GoogleTranslator
from pygame import mixer
from gtts import gTTS

import warnings
warnings.filterwarnings("ignore")
import os


with open("/Users/mukhtaralbinhamad/Desktop/Bootcamp/Meccacaption/model_weights/h_model.pkl", "rb") as model_3:
    loaded_model = pickle.load(model_3)
    loaded_model.make_predict_function()

model_temp = InceptionV3(weights="imagenet")

# Create a new model, by removing the last layer (output layer of 1000 classes) from the resnet50
model_resnet = Model(model_temp.input, model_temp.layers[-2].output)
model_resnet.make_predict_function()


    
# Load the word_to_idx and idx_to_word from disk

with open("/Users/mukhtaralbinhamad/Desktop/Bootcamp/Meccacaption/storage/wordtoix.pkl", "rb") as w2i:
    word_to_idx = pickle.load(w2i)

with open("/Users/mukhtaralbinhamad/Desktop/Bootcamp/Meccacaption/storage/ixtoword.pkl", "rb") as i2w:
    idx_to_word = pickle.load(i2w)
    

max_len = 24


def preprocess_image(img):
    img = image.load_img(img, target_size=(299, 299))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def encode_image(img):
    img = preprocess_image(img)
    feature_vector = model_resnet.predict(img)
    feature_vector = feature_vector.reshape(1, feature_vector.shape[1])
    return feature_vector


def predict_caption(photo):
    in_text = "startseq"

    for i in range(max_len):
        sequence = [word_to_idx[w] for w in in_text.split() if w in word_to_idx]
        sequence = pad_sequences([sequence], maxlen=max_len ,padding='post')
#         sequence = tf.stack(sequence)
        
        ypred = loaded_model.predict([photo,sequence])
        ypred = tf.stack(ypred)       
        ypred = np.argmax(ypred)
        word = idx_to_word[ypred]
        in_text+= ' ' + word

        if word =='endseq':
            break


    final_caption =  in_text.split()
    final_caption = final_caption[1:-1]
    final_caption = ' '.join(final_caption)

    return final_caption

def caption_this_image(input_img): 

    photo = encode_image(input_img)
    caption = predict_caption(photo)
    # keras.backend.clear_session()
    return caption

def ar_speech(img_path):
    mytext = caption_this_image(img_path)
    translated = GoogleTranslator(source='auto', target='ar').translate(mytext)
    language = 'ar'
    myobj = gTTS(text=translated, lang=language, slow=False)
    file_name = img_path[img_path.rfind("/")+1:img_path.rfind(".")] + "_ar_.mp3"
    myobj.save(file_name)
    mixer.init()
    mixer.music.load("./" + file_name)
    mixer.music.play()
    return translated
