import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD
import random

#definimos nuestas variables
words=[]
classes=[]
documents=[]
ignore_words=['?','!']
data_file=open('q&a.json').read()
intents = json.loads(data_file)

#Primero, importamos nuestros 'intents'
for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenizamos cada palabra de nuestras preguntas
        w = nltk.word_tokenize(pattern)
        #Agregamos w a nuestras palabras
        words.extend(w)
        #añadimos a documents la dupla de (palabra tokenizada, nombre de la clase que pertenece)
        documents.append((w, intent['tag']))

        #agregamos los tags a classes para categorizarlas
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

#limpiamos las palbras. En este caso solo son los caracteres '?' y '!' y ordenamos
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))
classes= sorted(list(set(classes)))

#Revisamos nuestras variables
print(len(documents), "documents")
print(len(classes), "classes", classes)
print(len(words), "unicas lemmatized words", words)

#Volvemos nuestras variables 'dummys'
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

#Ahora crearemos nuestros datos de entrenamiento
training=[]
output_empty=[0] * len(classes) 
for doc in documents:
    bag=[]
    pattern_words= doc[0]
    #convertimos a minusculas nuestras palabras
    pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

    #agregamosnlas palabras claves que se encuentran en nuestras preguntas
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1
    training.append([bag, output_row])

#entrenamos
random.shuffle(training)
training = np.array(training)
train_x = list(training[:,0])
train_y = list(training[:,1])
print("training data ceated")

#Creamos nuestra red neuronal de 3 capas
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0],), activation='softmax'))
print(model.summary())

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#Guardamos nuestro modelo y listo
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)
print("model created")