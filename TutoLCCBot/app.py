import logging
import telegram
import json
import random
import nltk
import pickle
import numpy as np
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from telebot.credentials import bot_token, bot_username
from tensorflow.keras.models import load_model
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

#Cargamos nuestros archivos
model = load_model('chatbot_model.h5')
intents = json.loads(open('q&a.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

#Configuración del loggin
logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger()
#TOKEN del bot
TOKEN = bot_token

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)

    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return(np.array(bag))

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    #print(p)
    res = model.predict(np.array([p]))[0]

    ERROR_THRESHOLD=0.25

    results = [[i, r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list=[]

    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})

    return return_list
    

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents=intents_json['intents']

    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

def start(update, context):
    #print(update)
    logger.info(f"El usuario {update.effective_user['id']}, a iniciado una conversación")
    name = update.effective_user['first_name']
    update.message.reply_text(f"Hola {name}, yo soy FilosoraptorBot!")

def echo(update, context):
    user_id = update.effective_user['id']
    logger.info(f"El usuario {user_id}, a enviado un mensaje de texto.")
    text = update.message.text
    if text != '':
        res = chatbot_response(text)
        context.bot.send_message(
            chat_id=user_id,
            parse_mode="MarkdownV2",
            text=f"{res}"
        )

#Para obtener info del bot
if __name__=='__main__':
    my_bot = telegram.Bot(token=TOKEN)
    #print(my_bot.getMe())

#Enlace entre el updater con el bot
updater = Updater(my_bot.token, use_context=True)
#Despachador
dp= updater.dispatcher

#Manejadores
dp.add_handler(CommandHandler("start", start))
dp.add_handler(MessageHandler(Filters.text, echo))

updater.start_polling()
print("Bot cargado")
updater.idle() #Para finalizar el bot con ctrl + c