from telebot.service_utils import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

import telebot

import os
from dotenv import load_dotenv

load_dotenv()


TOKEN = os.getenv('TOKEN')


bot = telebot.TeleBot(TOKEN)

@bot.message_handler(content_types=['text'])
def handle_text(message):
    bot.send_message(message.chat.id, "Смартфон vivo")

@bot.message_handler(content_types=['photo'])
def handle_photo(message):
        file_info = bot.get_file(message.photo[len(message.photo)-1].file_id)

        loaded_model = load_model('/home/kirill/Рабочий стол/ML/hyenaNcheetah.keras')
        downloaded_file = bot.download_file(file_info.file_path)

        src='/home/kirill/Рабочий стол/aot/'+file_info.file_path;
        with open(src, 'wb') as new_file:
           new_file.write(downloaded_file)


        img = image.load_img(src, target_size=(200, 200))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_preprocessed = img_array / 255.0

        prediction = loaded_model.predict(img_preprocessed)

        predicted_class_index = np.argmax(prediction)

        class_labels = ['C', 'H'] 
        predicted_class_label = class_labels[predicted_class_index]
        if predicted_class_label == 'C':
            bot.reply_to(message, "гепард")
        else:
            bot.reply_to(message, "гиена")


bot.polling(none_stop=True, interval=0)
