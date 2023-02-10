import telebot
import os
import torch

SAVED_IMG_DIR = "image.jpg"
IMAGE_FOLDER = "pred_img"


TOKEN = 'YOUR_TOKEN'
bot = telebot.TeleBot(TOKEN)

PATH = "yolov5l"

yolov5 = torch.hub.load('ultralytics/yolov5', PATH, pretrained=True)
print('MODEL LOADED SUCCESFULLY')

@bot.message_handler(commands=['start'])
def start(message):
    bot.send_message(message.chat.id, 'Pretrained YOLOV5 is integrated in this bot. Use command /help to see all commands. Send photo to get predicted image')

@bot.message_handler(commands=['help'])
def help(message):
    bot.send_message(message.chat.id, '/dataset - Get information about dataset.\n /model - Get inforamtion about model.\n /author - Get information about author\n ')

@bot.message_handler(commands=['model'])
def model(message):
    bot.send_message(message.chat.id, 'YOLOv5 (You Only Look Once version 5) is an object detection model developed by Alexey Bochkovskiy and the Ultralytics team. It is an improvement over previous versions of YOLO, with a focus on speed and accuracy.\n YOLOv5 uses a single convolutional neural network (CNN) to predict bounding boxes and class probabilities for objects in an image. \n It is trained on the COCO (Common Objects in Context) dataset and can detect up to 80 different object classes, including people, vehicles, and animals')

@bot.message_handler(commands=['dataset'])
def dataset(message):
    bot.send_message(message.chat.id, 'COCO (Common Objects in Context) is a large-scale object detection, segmentation, and captioning dataset. It was created by the Microsoft COCO Consortium, which includes researchers from institutions such as the University of California, Berkeley, and the Massachusetts Institute of Technology.\n The dataset contains over 330,000 images, each annotated with 80 object classes and more than 2.5 million object instances')

@bot.message_handler(commands=['author'])
def author(message):
    bot.send_message(message.chat.id, 'www.linkedin.com/in/AlikhanMukhatov - LinkedIn \n  https://github.com/esusos - GitHub \n alihanmuhatov@gmail.com - email \n @polotens - Telegram' )

@bot.message_handler(content_types=['photo'])
def predict(message):
    print('message.photo =', message.photo)
    fileID = message.photo[-1].file_id
    print('fileID =', fileID)
    file_info = bot.get_file(fileID)
    print('file.file_path =', file_info.file_path)
    downloaded_file = bot.download_file(file_info.file_path)

    filename = str(fileID) + ".jpg"
    
    with open(filename, 'wb') as new_file:
        new_file.write(downloaded_file)

    pred_img = open(filename, 'rb')
    print(pred_img)
    pred_img = yolov5(filename)

    SAVE_DIR = 'pred_img'

    pred_img.save(save_dir = SAVE_DIR)
    pred_img_dir = 'pred_img/' + filename

    final_image = open(pred_img_dir, 'rb')
    bot.send_photo(message.chat.id, final_image,)

    # remove file that was sent by user
    final_image.close()
    os.remove(filename)
    os.remove(pred_img_dir)
    os.rmdir(SAVE_DIR)

bot.infinity_polling()
