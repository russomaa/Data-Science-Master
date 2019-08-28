# -*- coding: utf-8 -*-
"""
RUBEBOT :(
"""
import pandas as pd
import numpy as np

import string
import csv
import os

from datetime import datetime, timedelta
from iexfinance.stocks import Stock
from iexfinance.stocks import get_historical_data

import seaborn as sns; sns.set()
import matplotlib.pyplot as plt
from matplotlib import cm
# from ggplot import *
# https://github.com/yhat/ggpy

# conda install -c r rpy2
from rpy2 import robjects
# from rpy2.robjects.packages import importr
# utils = importr('utils')
# utils.install_packages('tseries')
from telegram.ext import Updater, CommandHandler, CallbackQueryHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto, ReplyKeyboardMarkup, KeyboardButton
import logging
from io import BytesIO
from PIL import Image
# import PIL
# print(PIL.PILLOW_VERSION)

# RESOURCES:

# iex finance stocks:
# https://iextrading.com/apps/stocks/

# Bots:
# https://pypi.org/project/python-telegram-bot/
# https://python-telegram-bot.readthedocs.io/en/stable/
# https://telegramgeeks.com/2016/03/franz-messenger-for-telegram-whatsapp-and-other-im-services/

# ML:
# https://arxiv.org/pdf/1802.00500.pdf
# https://learningbot.chat/
# https://www.quora.com/How-is-machine-learning-applied-to-chat-bot
# https://www.td.org/insights/using-ai-chatbots-to-increase-learning-impact-and-value

### Defining tickers ####
# ticker_list = pd.read_excel("data/companies.xlsx", header = 0).iloc[:,0].values[0:20]
ticker_list = ["BPL", "CDLX", "VFF","YELP", "PBYI", "TRXC"]
# id = range(len(ticker_list))
duration = 365*2
prediction_file = 'data/prediction.csv'
# start = datetime.now() - timedelta(days=duration)
# end = datetime.now()#.strftime('%Y-%m-%d')
# # creating dictionary of data
# df = get_historical_data(ticker_list, start, end, output_format='pandas')
# idx = pd.IndexSlice
# df = df.loc[:,idx[:,"close"]]
# df.columns = ticker_list


### FUNCTIONS ###
def prediction_generator(investor, prediction_file = 'data/prediction.csv', horizon = 5):
    """
    Main function that will generate predictions getting data from IEXFINANCE
    """
    start = datetime.now() - timedelta(days=duration)
    end = datetime.now()
    # get data from IEXFINANCE, import data to a panda's dataframe
    df = get_historical_data(ticker_list, start, end, output_format='pandas')
    idx = pd.IndexSlice
    df = df.loc[:,idx[:,"close"]]
    df.columns = ticker_list
    hist_data = df.iloc[:,].to_csv("data/hist_data.csv") # columns must be selected
    robjects.r('''
        library(astsa)
        library(forecast)
        library(seasonal)
        library(tseries)
        library(readr)

        hist_data <- read_csv("data/hist_data.csv", col_names = TRUE)
        data <- as.ts(hist_data[,-1],start=1,frequency = 365)
        cn<-colnames(data)
        ns <- ncol(data)
        h=5
        fcast <- matrix(NA,nrow=h,ncol=ns)
        prediction <- sapply(data, function(x) forecast(auto.arima(x,seasonal = TRUE),h=h))

        for (i in 1:ns) fcast[,i] <- prediction[,i]$mean %>% as.vector()
        write.table(x = fcast, file = "data/prediction.csv",row.names = FALSE,col.names=cn,sep=",")
    '''
    )
    # importing predictions
    predictions = pd.read_csv(prediction_file, header = 0, delimiter = ",")
    # measuring growth to differentiate between risky and non-risky predictions
    new_df = df.iloc[df.shape[0]-1:,].reset_index(drop = True).append(predictions.iloc[:1,:].reset_index(drop = True)).reset_index(drop = True)
    growth = pd.DataFrame()
    for i in range(new_df.shape[1]):
        growth[ticker_list[i]] = [((new_df[ticker_list[i]].values[1]-new_df[ticker_list[i]].values[0])/new_df[ticker_list[i]].values[1])]
    growth = growth.iloc[0].sort_values(ascending = False)
    # differentiate ticker lists for risky (a) and non risky investors (b)
    ticker_list_a = growth.index[0:3]
    ticker_list_b = growth.index[3:]
    # OUTPUT
    if investor == "risky":
        predictions = predictions.loc[:, ticker_list_a]
        # saving a txt with text to be printed to the client
        with open('data/growth_info.txt', 'w', encoding = "utf-8") as f:
            print(*["Stock growth:\n" +" "+ ticker_list_a[i] + ": " + str(growth[ticker_list_a[i]]*100) + "%\n " if i == 0 else  ticker_list_a[i] + ":" + str(growth[ticker_list_a[i]]*100) + "%\n" for i in range(len(ticker_list_a))], file = f)
        # importing the txt file again to an object to be printed to the client
        with open("data/growth_info.txt","r", encoding = "utf-8") as f_open:
            growth_info = f_open.read()
        # generating text to print predictions to the client
        values_list = []
        for k in range(len(ticker_list_a)):
            values = ""
            # setting prediction horizon (5 days)
            days = ["Day "+str(i+1)+": €" for i in range(horizon)]
            for i in range(predictions.shape[0]):
                if i != predictions.shape[0] - 1:
                    values += days[i] + str(np.round(predictions.iloc[i,k], 2)) + ", "
                else:
                    values += days[i] + str(np.round(predictions.iloc[i,k], 2)) + "."
            values_list.append(values)
        # saving a txt with text to be printed to the client
        with open('data/predition_info.txt', 'w', encoding = "utf-8") as f:
            print(*[ticker_list_a[i] + "\n" + values_list[i] + "\n" if i==0 else "\n" +ticker_list_a[i] + "\n" + values_list[i] + "\n" for i in range(len(values_list))], file=f, )
        # importing the txt file again to an object to be printed to the client
        with open("data/predition_info.txt","r", encoding = "utf-8") as f_open:
            predicion_info = f_open.read()

    elif investor == "non-risky":
        # saving a txt with text to be printed to the client
        with open('data/growth_info.txt', 'w', encoding = "utf-8") as f:
            print(*["Stock growth:\n" +" "+ ticker_list_b[i] + ": " + str(growth[ticker_list_b[i]]*100) + "%\n" if i == 0 else  ticker_list_b[i] + ":" + str(growth[ticker_list_b[i]]*100) + "%\n" for i in range(len(ticker_list_b))], file = f)
        # importing the txt file again to an object to be printed to the client
        with open("data/growth_info.txt","r", encoding = "utf-8") as f_open:
            growth_info = f_open.read()
        # generating text to print predictions to the client
        values_list = []
        predictions = predictions.loc[:, ticker_list_b]
        for k in range(len(ticker_list_b)):
            values = ""
            # setting prediction horizon (5 days)
            days = ["Day "+str(i+1)+": €" for i in range(horizon)]
            for i in range(predictions.shape[0]):
                if i != predictions.shape[0] - 1:
                    values += days[i] + str(np.round(predictions.iloc[i,k], 2)) + ", "
                else:
                    values += days[i] + str(np.round(predictions.iloc[i,k], 2)) + "."
            values_list.append(values)
        # saving a txt with text to be printed to the client
        with open('data/predition_info.txt', 'w', encoding = "utf-8") as f:
            print(*[ticker_list_b[i] + "\n" + values_list[i] + "\n" if i==0 else "\n" +ticker_list_b[i] + "\n" + values_list[i] + "\n" for i in range(len(values_list))], file=f, )
        # importing the txt file again to an object to be printed to the client
        with open("data/predition_info.txt","r", encoding = "utf-8") as f_open:
            predicion_info = f_open.read()

    return growth_info + predicion_info


### BOT FUNCTIONS ###
logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                    level=logging.INFO)
logger = logging.getLogger(__name__)

# create userkeyboard, resize = true, autohide=false
keyboard_p1 = [[KeyboardButton("/help"), KeyboardButton("/stockstatus")],
               [KeyboardButton("/alarm"), KeyboardButton("/unset")],
               [KeyboardButton("/startinvesting")]]
reply_markup_p1 = ReplyKeyboardMarkup(keyboard_p1, True, False)

def help(bot, update):
    """
    Basic function to introduce the chat-bot to the client
    """
    with open("data/presentation.txt","r") as f_open:
        data = f_open.read()
    aux = "Hello {}".format(update.message.from_user.first_name)
    data = aux + data
    update.message.reply_text(data, reply_markup=reply_markup_p1)

def start_investing(bot, update):
    """
    This function will allow our bot to know the investment preferences of our clients
    """
    keyboard = [[InlineKeyboardButton("I'm a risky investor today📈", callback_data = "1"),
                InlineKeyboardButton("I'm not willing to take risks💆‍♂️", callback_data='2')]]
    reply_markup1 = InlineKeyboardMarkup(keyboard)
    update.message.reply_text('What kind of investor are you today?:', reply_markup=reply_markup1)

def stockstatus(bot, update):
    """
    Function that prints current stock status of available tickers
    """
    keyboard = [[InlineKeyboardButton(ticker_list[i], callback_data = ticker_list[i]) for i in range(len(ticker_list))]]
    # keyboard = build_menu(keyboard, n_cols = 2)
    reply_markup2 = InlineKeyboardMarkup(keyboard)
    update.message.reply_text('Please, select your ticker', reply_markup=reply_markup2)

def alarm(bot, job):
    """Send the alarm message."""
    with open("data/presentation_alarm.txt","r") as f_open:
        data = f_open.read()
    message = "Hi!, it’s time to check your portfolio💼"  + "\n" + data
    bot.send_message(job.context, text = message)

def set_timer(bot, update, args, job_queue, chat_data):
    """Add a job to the queue."""
    chat_id = update.message.chat_id
    try:
        # args[0] should contain the time for the timer in seconds
        due = int(args[0]) #60*60 time in hours
        if due < 0:
            update.message.reply_text('Sorry we can not go back to future!')
            return

        # Add job to queue
        job = job_queue.run_once(alarm, due, context=chat_id)
        chat_data['job'] = job

        update.message.reply_text('Alarm successfully set!')

    except (IndexError, ValueError):
        update.message.reply_text('Usage: /set <hours>')

def unset(bot, update, chat_data):
    """Remove the job if the user changed their mind."""
    if 'job' not in chat_data:
        update.message.reply_text('You have no active alarm, please use /set to activate an alarm')
        return

    job = chat_data['job']
    job.schedule_removal()
    del chat_data['job']

    update.message.reply_text('Timer successfully unset!')

def button(bot, update):
    """
    Main function controlling bot-client interactions
    """
    query = update.callback_query
    if str(query.data) == "1":
        # collecting forecast
        data = "These are my forecasts for a risky investor\n" + prediction_generator(investor = "risky")
        # plotting the best asset
        predictions = pd.read_csv(prediction_file, header = 0, delimiter = ",")
        # measuring growth to differentiate between risky and non-risky predictions
        start = datetime.now() - timedelta(days=duration)
        end = datetime.now()#.strftime('%Y-%m-%d')
        df = get_historical_data(ticker_list, start, end, output_format='pandas')
        idx = pd.IndexSlice
        df = df.loc[:,idx[:,"close"]]
        df.columns = ticker_list
        new_df = df.iloc[df.shape[0]-1:,].reset_index(drop = True).append(predictions.iloc[:1,:].reset_index(drop = True)).reset_index(drop = True)
        growth = pd.DataFrame()
        for i in range(new_df.shape[1]):
            growth[ticker_list[i]] = [((new_df[ticker_list[i]].values[1]-new_df[ticker_list[i]].values[0])/new_df[ticker_list[i]].values[1])]
        growth = growth.iloc[0].sort_values(ascending = False)
        # taking the best asset to provide an image with forecasts
        ticker_list_a = growth.index[0] #taking the riskiest predicion
        new_df = df.iloc[df.shape[0]-30:,].reset_index()
        new_df = new_df.loc[:, ["date", ticker_list_a]]
        new_dates = [new_df.date.values[len(new_df.date.values)-1].astype('M8[D]').astype('O') + timedelta(days=i+1) for i in range(5)]
        # new_df["date"] = new_df["date"].dt.strftime("%Y-%m-%d")
        predictions = predictions.loc[:,[ticker_list_a]]
        predictions["date"] = new_dates
        # ploting the image
        sns.set_style("darkgrid")
        fig = plt.figure(figsize=(8,8))
        plt.plot(new_df["date"].values, new_df[ticker_list_a].values, linestyle='--', marker='o')
        plt.plot(predictions["date"].values, predictions[ticker_list_a].values, linestyle='--', marker='o',color='r')
        plt.suptitle("Evolution of the recommended asset", fontsize=16)
        plt.title("Ticker name: " + ticker_list_a + "\n 30 days + 5 days forcasts (in red)", fontsize=14)
        plt.xlabel("Time")
        plt.ylabel("Stock price (€)")
        # plt.axis('equal')
        fig.savefig('data/forecast_image_risky.jpeg')
        plt.close(fig)

        bio = BytesIO()
        bio.name = 'data/forecast_image_risky.jpeg'
        im =  Image.open("data/forecast_image_risky.jpeg")
        im.save(bio, 'JPEG')
        bio.seek(0)

        # sending messages to client
        bot.send_photo(chat_id = query.message.chat_id, photo = bio)
        bot.edit_message_text(text=data,
                          chat_id=query.message.chat_id,
                          message_id=query.message.message_id)


    if str(query.data) == "2":
        data = "These are my predictions for a risk-averse stupid investor\n" + prediction_generator(investor = "non-risky")
        # plotting the best asset
        predictions = pd.read_csv(prediction_file, header = 0, delimiter = ",")
        # measuring growth to differentiate between risky and non-risky predictions
        start = datetime.now() - timedelta(days=duration)
        end = datetime.now()
        df = get_historical_data(ticker_list, start, end, output_format='pandas')
        idx = pd.IndexSlice
        df = df.loc[:,idx[:,"close"]]
        df.columns = ticker_list
        new_df = df.iloc[df.shape[0]-1:,].reset_index(drop = True).append(predictions.iloc[:1,:].reset_index(drop = True)).reset_index(drop = True)
        growth = pd.DataFrame()
        for i in range(new_df.shape[1]):
            growth[ticker_list[i]] = [((new_df[ticker_list[i]].values[1]-new_df[ticker_list[i]].values[0])/new_df[ticker_list[i]].values[1])]
        growth = growth.iloc[0].sort_values(ascending = False)
        # taking the best asset to provide an image with forecasts
        ticker_list_b = growth.index[4] #taking the best risk-averse asset??
        new_df = df.iloc[df.shape[0]-30:,].reset_index()
        new_df = new_df.loc[:, ["date", ticker_list_b]]
        new_dates = [new_df.date.values[len(new_df.date.values)-1].astype('M8[D]').astype('O') + timedelta(days=i+1) for i in range(5)]
        # new_df["date"] = new_df["date"].dt.strftime("%Y-%m-%d")
        predictions = predictions.loc[:,[ticker_list_b]]
        predictions["date"] = new_dates
        # ploting the image
        sns.set_style("darkgrid")
        fig = plt.figure(figsize=(8,8))
        plt.plot(new_df["date"].values, new_df[ticker_list_b].values, linestyle='--', marker='o')
        plt.plot(predictions["date"].values, predictions[ticker_list_b].values, linestyle='--', marker='o',color='r')
        plt.suptitle("Evolution of the recommended asset", fontsize=16)
        plt.title("Ticker name: " + ticker_list_b + "\n 30 days + 5 days forcasts (in red)", fontsize=14)
        plt.xlabel("Time")
        plt.ylabel("Stock price (€)")
        # plt.axis('equal')
        fig.savefig('data/forecast_image_risky.jpeg')
        plt.close(fig)

        bio = BytesIO()
        bio.name = 'data/forecast_image_risky.jpeg'
        im =  Image.open("data/forecast_image_risky.jpeg")
        im.save(bio, 'JPEG')
        bio.seek(0)

        # sending messages to client
        bot.send_photo(chat_id = query.message.chat_id, photo = bio)
        bot.edit_message_text(text=data,
                          chat_id=query.message.chat_id,
                          message_id=query.message.message_id)

    # stckstatus images
    for k in range(len(ticker_list)):
        # when query.data == ticker_list the query comes from stockstatus
        if str(query.data) == ticker_list[k]:
            start = datetime.now() - timedelta(days=30)
            end = datetime.now()#.strftime('%Y-%m-%d')
            # creating dictionary of data
            df = get_historical_data(str(query.data), start, end, output_format='pandas')
            df = df.loc[:,["close"]]

            sns.set_style("darkgrid")
            fig = plt.figure(figsize=(8,8))
            plt.plot(df.index, df.close.values, linestyle='--', marker='o')
            plt.title("Evolution of " + str(query.data) + "\n 30 days", fontsize=14)
            plt.xlabel("Time")
            plt.ylabel("Stock price (€)")
            plt.xticks(rotation=45)
            plt.axis('equal')
            fig.savefig('aux_image.jpeg')
            plt.close(fig)

            bio = BytesIO()
            bio.name = 'aux_image.jpeg'
            im =  Image.open("aux_image.jpeg")
            im.save(bio, 'JPEG')
            bio.seek(0)
            bot.send_photo(chat_id = query.message.chat_id, photo = bio)
            # update.message.reply_photo(query.message.chat_id, 'https://ih1.redbubble.net/image.59574941.7749/fc,550x550,grass_green.u7.jpg')
            bot.edit_message_text(text="Selected ticker: {}".format(query.data),
                              chat_id=query.message.chat_id,
                              message_id=query.message.message_id)


def error(bot, update, error):
    """
    Log Errors caused by Updates.
    """
    logger.warning('Update "%s" caused error "%s"', update, error)



def main():
    """
    Run the bot
    """
    updater = Updater("INSERT YOUR TOKEN HERE")
    updater.dispatcher.add_handler(CommandHandler("help", help))
    updater.dispatcher.add_handler(CommandHandler("start", help))
    updater.dispatcher.add_handler(CommandHandler("startinvesting", start_investing))
    updater.dispatcher.add_handler(CommandHandler("stockstatus", stockstatus))
    updater.dispatcher.add_handler(CommandHandler("set", set_timer,
                                    pass_args=True,
                                    pass_job_queue=True,
                                    pass_chat_data=True))
    updater.dispatcher.add_handler(CommandHandler("alarm", set_timer,
                                    pass_args=True,
                                    pass_job_queue=True,
                                    pass_chat_data=True))
    updater.dispatcher.add_handler(CommandHandler("unset", unset, pass_chat_data=True))
    updater.dispatcher.add_handler(CallbackQueryHandler(button))
    # log all errors
    updater.dispatcher.add_error_handler(error)
    # Start the bot
    updater.start_polling()
    updater.idle()

if __name__ == '__main__':
    main()




# save(self, filename, width=None, height=None, dpi=180)
# p = ggplot(aes(x = "date", y = "MX"), data = df )+geom_line()+labs(x = "HOla", y = "LALA", title = "jajajaja")
# p.save("prueba.png")


# new_df = df.iloc[df.shape[0]-30:,].reset_index()
# new_dates = [new_df.date.values[len(new_df.date.values)-1].astype('M8[D]').astype('O') + timedelta(days=i+1) for i in range(5)]
# predictions["date"] = new_dates
# new_df = new_df.append(predictions).reset_index()
# ggplot(aes(x = "date", y = "ENPH"),data = new_df)+geom_line()+geom_line(aes(x = "date", y = "ENPH", color = "red"), data = predictions)
