from __future__ import absolute_import, print_function

from tweepy.streaming import StreamListener
from tweepy import OAuthHandler
from tweepy import Stream
import tweepy
import json
from pymongo import MongoClient
import datetime

# Your credentials go here


class AvengersListener(StreamListener):

    def on_error(self, status_code):
        if status_code == 420:
            return False
        else:
            print('ERROR:' + repr(status_code))
            return True

    def on_data(self, raw_data):
        status = json.loads(raw_data)
        try:
            if 'delete' not in status:  # Tweepy también detecta cuando se ha eliminado un tweet
                if status['geo']:
                    created_at = status['created_at']
                    created_at = datetime.datetime.strptime(created_at, '%a %b %d %H:%M:%S +0000 %Y')
                    user_name = status['user']['screen_name']
                    text = status['text']
                    lat = str(status['coordinates']['coordinates'][1])
                    lon = str(status['coordinates']['coordinates'][0])
                    rts = status['retweet_count']
                    favs = status['favorite_count']
                    lang = status['user']['lang']
                    print(status['text'])
                    client = MongoClient('localhost', 27017)
                    db = client['Tweets']
                    collection = db['avengers']
                    tweet = {'date': created_at, 'user': user_name, 'tweet': text,
                             'latitude': lat, 'longitude': lon, 'language': lang, 'retweets':rts, 'favourites':favs}
                    collection.insert_one(tweet)
        except BaseException as e:
            print("Error on_data: %s" % str(e))


if __name__ == '__main__':

    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth)
    print(api.me().name)
    #
    # places = api.geo_search(query="USA", granularity="country")
    # place_id = places[0].id
    # tweets = api.search(q="place:%s" % place_id)

    av_stream = Stream(auth, AvengersListener())
    av_stream.filter(track=['#avengersendgame','#endgame'])
