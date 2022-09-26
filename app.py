import tweepy
import pandas as pd
import numpy as np
from flask import Flask,render_template,request,redirect,Response, send_file, make_response, url_for
from flask.wrappers import Request
import pandas as pd
from twilio.rest import Client
import plotly.graph_objs as go
import io
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import plotly.graph_objs as go
import plotly.express as px
import text2emotion as te
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import seaborn as sns


API_KEY="eM8GCF0S3xZj0AmEsMsrjB9kZ"
API_SECRET="GWN4IWrAGnQULkOcTDQeiAuztl9a23su8pDIt8rlwteq8oVgey"
BEARER_TOKEN="AAAAAAAAAAAAAAAAAAAAAJmcgQEAAAAAHgYDxP4tImA1MRPb%2FBmnyhQPtXA%3Dz0YQvG5Iei2r8asfdxlclfMkM4X5dbeCCJKTkASw0khA8PB9Vs"
ACCESS_TOKEN="1562759237326934016-vkLba69MYj0bxFOdYVXwlqDL9RAPMB"
ACCESS_TOKEN_SECRET="pDwezo2u6ZAsAinPsLpsR0QrccK2R3vVbyNuYbUka0kkE"

# authentication
auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)

api = tweepy.API(auth)

public_tweets = api.home_timeline()


# create dataframe
columns = ['id','Time', 'User', 'Tweet','Total Likes','Did you retweet','Did I like','Talking about Place']
data = []
for tweet in public_tweets:
    data.append([tweet.id,tweet.created_at, tweet.user.screen_name, tweet.text, tweet.favorite_count,tweet.retweeted,tweet.favorited,tweet.place])

df = pd.DataFrame(data, columns=columns)



sid = SentimentIntensityAnalyzer()



nltk.download('words')
words = set(nltk.corpus.words.words())

sentence = df['Tweet'][1]
sid.polarity_scores(sentence)['compound']

#Cleaning Tweets and creating new dataframe
def cleaner(tweet):
    tweet = re.sub("@[A-Za-z0-9]+","",tweet) #Remove @ sign
    tweet = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", tweet) #Remove http links
    tweet = " ".join(tweet.split())
    tweet = tweet.replace("#", "").replace("_", " ") #Remove hashtag sign but keep the text
    tweet = " ".join(w for w in nltk.wordpunct_tokenize(tweet)
         if w.lower() in words or not w.isalpha())
    return tweet


df['tweet_clean'] = df['Tweet'].apply(cleaner)
word_dict = {'manipulate':-1,'manipulative':-1,'jamescharlesiscancelled':-1,'jamescharlesisoverparty':-1,
            'pedophile':-1,'pedo':-1,'cancel':-1,'cancelled':-1,'cancel culture':0.4,'teamtati':-1,'teamjames':1,
            'teamjamescharles':1,'liar':-1}

import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer

sid = SentimentIntensityAnalyzer()
sid.lexicon.update(word_dict)

list1 = []
for i in df['tweet_clean']:
    list1.append((sid.polarity_scores(str(i)))['compound'])

df['sentiment'] = pd.Series(list1)

def sentiment_category(sentiment):
    label = ''
    if(sentiment>0):
        label = 'positive'
    elif(sentiment == 0):
        label = 'neutral'
    else:
        label = 'negative'
    return(label)

df['sentiment_category'] = df['sentiment'].apply(sentiment_category)

df['date'] = pd.to_datetime(df['Time']).dt.date
df['date']=df['date'].astype(str)
df['Time'] = pd.to_datetime(df['Time']).dt.time


#For Visualisation 4
neg = df[df['sentiment_category']=='negative']
neg = neg.groupby(['date'],as_index=False).mean()

pos = df[df['sentiment_category']=='positive']
#No. of users on a particular date with pos tweets
pos = pos.groupby(['date'],as_index=False).mean()


pos = pos[['date','sentiment']]
neg = neg[['date','sentiment']]
pos.rename(columns = {'sentiment':'pos_sent'}, inplace = True)
neg.rename(columns = {'sentiment':'neg_sent'}, inplace = True)

final=pd.merge(pos, neg,how='outer', on='date')



#Emotional Analysis Graph
happy=[]
angry=[]
surprise=[]
sad=[]
fear=[]

for i in range(len(df)):
    temp=te.get_emotion(df['tweet_clean'][i])
    tempVal= list(temp.values())
    if(tempVal[0]==0 and tempVal[1]==0 and tempVal[2]==0 and tempVal[3]==0 and tempVal[4]==0):
        pass
    else:
        happy.append(tempVal[0])
        angry.append(tempVal[1])
        surprise.append(tempVal[2])
        sad.append(tempVal[3])
        fear.append(tempVal[4])

    
happy=sum(happy)/(len(df)-6)
angry=sum(angry)/(len(df)-6)
surp=sum(surprise)/(len(df)-6)
sad=sum(sad)/(len(df)-6)
fear=sum(fear)/(len(df)-6)



#Routes Start From Here
app = Flask(__name__)


@app.route('/')
def hello_world():
    df1=df.head(5)
    df1=df1.drop(['id','Time','Tweet','Talking about Place','tweet_clean'], axis=1)

    #List of people following me
    followers = api.get_follower_ids()
    followers=len(followers)

    #List of people following me
    friendList = api.get_friend_ids()
    friendList=len(friendList)

    #Muted Ids
    mutedIdsCount = api.get_muted_ids()
    mutedIdsCount=len(mutedIdsCount)

    #Cards
    maxSent=df['sentiment'].max()
    minSent=df['sentiment'].min()
    avgSent=df['sentiment'].mean()
    totSent=df['sentiment'].count()


    return render_template('index.html', followers=followers,friendList=friendList,maxSent=maxSent,minSent=minSent,avgSent=avgSent,totSent=totSent,mutedIdsCount=mutedIdsCount,tables=[df1.to_html(classes='data')])

#Visualisation Renders
@app.route('/1.png')
def plot_pngFinal1():
    fig, ax = plt.subplots(figsize =(3.5,3.5))
    v=df['sentiment_category'].value_counts()
    labels =  ['Positive','Negative',"Nuetral"]
    colors = ['pink', 'silver', 'steelblue']
    explode = [0,0.1,0.1]
    wedge_properties = {"edgecolor":"k",'linewidth': 2}

    plt.pie(v, labels=labels, explode=explode, colors=colors, startangle=30,
               counterclock=False, shadow=True, wedgeprops=wedge_properties,
               autopct="%1.1f%%", pctdistance=0.7, textprops={'fontsize': 10})

    plt.title("Sentiment Percentage",fontsize=15)
    plt.legend(fontsize=12)   
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/2.png')
def plot_pngFinal2():
    fig, ax = plt.subplots(figsize =(4,4))
    #Here if any of the emotions is higher than a certain threshold for any tweet they can be deleted.

    y = np.array([happy, angry, surp, sad,fear])
    labels = ["Happy", "Angry", "Surprised", "Sad","Fear"]
    colors = ['#eca1a6', '#bdcebe', '#bdcebe','#ada397','#c94c4c']
    explode = [0.15,0.15,0.15,0.15,0.15]
    wedge_properties = {"edgecolor":"k",'linewidth': 2}

    plt.pie(y, labels=labels, explode=explode, colors=colors, startangle=30,
               counterclock=False, shadow=True, wedgeprops=wedge_properties,
               autopct="%1.1f%%", pctdistance=0.7, textprops={'fontsize': 10})

    plt.title("Emotion Percentage",fontsize=15)
    plt.legend(fontsize=7)      
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/3.png')
def plot_pngFinal3():
    fig, ax = plt.subplots(figsize =(7,7))
    sns.boxplot(x='User', y='sentiment', notch = True,
            data=df, showfliers=False,palette="Set2").set(title='Sentiment Score by User')
    #modify axis labels
    plt.xlabel('User')
    plt.ylabel('Sentiment Score')
    plt.xticks(rotation=90)       
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/4.png')
def plot_pngFinal4():
    fig, ax = plt.subplots(figsize =(5,5))

    X = final['date']
    posSent = final['pos_sent']
    negSent = final['neg_sent']

    plt.axhline(posSent.mean(), color='red', ls='dotted')
    plt.axhline(negSent.mean(), color='red', ls='dotted')
    plt.axhline(0, color='black')
    X_axis = np.arange(len(X))
    
    plt.bar(X_axis - 0.2, posSent, 0.4, label = 'Pos')
    plt.bar(X_axis + 0.2, negSent, 0.4, label = 'Neg')

    
    plt.xticks(X_axis, X)
    plt.xlabel("Date")
    plt.ylabel("Sentiment Score")
    plt.title("Sentiment Score by date")
    plt.legend()   
    output = io.BytesIO()
    FigureCanvas(fig).print_png(output)
    return Response(output.getvalue(), mimetype='image/png')

@app.route('/5.png')
def plot_pngFinal5():
    #fig, ax = plt.subplots(figsize =(4,4))
    positive = df[df['sentiment_category']=='positive']
    wordcloud = WordCloud(max_font_size=50, max_words=500, background_color="white").generate(str(positive['tweet_clean']))
    plt.figure() 
    plt.axis("off")
    img = io.BytesIO()
    plt.savefig(img)
    img.seek(0)
    return send_file(img, mimetype='image/png')

#Filtering Tweets
@app.route('/FilterTweets',methods=['GET','POST'])
def filterFormGet():
    if request.method=='POST':
        #Get from form
        filterForm=request.form.get('dropdownMenuButton2')
        specification=request.form['specification']

        #Function to generate twitter link
        def generateURL(User,ID):
            links=[]
            for i in range(len(User)):
                links.append(f"twitter.com/{User[i]}/status/{ID[i]}")
            return links
        
        #Emotion Filter chosen
        if(filterForm=='emotionFilter'):
            happyId=[]
            angryId=[]
            surpriseId=[]
            sadId=[]
            fearId=[]

            happyUser=[]
            angryUser=[]
            surpriseUser=[]
            sadUser=[]
            fearUser=[]

            for i in range(len(df)):
                temp=te.get_emotion(df['tweet_clean'][i])
                tempVal= list(temp.values())
                if(tempVal[0]==0 and tempVal[1]==0 and tempVal[2]==0 and tempVal[3]==0 and tempVal[4]==0):
                    pass
                else:
                    if(tempVal[0]>0):
                        happyId.append(df['id'][i])
                        happyUser.append(df['User'][i])
                    if(tempVal[1]>0):
                        angryId.append(df['id'][i])
                        angryUser.append(df['User'][i])
                    if(tempVal[2]>0):
                        surpriseId.append(df['id'][i])
                        surpriseUser.append(df['User'][i])
                    if(tempVal[3]>0):
                        sadId.append(df['id'][i])
                        sadUser.append(df['User'][i])
                    if(tempVal[4]>0):
                        fearId.append(df['id'][i])
                        fearUser.append(df['User'][i])
                    else:
                        pass
            #Filter tweets by emotion
            emotion=specification
            if(emotion=='Happy'):
                print(generateURL(happyId,happyUser))
            elif(emotion=="Angry"):
                print(generateURL(angryId,angryUser))
            elif(emotion=="Surprised"):
                print(generateURL(surpriseId,surpriseUser))
            elif(emotion=="Sad"):
                print(generateURL(sadId,sadUser))
            elif(emotion=="Fear"):
                print(generateURL(fearId,fearUser))


        elif(filterForm=='wordFilter'):
            print('2')
        elif(filterForm=='likesFilter'):
            print('3')
        elif(filterForm=='sentimentFilter'):
            print('4')
        elif(filterForm=='dateFilter'):
            print('5')
        else:
            print('Filter by date')
    return render_template('FilterTweets.html')

@app.route('/TwilioForm',methods=['GET','POST'])
def TwilioForm():
    if request.method=="POST":
        account_sid=request.form['sid']
        auth_token=request.form['token']
        twwpfrom=request.form['wpfrom']
        twwpto=request.form['wpto']
        twmsg=request.form['msg']


        client = Client(account_sid, auth_token)

        message = client.messages \
        .create(
            body=twmsg,
            from_='whatsapp:'+twwpfrom,
            to='whatsapp:'+twwpto
        )

        print(message.sid)
        

    return render_template('TwilioForm.html')

# main driver function
if __name__ == '__main__':
	app.run()
