from flask import Flask
from flask import request
from flask import render_template
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
import operator
import pandas as pd

stop_words = ['i','me','my','myself','we','our','ours','ourselves','you',"you're","you've","you'll","you'd",'your','yours','yourself',
 'yourselves','he','him','his','himself','she',"she's",'her','hers','herself','it',"it's",'its','itself','they',
 'them','their','theirs','themselves','what','which','who','whom','this','that',"that'll",'these','those','am',
 'is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but',
 'if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through',
 'during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again',
 'further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most',
 'other','some','such','no','nor','not','only','own','same','so','than','too','very','s','t','can','will','just',
 'don',"don't",'should',"should've",'now','d','ll','m','o','re','ve','y','ain','aren',"aren't",'couldn',"couldn't",
 'didn',"didn't",'doesn',"doesn't",'hadn',"hadn't",'hasn',"hasn't",'haven',"haven't",'isn',"isn't",'ma','mightn',
 "mightn't",'mustn',"mustn't",'needn',"needn't",'shan',"shan't",'shouldn',"shouldn't",'wasn',"wasn't",'weren',
 "weren't",'won',"won't",'wouldn',"wouldn't",'would','said']

file = open("tfidf_final.pickle",'r')
tfidf_fit = pickle.load(file)

file = open("model_final.pickle", 'r')
model = pickle.load(file)

file = open("tfidf_vectorizer_final.pickle", 'r')
tfidf_vectorizer = pickle.load(file)

odds = [str(num) for num in range(1,51,2)]

app = Flask(__name__)

@app.route('/')
def my_form():
	return render_template("index.html")

@app.route('/', methods=['POST'])
def my_form_post():
	first_odd = []
	text = request.form['text']
	if not [x.lower() for x in text.split() if x.lower() not in stop_words]:
		return "You only entered stop words!  Please try again."
	foo = tfidf_fit.transform([text])
	if text.split()[0] in odds:
	    first_odd.append(1)
	else:
	    first_odd.append(0)
	df = pd.DataFrame(foo.todense(), columns=[tfidf_vectorizer.get_feature_names()])
	df['first_odd'] = first_odd
	df = pd.get_dummies(df, columns=['first_odd'])
	if 'first_odd_0' in df.columns:
	    df['first_odd_1'] = 0.0
	else:
	    df['first_odd_0'] = 0.0
	probs = model.predict_proba(df.iloc[0])[0]
	result1 = "<h1>Probability that this is clickbaity is %f </h1><br/>" %probs[1]
	if probs[1] > probs[0]:
		result2 = "<h1>This is probably not important.  Come back to it later?</h1>"
	else:
		result2 = "<h1>This is probably news you might be interested in learning more about!</h1>"


	return result1 + result2

if __name__ == '__main__':
    app.run(debug=True)