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