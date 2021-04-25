from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap 
import pandas as pd 

#para facilitar a manipulação com números e ser utilizado com o Naive Bayes
from sklearn.feature_extraction.text import CountVectorizer

#carregar o modelo
from sklearn.externals import joblib

app = Flask(__name__)
Bootstrap(app)


@app.route('/')
def index():
	return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
	df= pd.read_csv("data/nomes.csv")
	# Atributos e labels
	df_X = df.name
	df_Y = df.sex
    
    # vetorização dos dados e transformação em matriz de frequência. Retorna a frequência
	nomes = df_X
	cv = CountVectorizer()
	X = cv.fit_transform(nomes) 
	
	# Carregando modelo 
	naivebayes_model = open("models/naivebayesmodel.pkl","rb")
	clf = joblib.load(naivebayes_model)
	
	# Recebe a query de entrada do form
	if request.method == 'POST':
		namequery = request.form['namequery']
		data = [namequery]
		vect = cv.transform(data).toarray()
		my_prediction = clf.predict(vect)
	return render_template('results.html',prediction = my_prediction,name = namequery.upper())


if __name__ == '__main__':
	app.run(debug=True)
