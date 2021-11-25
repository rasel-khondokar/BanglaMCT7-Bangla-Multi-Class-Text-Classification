from flask import Flask, request, jsonify, render_template

from training.prediction import get_category

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():

    article = request.form['news']
    if article:
        try:
            output = get_category(article)
        except Exception as e:
            output = 'Something went wrong!'
    else:
        output = 'নিউজটি বাংলায় লিখুন!'


    return render_template('index.html', prediction_text=output)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8888)
