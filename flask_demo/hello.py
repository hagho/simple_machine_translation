from flask import Flask
from flask import render_template


app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/demo/')
def demo():
    return render_template('demo.html')


if __name__ == '__main__':
    app.run(host = '0.0.0.0', debug = True)