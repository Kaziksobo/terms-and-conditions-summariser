from flask import Flask, request, render_template
from summary_functions import ab_summary, ex_summary

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summary', methods=['POST'])
def summarise():
    text_input = request.form['text']
    

    summary_a = ab_summary(text_input)
    # summary_e = ex_summary(text_input)
    summary_e = None

    return render_template('summary.html', summary_a=summary_a, summary_e=summary_e)

if __name__ == '__main__':
    app.run(debug=True)