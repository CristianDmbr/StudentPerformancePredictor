from flask import Flask, redirect, url_for

app = Flask(__name__)

@app.route("/")
def welcome():
    return ("The main page")

@app.route("/success/<int:score>")
def success(score):
    return "<html><body><h1>The student has passed </h1></body></html>"

@app.route("/fail/<int:score>")
def fail(score):
    return "<html><body><h1> The student has failed </h1></body></html>"

@app.route("/amazing/<int:score>")
def amazing(score):
    return "<html><body><h1> The student has smashed the exam </h1></body></html>" 


# Results checker
@app.route("/results/<int:marks>")
def results(marks):
    result =""
    if marks < 50:
        result = "fail"
    elif marks < 90 :
        result = "success"
    else :
        result = "amazing"
    return redirect(url_for(result,score=marks))

if __name__=='__main__':
    app.run(debug=True)