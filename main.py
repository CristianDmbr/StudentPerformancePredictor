from flask import Flask, redirect, url_for, render_template, request



app = Flask(__name__)

@app.route("/")
def welcome():
    return render_template("index.html")

@app.route("/success/<int:score>")
def success(score):
    res = ""
    if score >=50 :
        res = "PASS"
    else : 
        res = "FAIL"
    exp={"score" : score, "res" : res}
    return render_template("result.html", result = exp)


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

@app.route("/submit", methods = ["Post", "GET"])
def submit():
    total_score = 0
    if request.method == "POST":
        science = float(request.form['science'])
        maths = float(request.form['maths'])
        c = float(request.form['c'])
        data_science = float(request.form['datascience'])
        total_score = (science + maths + c + data_science + total_score) / 4
    return redirect(url_for("success",score = total_score))

if __name__=='__main__':
    app.run(debug=True)