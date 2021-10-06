from flask import Flask,render_template,request,redirect
from TableManager import TableManager

def is_number(string):
    try:
        float(string)
        return True
    except ValueError:
        return False

app= Flask(__name__)

table_manager=TableManager("static/data/filter_importances.csv")

@app.route("/")
def login():
	return render_template("index.html",table=None,fmin=0.0)

@app.route("/check/",methods=["POST","GET"])
@app.route("/<low_freq>/<high_freq>/<delta_freq>/check/",methods=["POST","GET"])
def check(low_freq=0,high_freq=1000.0,delta_freq=1000.0):
	if request.method=="POST":
		form_data = request.form
		print(form_data)
		if is_number(form_data["low_freq"]):
			low_freq=form_data["low_freq"]
		if is_number(form_data["high_freq"]):
			print(form_data["high_freq"])
			high_freq=form_data["high_freq"]
		else:
			print("fail")
			print(form_data["high_freq"])
		if is_number(form_data["delta_freq"]):
			delta_freq=form_data["delta_freq"]
		target=form_data["target"]
		return redirect(f"/{low_freq}/{high_freq}/{delta_freq}/{target}")
	else:
		return redirect("/")

@app.route("/<low_freq>/<high_freq>/<delta_freq>/<target>")
def search_result(low_freq,high_freq,delta_freq,target):
	result_table=table_manager.get_filtered_table(low_freq,high_freq,delta_freq,target)
	return render_template("index.html",table=result_table.to_html(classes="general-table",index=False),fmin=low_freq)

if __name__ == "__main__":
    app.run()