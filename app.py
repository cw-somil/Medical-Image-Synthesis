from flask import Flask, render_template, request, send_from_directory
import os
from predict import *

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

model = load_model()

global graph
graph = tf.get_default_graph()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/form")
def form():
    return render_template("form.html")


@app.route("/upload", methods=['POST'])
def upload():
    target = os.path.join(APP_ROOT, 'images/')
    print(target)
    if not os.path.isdir(target):
        os.mkdir(target)

#	for file in request.files.getlist("file"):
#		print(file)
#		filename = file.filename
#		destination = "/".join([target,filename])
#		print(destination)
#		file.save(destination)

    file = request.files['file']
    filename = file.filename
    destination = "/".join([target, filename])
    file.save(destination)
    with graph.as_default():
        outname = predict(filename, model)

    return render_template("complete.html", prev_name=filename, res_name=outname)


@app.route('/upload/<filename>')
def prev_img(filename):
    return send_from_directory('images', filename)


@app.route('/res/<filename>')
def res_img(filename):
    return send_from_directory('results', filename)


if __name__ == "__main__":
    app.run(port=4555, debug=True)
