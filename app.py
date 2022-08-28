#!/usr/bin/env python3

"""
@author: Jithin Sasikumar

Script to create a web application that wraps the trained model to be used for inference using
`FLASK API`. It facilitates the application to run from a server which defines every routes and
functions to perform. The front-end is designed using `./templates/page.html` and its styles in
`./static/page.css`

Note:
    Make sure to define all the variables and valid paths in `.config_dir/config.yaml` to run
    this script without errors and issues.
"""

from flask import Flask, render_template, request, flash, abort
from omegaconf import OmegaConf
from src import data
from src.inference import KeywordSpotter
from src.exception_handler import NotFoundError

app = Flask(__name__)
app.config["SECRET_KEY"] = "MyKWSAppSecretKey"
cfg = OmegaConf.load('./config_dir/config.yaml')

@app.route('/')
def home():
    """
    Returns the result of calling render_template() with page.html
    """
    return render_template('page.html')

@app.route("/transcribe", methods = ["POST"])
def transcribe():
    """
    Returns the prediction from trained model artifact whenever transcribe route is called.
    It accepts file input (.wav) whenever user uploads the file, and make prediction using it.
    The `app.route()` decorator does the job of event handling by means of `jinja2` template
    engine.

    Raises
    ------
    NotFoundError: Exception
        404 error, if any exception occurs.
    """

    recognized_keyword = ""
    if request.method == "POST":
        audio_file = request.files["file"]
        if audio_file.filename == "":
            flash("File not found !!!", category="error")
            return render_template("page.html")
       
        elif not data.check_fileType(filename=audio_file.filename, extension=".wav"):
            flash("Unsupported file format. Please use only .wav files", category="error")
            return render_template("page.html")

        else:
            try:
                recognizer = KeywordSpotter(audio_file,
                                            cfg.paths.model_artifactory_dir,
                                            cfg.params.n_mfcc,
                                            cfg.params.mfcc_length,
                                            cfg.params.sampling_rate)
                recognized_keyword, label_probability = recognizer.predict()

            except NotFoundError:
                abort(404, description = "Sorry, something went wrong. Cannot predict from the model. Please try again !!!")

    return render_template(
                "page.html",
                 recognized_keyword = f"Transcribed keyword: {recognized_keyword.title()}",
                 label_probability = f"Predicted probability: {label_probability}"
                 )

if __name__ == "__main__":
    app.run(debug=False)