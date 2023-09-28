"""
Core of the DEMO-app
Marta Dzięgielewska
"""


from flask import Flask

UPLOAD_FOLDER = '/media/madziegielewska/Seagate Expansion Drive/Diploma-Project/Demo-App/static/uploads'

app = Flask(__name__)

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024