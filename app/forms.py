from flask_wtf import FlaskForm
from flask_wtf.file import FileField, FileRequired, FileAllowed
from wtforms import SubmitField

class FileUploadForm(FlaskForm):
    file = FileField('Upload Dataset (.txt)', 
                    validators=[
                        FileRequired(),
                        FileAllowed(['txt'], 'Text files only!')
                    ])
    submit = SubmitField('Upload and Predict')
