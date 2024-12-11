from flask import Blueprint, render_template, request, redirect, url_for, flash
from werkzeug.utils import secure_filename
import csv
from io import StringIO
from .forms import FileUploadForm
from .model import CRISPRPredictor
from config import Config

main = Blueprint('main', __name__)
predictor = CRISPRPredictor(Config.MODEL_PATH, Config.TOKENIZER_PATH)

@main.route('/', methods=['GET', 'POST'])
def index():
    form = FileUploadForm()
    if form.validate_on_submit():
        try:
            file = form.file.data
            content = file.read().decode('utf-8')
            
            results = []
            for line in StringIO(content):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        target_seq, offtarget_seq, _ = line.split(',')
                        # Predict off-target risk
                        risk_probability = predictor.predict(target_seq, offtarget_seq)
                        results.append({
                            'target_seq': target_seq,
                            'offtarget_seq': offtarget_seq,
                            'risk_prob': risk_probability
                        })
                    except Exception as e:
                        print(f"Error processing line: {line}")
                        print(f"Error: {str(e)}")
                        continue
            
            if not results:
                flash("No valid predictions could be made from the uploaded file.", "warning")
                return redirect(url_for('main.index'))
                
            return render_template('results.html', results=results)
            
        except Exception as e:
            flash(f"Error processing file: {str(e)}", "error")
            return redirect(url_for('main.index'))
    
    return render_template('index.html', form=form)
    # return render_template('index.html', form=form)