import os
from app import app
import urllib.request
from werkzeug.utils import secure_filename
from flask import Flask, flash, request, redirect, url_for, render_template

from prediction import test

ALLOWED_EXTENSIONS = set(["dcm"])

def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
	
@app.route('/')
def upload_form():
	file_names = ["a.jpg","b.jpg","c.jpg"]
	return render_template('home.html',title="Thành viên", name1 ="Lưu Minh Quân", name2 ="Trần Tấn Thành", name3 ="Trần Hoàng Vũ" ,filenames=file_names )

@app.route('/', methods=['POST'])
def upload_image():
	file = request.files['file']
	if file and allowed_file(file.filename):
		filename = secure_filename(file.filename)
		nameFile = filename[:filename.rfind('.')]
		file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
		a,b,c = test(file_path)
		file_names = [nameFile+a,nameFile+b,nameFile+c]
	print(file_names)
	return render_template('home.html',title="Kết quả", name1 ="Image", name2 ="Segmentation", name3 ="Image_Segmentation", filenames=file_names)

@app.route('/display/<filename>')
def display_image(filename):
	print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run(debug=True)