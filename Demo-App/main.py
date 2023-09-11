import os

from werkzeug import run_simple
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import event
import segmentation


@app.route('/')
def upload_form():
	return render_template('upload.html')


@app.route('/results', methods=['GET','POST'])
def upload_video():
	if 'file' not in request.files:
		flash('Brak pliku bądź rozwiązania')
		return redirect(request.url)
	file = request.files['file']
	selected_option = request.form.get('dropdown')

	if file.filename == '':
		flash('No image selected for uploading')
		return redirect(request.url)
	else:
		filename = secure_filename(file.filename)
		file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
	
		if selected_option == 'move':
			current_res = 1
			result_num = 2

			file, file_extension =  os.path.splitext(f'{filename}')
			px = segmentation.test_segmentation(filename)

			filename = f'{file}_segmented.mp4'
			res = [px]
		elif selected_option == 'event':
			current_res = 2
			result_num = 1

			transnet_results = event.predict_transnetv2(filename)
			scenedetect_results = event.predict_scenedetect(filename)

			file, file_extension =  os.path.splitext(f'{filename}')
			res = [transnet_results, scenedetect_results, file]

		return render_template('results.html', filename=filename, curr=current_res, result=result_num, res=res)


@app.route('/results/current_res=<current_res>/filename=<filename>', methods=['GET', 'POST'])
def get_new(current_res, filename):
	if current_res == '1':
		transnet_results = event.predict_transnetv2(filename)
		scenedetect_results = event.predict_scenedetect(filename)

		file, file_extension =  os.path.splitext(f'{filename}')
		res = [transnet_results, scenedetect_results, file]

		return render_template('results.html', filename=filename, curr=2, result=current_res, res=res)
	elif current_res == '2':
		file, file_extension =  os.path.splitext(f'{filename}')
		res = [[121212, 678],[787, 678], file]

		return render_template('results.html', filename=filename, curr=1, result=current_res, res=res)


@app.route('/display/<filename>')
def display_video(filename):
	# print('display_video filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/static/graphs/<filename>.mp4/mode=<mode>')
def display_graph(filename, mode):
	file, file_extension =  os.path.splitext(f'{filename}')

	if mode == "transnet":
		return redirect(url_for('static', filename=f'graphs/{file}_transnet.jpg'), code=301)
	else:
		return redirect(url_for('static', filename=f'graphs/{file}.jpg'), code=301)
	

if __name__ == "__main__":
    app.run()
    run_simple("localhost", 5000, app)