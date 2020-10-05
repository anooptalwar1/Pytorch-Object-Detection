from detecto.core import Model
from detecto.utils import read_image
from detecto.visualize import show_labeled_image, detect_video, plot_prediction_grid
import torch
from constants import model_constants
from flask_cors import CORS
from flask import Flask, flash, request, redirect, Response, render_template, abort, jsonify, send_from_directory, send_file, safe_join, abort
import os
from controllers import model_controller, database_controller
from utilities import xml_convert, image_transform
import multiprocessing
from werkzeug.utils import secure_filename
import json
import cv2



app = Flask(__name__)
CORS(app)
app.secret_key = os.urandom(24)

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print ("running on the GPU")
else:
    device = torch.device("cpu")
    print ("running on the CPU")

torch.cuda.empty_cache()

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'bmp', 'mp4', 'avi'])



def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

""" Create user directories """
def create_user_dir(userid, project):
    userdir = os.path.join("users", userid)
    projectdir = os.path.join("users", userid, project)
    imagedir = os.path.join("users", userid, project, "images")
    outputdir = os.path.join("users",userid, project, "output")
    if not os.path.exists(userdir):
        os.mkdir(userdir)
    if not os.path.exists(projectdir):
        os.mkdir(projectdir)
    if not os.path.exists(imagedir):
        os.mkdir(imagedir)
    if not os.path.exists(outputdir):
        os.mkdir(outputdir)

""" Model training API  """
@app.route('/app/train/<userid>', methods=['POST'])
def train_model(userid):
    if request.method == 'POST':
        content = request.json
        userid = content['userid']
        project = content['project']
        epochs = content['epochs']
        classes = content['classes']
        sequence = multiprocessing.Process(target=model_controller.modeltrain, args=(userid, project, epochs, classes))
        sequence.start()
        return jsonify("training initialised")
        
""" Retrive Last Model training based on userid and project """
@app.route('/app/train/<userid>', methods=['GET'])
def train_info(userid):
    if request.method == 'GET':
        content = request.json
        userid = content['userid']
        project = content['project']
        epochs = content['epochs']
        classes = content['classes']
        return Response(database_controller.last_train_info_get_by_user(userid, project, classes, epochs))

""" Retrive All Model training based on userid """
@app.route('/app/train/all/<userid>', methods=['GET'])
def all_train_info_by_user(userid):
    if request.method == 'GET':
        return Response(database_controller.train_info_get_by_user(userid))

""" Recieve file and store input data for user """
@app.route('/app/annotate/files/<userid>', methods=['GET', 'POST'])
def receive_images(userid):
    if request.method == 'POST':


        content = json.loads(request.form['jsondata'])
        project = content['project']

        create_user_dir(userid, project)
        
        file_name = content['data'][0]['annotation']['filename']
        size = content['data'][0]['annotation']['size']
        object_data = content['data'][0]['annotation']['object']
        img_classes = [li['name'] for li in object_data]
        strcontent = json.dumps(content)
        imagename = file_name.rsplit('.', 1)[0]
        PROJECT_DIRECTORY = "users" + "/" + userid + "/" + project + "/"
        xml_convert.convertJsonToPascal(strcontent, PROJECT_DIRECTORY  + "images", imagename)

        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(PROJECT_DIRECTORY  + "images", filename))
            database_controller.project_file_info(userid, project, filename, imagename + ".xml", size, object_data)
            flash('File successfully uploaded')
            return jsonify(image_received_name = filename,xml_file_name = imagename + ".xml",file_path = PROJECT_DIRECTORY  + "images" + "/" + imagename + ".xml", image_classes = img_classes)
        
        else:
            flash('Allowed file types are png, jpg, jpeg')
            return redirect(request.url)
                

""" Predict image to test a model """
@app.route('/app/predict/image/<userid>', methods=['POST'])
def predict_image(userid):
    if request.method == 'POST':
        content = json.loads(request.form['jsondata'])
        userid = content['userid']
        project = content['project']
        classes = content['classes']
        modelname = content['modelname']

        PROJECT_DIRECTORY = "users" + "/" + userid + "/" + project + "/"

        model = Model.load(PROJECT_DIRECTORY + modelname, classes)

        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(PROJECT_DIRECTORY, filename))
            flash('File successfully uploaded')
            image = read_image(PROJECT_DIRECTORY + filename)
            labels, boxes, scores = model.predict_top(image)
            # strlabels = str(labels)
            # strboxes = str(boxes)
            # strscores = str(scores)
            show_labeled_image(image, boxes, PROJECT_DIRECTORY + filename, labels)
            # return jsonify(labels = strlabels, boxes = strboxes, scores = strscores )
            return send_from_directory(PROJECT_DIRECTORY, filename, as_attachment=True)
        
        else:
            flash('Allowed file types are png, jpg, jpeg')
            return redirect(request.url)

""" Predict video to test a model """
@app.route('/app/predict/video/<userid>', methods=['POST'])
def predict_video(userid):
    if request.method == 'POST':
        content = json.loads(request.form['jsondata'])
        userid = content['userid']
        project = content['project']
        classes = content['classes']
        modelname = content['modelname']

        PROJECT_DIRECTORY = "users" + "/" + userid + "/" + project + "/"

        model = Model.load(PROJECT_DIRECTORY + modelname, classes)

        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(PROJECT_DIRECTORY, filename))
            flash('File successfully uploaded')
            video_name = filename.rsplit('.', 1)[0]
            detect_video(model, PROJECT_DIRECTORY + filename, PROJECT_DIRECTORY + video_name + "_detected" + ".mp4", score_filter=0.6)
            return send_from_directory(PROJECT_DIRECTORY, video_name + "_detected" + ".mp4", as_attachment=True)
        
        else:
            flash('Allowed file types are png, jpg, jpeg, mp4, avi')
            return redirect(request.url)


""" Transform Images based on userid """
@app.route('/app/transform/image/<userid>', methods=['POST'])
def image_resize(userid):
    if request.method == 'POST':
        content = json.loads(request.form['jsondata'])
        userid = content['userid']
        project = content['project']

        create_user_dir(userid, project)

        PROJECT_DIRECTORY = "users" + "/" + userid + "/" + project + "/"

        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(PROJECT_DIRECTORY + "images", filename))
            flash('File successfully uploaded')
            image_transform.image_resize(PROJECT_DIRECTORY + "images", filename)
            return jsonify(PROJECT_DIRECTORY + "images" + "/" + filename )
        
        else:
            flash('Allowed file types are png, jpg, jpeg')
            return redirect(request.url)


""" Receive transformed image """
@app.route('/app/transform/image/<userid>', methods=['GET'])
def image_retrieve(userid):
    if request.method == 'GET':
        content = request.json
        userid = content['userid']
        project = content['project']
        image_name = content['image']

        IMAGE_DIRECTORY = "users" + "/" + userid + "/" + project + "/" + "images"
        return send_from_directory(IMAGE_DIRECTORY, image_name, as_attachment=True)


""" Predict image to test a model """
@app.route('/app/predict/grid/<userid>', methods=['POST'])
def predict_grid(userid):
    if request.method == 'POST':
        content = json.loads(request.form['jsondata'])
        userid = content['userid']
        project = content['project']
        classes = content['classes']
        modelname = content['modelname']

        PROJECT_DIRECTORY = "users" + "/" + userid + "/" + project + "/"
    
        model = Model.load(PROJECT_DIRECTORY + modelname, classes)
        images = []

        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        
        if file.filename == '':
            flash('No file selected for uploading')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(PROJECT_DIRECTORY, filename))
            flash('File successfully uploaded')
            for i in range(1):
                image = read_image(PROJECT_DIRECTORY + filename.format(i))
                images.append(image)
            plot_prediction_grid(model, images, PROJECT_DIRECTORY + filename, dim=(1, 1), figsize=(8, 8), score_filter=0.7)
            return send_from_directory(PROJECT_DIRECTORY, filename, as_attachment=True)
        
        else:
            flash('Allowed file types are png, jpg, jpeg')
            return redirect(request.url)

                
# @app.route('/app/annotate/files/<userid>', methods=['GET', 'POST'])
# def receive_json(userid):
#     if request.method == 'POST':
#         content = request.json
#         project_name = content['project']
#         file_name = content['data'][0]['annotation']['filename']
#         imagename = file_name.rsplit('.', 1)[0]
#         strcontent = json.dumps(content)
#         xml_convert.convertJsonToPascal(strcontent, userid, imagename)
#         return jsonify(project = project_name, xml_file_name = imagename + ".xml")




if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6543, debug=True)