# pip install Flask flask_session flask_wtf werkzeug
# pip install pyvips

import os, cv2, shutil, zipfile, gzip, tarfile, queue, threading, time, json, uuid
from datetime import datetime
import pandas as pd
import numpy as np
from flask import (
  Flask, render_template, request, redirect, url_for, flash, session, send_file, jsonify,
  send_from_directory,
)
from flask_wtf.csrf import CSRFProtect, CSRFError
from flask_session import Session
from werkzeug.utils import secure_filename

from Helper import *
from Threads import *
from Database import *
from Validations import *

MakeFolders()

app = Flask(__name__)
app.secret_key = "HOSSAM MAGDY BALAHA"
app.config["SESSION_PERMANENT"] = False
app.config["SESSION_TYPE"] = "filesystem"
app.config['MAX_CONTENT_LENGTH'] = 5 * 1024 ** 3  # 5 GB

csrf = CSRFProtect(app)
csrf.init_app(app)
Session(app)

stopQueueThread = False
queueThread = threading.Thread(target=QueueThreadHandler)


@app.route('/')
def landing():
  MakeFolders()
  return render_template('landing.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
  MakeFolders()

  if (session.get("username") is not None and session.get("is_authenticated")):
    return redirect(url_for('landing'))

  if (request.method == 'POST'):
    username = request.form.get('username')
    password = request.form.get('password')

    validation = ValidateFields()
    usernameValidation = validation.Username(username, minLength=6, maxLength=100)
    passwordValidation = validation.Password(password, minLength=6, maxLength=100)

    errors = []
    if (not usernameValidation[0]):
      errors.append(usernameValidation[1])
    if (not passwordValidation[0]):
      errors.append(passwordValidation[1])

    if (len(errors) > 0):
      for error in errors:
        flash(error, 'error')

      session['loginForm'] = request.form
      return redirect(url_for('login'))

    if (not ProjectHandler().Authenticate(username, password)):
      error = 'Invalid Credentials. Please try again.'
      flash(error, 'error')

      session['loginForm'] = request.form
      return redirect(url_for('login'))
    else:
      session.clear()
      session['username'] = username
      session['is_authenticated'] = True
      return redirect('/')
  else:
    formData = session.get('loginForm', None)
    session.pop('loginForm', None)
    return render_template('login.html', loginForm=formData)


@app.route('/register', methods=['GET', 'POST'])
def register():
  MakeFolders()

  if (session.get("username") is not None and session.get("is_authenticated")):
    return redirect(url_for('landing'))

  if (request.method == 'POST'):
    username = request.form.get('username')
    password = request.form.get('password')
    confirm = request.form.get('confirm')

    validation = ValidateFields()
    usernameValidation = validation.Username(username, minLength=6, maxLength=100)
    passwordValidation = validation.Password(password, minLength=6, maxLength=100)
    confirmValidation = validation.ConfirmPassword(password, confirm)

    errors = []
    if (not usernameValidation[0]):
      errors.append(usernameValidation[1])
    if (not passwordValidation[0]):
      errors.append(passwordValidation[1])
    if (not confirmValidation[0]):
      errors.append(confirmValidation[1])

    if (len(errors) > 0):
      for error in errors:
        flash(error, 'error')

      session['registerForm'] = request.form
      return redirect(url_for('register'))

    registerConfirmation = ProjectHandler().AddNewUser(username, password, confirm)

    if (not registerConfirmation['success']):
      session.clear()
      flash(registerConfirmation['message'], 'error')
      session['registerForm'] = request.form
      return redirect(url_for('register'))
    else:
      session.clear()
      flash(registerConfirmation['message'], 'success')
      return redirect(url_for('login'))
  else:
    formData = session.get('registerForm', None)
    session.pop('registerForm', None)
    return render_template('register.html', registerForm=formData)


@app.route('/logout', methods=['GET'])
def logout():
  session.clear()
  return redirect('/')


@app.route('/viewer', methods=['GET'])
def viewer():
  MakeFolders()

  if (session.get("username") is None or not session.get("is_authenticated")):
    return redirect(url_for('landing'))

  username = session.get("username")

  return render_template('viewer.html', username=username)


@app.route('/wsi', methods=['GET'])
def wsi():
  MakeFolders()

  if (session.get("username") is None or not session.get("is_authenticated")):
    return redirect(url_for('landing'))

  return render_template(
    'wsi.html',
    queueCount=workingQueue.qsize(),
  )


@app.route('/dzi', methods=['GET'])
def dzi():
  MakeFolders()

  if (session.get("username") is None or not session.get("is_authenticated")):
    return redirect(url_for('landing'))

  conversions = len([f for f in os.listdir(DZI_PATH) if f.endswith(".dzi")])

  return render_template(
    'dzi.html',
    queueCount=workingQueue.qsize(),
    conversions=conversions,
  )


@app.route('/augmentation', methods=['GET', 'POST'])
def augmentation():
  MakeFolders()

  if (session.get("username") is None or not session.get("is_authenticated")):
    return redirect(url_for('landing'))

  if (request.method == 'POST'):
    return
  else:
    return render_template('augmentation.html')


@app.route('/classification', methods=['GET', 'POST'])
def classification():
  global stopQueueThread, queueThread
  MakeFolders()

  if (session.get("username") is None or not session.get("is_authenticated")):
    return redirect(url_for('landing'))

  username = session.get("username")
  datasetsRecords = ProjectHandler().GetUserDatasets(username)
  datasetsDict = {record['store_name']: record['name'] for record in datasetsRecords if record['is_file'] == 0}
  datasetsStoreNames = [record['store_name'] for record in datasetsRecords if record['is_file'] == 0]

  if (request.method == 'POST'):
    title = request.form.get('title')
    model = request.form.get('model')
    optimizer = request.form.get('optimizer')
    shape = request.form.get('shape')
    activation = request.form.get('activation')
    trainable = request.form.get('trainable')
    tlTrainingRatio = request.form.get('tlTrainingRatio')
    batchSize = request.form.get('batchSize')
    trainingRatio = request.form.get('trainingRatio')
    epochs = request.form.get('epochs')
    classes = request.form.get('classes')
    dataset = request.form.get('dataset')
    scaler = request.form.get('scaler')

    validation = ValidateFields()
    rules = [
      validation.Title(title, minLength=6, maxLength=100),
      validation.In("Model", model, MODELS_NAMES),
      validation.In("Optimizer", optimizer, OPTIMIZERS_NAMES),
      validation.In("Activation", activation, HIDDEN_ACTIVATIONS_NAMES),
      validation.In("Scaler", scaler, SCALERS_NAMES),
      validation.In("Dataset", dataset, datasetsStoreNames),
      validation.Shape(shape),
      validation.Trainable(trainable),
      validation.TLTrainingRatio(tlTrainingRatio),
      validation.BatchSize(batchSize),
      validation.TrainingRatio(trainingRatio),
      validation.Epochs(epochs),
      validation.Classes(classes),
    ]

    errors = []
    for rule in rules:
      if (not rule[0]):
        errors.append(rule[1])

    if (len(errors) > 0):
      for error in errors:
        flash(error, 'error')
      session['classificationForm'] = request.form
      return redirect(url_for('classification'))

    configurations = {
      "Title"                     : title,
      "Model"                     : model,
      "Optimizer"                 : optimizer,
      "Shape"                     : shape,
      "Hidden Activation Function": activation,
      "Is Trainable?"             : trainable,
      "TL Training Ratio"         : tlTrainingRatio,
      "Batch Size"                : batchSize,
      "Train-to-test Ratio"       : trainingRatio,
      "Epochs"                    : epochs,
      "Classes"                   : classes,
      "Dataset"                   : datasetsDict[dataset],
      "Scaler Technique"          : scaler,
    }

    storeName = ProjectHandler().GetUniqueClassificationFilename(None)
    relativePath = ""
    result = ProjectHandler().AddUserClassification(username, title, storeName, relativePath, configurations)

    if (not result['success']):
      flash(result['message'], 'error')
      session['classificationForm'] = request.form
      return redirect(url_for('classification'))

    allFields = [
      title, storeName, model, optimizer, shape, activation, trainable, tlTrainingRatio, batchSize, trainingRatio,
      epochs, classes, dataset, scaler, username
    ]

    workingQueue.put(("CLS_PRS", *allFields))

    if (stopQueueThread):
      stopQueueThread = False
      del queueThread
      queueThread = threading.Thread(target=QueueThreadHandler)

    try:
      if (not queueThread.is_alive()):
        queueThread.start()
    except Exception as e:
      print("Error: ", e)

    # session['classificationForm'] = request.form
    return redirect(url_for('classification'))

  else:

    formData = session.get('classificationForm', None)
    session.pop('classificationForm', None)
    return render_template(
      'classification.html',
      classificationForm=formData,
      datasets=datasetsDict,
      queueCount=workingQueue.qsize(),
      scalers=SCALERS_NAMES,
      activations=HIDDEN_ACTIVATIONS_NAMES,
      optimizers=OPTIMIZERS_NAMES,
      models=MODELS_NAMES,
    )


@app.route('/classification-history', methods=['GET'])
def classificationHistory():
  global stopQueueThread, queueThread
  MakeFolders()

  if (session.get("username") is None or not session.get("is_authenticated")):
    return redirect(url_for('landing'))

  return render_template(
    'classification_history.html',
  )


@app.route('/classification-inference', methods=['GET'])
def classificationInference():
  MakeFolders()

  if (session.get("username") is None or not session.get("is_authenticated")):
    return redirect(url_for('landing'))

  username = session.get("username")
  classificationsRecords = ProjectHandler().GetUserClassifications(username)
  classificationsRecords = [el for el in classificationsRecords if el['is_success']]

  return render_template(
    'classification_inference.html',
    classificationsRecords=classificationsRecords,
  )


@app.route('/segmentation', methods=['GET', 'POST'])
def segmentation():
  global stopQueueThread, queueThread
  MakeFolders()

  if (session.get("username") is None or not session.get("is_authenticated")):
    return redirect(url_for('landing'))

  username = session.get("username")
  wsiRecords = ProjectHandler().GetUserWSIs(username, amount=None, searchFor="SVS")
  wsiDict = {record['store_name']: record['name'] for record in wsiRecords if record['type'] == "SVS"}
  wsiStoreNames = [record['store_name'] for record in wsiRecords if record['type'] == "SVS"]

  if (request.method == 'POST'):
    title = request.form.get('title')
    model = request.form.get('model')
    activation = request.form.get('activation')
    scaler = request.form.get('scaler')
    batchSize = request.form.get('batchSize')
    trainingRatio = request.form.get('trainingRatio')
    epochs = request.form.get('epochs')
    shape = request.form.get('shape')
    wsiFile = request.form.get('wsiFile')
    annotations = request.form.get('annotations')
    shuffle = request.form.get('shuffle')
    qupathProject = request.files['qupathProject']

    validation = ValidateFields()
    rules = [
      validation.Title(title, minLength=6, maxLength=100),
      validation.In("Model", model, UNET_MODELS_NAMES),
      validation.In("Activation", activation, HIDDEN_ACTIVATIONS_NAMES),
      validation.In("Scaler", scaler, SCALERS_NAMES),
      validation.In("Whole Slide Image (WSI) File", wsiFile, wsiStoreNames),
      validation.Shape(shape),
      validation.BatchSize(batchSize),
      validation.TrainingRatio(trainingRatio),
      validation.Epochs(epochs),
      validation.Annotations(annotations),
      validation.Shuffle(shuffle),
    ]

    errors = []
    for rule in rules:
      if (not rule[0]):
        errors.append(rule[1])

    qupathProjectFilename = secure_filename(qupathProject.filename)
    ext = qupathProjectFilename.split('.')[-1]
    if (ext != "zip"):
      errors.append("QuPath project file must be of type .zip")

    if (len(errors) > 0):
      for error in errors:
        flash(error, 'error')
      session['segmentationForm'] = request.form
      return redirect(url_for('segmentation'))

    configurations = {
      "Title"                       : title,
      "Model"                       : model,
      "Shape"                       : shape,
      "Hidden Activation Function"  : activation,
      "Batch Size"                  : batchSize,
      "Train-to-test Ratio"         : trainingRatio,
      "Epochs"                      : epochs,
      "Annotations"                 : annotations,
      "Whole Slide Image (WSI) File": wsiDict[wsiFile],
      "Scaler Technique"            : scaler,
      "Shuffle"                     : shuffle,
    }

    storeName = ProjectHandler().GetUniqueSegmentationFilename(None)
    relativePath = ""
    result = ProjectHandler().AddUserSegmentation(username, title, storeName, relativePath, configurations)

    if (not result['success']):
      flash(result['message'], 'error')
      session['segmentationForm'] = request.form
      return redirect(url_for('segmentation'))

    storePath = os.path.join(SEGMENTATIONS_PATH, storeName)
    inferencePath = os.path.join(storePath, "Inference")
    os.makedirs(storePath, exist_ok=True)
    os.makedirs(inferencePath, exist_ok=True)
    qupathProjectSavePath = os.path.join(storePath, qupathProjectFilename)
    qupathProject.save(qupathProjectSavePath)

    allFields = [
      title, storeName, wsiFile, annotations, model, activation, batchSize, trainingRatio, epochs, scaler,
      shape, shuffle, username, qupathProjectFilename
    ]

    workingQueue.put(("SEG_WEK_PRS", *allFields))

    if (stopQueueThread):
      stopQueueThread = False
      del queueThread
      queueThread = threading.Thread(target=QueueThreadHandler)

    try:
      if (not queueThread.is_alive()):
        queueThread.start()
    except Exception as e:
      print("Error: ", e)

    return redirect(url_for('segmentation'))

  else:

    formData = session.get('segmentationForm', None)
    session.pop('segmentationForm', None)
    return render_template(
      'segmentation.html',
      segmentationForm=formData,
      wsis=wsiDict,
      queueCount=workingQueue.qsize(),
      scalers=SCALERS_NAMES,
      activations=HIDDEN_ACTIVATIONS_NAMES,
      models=UNET_MODELS_NAMES,
    )

    # if (qupathProject):
    #   filename = secure_filename(qupathProject.filename)
    #   newFileName = filename
    #   filePath = os.path.join(QUPATH_COMPRESSED_PATH, newFileName)
    #
    #   counter = 1
    #   while (os.path.exists(filePath)):
    #     newFileName = ".".join(filename.split(".")[:-1]) + " (" + str(counter) + ")." + filename.split(".")[-1]
    #     filePath = os.path.join(QUPATH_COMPRESSED_PATH, newFileName)
    #     counter += 1
    #
    #   filename = newFileName
    #   qupathProject.save(filePath)
    #   allFields.append(filename)
    #
    #   with zipfile.ZipFile(filePath, 'r') as zipRef:
    #     path = os.path.join(QUPATH_PROJECTS_PATH, newFileName)
    #     path = ".".join(path.split(".")[:-1])
    #     zipRef.extractall(path)
    # else:
    #   flash("QuPath project is not found.", "error")
    #   return redirect(url_for('segmentation'))
    #
    # workingQueue.put(("SEG_WEK_PRS", *allFields))
    #
    # if (stopQueueThread):
    #   stopQueueThread = False
    #   del queueThread
    #   queueThread = threading.Thread(target=QueueThreadHandler)
    #
    # try:
    #   if (not queueThread.is_alive()):
    #     queueThread.start()
    # except Exception as e:
    #   print("Error: ", e)
    #
    # return redirect(url_for('segmentation'))
    #
    # else:
    #   wsiFiles = sorted(os.listdir(WSI_PATH))
    #
    #
    #
    #   return render_template(
    #     'segmentation.html',
    #     wsiFiles=wsiFiles,
    #     queueCount=workingQueue.qsize(),
    #     histories=histories,
    #   )


@app.route('/segmentation-history', methods=['GET'])
def segmentationHistory():
  global stopQueueThread, queueThread
  MakeFolders()

  if (session.get("username") is None or not session.get("is_authenticated")):
    return redirect(url_for('landing'))

  return render_template(
    'segmentation_history.html',
  )


@app.errorhandler(CSRFError)
def handleCsrfRrror(e):
  return render_template('csrf_error.html', reason=e.description), 400


# =================================================== #
# ==================== API ROUTES =================== #
# =================================================== #

@app.route('/api/datasets/delete', methods=['POST'])
def deleteDatasetsAPI():
  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  if ('item' not in request.form):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target file is not defined.",
    }
    return response

  if ('path' not in request.form):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Path is not defined.",
    }
    return response

  path = request.form['path']
  item = "-".join(request.form['item'].split("-")[2:])
  username = session.get("username")

  if (path == ""):
    record = ProjectHandler().SelectUserDatasetByStoreName(username, item)
    storeName = record["store_name"]
    relativePath = record["relative_path"]
    absolutePath = os.path.join(DATASETS_PATH, relativePath, storeName)
    isBelong = ProjectHandler().IsUserDatasetBelongToUser(username, item)
  else:
    absolutePath = os.path.join(DATASETS_PATH, item)
    isBelong = ProjectHandler().IsUserDatasetBelongToUser(username, path.split("/")[0])

  if (not isBelong):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target file does not belong to you.",
    }
    return response

  if (os.path.exists(absolutePath)):
    if os.path.isfile(absolutePath) or os.path.islink(absolutePath):
      os.remove(absolutePath)
      if (path == ""):
        ProjectHandler().DeleteUserDataset(username, record['id'])
    elif os.path.isdir(absolutePath):
      shutil.rmtree(absolutePath)
      if (path == ""):
        ProjectHandler().DeleteUserDataset(username, record['id'])

    response = {
      "result"    : None,
      "is_success": True,
      "message"   : "Successfully deleted.",
    }
    return response
  else:
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "File does not exist.",
    }
    return response


@app.route('/api/datasets/uncompress', methods=['POST'])
def uncompressDatasetsAPI():
  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  if ('item' not in request.form):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target file is not defined.",
    }
    return response

  if ('path' not in request.form):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Path is not defined.",
    }
    return response

  item = "-".join(request.form['item'].split("-")[2:])
  username = session.get("username")
  record = ProjectHandler().SelectUserDatasetByStoreName(username, item)

  if (record is None):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "File does not exist.",
    }
    return response

  storeName = record["store_name"]
  relativePath = record["relative_path"]
  absolutePath = os.path.join(DATASETS_PATH, relativePath, storeName)
  extension = storeName.split(".")[-1]
  uniqueFilename = ProjectHandler().GetUniqueDatasetFilename(extension, includeExtension=False)
  uncompressPath = os.path.join(DATASETS_PATH, relativePath, uniqueFilename)
  isFile = False

  result = ProjectHandler().AddUserDataset(username, record["name"], uniqueFilename, relativePath, isFile)
  if (result["success"]):
    if (os.path.exists(absolutePath)):
      if os.path.isfile(absolutePath) or os.path.islink(absolutePath):
        if (absolutePath.endswith(".zip")):
          with zipfile.ZipFile(absolutePath, 'r') as zipRef:
            zipRef.extractall(uncompressPath)
        elif (absolutePath.endswith(".tar")):
          with tarfile.open(absolutePath) as tar:
            tar.extractall(uncompressPath)
        elif (absolutePath.endswith(".gz")):
          with gzip.open(absolutePath, 'rb') as fIn:
            with open(uncompressPath, 'wb') as fOut:
              shutil.copyfileobj(fIn, fOut)
        else:
          with zipfile.ZipFile(absolutePath, 'r') as zipRef:
            zipRef.extractall(uncompressPath)

        response = {
          "result"    : None,
          "is_success": True,
          "message"   : "Successfully uncompressed.",
        }
      else:
        response = {
          "result"    : None,
          "is_success": False,
          "message"   : "Target item can't be uncompressed.",
        }
      return response
    else:
      response = {
        "result"    : None,
        "is_success": False,
        "message"   : "File does not exist.",
      }
      return response


@app.route('/api/datasets/upload', methods=['POST'])
def uploadDatasetsAPI():
  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  if ('file' not in request.files):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "File is not found.",
    }
    return response

  if ('path' not in request.form):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Path is not defined.",
    }
    return response

  file = request.files['file']
  if (file.filename == ''):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "File is not selected.",
    }
    return response

  ext = file.filename.split('.')[-1].lower()
  if (ext not in COMP_TYPES):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "File extension is not supported.",
    }
    return response

  if (len(request.form.get("path"))):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Uploading is supported only in the root directory.",
    }
    return response

  if (file):
    path = request.form.get("path")
    filename = secure_filename(file.filename)
    filenameNoExt = ".".join(filename.split(".")[:-1])
    absolutePath = os.path.join(DATASETS_PATH, path)
    uniqueFilename = ProjectHandler().GetUniqueDatasetFilename(ext, includeExtension=True)
    savePath = os.path.join(absolutePath, uniqueFilename)
    username = session.get("username")
    isFile = True

    result = ProjectHandler().AddUserDataset(username, filenameNoExt, uniqueFilename, path, isFile)
    if (result["success"]):
      file.save(savePath)

    response = {
      "result"    : None,
      "is_success": result["success"],
      "message"   : result["message"],
    }
    return response


@app.route('/api/datasets/image', methods=['GET'])
def imageDatasetsAPI():
  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  if ('path' not in request.args):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Path is not defined.",
    }
    return response

  path = request.args.get("path")
  username = session.get("username")

  isBelong = ProjectHandler().IsUserDatasetBelongToUser(username, path.split("/")[0])
  if (not isBelong):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target file does not belong to you.",
    }
    return response

  filePath = os.path.join(DATASETS_PATH, path)

  if (os.path.exists(filePath)):
    if (os.path.isfile(filePath)):
      encodedString = EncodeImage(filePath).decode('utf-8')
      response = {
        "result"    : encodedString,
        "is_success": True,
        "message"   : "Image is loaded successfully.",
      }
      return response
    else:
      response = {
        "result"    : None,
        "is_success": False,
        "message"   : "Target item is not a file.",
      }
      return response
  else:
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "File does not exist.",
    }
    return response


@app.route('/api/datasets/browse', methods=['GET'])
def browseDatasetsAPI():
  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  path = request.args.get('path')
  fromNumber = int(request.args.get('from', 0))
  stepNumber = int(request.args.get('step', 10))
  username = session.get("username")

  if (path != ""):
    isBelong = ProjectHandler().IsUserDatasetBelongToUser(username, path.split("/")[0])
    if (isBelong):
      targetFolder = os.path.join(DATASETS_PATH, path)
      if (os.path.exists(targetFolder)):
        if (os.path.isdir(targetFolder)):
          items = os.listdir(targetFolder)
          noOfElements = len(items)
          items.sort()
          items = items[fromNumber:fromNumber + stepNumber]

          results = []
          for item in items:
            itemPath = os.path.join(targetFolder, item)
            size = os.path.getsize(itemPath)
            modifiedAt = datetime.fromtimestamp(os.path.getmtime(itemPath)).strftime('%Y-%m-%d %H:%M:%S')
            createdAt = datetime.fromtimestamp(os.path.getctime(itemPath)).strftime('%Y-%m-%d %H:%M:%S')
            accessedAt = datetime.fromtimestamp(os.path.getatime(itemPath)).strftime('%Y-%m-%d %H:%M:%S')
            recordType = "file" if os.path.isfile(itemPath) else "folder"
            isImage = (item.split(".")[-1].lower() in IMAGE_TYPES) and (recordType == "file")
            # isCompressed = (item.split(".")[-1].lower() in COMP_TYPES) and (recordType == "file")
            name = item
            handlePath = os.path.join(path, item)

            results.append({
              "name"        : name,
              "path"        : handlePath,
              "type"        : recordType,
              "isImage"     : isImage,
              "isCompressed": False,
              "size"        : size,
              "modifiedAt"  : modifiedAt,
              "createdAt"   : createdAt,
              "accessedAt"  : accessedAt,
            })

          response = {
            "result"    : {"elements": results, "count": noOfElements},
            "is_success": True,
            "message"   : "Success loading the elements.",
          }

          return response
        else:
          response = {
            "result"    : None,
            "is_success": False,
            "message"   : "Target item is not a folder.",
          }
          return response
    else:
      response = {
        "result"    : None,
        "is_success": False,
        "message"   : "Path is not valid.",
      }
  else:
    records = ProjectHandler().GetUserDatasets(username, path, fromNumber, stepNumber)
    noOfElements = ProjectHandler().CountUserDatasets(username, path)

    results = []
    for record in records:
      storePath = os.path.join(DATASETS_PATH, record["relative_path"], record["store_name"])
      if (not os.path.exists(storePath)):
        ProjectHandler().DeleteUserDataset(username, record["id"])
        continue

      size = os.path.getsize(storePath)
      modifiedAt = datetime.fromtimestamp(os.path.getmtime(storePath)).strftime('%Y-%m-%d %H:%M:%S')
      createdAt = datetime.fromtimestamp(os.path.getctime(storePath)).strftime('%Y-%m-%d %H:%M:%S')
      accessedAt = datetime.fromtimestamp(os.path.getatime(storePath)).strftime('%Y-%m-%d %H:%M:%S')
      recordType = "file" if record["is_file"] else "folder"
      isImage = (record["store_name"].split(".")[-1].lower() in IMAGE_TYPES) and (recordType == "file")
      isCompressed = (record["store_name"].split(".")[-1].lower() in COMP_TYPES) and (recordType == "file")
      name = record["name"]
      handlePath = os.path.join(record["relative_path"], record["store_name"])

      results.append({
        "name"        : name,
        "path"        : handlePath,
        "type"        : recordType,
        "isImage"     : isImage,
        "isCompressed": isCompressed,
        "size"        : size,
        "modifiedAt"  : modifiedAt,
        "createdAt"   : createdAt,
        "accessedAt"  : accessedAt,
      })

    response = {
      "result"    : {"elements": results, "count": noOfElements},
      "is_success": True,
      "message"   : "Success loading the elements.",
    }
    return response


@app.route('/api/wsi/browse', methods=['GET'])
def browseWSIAPI():
  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  fromNumber = int(request.args.get('from', 0))
  stepNumber = int(request.args.get('step', 10))
  username = session.get("username")

  records = ProjectHandler().GetUserWSIs(username, fromNumber, stepNumber, searchFor="SVS")
  noOfElements = ProjectHandler().CountUserWSIs(username, searchFor="SVS")

  results = []
  for record in records:
    storePath = os.path.join(WSI_PATH, record["store_name"])
    if (not os.path.exists(storePath)):
      ProjectHandler().DeleteUserWSI(username, record["id"])
      continue

    size = os.path.getsize(storePath)
    modifiedAt = datetime.fromtimestamp(os.path.getmtime(storePath)).strftime('%Y-%m-%d %H:%M:%S')
    createdAt = datetime.fromtimestamp(os.path.getctime(storePath)).strftime('%Y-%m-%d %H:%M:%S')
    accessedAt = datetime.fromtimestamp(os.path.getatime(storePath)).strftime('%Y-%m-%d %H:%M:%S')
    recordType = record["type"]
    name = record["name"]
    handlePath = record["store_name"]

    results.append({
      "name"      : name,
      "path"      : handlePath,
      "type"      : recordType,
      "size"      : size,
      "modifiedAt": modifiedAt,
      "createdAt" : createdAt,
      "accessedAt": accessedAt,
    })

  response = {
    "result"    : {"elements": results, "count": noOfElements},
    "is_success": True,
    "message"   : "Success loading the elements.",
  }
  return response


@app.route('/api/wsi/upload', methods=['POST'])
def uploadWSIAPI():
  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  if ('file' not in request.files):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "File is not found.",
    }
    return response

  file = request.files['file']
  if (file.filename == ''):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "File is not selected.",
    }
    return response

  ext = file.filename.split('.')[-1].lower()
  if (ext not in WSI_TYPES):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "File extension is not supported.",
    }
    return response

  if (file):
    filename = secure_filename(file.filename)
    uniqueFilename = ProjectHandler().GetUniqueWSIFilename(ext, includeExtension=True)
    savePath = os.path.join(WSI_PATH, uniqueFilename)
    username = session.get("username")

    result = ProjectHandler().AddUserWSI(username, filename, uniqueFilename, ext.upper())
    if (result["success"]):
      file.save(savePath)

    response = {
      "result"    : None,
      "is_success": result["success"],
      "message"   : result["message"],
    }
    return response

  # if (file):
  #   filename = secure_filename(file.filename)
  #   filePath = os.path.join(WSI_PATH, filename)
  #
  #   counter = 1
  #   while (os.path.exists(filePath)):
  #     newFileName = ".".join(filename.split(".")[:-1]) + " (" + str(counter) + ")." + filename.split(".")[-1]
  #     filePath = os.path.join(WSI_PATH, newFileName)
  #     counter += 1
  #
  #   file.save(filePath)
  #   response = {
  #     "result"    : None,
  #     "is_success": True,
  #     "message"   : "File is uploaded successfully.",
  #   }
  #   return response


@app.route('/api/wsi/delete', methods=['POST'])
def deleteWSIAPI():
  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  if ('item' not in request.form):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target file is not defined.",
    }
    return response

  username = session.get("username")
  item = "-".join(request.form['item'].split("-")[3:])
  record = ProjectHandler().SelectUserWSIByStoreName(username, item, "SVS")
  storeName = record["store_name"]
  absolutePath = os.path.join(WSI_PATH, storeName)
  isBelong = ProjectHandler().IsUserWSIBelongToUser(username, item)

  if (not isBelong):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target file does not belong to you.",
    }
    return response

  if (os.path.exists(absolutePath)):
    if os.path.isfile(absolutePath) or os.path.islink(absolutePath):
      os.remove(absolutePath)
      ProjectHandler().DeleteUserWSI(username, record['id'])
    elif os.path.isdir(absolutePath):
      shutil.rmtree(absolutePath)
      ProjectHandler().DeleteUserWSI(username, record['id'])

    response = {
      "result"    : None,
      "is_success": True,
      "message"   : "Successfully deleted.",
    }
    return response
  else:
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "File does not exist.",
    }
    return response


@app.route('/api/wsi/dzi', methods=['POST'])
def dziWSIAPI():
  global stopQueueThread, queueThread

  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  title = request.form.get('title')
  size = request.form.get('size')
  overlap = request.form.get('overlap')

  validation = ValidateFields()

  titleValidation = validation.Title(title, minLength=6, maxLength=100)
  overlapValidation = validation.Overlap(overlap, minValue=0, maxValue=100)
  sizeValidation = validation.Size(overlap, minValue=1, maxValue=2048)

  if (not titleValidation[0]):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : titleValidation[1],
    }
    return response
  if (not overlapValidation[0]):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : overlapValidation[1],
    }
    return response
  if (not sizeValidation[0]):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : sizeValidation[1],
    }
    return response

  size = int(size)
  overlap = int(overlap)

  username = session.get("username")
  item = "-".join(request.form['item'].split("-")[2:])
  record = ProjectHandler().SelectUserWSIByStoreName(username, item, "SVS")
  storeName = record["store_name"]
  absolutePath = os.path.join(WSI_PATH, storeName)
  isBelong = ProjectHandler().IsUserWSIBelongToUser(username, item)

  if (not isBelong):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target file does not belong to you.",
    }
    return response

  uniqueDZIName = ProjectHandler().GetUniqueWSIFilename("dzi", includeExtension=False)

  if (title is not None and title != ""):
    dziItem = title
  else:
    dziItem = record["name"]

  if (os.path.exists(absolutePath)):
    dziPath = os.path.join(DZI_PATH, uniqueDZIName)

    allFields = [absolutePath, dziPath, dziItem, size, overlap]
    workingQueue.put(("WSI_CNV", *allFields))

    result = ProjectHandler().AddUserWSI(username, dziItem, uniqueDZIName, "DZI")

    if (stopQueueThread):
      stopQueueThread = False
      del queueThread
      queueThread = threading.Thread(target=QueueThreadHandler)

    try:
      if (not queueThread.is_alive()):
        queueThread.start()
    except Exception as e:
      print("Error: ", e)

    response = {
      "result"    : None,
      "is_success": True,
      "message"   : "Successfully started the conversion process.",
    }
    return response
  else:
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "File does not exist.",
    }
    return response


@app.route('/api/dzi/browse', methods=['GET'])
def browseDZIAPI():
  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  fromNumber = int(request.args.get('from', 0))
  stepNumber = int(request.args.get('step', 10))
  username = session.get("username")

  records = ProjectHandler().GetUserWSIs(username, fromNumber, stepNumber, searchFor="DZI")
  noOfElements = ProjectHandler().CountUserWSIs(username, searchFor="DZI")

  results = []
  for record in records:
    storePath = os.path.join(DZI_PATH, record["store_name"])
    if (not os.path.exists(storePath)):
      ProjectHandler().DeleteUserWSI(username, record["id"])
      continue

    if (not os.path.exists(os.path.join(DZI_PATH, record["store_name"], record["name"] + ".dzi"))):
      continue

    modifiedAt = datetime.fromtimestamp(os.path.getmtime(storePath)).strftime('%Y-%m-%d %H:%M:%S')
    createdAt = datetime.fromtimestamp(os.path.getctime(storePath)).strftime('%Y-%m-%d %H:%M:%S')
    accessedAt = datetime.fromtimestamp(os.path.getatime(storePath)).strftime('%Y-%m-%d %H:%M:%S')
    recordType = record["type"]
    name = record["name"]
    handlePath = os.path.join(record["store_name"], record["name"] + ".dzi")

    segmentations = ProjectHandler().GetUserSegmentations(username, amount=None)
    segmentations = {s["store_name"]: s["name"] for s in segmentations if s["is_success"]}

    results.append({
      "name"         : name,
      "outerPath"    : record["store_name"],
      "path"         : handlePath,
      "type"         : recordType,
      "segmentations": list(segmentations.items()),
      "modifiedAt"   : modifiedAt,
      "createdAt"    : createdAt,
      "accessedAt"   : accessedAt,
    })

  response = {
    "result"    : {"elements": results, "count": noOfElements},
    "is_success": True,
    "message"   : "Success loading the elements.",
  }
  return response


@app.route('/api/dzi/delete', methods=['POST'])
def deleteDZIAPI():
  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  if ('item' not in request.form):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target file is not defined.",
    }
    return response

  username = session.get("username")
  item = "-".join(request.form['item'].split("-")[3:])
  record = ProjectHandler().SelectUserWSIByStoreName(username, item, "DZI")
  storeName = record["store_name"]
  absolutePath = os.path.join(DZI_PATH, storeName)
  isBelong = ProjectHandler().IsUserWSIBelongToUser(username, item)

  if (not isBelong):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target file does not belong to you.",
    }
    return response

  if (os.path.exists(absolutePath)):
    if os.path.isfile(absolutePath) or os.path.islink(absolutePath):
      os.remove(absolutePath)
      ProjectHandler().DeleteUserWSI(username, record['id'])
    elif os.path.isdir(absolutePath):
      shutil.rmtree(absolutePath)
      ProjectHandler().DeleteUserWSI(username, record['id'])

    response = {
      "result"    : None,
      "is_success": True,
      "message"   : "Successfully deleted.",
    }
    return response
  else:
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "File does not exist.",
    }
    return response


@app.route('/api/classification/browse', methods=['GET'])
def browseClassificationAPI():
  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  fromNumber = int(request.args.get('from', 0))
  stepNumber = int(request.args.get('step', 10))
  username = session.get("username")

  records = ProjectHandler().GetUserClassifications(username, fromNumber, stepNumber)
  noOfElements = ProjectHandler().CountUserClassifications(username)

  results = []
  for record in records:
    storePath = os.path.join(CLASSIFICATIONS_PATH, record["store_name"])
    if (not os.path.exists(storePath)):
      ProjectHandler().DeleteUserClassification(username, record["id"])
      continue

    if (not os.path.exists(os.path.join(CLASSIFICATIONS_PATH, record["store_name"]))):
      continue

    createdAt = record["created_at"]
    updatedAt = record["updated_at"]
    name = record["name"]
    storeName = record["store_name"]

    results.append({
      "name"     : name,
      "isSuccess": record["is_success"],
      "message"  : record["message"],
      "storeName": storeName,
      "createdAt": createdAt,
      "updatedAt": updatedAt,
    })

  response = {
    "result"    : {"elements": results, "count": noOfElements},
    "is_success": True,
    "message"   : "Success loading the elements.",
  }
  return response


@app.route('/api/classification/configs/init', methods=['POST'])
def initConfigurationsClassificationAPI():
  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  if ('target' not in request.form):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target classifier is not defined.",
    }
    return response

  classifierStoreName = request.form.get('target')
  username = session.get("username")
  isBelong = ProjectHandler().IsUserClassificationBelongToUser(username, classifierStoreName)

  if (not isBelong):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target classifier does not belong to you.",
    }
    return response

  record = ProjectHandler().GetUserClassificationByStoreName(username, classifierStoreName)
  initConfigs = record["configurations"]

  response = {
    "result"    : list(json.loads(initConfigs).items()),
    "is_success": True,
    "message"   : "Success loading the elements.",
  }
  return response


@app.route('/api/classification/configs/post', methods=['POST'])
def postConfigurationsClassificationAPI():
  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  if ('target' not in request.form):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target classifier is not defined.",
    }
    return response

  classifierStoreName = request.form.get('target')
  username = session.get("username")
  isBelong = ProjectHandler().IsUserClassificationBelongToUser(username, classifierStoreName)

  if (not isBelong):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target classifier does not belong to you.",
    }
    return response

  record = ProjectHandler().GetUserClassificationByStoreName(username, classifierStoreName)
  if (not record['is_success']):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Can not handle failed classifier.",
    }
    return response

  postConfigs = record["post_configurations"]

  response = {
    "result"    : list(json.loads(postConfigs).items()),
    "is_success": True,
    "message"   : "Success loading the elements.",
  }
  return response


@app.route('/api/classification/delete', methods=['POST'])
def deleteClassificationAPI():
  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  if ('target' not in request.form):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target classifier is not defined.",
    }
    return response

  classifierStoreName = request.form.get('target')
  username = session.get("username")
  isBelong = ProjectHandler().IsUserClassificationBelongToUser(username, classifierStoreName)

  if (not isBelong):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target classifier does not belong to you.",
    }
    return response

  absolutePath = os.path.join(CLASSIFICATIONS_PATH, classifierStoreName)
  if (os.path.exists(absolutePath)):
    result = ProjectHandler().DeleteUserClassificationByStoreName(username, classifierStoreName)
    if (result["success"]):
      shutil.rmtree(absolutePath)

      response = {
        "result"    : None,
        "is_success": True,
        "message"   : "Target classifier is deleted successfully.",
      }
      return response

  response = {
    "result"    : None,
    "is_success": False,
    "message"   : "Something went wrong during the deletion process.",
  }
  return response


@app.route('/api/classification/history', methods=['POST'])
def historyClassificationAPI():
  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  if ('target' not in request.form):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target classifier is not defined.",
    }
    return response

  classifierStoreName = request.form.get('target')
  username = session.get("username")
  isBelong = ProjectHandler().IsUserClassificationBelongToUser(username, classifierStoreName)

  if (not isBelong):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target classifier does not belong to you.",
    }
    return response

  record = ProjectHandler().GetUserClassificationByStoreName(username, classifierStoreName)
  if (not record['is_success']):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Can not handle failed classifier.",
    }
    return response

  absolutePath = os.path.join(CLASSIFICATIONS_PATH, classifierStoreName, "history.json")
  if (os.path.exists(absolutePath)):
    df = pd.read_json(absolutePath)
    columns = df.columns.tolist()
    history = np.round(df.values.tolist(), 4).tolist()

    response = {
      "result"    : {"columns": columns, "history": history},
      "is_success": True,
      "message"   : "Success loading the history.",
    }
    return response

  response = {
    "result"    : None,
    "is_success": False,
    "message"   : "Something went wrong.",
  }
  return response


@app.route('/api/classification/history/plot', methods=['POST'])
def plotHistoryClassificationAPI():
  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  if ('target' not in request.form):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target classifier is not defined.",
    }
    return response

  classifierStoreName = request.form.get('target')
  username = session.get("username")
  isBelong = ProjectHandler().IsUserClassificationBelongToUser(username, classifierStoreName)

  if (not isBelong):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target classifier does not belong to you.",
    }
    return response

  record = ProjectHandler().GetUserClassificationByStoreName(username, classifierStoreName)
  if (not record['is_success']):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Can not handle failed classifier.",
    }
    return response

  absolutePath = os.path.join(CLASSIFICATIONS_PATH, classifierStoreName, "history.json")
  if (os.path.exists(absolutePath)):
    df = pd.read_json(absolutePath)
    result = GetTrainingHistoryPlot(df, metrics=["loss", "accuracy", "Precision", "Recall", "AUC"])

    response = {
      "result"    : result,
      "is_success": True,
      "message"   : "Success loading the history plot.",
    }
    return response

  response = {
    "result"    : None,
    "is_success": False,
    "message"   : "Something went wrong.",
  }
  return response


@app.route('/api/classification/evaluations', methods=['POST'])
def evaluationsClassificationAPI():
  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  if ('target' not in request.form):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target classifier is not defined.",
    }
    return response

  classifierStoreName = request.form.get('target')
  username = session.get("username")
  isBelong = ProjectHandler().IsUserClassificationBelongToUser(username, classifierStoreName)

  if (not isBelong):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target classifier does not belong to you.",
    }
    return response

  record = ProjectHandler().GetUserClassificationByStoreName(username, classifierStoreName)
  if (not record['is_success']):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Can not handle failed classifier.",
    }
    return response

  absolutePath = os.path.join(CLASSIFICATIONS_PATH, classifierStoreName, "evaluations.json")
  if (os.path.exists(absolutePath)):
    df = pd.read_json(absolutePath)
    tmp = df.select_dtypes(include=[np.number])
    df.loc[:, tmp.columns] = np.round(tmp, 4)
    columns = df.columns.tolist()
    newCols = columns[-1:] + columns[:-1]
    evaluations = df[newCols].values.tolist()
    columns = df[newCols].columns.tolist()

    response = {
      "result"    : {"columns": columns, "evaluations": evaluations},
      "is_success": True,
      "message"   : "Success loading the evaluations.",
    }
    return response

  response = {
    "result"    : None,
    "is_success": False,
    "message"   : "Something went wrong.",
  }
  return response


@app.route('/api/classification/inference', methods=['POST'])
def classifyClassificationAPI():
  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  if ('selectedImageFile' not in request.files):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Image is not found.",
    }
    return response

  if ('target' not in request.form):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target classification run is not defined.",
    }
    return response

  file = request.files['selectedImageFile']
  if (file.filename == ''):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Image is not selected.",
    }
    return response

  ext = file.filename.split('.')[-1].lower()
  if (ext not in OPENCV_IMAGE_TYPES):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Image extension is not supported.",
    }
    return response

  classifierStoreName = request.form.get('target')
  username = session.get("username")
  isBelong = ProjectHandler().IsUserClassificationBelongToUser(username, classifierStoreName)

  if (not isBelong):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target classifier does not belong to you.",
    }
    return response

  record = ProjectHandler().GetUserClassificationByStoreName(username, classifierStoreName)
  if (not record['is_success']):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Can not handle failed classifier.",
    }
    return response

  if (file):
    filename = secure_filename(file.filename)
    inferencePath = os.path.join(CLASSIFICATIONS_PATH, classifierStoreName, "Inference")
    uniqueName = str(uuid.uuid4()) + "." + ext
    imagePath = os.path.join(inferencePath, uniqueName)
    file.save(imagePath)

    configs = record['post_configurations']
    cls, prob = ManipulateClassificationInference(imagePath, classifierStoreName, configs)

    print(cls, prob)
    response = {
      "result"    : {"class": cls, "probability": prob},
      "is_success": True,
      "message"   : "Success classifying the image.",
    }
    return response

  response = {
    "result"    : None,
    "is_success": False,
    "message"   : "Something went wrong.",
  }
  return response


@app.route('/api/segmentation/browse', methods=['GET'])
def browseSegmentationAPI():
  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  fromNumber = int(request.args.get('from', 0))
  stepNumber = int(request.args.get('step', 10))
  username = session.get("username")

  records = ProjectHandler().GetUserSegmentations(username, fromNumber, stepNumber)
  noOfElements = ProjectHandler().CountUserSegmentations(username)

  results = []
  for record in records:
    storePath = os.path.join(SEGMENTATIONS_PATH, record["store_name"])
    if (not os.path.exists(storePath)):
      ProjectHandler().DeleteUserSegmentation(username, record["id"])
      continue

    if (not os.path.exists(os.path.join(SEGMENTATIONS_PATH, record["store_name"]))):
      continue

    createdAt = record["created_at"]
    updatedAt = record["updated_at"]
    name = record["name"]
    storeName = record["store_name"]

    results.append({
      "name"     : name,
      "isSuccess": record["is_success"],
      "message"  : record["message"],
      "storeName": storeName,
      "createdAt": createdAt,
      "updatedAt": updatedAt,
    })

  response = {
    "result"    : {"elements": results, "count": noOfElements},
    "is_success": True,
    "message"   : "Success loading the elements.",
  }
  return response


@app.route('/api/segmentation/configs/init', methods=['POST'])
def initConfigurationsSegmentationAPI():
  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  if ('target' not in request.form):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target segmentation is not defined.",
    }
    return response

  segmentationStoreName = request.form.get('target')
  username = session.get("username")
  isBelong = ProjectHandler().IsUserSegmentationBelongToUser(username, segmentationStoreName)

  if (not isBelong):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target segmentation does not belong to you.",
    }
    return response

  record = ProjectHandler().GetUserSegmentationByStoreName(username, segmentationStoreName)
  initConfigs = record["configurations"]

  response = {
    "result"    : list(json.loads(initConfigs).items()),
    "is_success": True,
    "message"   : "Success loading the elements.",
  }
  return response


@app.route('/api/segmentation/configs/post', methods=['POST'])
def postConfigurationsSegmentationAPI():
  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  if ('target' not in request.form):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target segmentation is not defined.",
    }
    return response

  segmentationStoreName = request.form.get('target')
  username = session.get("username")
  isBelong = ProjectHandler().IsUserSegmentationBelongToUser(username, segmentationStoreName)

  if (not isBelong):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target segmentation does not belong to you.",
    }
    return response

  record = ProjectHandler().GetUserSegmentationByStoreName(username, segmentationStoreName)
  if (not record['is_success']):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Can not handle failed segmentation.",
    }
    return response

  postConfigs = record["post_configurations"]

  response = {
    "result"    : list(json.loads(postConfigs).items()),
    "is_success": True,
    "message"   : "Success loading the elements.",
  }
  return response


@app.route('/api/segmentation/delete', methods=['POST'])
def deleteSegmentationAPI():
  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  if ('target' not in request.form):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target segmentation is not defined.",
    }
    return response

  segmentationStoreName = request.form.get('target')
  username = session.get("username")
  isBelong = ProjectHandler().IsUserSegmentationBelongToUser(username, segmentationStoreName)

  if (not isBelong):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target segmentation does not belong to you.",
    }
    return response

  absolutePath = os.path.join(SEGMENTATIONS_PATH, segmentationStoreName)
  if (os.path.exists(absolutePath)):
    result = ProjectHandler().DeleteUserSegmentationByStoreName(username, segmentationStoreName)
    if (result["success"]):
      shutil.rmtree(absolutePath)

      response = {
        "result"    : None,
        "is_success": True,
        "message"   : "Target segmentation is deleted successfully.",
      }
      return response

  response = {
    "result"    : None,
    "is_success": False,
    "message"   : "Something went wrong during the deletion process.",
  }
  return response


@app.route('/api/segmentation/history', methods=['POST'])
def historySegmentationAPI():
  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  if ('target' not in request.form):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target segmentation is not defined.",
    }
    return response

  segmentationStoreName = request.form.get('target')
  username = session.get("username")
  isBelong = ProjectHandler().IsUserSegmentationBelongToUser(username, segmentationStoreName)

  if (not isBelong):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target segmentation does not belong to you.",
    }
    return response

  record = ProjectHandler().GetUserSegmentationByStoreName(username, segmentationStoreName)
  if (not record['is_success']):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Can not handle failed segmentation.",
    }
    return response

  absolutePath = os.path.join(SEGMENTATIONS_PATH, segmentationStoreName, "history.json")
  if (os.path.exists(absolutePath)):
    df = pd.read_json(absolutePath)
    columns = df.columns.tolist()
    history = np.round(df.values.tolist(), 4).tolist()

    response = {
      "result"    : {"columns": columns, "history": history},
      "is_success": True,
      "message"   : "Success loading the history.",
    }
    return response

  response = {
    "result"    : None,
    "is_success": False,
    "message"   : "Something went wrong.",
  }
  return response


@app.route('/api/segmentation/history/plot', methods=['POST'])
def plotHistorySegmentationAPI():
  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  if ('target' not in request.form):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target segmentation is not defined.",
    }
    return response

  segmentationStoreName = request.form.get('target')
  username = session.get("username")
  isBelong = ProjectHandler().IsUserSegmentationBelongToUser(username, segmentationStoreName)

  if (not isBelong):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target segmentation does not belong to you.",
    }
    return response

  record = ProjectHandler().GetUserSegmentationByStoreName(username, segmentationStoreName)
  if (not record['is_success']):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Can not handle failed segmentation.",
    }
    return response

  absolutePath = os.path.join(SEGMENTATIONS_PATH, segmentationStoreName, "history.json")
  if (os.path.exists(absolutePath)):
    df = pd.read_json(absolutePath)
    result = GetTrainingHistoryPlot(
      df,
      metrics=["loss", "accuracy", "Precision", "Recall", "AUC", "DiceCoeff", "IOUCoeff"]
    )

    response = {
      "result"    : result,
      "is_success": True,
      "message"   : "Success loading the history plot.",
    }
    return response

  response = {
    "result"    : None,
    "is_success": False,
    "message"   : "Something went wrong.",
  }
  return response


@app.route('/api/segmentation/evaluations', methods=['POST'])
def evaluationsSegmentationAPI():
  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  if ('target' not in request.form):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target segmentation is not defined.",
    }
    return response

  segmentationStoreName = request.form.get('target')
  username = session.get("username")
  isBelong = ProjectHandler().IsUserSegmentationBelongToUser(username, segmentationStoreName)

  if (not isBelong):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Target segmentation does not belong to you.",
    }
    return response

  record = ProjectHandler().GetUserSegmentationByStoreName(username, segmentationStoreName)
  if (not record['is_success']):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Can not handle failed segmentation.",
    }
    return response

  absolutePath = os.path.join(SEGMENTATIONS_PATH, segmentationStoreName, "evaluations.json")
  if (os.path.exists(absolutePath)):
    df = pd.read_json(absolutePath)
    tmp = df.select_dtypes(include=[np.number])
    df.loc[:, tmp.columns] = np.round(tmp, 4)
    columns = df.columns.tolist()
    newCols = columns[-1:] + columns[:-1]
    evaluations = df[newCols].values.tolist()
    columns = df[newCols].columns.tolist()

    response = {
      "result"    : {"columns": columns, "evaluations": evaluations},
      "is_success": True,
      "message"   : "Success loading the evaluations.",
    }
    return response

  response = {
    "result"    : None,
    "is_success": False,
    "message"   : "Something went wrong.",
  }
  return response


@app.route('/api/tile/dip', methods=['POST'])
def applyTileDIP():
  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  if ('tile' not in request.form):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Tile is not defined.",
    }
    return response

  if ('threshold' not in request.form):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Threshold value is not defined.",
    }
    return response

  if ('medblur' not in request.form):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Median blur value is not defined.",
    }
    return response

  tileURL = request.form['tile']
  tileParts = tileURL.split("/")
  tileName = tileParts[-1]
  tileFolder = tileParts[-2]
  tileParent = tileParts[-3]

  tilePath = os.path.join(DZI_PATH, tileParent, tileFolder, tileName)

  if (not os.path.exists(tilePath)):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Tile does not exist.",
    }
    return response

  threshold = int(request.form['threshold'])
  medblur = int(request.form['medblur'])

  tile = cv2.imread(tilePath)
  tileGray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY)
  _, backgroundMask = cv2.threshold(tileGray, threshold, 255, cv2.THRESH_BINARY_INV)
  backgroundMaskBGR = cv2.cvtColor(backgroundMask, cv2.COLOR_GRAY2BGR)

  if (medblur > 0):
    backgroundMaskBGR = cv2.medianBlur(backgroundMaskBGR, medblur)

  encImg = EncodeImageFromOpenCV(backgroundMaskBGR).decode('utf-8')

  response = {
    "result"    : encImg,
    "is_success": True,
    "message"   : "Successfully.",
  }
  return response


@app.route('/api/tile/segment', methods=['POST'])
def applyTileSegment():
  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  if ('tile' not in request.form):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Tile is not defined.",
    }
    return response

  if ('segmenter' not in request.form):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "The segmentation name is not defined.",
    }
    return response

  tileURL = request.form['tile']
  tileURL.replace("\\", "/")
  tileParts = tileURL.split("/")
  tileName = tileParts[-1]
  tileFolder = tileParts[-2]
  tileParent = tileParts[-3]

  tilePath = os.path.join(DZI_PATH, tileParent, tileFolder, tileName)

  if (not os.path.exists(tilePath)):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Tile does not exist.",
    }
    return response

  segmenter = request.form['segmenter']

  record = ProjectHandler().GetUserSegmentationByStoreName(session.get("username"), segmenter)
  configs = record['post_configurations']

  maskImage = ManipulateWeaklyUNetSegmentationInference(tilePath, segmenter, configs)
  encImg = EncodeImageFromOpenCV(maskImage).decode('utf-8')

  response = {
    "result"    : encImg,
    "is_success": True,
    "message"   : "Successfully.",
  }
  return response


@app.route('/api/augmentation/live', methods=['POST'])
def liveAugmentationAPI():
  global stopQueueThread, queueThread

  if (session.get("username") is None or not session.get("is_authenticated")):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Not authenticated.",
    }
    return response

  if ('selectedImageFile' not in request.files):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Selected image to augment is not found.",
    }
    return response

  selectedImageFile = request.files['selectedImageFile']
  if (selectedImageFile.filename == ''):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Selected image to augment is not selected.",
    }
    return response

  ext = selectedImageFile.filename.split('.')[-1].lower()
  if (ext not in ["jpg", "jpeg", "png"]):
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Image extension is not supported.",
    }
    return response

  rotation = request.form.get("rotation")
  widthShift = request.form.get("widthShift")
  heightShift = request.form.get("heightShift")
  zoom = request.form.get("zoom")
  verticalFlip = request.form.get("verticalFlip")
  horizontalFlip = request.form.get("horizontalFlip")
  shear = request.form.get("shear")
  brightness = request.form.get("brightness")
  saturation = request.form.get("saturation")
  hue = request.form.get("hue")
  specific = request.form.get("specific")
  single = request.form.get("single")

  validation = ValidateFields()
  validations = [
    validation.Rotation(rotation, minValue=0, maxValue=360),
    validation.WidthShift(widthShift, minValue=-1, maxValue=1),
    validation.HeightShift(heightShift, minValue=-1, maxValue=1),
    validation.Zoom(zoom, minValue=-1, maxValue=1),
    validation.Shear(shear, minValue=-1, maxValue=1),
    validation.VerticalFlip(verticalFlip),
    validation.HorizontalFlip(horizontalFlip),
    validation.Brightness(brightness, minValue=0, maxValue=100),
    validation.Saturation(saturation, minValue=0, maxValue=100),
    validation.Hue(hue, minValue=0, maxValue=100),
    validation.SpecificRandom(specific),
    validation.SingleMultiple(single),
  ]

  for val in validations:
    if (not val[0]):
      response = {
        "result"    : None,
        "is_success": False,
        "message"   : val[1],
      }
      return response

  if (selectedImageFile):
    filename = secure_filename(selectedImageFile.filename)
    uniqueAugFilename = ProjectHandler().GetUniqueAugmentationFilename(ext, includeExtension=True)
    filePath = os.path.join(LIVE_AUGMENTATIONS_PATH, uniqueAugFilename)
    username = session.get("username")

    selectedImageFile.save(filePath)

    image, whichIsRun, configs = LiveImageAugmentation(
      filePath,
      encode=True,
      rotation=int(rotation),
      zoom=float(zoom),
      widthShift=float(widthShift),
      heightShift=float(heightShift),
      shear=float(shear),
      verticalFlip=(verticalFlip == "1"),
      horizontalFlip=(horizontalFlip == "1"),
      brightness=int(brightness),
      saturation=int(saturation),
      hue=int(hue),
      isSpecific=(specific == "0"),
      isSingle=(single == "0"),
    )

    configsDict = {el[0]: el[1] for el in configs}
    configsDict["Is Specific"] = specific
    configsDict["Is Single"] = single

    result = ProjectHandler().AddUserAugmentation(username, filename, uniqueAugFilename, configsDict)
    if (result["success"]):
      response = {
        "result"    : {
          "image"    : image,
          "selection": whichIsRun,
          "notes"    : configs,
        },
        "is_success": True,
        "message"   : "Augmentation is performed successfully.",
      }

      return response

  else:
    response = {
      "result"    : None,
      "is_success": False,
      "message"   : "Unsupported file.",
    }
    return response
