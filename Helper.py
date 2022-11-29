import base64, os, sys, cv2, time, pickle, zipfile
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import Sequence

from Database import *

BASE_FOLDERS_PATH = os.path.join(os.getcwd(), "static", "Assets")
CLASSIFICATIONS_PATH = os.path.join(BASE_FOLDERS_PATH, "Classifications")
SEGMENTATIONS_PATH = os.path.join(BASE_FOLDERS_PATH, "Segmentations")
AUGMENTATIONS_PATH = os.path.join(BASE_FOLDERS_PATH, "Augmentations")
LIVE_AUGMENTATIONS_PATH = os.path.join(AUGMENTATIONS_PATH, "Live")
DATASETS_PATH = os.path.join(BASE_FOLDERS_PATH, "Datasets")
WSI_PATH = os.path.join(BASE_FOLDERS_PATH, "WSI Datasets")
DZI_PATH = os.path.join(BASE_FOLDERS_PATH, "DZI Datasets")
IMAGE_TYPES = ["jpg", "jpeg", "png", "gif", "bmp", "svg"]
COMP_TYPES = ["zip", "rar", "7z", "tar", "gz", "bz2", "xz"]
OPENCV_IMAGE_TYPES = ["jpg", "jpeg", "png"]
QUPATH_COMP_TYPES = ["zip"]
WSI_TYPES = ["svs"]
MAX_QUEUE_SIZE = 100

MODELS_NAMES = [
  "VGG16",
  "VGG19",
  "MobileNet",
  "MobileNetV2",
  "Xception",
  "InceptionV3",
  "InceptionResNetV2",
  "ResNet50",
  "ResNet101",
  "ResNet152",
  "ResNet50V2",
  "ResNet101V2",
  "ResNet152V2",
  "DenseNet121",
  "DenseNet169",
  "DenseNet201",
  "NASNetLarge",
  "NASNetMobile",
]

OPTIMIZERS_NAMES = [
  "adam",
  "sgd",
  "rmsprop",
  "adagrad",
  "adadelta",
  "adamax",
  "nadam",
]

HIDDEN_ACTIVATIONS_NAMES = [
  "relu",
  "tanh",
  "selu",
  "elu",
  "exponential",
  "leakyrelu",
]

SCALERS_NAMES = [
  "MaxNormalization",
  "L1Normalization",
  "L2Normalization",
  "Standardization",
  "MinMaxScaler",
  "MaxAbsScaler",
  "RobustQ1Scaler",
  "RobustMedianScaler",
  "None",
]

UNET_MODELS_NAMES = [
  "Manual U-Net (Input is 256x256x3)",
]


def MakeFolders():
  if (not os.path.exists(BASE_FOLDERS_PATH)):
    os.makedirs(BASE_FOLDERS_PATH)
  if (not os.path.exists(CLASSIFICATIONS_PATH)):
    os.makedirs(CLASSIFICATIONS_PATH)
  if (not os.path.exists(SEGMENTATIONS_PATH)):
    os.makedirs(SEGMENTATIONS_PATH)
  if (not os.path.exists(AUGMENTATIONS_PATH)):
    os.makedirs(AUGMENTATIONS_PATH)
  if (not os.path.exists(LIVE_AUGMENTATIONS_PATH)):
    os.makedirs(LIVE_AUGMENTATIONS_PATH)
  if (not os.path.exists(DATASETS_PATH)):
    os.makedirs(DATASETS_PATH)
  if (not os.path.exists(WSI_PATH)):
    os.makedirs(WSI_PATH)
  if (not os.path.exists(DZI_PATH)):
    os.makedirs(DZI_PATH)


def EncodeImage(path):
  with open(path, "rb") as imageFile:
    encodedString = base64.b64encode(imageFile.read())
  return encodedString


def EncodeImageFromOpenCV(img):
  _, imArr = cv2.imencode('.jpg', img)  # im_arr: image in Numpy one-dim array format.
  imBytes = imArr.tobytes()
  encodedString = base64.b64encode(imBytes)
  return encodedString


def IsInt(s):
  try:
    int(s)
    return True
  except ValueError:
    return False


def IsNumber(s):
  try:
    float(s)
    return True
  except ValueError:
    return False


def WSI2PNG(wsiPath, pngPath, level, downsample):
  import openslide
  from openslide.deepzoom import DeepZoomGenerator

  wsi = openslide.open_slide(wsiPath)
  dz = DeepZoomGenerator(wsi, tile_size=256, overlap=0, limit_bounds=False)
  dz.get_tile(dz.level_count - 1, (0, 0)).save(pngPath, "PNG")


def WSI2VipsDZI(wsiPath, dziPath, tileSize=128, overlap=0, exportFormat="png"):
  # https://github.com/libvips/build-win64-mxe/releases/tag/v8.13.2
  # https://www.libvips.org/API/current/Making-image-pyramids.html
  # https://stackoverflow.com/questions/89228/how-do-i-execute-a-program-or-call-a-system-command
  import subprocess
  BASE_PATH = os.path.join(os.getcwd(), "Packages", "vips-dev-8.13", "bin")
  subprocess.call(
    [
      f"{BASE_PATH}\\vips", "dzsave", f"{wsiPath}", f"{dziPath}",
      "--tile-size", str(tileSize),
      "--overlap", str(overlap),
      "--suffix", "." + exportFormat,
      "--layout", "dz",
    ]
  )


def GetTrainingHistoryPlot(history, metrics=["loss", "accuracy"]):
  # https://stackoverflow.com/questions/38061267/matplotlib-graphic-image-to-base64
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  import io

  plt.figure(figsize=(10 * len(metrics), 10), dpi=350)
  for i, metric in enumerate(metrics):
    plt.subplot(1, len(metrics), i + 1)
    plt.plot(history[metric], label=metric.capitalize(), marker="o")
    plt.plot(history["val_" + metric], label="Validation " + metric.capitalize(), marker="o")
    plt.xlabel("Epoch Number", fontsize=12)
    plt.ylabel("Score Value", fontsize=12)
    plt.grid()
    plt.tight_layout()
    plt.legend()

  ioBytes = io.BytesIO()
  plt.savefig(ioBytes, format="png")
  ioBytes.seek(0)
  ioString = base64.b64encode(ioBytes.read())
  ioString = ioString.decode("utf-8")
  plt.close()

  return ioString


def ManipulateWSIConversion(allFields):
  (filePath, dziPath, title, size, overlap) = allFields

  if not os.path.exists(filePath):
    return

  print(f"Converting {filePath} to {dziPath}", flush=True)

  os.makedirs(dziPath, exist_ok=True)
  storePath = os.path.join(dziPath, title)

  WSI2VipsDZI(
    filePath,
    storePath,
    tileSize=size,
    overlap=overlap,
    exportFormat="png",
  )


def ManipulateClassificationThread(allFields):
  import cv2, pickle, json
  import numpy as np
  import pandas as pd
  import matplotlib.pyplot as plt
  from tensorflow.keras.utils import to_categorical
  from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
  from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout, GlobalAveragePooling2D
  from tensorflow.keras import applications
  from datetime import datetime
  from tensorflow.keras.models import Model
  from sklearn.model_selection import train_test_split
  from tensorflow.keras.backend import clear_session
  from tensorflow.keras.metrics import (
    TruePositives,
    FalsePositives,
    TrueNegatives,
    FalseNegatives,
    Precision,
    Recall,
    AUC,
  )

  clear_session()

  (
    title, storeName, model, optimizer, shape, activation, trainable, tlTrainingRatio,
    batchSize, trainingRatio, epochs, classes, datasetStoreName, scaler, username,
  ) = allFields

  classes = [el.strip() for el in classes.split(",")]
  noOfClasses = len(classes)
  shape = shape.split(",")
  shape = [int(i) for i in shape]

  storePath = os.path.join(CLASSIFICATIONS_PATH, storeName)
  inferencePath = os.path.join(CLASSIFICATIONS_PATH, storeName, "Inference")
  os.makedirs(storePath, exist_ok=True)
  os.makedirs(inferencePath, exist_ok=True)

  datasetPickleFile = os.path.join(storePath, f"dataset.p")
  checkpointPath = os.path.join(storePath, f"model.h5")
  logPath = os.path.join(storePath, f"log.csv")
  historyPath = os.path.join(storePath, f"history.json")
  evaluationsPath = os.path.join(storePath, f"evaluations.json")

  datasetRecord = ProjectHandler().SelectUserDatasetByStoreName(username, datasetStoreName)
  datasetPath = os.path.join(DATASETS_PATH, datasetRecord["store_name"], datasetRecord["name"])

  try:
    if (os.path.exists(datasetPickleFile)):
      X, y = pickle.load(open(datasetPickleFile, "rb"))
    else:
      X, y = [], []
      for cls in classes:
        clsPath = os.path.join(datasetPath, cls)
        files = os.listdir(clsPath)
        for file in files:
          if (file.split(".")[-1].lower() not in OPENCV_IMAGE_TYPES):
            continue
          filePath = os.path.join(clsPath, file)
          try:
            image = cv2.imread(filePath)
            if (len(shape) == 3 and len(image.shape) == 2):
              image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            image = cv2.resize(image, (int(shape[0]), int(shape[1])))
            X.append(image)
            y.append(classes.index(cls))
          except Exception as e:
            print(e, "Problem with file:", filePath, flush=True)

      pickle.dump((X, y), open(datasetPickleFile, "wb"))

    X = np.array(X)
    y = np.array(y)

    X = ScaleImage(X, scaler)

    # Split dataset.
    xTrain, xTest, yTrain, yTest = train_test_split(X, y, train_size=float(trainingRatio), stratify=y)
    xTrain, xVal, yTrain, yVal = train_test_split(xTrain, yTrain, train_size=float(trainingRatio), stratify=yTrain)

    modelShape = X.shape[1:]
    if (len(shape) == 2):
      modelShape = (*X.shape[1:], 1)
    print("Model Input Shape:", modelShape, flush=True)

    # Create the model.
    if (model == "VGG16"):
      baseModel = applications.VGG16(include_top=False, weights="imagenet", input_shape=(*modelShape,))
    elif (model == "VGG19"):
      baseModel = applications.VGG19(include_top=False, weights="imagenet", input_shape=(*modelShape,))
    elif (model == "MobileNet"):
      baseModel = applications.MobileNet(include_top=False, weights="imagenet", input_shape=(*modelShape,))
    elif (model == "MobileNetV2"):
      baseModel = applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(*modelShape,))
    elif (model == "Xception"):
      baseModel = applications.Xception(include_top=False, weights="imagenet", input_shape=(*modelShape,))
    elif (model == "InceptionV3"):
      baseModel = applications.InceptionV3(include_top=False, weights="imagenet", input_shape=(*modelShape,))
    elif (model == "InceptionResNetV2"):
      baseModel = applications.InceptionResNetV2(include_top=False, weights="imagenet", input_shape=(*modelShape,))
    elif (model == "ResNet50"):
      baseModel = applications.ResNet50(include_top=False, weights="imagenet", input_shape=(*modelShape,))
    elif (model == "ResNet101"):
      baseModel = applications.ResNet101(include_top=False, weights="imagenet", input_shape=(*modelShape,))
    elif (model == "ResNet152"):
      baseModel = applications.ResNet152(include_top=False, weights="imagenet", input_shape=(*modelShape,))
    elif (model == "ResNet50V2"):
      baseModel = applications.ResNet50V2(include_top=False, weights="imagenet", input_shape=(*modelShape,))
    elif (model == "ResNet101V2"):
      baseModel = applications.ResNet101V2(include_top=False, weights="imagenet", input_shape=(*modelShape,))
    elif (model == "ResNet152V2"):
      baseModel = applications.ResNet152V2(include_top=False, weights="imagenet", input_shape=(*modelShape,))
    elif (model == "DenseNet121"):
      baseModel = applications.DenseNet121(include_top=False, weights="imagenet", input_shape=(*modelShape,))
    elif (model == "DenseNet169"):
      baseModel = applications.DenseNet169(include_top=False, weights="imagenet", input_shape=(*modelShape,))
    elif (model == "DenseNet201"):
      baseModel = applications.DenseNet201(include_top=False, weights="imagenet", input_shape=(*modelShape,))
    elif (model == "NASNetLarge"):
      baseModel = applications.NASNetLarge(include_top=False, weights="imagenet", input_shape=(*modelShape,))
    elif (model == "NASNetMobile"):
      baseModel = applications.NASNetMobile(include_top=False, weights="imagenet", input_shape=(*modelShape,))
    else:
      print("Unsupported model.", flush=True)

    # Freeze the layers.
    for layer in baseModel.layers:
      layer.trainable = False
    noOfLayers = len(baseModel.layers)
    noOfLayersToUnfreeze = int(noOfLayers * float(tlTrainingRatio))
    if (noOfLayersToUnfreeze > 0):
      for layer in baseModel.layers[-noOfLayersToUnfreeze:]:
        layer.trainable = True

    # Create the model.
    x = baseModel.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation=activation.lower())(x)

    predictions = Dense(noOfClasses, activation='softmax')(x) if \
      (noOfClasses > 2) else Dense(1, activation='sigmoid')(x)

    workingModel = Model(inputs=baseModel.input, outputs=predictions)

    metrics = [
      "accuracy",
      TruePositives(name="TP"),
      FalsePositives(name="FP"),
      TrueNegatives(name="TN"),
      FalseNegatives(name="FN"),
      Precision(name="Precision"),
      Recall(name="Recall"),
      AUC(name="AUC"),
    ]

    loss = 'categorical_crossentropy' if (noOfClasses > 2) else 'binary_crossentropy'

    workingModel.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    callbacks = [
      EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
      ModelCheckpoint(filepath=checkpointPath, monitor="val_loss", save_best_only=True, verbose=1),
      CSVLogger(logPath, separator=",", append=True),
    ]

    # Train the model.
    history = workingModel.fit(
      xTrain, yTrain,
      epochs=int(epochs),
      batch_size=int(batchSize),
      validation_data=(xVal, yVal),
      callbacks=callbacks,
      verbose=1,
    )

    historyDF = pd.DataFrame(history.history)
    with open(historyPath, mode='w') as f:
      historyDF.to_json(f)

    workingModel.load_weights(checkpointPath)

    # Evaluate the model.
    trainResults = workingModel.evaluate(xTrain, yTrain, batch_size=int(batchSize), verbose=1, return_dict=True)
    valResults = workingModel.evaluate(xVal, yVal, batch_size=int(batchSize), verbose=1, return_dict=True)
    testResults = workingModel.evaluate(xTest, yTest, batch_size=int(batchSize), verbose=1, return_dict=True)

    trainResults["Type"] = "Train"
    valResults["Type"] = "Validation"
    testResults["Type"] = "Test"

    evalsDF = pd.DataFrame([trainResults, valResults, testResults])
    with open(evaluationsPath, mode='w') as f:
      evalsDF.to_json(f)

    configs = {
      "Title"                    : title,
      "Model Name"               : model,
      "Optimizer"                : optimizer,
      "Given Shape"              : str(shape),
      "Model Shape"              : str(modelShape),
      "Hidden Activation"        : activation,
      "Is Trainable?"            : trainable,
      "TL Training Ratio"        : tlTrainingRatio,
      "Batch Size"               : batchSize,
      "Training Ratio"           : trainingRatio,
      "Epochs"                   : epochs,
      "Classes"                  : str(classes),
      "Dataset Name"             : datasetRecord["name"],
      "Scaler Technique"         : scaler,
      "Loss Function"            : loss,
      "Pretrained Weights"       : "imagenet",
      "No. of Classes"           : noOfClasses,
      "No. of Layers"            : noOfLayers,
      "No. of Layers to Unfreeze": noOfLayersToUnfreeze,
      "xTrain Shape"             : str(xTrain.shape),
      "yTrain Shape"             : str(yTrain.shape),
      "xVal Shape"               : str(xVal.shape),
      "yVal Shape"               : str(yVal.shape),
      "xTest Shape"              : str(xTest.shape),
      "yTest Shape"              : str(yTest.shape),
      "X Shape"                  : str(X.shape),
      "y Shape"                  : str(y.shape),
    }

    ProjectHandler().UpdateUserClassificationSuccess(username, title, storeName, True, "")
    ProjectHandler().UpdateUserClassificationPostConfigurations(username, title, storeName, configs)

  except Exception as e:
    print("Exception:", e, flush=True)
    ProjectHandler().UpdateUserClassificationSuccess(username, title, storeName, False, str(e))

  sys.stdout.flush()


def ManipulateClassificationInferenceThread(allFields):
  import json, cv2
  import numpy as np
  from tensorflow.keras.backend import clear_session
  from tensorflow.keras.models import load_model

  (target, filename) = allFields

  clear_session()

  try:
    workingFolder = os.path.join(CLASSIFICATIONS_HISTORY_PATH, target)
    configsPath = os.path.join(workingFolder, "configs.json")
    predictionsPath = os.path.join(workingFolder, "predictions.csv")
    imagePath = os.path.join(CLASSIFICATIONS_INFERENCE_PATH, target, filename)

    with open(configsPath, mode='r') as f:
      configs = json.load(f)

    modelPath = configs["Checkpoint Path"]
    scaler = configs["Scaler Technique"]
    modelShape = [int(el) for el in configs["Model Shape"].replace("(", "").replace(")", "").split(", ")]
    noOfClasses = int(configs["No. of Classes"])
    classes = [el.strip() for el in configs["Classes"].replace("[", "").replace("]", "").split(",")]

    image = cv2.imread(imagePath)
    if (len(modelShape) == 2):
      image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.resize(image, (int(modelShape[0]), int(modelShape[1])))

    image = ScaleImage(image, scaler)

    modelObj = load_model(modelPath)
    imageForModel = image.reshape(1, *modelShape)
    prediction = modelObj.predict(imageForModel)

    if (noOfClasses == 2):
      prediction = prediction[0][0]
      clsIndex = 0 if (prediction < 0.5) else 1
      cls = classes[clsIndex]
      prob = prediction  # if (clsIndex == 0) else (1 - prediction)
    else:
      clsIndex = np.argmax(prediction[0])
      cls = classes[clsIndex]
      prob = prediction[0][clsIndex]

    if (not os.path.exists(predictionsPath)):
      with open(predictionsPath, mode='w') as f:
        f.write("Image,Probability,Class\n")

    with open(predictionsPath, mode='a') as f:
      f.write(f"{filename},{prob},{cls}\n")

  except Exception as e:
    print("Exception:", e, flush=True)


def ScaleImage(image, scaler, axis=(1, 2, 3)):
  if (scaler == "MaxNormalization"):
    image = image / 255.0
  elif (scaler == "MinMaxNormalization"):
    minX = image.min(axis=axis, keepdims=True)
    maxX = image.max(axis=axis, keepdims=True)
    image = (image - minX) / (maxX - minX)
  elif (scaler == "MaxAbsNormalization"):
    maxX = image.max(axis=axis, keepdims=True)
    image = image / np.abs(maxX)
  elif (scaler == "Standardization"):
    mean = image.mean(axis=axis, keepdims=True)
    std = image.std(axis=axis, keepdims=True)
    image = (image - mean) / std
  elif (scaler == "L1Normalization"):
    factor = np.sum(np.abs(image), axis=axis, keepdims=True)
    image = image / factor
  elif (scaler == "L2Normalization"):
    factor = np.sqrt(np.sum(image ** 2, axis=axis, keepdims=True))
    image = image / factor
  elif (scaler == "RobustQ1Scaler"):
    q75, q25 = np.percentile(image, [75, 25], axis=axis, keepdims=True)
    iqr = q75 - q25
    image = (image - q25) / iqr
  elif (scaler == "RobustMedianScaler"):
    median = np.median(image, axis=axis, keepdims=True)
    q75, q25 = np.percentile(image, [75, 25], axis=axis, keepdims=True)
    iqr = q75 - q25
    image = (image - median) / iqr
  else:
    print("Unsupported scaler.", flush=True)
  return image


class TilesMasksDataGenerator(Sequence):
  def __init__(self, ids, path, scaler, batchSize=8, imageSize=100):
    self.ids = np.random.permutation(ids)
    self.path = path
    self.batchSize = batchSize
    self.imageSize = imageSize
    self.scaler = scaler
    self.on_epoch_end()

  def __load__(self, idName):
    imagePath = os.path.join(self.path, idName)
    image = cv2.imread(imagePath)
    w_ = int(image.shape[1] / 2)
    mask_ = image[:, w_:, :]
    im_ = image[:, :w_, :]
    mask_ = cv2.cvtColor(mask_, cv2.COLOR_BGR2GRAY)
    im_ = ScaleImage(im_, self.scaler)
    mask_ = ScaleImage(mask_, self.scaler)
    return im_, mask_

  def __getitem__(self, index):
    if ((index + 1) * self.batchSize > len(self.ids)):
      self.batchSize = len(self.ids) - index * self.batchSize
    filesBatch = self.ids[index * self.batchSize: (index + 1) * self.batchSize]
    images, masks = [], []
    for idName in filesBatch:
      img_, mask_ = self.__load__(idName)
      images.append(img_)
      masks.append(mask_)
    images = np.array(images)
    masks = np.array(masks)
    return images, masks

  def on_epoch_end(self):
    self.ids = np.random.permutation(self.ids)

  def __len__(self):
    return int(np.ceil(len(self.ids) / float(self.batchSize)))


def HandleWeakSegmentationROI(
    projectPath, imagePath, annotationName, saveFolder,
    imageLevel=0, tileWidth=256, tileHeight=256
):
  os.add_dll_directory(os.path.join(os.getcwd(), "Packages", "openslide-win64-20221111", "bin"))

  import openslide
  from paquo.projects import QuPathProject

  print("Working on the annotation:", annotationName, flush=True)

  # Read the QuPath project.
  project = QuPathProject(projectPath, mode='r')

  # Get the image.
  image = project.images[0]
  print("Image:", image)

  isAnnotationFound = False
  annotationIndex = -1

  # Get the annotations.
  annotations = image.hierarchy.annotations
  for i, annotation in enumerate(annotations):
    print("Found the annotation:", annotation.name, flush=True)  # , annotation.path_class
    if (str(annotation.name).strip() == annotationName.strip()):
      isAnnotationFound = True
      annotationIndex = i
      break

  if (not isAnnotationFound):
    print(f"Annotation '{annotationName}' is not found.", flush=True)
    return

  # Get the annotations.
  annotation = annotations[annotationIndex]
  roi = annotation.parent

  # Get the dimensions of the working ROIs.
  xy = roi.roi.exterior.xy
  xLeft = int(xy[0][0])
  xRight = int(xy[0][2])
  yTop = int(xy[1][0])
  yBottom = int(xy[1][1])

  print("ROI Dimensions:", xLeft, xRight, yTop, yBottom, flush=True)

  # Get the ROIs.
  roiAnnotation = annotation.roi

  # Get the coords.
  xList = [np.array(el.exterior.xy[0]) for el in roiAnnotation.geoms]
  yList = [np.array(el.exterior.xy[1]) for el in roiAnnotation.geoms]

  s = openslide.OpenSlide(imagePath)
  print("Image Dimensions:", s.dimensions, flush=True)

  # Get the image dimensions.
  imageWidth, imageHeight = s.dimensions[0], s.dimensions[1]

  # Get the image downsample.
  imageDownsample = int(s.level_downsamples[imageLevel])

  # Get the image size.
  imageSize = s.level_dimensions[imageLevel]

  # Read the image.
  image = s.read_region((0, 0), imageLevel, imageSize)

  # Downsample the x and y coords.
  xList = [x_ / imageDownsample for x_ in xList]
  yList = [y_ / imageDownsample for y_ in yList]

  polygons = [
    np.array([(x_, y_) for x_, y_ in zip(xList[i], yList[i])]).astype(np.int32)
    for i in range(len(xList))
  ]
  # Cast the polygons.
  polygons = np.array(polygons, dtype=object)

  print("Number of Polygons:", len(polygons), flush=True)

  # Downsample the dimensions of the ROIs.
  xLeft = int(xLeft / imageDownsample)
  xRight = int(xRight / imageDownsample)
  yTop = int(yTop / imageDownsample)
  yBottom = int(yBottom / imageDownsample)

  # Convert the image to a NumPy array.
  imageLR = np.array(image).astype(np.uint8)

  # Convert the image to BGR.
  imageLR = cv2.cvtColor(imageLR, cv2.COLOR_RGB2BGR)

  # Create a mask.
  mask = np.zeros(imageLR.shape, dtype=np.uint8)

  # Fill the mask with polygons.
  cv2.fillPoly(mask, polygons, (255, 255, 255))

  # Draw the polygons.
  # imageLRContours = cv2.drawContours(imageLR, polygons, -1, (0, 255, 0), 1)

  croppedImageLR = imageLR[yTop:yBottom, xLeft:xRight, :]
  croppedMaskLR = mask[yTop:yBottom, xLeft:xRight, :]

  print("Cropped Image Dimensions:", croppedImageLR.shape, flush=True)
  print("Cropped Mask Dimensions:", croppedMaskLR.shape, flush=True)

  # Display the image.
  # cv2.imshow("ROI LR Image", croppedImageLR)
  # cv2.waitKey(0)

  # Get the number of tiles.
  numXTiles = int((xRight - xLeft) / tileWidth)
  numYTiles = int((yBottom - yTop) / tileHeight)

  print("Number of Tiles:", numXTiles, numYTiles, flush=True)

  # Extract the tiles.
  for i in range(numXTiles):
    for j in range(numYTiles):
      xLeft_ = i * tileWidth
      xRight_ = xLeft_ + tileWidth
      yTop_ = j * tileHeight
      yBottom_ = yTop_ + tileHeight
      tile_ = croppedImageLR[yTop_:yBottom_, xLeft_:xRight_, :]
      mask_ = croppedMaskLR[yTop_:yBottom_, xLeft_:xRight_, :]
      conc_ = cv2.hconcat([tile_, mask_])
      cv2.imwrite(os.path.join(saveFolder, f"Tile_Mask_{annotationName}_{i + 1}_{j + 1}.png"), conc_)


def UNetModel(inputShape, activation):
  from tensorflow.keras.layers import Conv2D, MaxPool2D, Input, UpSampling2D, Concatenate
  from tensorflow.keras.models import Model
  from tensorflow.keras.backend import clear_session

  def _downBlock(x, filters, kernelSize=(3, 3), padding="same", strides=1):
    c = Conv2D(filters, kernelSize, padding=padding, strides=strides, activation=activation)(x)
    c = Conv2D(filters, kernelSize, padding=padding, strides=strides, activation=activation)(c)
    p = MaxPool2D((2, 2), (2, 2))(c)
    return c, p

  def _upBlock(x, skip, filters, kernelSize=(3, 3), padding="same", strides=1):
    us = UpSampling2D((2, 2))(x)
    concat = Concatenate()([us, skip])
    c = Conv2D(filters, kernelSize, padding=padding, strides=strides, activation=activation)(concat)
    c = Conv2D(filters, kernelSize, padding=padding, strides=strides, activation=activation)(c)
    return c

  def _bottleneck(x, filters, kernelSize=(3, 3), padding="same", strides=1):
    c = Conv2D(filters, kernelSize, padding=padding, strides=strides, activation=activation)(x)
    c = Conv2D(filters, kernelSize, padding=padding, strides=strides, activation=activation)(c)
    return c

  clear_session()
  f = [16, 16, 32, 64, 128, 256]
  inputs = Input(inputShape)

  p0 = inputs
  c1, p1 = _downBlock(p0, f[0])  # 128 -> 64
  c2, p2 = _downBlock(p1, f[1])  # 64 -> 32
  c3, p3 = _downBlock(p2, f[2])  # 32 -> 16
  c4, p4 = _downBlock(p3, f[3])  # 16->8
  c5, p5 = _downBlock(p4, f[4])  # 16->8

  bn = _bottleneck(p5, f[5])

  u0 = _upBlock(bn, c5, f[4])  # 8 -> 16
  u1 = _upBlock(u0, c4, f[3])  # 8 -> 16
  u2 = _upBlock(u1, c3, f[2])  # 16 -> 32
  u3 = _upBlock(u2, c2, f[1])  # 32 -> 64
  u4 = _upBlock(u3, c1, f[0])  # 64 -> 128

  outputs = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
  model = Model(inputs, outputs)
  return model


# https://notebook.community/cshallue/models/samples/outreach/blogs/segmentation_blogpost/image_segmentation
def DiceCoeff(yTrue, yPred, smooth=1):
  import tensorflow as tf

  yTrue_ = tf.reshape(yTrue, [-1])
  yPred_ = tf.reshape(yPred, [-1])
  intersection = tf.reduce_sum(yTrue_ * yPred_)
  score = (2.0 * intersection + smooth) / (tf.reduce_sum(yTrue_) + tf.reduce_sum(yPred_) + smooth)
  return score


def DiceLoss(yTrue, yPred):
  loss = 1 - DiceCoeff(yTrue, yPred)
  return loss


def BCEDiceLoss(yTrue, yPred):
  from tensorflow.keras.losses import binary_crossentropy

  loss = binary_crossentropy(yTrue, yPred) + DiceCoeff(yTrue, yPred)
  return loss


def IOUCoeff(yTrue, yPred, smooth=1):
  import tensorflow as tf

  intersection = tf.reduce_sum(yTrue * yPred)
  union = tf.reduce_sum(yTrue) + tf.reduce_sum(yPred) - intersection
  iou = (intersection + smooth) / (union + smooth)
  return iou


def IOULoss(yTrue, yPred):
  loss = 1 - IOUCoeff(yTrue, yPred)
  return loss


def BCEIOULoss(yTrue, yPred):
  from tensorflow.keras.losses import binary_crossentropy

  loss = binary_crossentropy(yTrue, yPred) + IOULoss(yTrue, yPred)
  return loss


def StorePickle(data, path):
  with open(path, 'wb') as f:
    pickle.dump(data, f)


def LoadPickle(path):
  with open(path, 'rb') as f:
    data = pickle.load(f)
  return data


def ManipulateWeaklyUNetSegmentation(allFields):
  import json, cv2, glob
  import numpy as np
  import pandas as pd
  from datetime import datetime
  from tensorflow.keras.layers import Conv2D, MaxPool2D, Input, UpSampling2D, Concatenate
  from tensorflow.keras.models import Model
  from tensorflow.keras.backend import clear_session
  from tensorflow.keras.models import load_model
  from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger
  from sklearn.model_selection import train_test_split
  from tensorflow.keras.metrics import (
    TruePositives,
    FalsePositives,
    TrueNegatives,
    FalseNegatives,
    Precision,
    Recall,
    AUC,
  )

  (
    title, storeName, wsiFile, annotations, model, activation, batchSize, trainingRatio, epochs, scaler,
    shape, shuffle, username, qupathProjectFilename
  ) = allFields

  clear_session()

  storePath = os.path.join(SEGMENTATIONS_PATH, storeName)
  inferencePath = os.path.join(storePath, "Inference")
  os.makedirs(storePath, exist_ok=True)
  os.makedirs(inferencePath, exist_ok=True)

  checkpointPath = os.path.join(storePath, f"model.h5")
  logPath = os.path.join(storePath, f"log.csv")
  historyPath = os.path.join(storePath, f"history.json")
  evaluationsPath = os.path.join(storePath, f"evaluations.json")

  qupathProjectFilenameNoExt = ".".join(qupathProjectFilename.split(".")[:-1])
  qupathProjectSavePath = os.path.join(storePath, qupathProjectFilename)
  qupathProjectSavePathNoExt = os.path.join(storePath, qupathProjectFilenameNoExt)

  with zipfile.ZipFile(qupathProjectSavePath, 'r') as zipRef:
    zipRef.extractall(qupathProjectSavePathNoExt)

  try:
    isFound = False
    projectPath = None
    for root, dirs, files in os.walk(qupathProjectSavePathNoExt):
      for file in files:
        isFound = True
        projectPath = os.path.join(root, file)
        break
      if (isFound):
        break

    if (not isFound):
      raise Exception("Project file not found.")

    shape = shape.split(",")
    shape = [int(i) for i in shape]
    shape = (256, 256, 3) if (model == "Manual U-Net (Input is 256x256x3)") else [int(el) for el in shape.split(",")]
    annotationsList = annotations.split(",")
    annotationsList = [annotation.strip() for annotation in annotationsList]
    tileWidth, tileHeight = shape[:2]
    imageLevel = 0
    imagePath = os.path.join(WSI_PATH, wsiFile)
    shuffleFlag = True if (shuffle == "1") else False

    # Split dataset into training, testing, and validation.
    datasetPath = os.path.join(storePath, "Dataset")
    os.makedirs(datasetPath, exist_ok=True)

    for annotation in annotationsList:
      HandleWeakSegmentationROI(
        projectPath, imagePath, annotation, datasetPath,
        imageLevel=imageLevel, tileWidth=tileWidth, tileHeight=tileHeight,
      )

    datasetFiles = os.listdir(datasetPath)

    if (shuffleFlag):
      np.random.shuffle(datasetFiles)

    trainFiles, testFiles = train_test_split(datasetFiles, train_size=float(trainingRatio), shuffle=shuffleFlag)
    trainFiles, valFiles = train_test_split(trainFiles, train_size=float(trainingRatio), shuffle=shuffleFlag)

    StorePickle(
      {"train": trainFiles, "test": testFiles, "val": valFiles},
      os.path.join(storePath, "Split.p")
    )

    trainDG = TilesMasksDataGenerator(trainFiles, datasetPath, scaler, imageSize=shape, batchSize=int(batchSize))
    validationDG = TilesMasksDataGenerator(valFiles, datasetPath, scaler, imageSize=shape, batchSize=int(batchSize))
    testDG = TilesMasksDataGenerator(testFiles, datasetPath, scaler, imageSize=shape, batchSize=int(batchSize))

    trainSteps = len(trainDG) // int(batchSize)
    validationSteps = len(validationDG) // int(batchSize)
    if (validationSteps <= 0):
      validationSteps = 1

    metrics = [
      "accuracy",
      DiceCoeff,
      IOUCoeff,
      TruePositives(name="TP"),
      FalsePositives(name="FP"),
      TrueNegatives(name="TN"),
      FalseNegatives(name="FN"),
      Precision(name="Precision"),
      Recall(name="Recall"),
      AUC(name="AUC"),
    ]

    if (model == "Manual U-Net (Input is 256x256x3)"):
      workingModel = UNetModel(shape, activation)
      workingModel.compile(optimizer="adam", loss="binary_crossentropy", metrics=metrics)
      workingModel.summary()
    else:
      raise Exception("Model not found.")

    callbacks = [
      EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
      CSVLogger(logPath, separator=",", append=True),
      ModelCheckpoint(checkpointPath, save_best_only=True, verbose=1, monitor="val_loss", mode="min"),
    ]

    history = workingModel.fit(
      trainDG,
      steps_per_epoch=trainSteps,
      validation_data=validationDG,
      validation_steps=validationSteps,
      epochs=int(epochs),
      verbose=2,
      callbacks=callbacks,
    )

    historyDF = pd.DataFrame(history.history)
    with open(historyPath, mode='w') as f:
      historyDF.to_json(f)

    workingModel.load_weights(checkpointPath)

    # Evaluate the model.
    trainResults = workingModel.evaluate(trainDG, batch_size=int(batchSize), verbose=1, return_dict=True)
    valResults = workingModel.evaluate(validationDG, batch_size=int(batchSize), verbose=1, return_dict=True)
    testResults = workingModel.evaluate(testDG, batch_size=int(batchSize), verbose=1, return_dict=True)

    trainResults["Type"] = "Train"
    valResults["Type"] = "Validation"
    testResults["Type"] = "Test"

    evalsDF = pd.DataFrame([trainResults, valResults, testResults])
    with open(evaluationsPath, mode='w') as f:
      evalsDF.to_json(f)

    configs = {
      "Title"            : title,
      "Model Name"       : model,
      "Working Shape"    : str(shape),
      "Hidden Activation": activation,
      "Batch Size"       : batchSize,
      "Training Ratio"   : trainingRatio,
      "Shuffle Dataset?" : shuffle,
      "Epochs"           : epochs,
      "Annotations"      : str(annotations),
      "Scaler Technique" : scaler,
      "Loss Function"    : "Binary Cross-entropy (BCE)",
      "Train Shape"      : str(len(trainFiles)),
      "Validation Shape" : str(len(valFiles)),
      "Test Shape"       : str(len(testFiles)),
    }

    ProjectHandler().UpdateUserSegmentationSuccess(username, title, storeName, True, "")
    ProjectHandler().UpdateUserSegmentationPostConfigurations(username, title, storeName, configs)

  except Exception as e:
    print("Exception:", e, flush=True)
    ProjectHandler().UpdateUserSegmentationSuccess(username, title, storeName, False, str(e))

  sys.stdout.flush()


def ManipulateWeaklyUNetSegmentationInference(imagePath, segmenter, configs):
  import json, cv2
  import numpy as np
  from tensorflow.keras.backend import clear_session
  from tensorflow.keras.models import load_model

  configs = json.loads(configs)

  modelPath = os.path.join(SEGMENTATIONS_PATH, segmenter, "model.h5")
  scaler = configs["Scaler Technique"]
  modelShape = [int(el) for el in configs["Working Shape"].replace("(", "").replace(")", "").split(", ")]

  image = cv2.imread(imagePath)
  if (len(modelShape) == 2):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
  image = cv2.resize(image, (int(modelShape[0]), int(modelShape[1])))

  image = ScaleImage(image, scaler)

  modelObj = load_model(modelPath, custom_objects={"DiceCoeff": DiceCoeff, "IOUCoeff": IOUCoeff})
  imageForModel = image.reshape(1, *modelShape)
  prediction = modelObj.predict(imageForModel)

  maskImg = np.squeeze(prediction[0])
  maskImg = cv2.resize(maskImg, (image.shape[1], image.shape[0]))
  maskImg = (maskImg * 255).astype(np.uint8)  # Is suitable with other scalers?

  return maskImg


def ManipulateWeaklyUNetSegmentationInferenceThread(allFields):
  (target, filename) = allFields

  clear_session()

  try:
    workingFolder = os.path.join(SEGMENTATIONS_HISTORY_PATH, target)
    configsPath = os.path.join(workingFolder, "configs.json")
    imagePath = os.path.join(SEGMENTATIONS_INFERENCE_PATH, target, filename)
    maskPath = os.path.join(SEGMENTATIONS_INFERENCE_PATH, target, "Mask_" + filename)

    maskImg = ManipulateWeaklyUNetSegmentationInference(imagePath, configsPath)
    cv2.imwrite(maskPath, maskImg)

  except Exception as e:
    print("Exception:", e, flush=True)


def ManipulateClassificationInference(imagePath, classifierStoreName, configs):
  import json, cv2
  import numpy as np
  from tensorflow.keras.backend import clear_session
  from tensorflow.keras.models import load_model

  configs = json.loads(configs)

  modelPath = os.path.join(CLASSIFICATIONS_PATH, classifierStoreName, "model.h5")
  scaler = configs["Scaler Technique"]
  classes = configs["Classes"].strip('][').split(', ')
  noOfClasses = configs["No. of Classes"]
  modelShape = [int(el) for el in configs["Model Shape"].replace("(", "").replace(")", "").split(", ")]

  image = cv2.imread(imagePath)
  if (len(modelShape) == 2):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
  image = cv2.resize(image, (int(modelShape[0]), int(modelShape[1])))

  image = ScaleImage(image, scaler, axis=(1, 2))

  modelObj = load_model(modelPath, custom_objects={})
  imageForModel = image.reshape(1, *modelShape)
  prediction = modelObj.predict(imageForModel)

  if (noOfClasses == 2):
    prediction = prediction[0][0]
    clsIndex = 0 if (prediction < 0.5) else 1
    cls = classes[clsIndex]
    prob = prediction if (clsIndex == 1) else (1 - prediction)
  else:
    clsIndex = np.argmax(prediction[0])
    cls = classes[clsIndex]
    prob = prediction[0][clsIndex]

  prob = np.round(prob, 4)

  return str(cls), str(prob)


def LiveImageAugmentation(
    imagePath,
    rotation=0,
    widthShift=0,
    heightShift=0,
    horizontalFlip=False,
    verticalFlip=False,
    zoom=0,
    shear=0,
    brightness=0,
    saturation=0,
    hue=0,
    isSpecific=False,
    isSingle=False,
    encode=True
):
  image = cv2.imread(imagePath)

  def _ChangeRotation(image, angle=-25, deltaWindowSize=(0, 0)):
    numRows, numCols = image.shape[:2]
    rotationMatrix = cv2.getRotationMatrix2D((numCols / 2.0, numRows / 2.0), angle, 1)
    result = cv2.warpAffine(image, rotationMatrix, (numCols + deltaWindowSize[0], numRows + deltaWindowSize[1]))
    result = cv2.merge(cv2.split(result))
    return result

  def _ChangeZoom(img, value=0):
    h, w = img.shape[:2]
    M = np.float32([[1 + value, 0, 0], [0, 1 + value, 0]])
    img = cv2.warpAffine(img, M, (w, h))
    return img

  def _ChangeShear(img, value=0):
    h, w = img.shape[:2]
    M = np.array([[1, value, 0], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (w, h))
    return img

  def _ChangeWidthShift(img, value=0):
    h, w = img.shape[:2]
    M = np.array([[1, 0, value * w], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (w, h))
    return img

  def _ChangeHeightShift(img, value=0):
    h, w = img.shape[:2]
    M = np.array([[1, 0, 0], [0, 1, value * h]])
    img = cv2.warpAffine(img, M, (w, h))
    return img

  def _ChangeHorizontalFlip(img):
    return cv2.flip(img, 1)

  def _ChangeVerticalFlip(img):
    return cv2.flip(img, 0)

  def _ChangeBrightness(img, value=0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value
    hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

  def _ChangeSaturation(img, value=0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 255 - value
    s[s > lim] = 255
    s[s <= lim] += value
    hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

  def _ChangeHue(img, value=0):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lim = 180 - value
    h[h > lim] = 180
    h[h <= lim] += value
    hsv = cv2.merge((h, s, v))
    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img

  whichIsRun = ""
  configsDict = {
    "Rotation"       : rotation if (isSpecific or rotation == 0) else np.random.randint(-rotation, rotation),
    "Width Shift"    : widthShift if (isSpecific or widthShift == 0) else np.round(
      np.random.uniform(-widthShift, widthShift), 3),
    "Height Shift"   : heightShift if (isSpecific or heightShift == 0) else np.round(np.random.uniform(
      -heightShift, heightShift), 3),
    "Horizontal Flip": horizontalFlip if (isSpecific or horizontalFlip == 0) else np.random.randint(0, 2),
    "Vertical Flip"  : verticalFlip if (isSpecific or verticalFlip == 0) else np.random.choice([True, False]),
    "Zoom"           : zoom if (isSpecific or zoom == 0) else np.round(np.random.uniform(-zoom, zoom), 3),
    "Shear"          : shear if (isSpecific or shear == 0) else np.round(np.random.uniform(-shear, shear), 3),
    "Brightness"     : brightness if (isSpecific or brightness == 0) else np.random.randint(0, brightness),
    "Saturation"     : saturation if (isSpecific or saturation == 0) else np.random.randint(0, saturation),
    "Hue"            : hue if (isSpecific or hue == 0) else np.random.randint(0, hue),
  }

  if (isSingle):
    idxList = [i for i, el in enumerate(configsDict.keys()) if configsDict[el] != 0]
    if (len(idxList) > 0):
      randomIndex = np.random.choice(idxList)
      if (randomIndex == 0):
        whichIsRun = "Rotation"
        image = _ChangeRotation(image, configsDict[whichIsRun])
      elif (randomIndex == 1):
        whichIsRun = "Width Shift"
        image = _ChangeWidthShift(image, configsDict[whichIsRun])
      elif (randomIndex == 2):
        whichIsRun = "Height Shift"
        image = _ChangeHeightShift(image, configsDict[whichIsRun])
      elif (randomIndex == 3):
        whichIsRun = "Horizontal Flip"
        if (configsDict[whichIsRun]):
          image = _ChangeHorizontalFlip(image)
      elif (randomIndex == 4):
        whichIsRun = "Vertical Flip"
        if (configsDict[whichIsRun]):
          image = _ChangeVerticalFlip(image)
      elif (randomIndex == 5):
        whichIsRun = "Zoom"
        image = _ChangeZoom(image, configsDict[whichIsRun])
      elif (randomIndex == 6):
        whichIsRun = "Shear"
        image = _ChangeShear(image, configsDict[whichIsRun])
      elif (randomIndex == 7):
        whichIsRun = "Brightness"
        image = _ChangeBrightness(image, configsDict[whichIsRun])
      elif (randomIndex == 8):
        whichIsRun = "Saturation"
        image = _ChangeSaturation(image, configsDict[whichIsRun])
      elif (randomIndex == 9):
        whichIsRun = "Hue"
        image = _ChangeHue(image, configsDict[whichIsRun])
  else:
    if (brightness > 0):
      image = _ChangeBrightness(image, configsDict["Brightness"])
    if (saturation > 0):
      image = _ChangeSaturation(image, configsDict["Saturation"])
    if (hue > 0):
      image = _ChangeHue(image, configsDict["Hue"])
    if (rotation > 0):
      image = _ChangeRotation(image, configsDict["Rotation"])
    if (widthShift > 0):
      image = _ChangeWidthShift(image, configsDict["Width Shift"])
    if (heightShift > 0):
      image = _ChangeHeightShift(image, configsDict["Height Shift"])
    if (horizontalFlip):
      if (configsDict["Horizontal Flip"]):
        image = _ChangeHorizontalFlip(image)
    if (verticalFlip):
      if (configsDict["Vertical Flip"]):
        image = _ChangeVerticalFlip(image)
    if (zoom > 0):
      image = _ChangeZoom(image, configsDict["Zoom"])
    if (shear > 0):
      image = _ChangeShear(image, configsDict["Shear"])

  imagePathNotExt = os.path.splitext(imagePath)[0]
  imagePathExt = os.path.splitext(imagePath)[1]
  imagePathAugmented = imagePathNotExt + "_augmented" + imagePathExt

  cv2.imwrite(imagePathAugmented, image)

  print("Augmented image saved to:", imagePathAugmented, flush=True)

  if (encode):
    image = EncodeImageFromOpenCV(image)
    image = image.decode('utf-8')

  for k in configsDict.keys():
    configsDict[k] = str(configsDict[k])
  configs = list(configsDict.items())

  return image, whichIsRun, configs
