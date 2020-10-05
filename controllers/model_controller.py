from detecto import core, utils
from torchvision import transforms
import matplotlib.pyplot as plt
import torch
from constants import model_constants
from utilities import split_files
import os
from datetime import datetime
from controllers import database_controller
import logging
import logging.handlers
import sys
import io
import pathlib
from subprocess import Popen, PIPE



def modeltrain(userid, project, epochs, classes):
    torch.cuda.empty_cache()
    
    xml = "xml"
    PROJECT_DIRECTORY = "users" + "/" + userid + "/" + project + "/"
    split_files.iterate_dir(PROJECT_DIRECTORY + "images", PROJECT_DIRECTORY + "output", model_constants.SPLIT_RATIO, xml)

    utils.xml_to_csv(PROJECT_DIRECTORY + model_constants.TRAIN_LABELS, PROJECT_DIRECTORY + model_constants.CSV_TRAIN_LABELS)
    utils.xml_to_csv(PROJECT_DIRECTORY + model_constants.VALIDATE_LABELS, PROJECT_DIRECTORY + model_constants.CSV_VALIDATE_LABELS)

    custom_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(800),
        transforms.ColorJitter(saturation=0.3),
        transforms.ToTensor(),
        utils.normalize_transform(),
    ])
        
    dataset = core.Dataset(PROJECT_DIRECTORY + model_constants.CSV_TRAIN_LABELS, PROJECT_DIRECTORY + model_constants.TRAIN_IMAGES,
                    transform=custom_transforms)
                       
    val_dataset = core.Dataset(PROJECT_DIRECTORY + model_constants.CSV_VALIDATE_LABELS, PROJECT_DIRECTORY + model_constants.VALIDATE_IMAGES)

    loader = core.DataLoader(dataset, batch_size=model_constants.BATCH_SIZE, shuffle=True)

    model = core.Model(classes)

    startnow = datetime.now()
    dt_start = startnow.strftime("%d-%m-%Y-%H-%M-%S")

    status_start = 'started'
    modelname = model_constants.MODEL_NAME + dt_start + ".pth"
    print (modelname)
    database_controller.statusenter(userid, project, status_start, modelname, classes, dt_start, epochs)
    
    if True:
        losses = model.fit(loader, userid, project, modelname, classes, dt_start, val_dataset, epochs=epochs,learning_rate=model_constants.LEARNING_RATE, verbose=True)
        status_end = 'completed'
    else:
        status_end = 'error'
        print ("Training Error")

    
    model.save(PROJECT_DIRECTORY + model_constants.MODEL_NAME + dt_start + ".pth")
    
    modelfile = pathlib.Path(PROJECT_DIRECTORY + model_constants.MODEL_NAME + dt_start + ".pth")
    if modelfile.exists():
        status_end = 'completed'
        print ("Model Saved")
    else:
        status_end = 'error'
        print ("File not exist")

    stopnow = datetime.now()
    # status_end = 'completed'
    dt_end = stopnow.strftime("%d-%m-%Y-%H-%M-%S")
    database_controller.statusupdate(userid, project, status_end, modelname, epochs)

    database_controller.traininfoinsert(userid, project, status_end, dt_start, dt_end, losses, modelname,classes, epochs)

    del losses, model, loader, dataset, val_dataset, custom_transforms

