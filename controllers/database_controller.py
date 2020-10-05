import pymongo
from bson.json_util import dumps

dbconnection = pymongo.MongoClient("mongodb://localhost:27017/")

dblist = dbconnection.list_database_names()


if "modeldatabase" in dblist:
    print("The database exists.")
    db = dbconnection["modeldatabase"]
    if "statustable" in db.list_collection_names() and "traininginfo" in db.list_collection_names():
        print ("Tables Exists")
    statuscol = db["statustable"]
    traincol  = db["traininginfo"]
    projectcol = db["projectinfo"]
    losscol = db["lossinfo"]
else:
    db = dbconnection["modeldatabase"]
    statuscol = db["statustable"]
    traincol  = db["traininginfo"]
    projectcol = db["projectinfo"]
    losscol = db["lossinfo"]


""" Insert status in database collection statustable when training begins """
def statusenter(userid, project, status_start, modelname, classes, dt_start, epochs):  

    statusquery = { "userid": userid, "project": project, "status": status_start, "modelname" : modelname, "classes": classes, "date_start" : dt_start, "epochs_recieved": epochs}
    statuscol.insert_one(statusquery)


""" Update status in database collection statustable when training ends """
def statusupdate(userid, project, status_end, modelname, epochs):  

    statuscheck = { "userid": userid, "project": project, "modelname" : modelname, "epochs_recieved": epochs}
    statuscol.find_one_and_update(statuscheck,{"$set": {"status": status_end}}, upsert = True)

""" Insert loss in database collection lossinfo when training begins """
def lossinsert(userid, project, epoch, loss, modelname, classes, dt_start, epochs):  

    lossquery = { "userid": userid, "project": project, "epoch": epoch, "loss": loss, "modelname" : modelname, "classes": classes, "date_start" : dt_start, "epochs_recieved": epochs}
    losscol.insert_one(lossquery)

""" Insert status in database collection traininginfo when training ends """
def traininfoinsert(userid, project, status_end, dt_start, dt_end, losses, modelname, classes, epochs):

    trainquery = { "userid": userid, "project": project,  "status": status_end,  "date_start" : dt_start, "date_end" : dt_end, "losses" : losses, "modelname" : modelname, "classes": classes, "epochs_recieved": epochs }
    traincol.insert_one(trainquery)


""" fetch last training status from database collection traininginfo when training ends """
def last_train_info_get_by_user(userid, project, classes, epochs):

    statusfind = { "userid": userid, "project": project, "classes": classes, "epochs_recieved": epochs}
    
    
    statq = statuscol.find_one(statusfind,sort=[('date_start', pymongo.DESCENDING)])

    
    if statq['status'] == 'started' or statq['status'] == 'error':
        modelname = statq['modelname']
        date_start = statq['date_start']
        lossq = { "userid": userid, "project": project, "date_start" : date_start, "modelname" : modelname, "classes": classes, "epochs_recieved": epochs}
        return dumps(losscol.find(lossq))
        
    elif statq['status'] == 'completed':
        modelname = statq['modelname']
        date_start = statq['date_start']
        trainfind = { "userid": userid, "project": project, "date_start" : date_start, "modelname" : modelname, "classes": classes, "epochs_recieved": epochs }
        return dumps(traincol.find_one(trainfind))

def train_info_get_by_user(userid):

    statusfind = {"userid": userid}
    return dumps(statuscol.find(statusfind))


""" Insert file information in database based on project """    
def project_file_info(userid, project, image_filename, xml_filename, size, object_data):

    project_file_query = { "userid": userid, "project": project,  "image_filename": image_filename,  "xml_filename" : xml_filename, "size": size, "object": object_data }
    projectcol.insert_one(project_file_query)

