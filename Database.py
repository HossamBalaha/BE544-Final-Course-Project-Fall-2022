import mysql.connector


class DatabaseHandler(object):
  HOST = 'localhost'
  USER = 'root'
  PASSWORD = ''
  DATABASE = 'be544'

  def __init__(self):
    self.connection = mysql.connector.connect(
      host=self.HOST,
      user=self.USER,
      password=self.PASSWORD,
      database=self.DATABASE
    )
    self.cursor = self.connection.cursor()

  def insert(self, query, values):
    self.cursor.execute(query, values)
    self.connection.commit()
    return self.cursor.lastrowid, self.cursor.rowcount, self.cursor.statement

  def select(self, query, values):
    self.cursor.execute(query, values)
    columns = [col[0] for col in self.cursor.description]
    rows = [dict(zip(columns, row)) for row in self.cursor.fetchall()]
    return rows

  def singleSelect(self, query, values):
    self.cursor.execute(query, values)
    columns = [col[0] for col in self.cursor.description]
    record = self.cursor.fetchone()
    if record is not None:
      return dict(zip(columns, record))
    else:
      return None

  def update(self, query, values):
    self.cursor.execute(query, values)
    self.connection.commit()
    return self.cursor.lastrowid, self.cursor.rowcount, self.cursor.statement

  def delete(self, query, values):
    self.cursor.execute(query, values)
    self.connection.commit()
    return self.cursor.lastrowid, self.cursor.rowcount, self.cursor.statement

  def __del__(self):
    self.connection.close()


class ProjectHandler(object):
  def Authenticate(self, username, password):
    db = DatabaseHandler()
    query = "SELECT * FROM users WHERE username = %s AND password = %s;"
    values = (username, password)
    result = db.singleSelect(query, values)
    if (result is None):
      return False
    else:
      return True

  def AddNewUser(self, username, password, confirm):
    if (password != confirm):
      return {"success": False, "message": "Password and its confirmation do not match."}

    db = DatabaseHandler()

    query = "SELECT * FROM users WHERE username = %s;"
    values = (username,)
    result = db.singleSelect(query, values)
    if (result is not None):
      return {"success": False, "message": "Username already exists."}

    query = "INSERT INTO users (username, password) VALUES (%s, %s);"
    values = (username, password)
    lastRowID, rowCount, statement = db.insert(query, values)
    if (rowCount == 1):
      return {"success": True, "message": "User is added successfully."}
    else:
      return {"success": False, "message": "Could not add new user."}

  def GenerateUniqueFilename(self, table, extension, includeExtension=True):
    import uuid
    db = DatabaseHandler()
    query = "SELECT * FROM " + table + " WHERE store_name = %s;"
    while True:
      uniqueValue = str(uuid.uuid4())
      if (extension is not None):
        filename = uniqueValue + "." + extension
      else:
        filename = uniqueValue
      values = (filename,)
      result = db.singleSelect(query, values)
      if (result is None):
        if (includeExtension):
          return filename
        else:
          return uniqueValue

  def GetUniqueDatasetFilename(self, extension, includeExtension=True):
    return self.GenerateUniqueFilename("user_datasets", extension, includeExtension=includeExtension)

  def CountUserDatasets(self, username, relativePath=""):
    db = DatabaseHandler()
    query = "SELECT COUNT(*) as K FROM user_datasets WHERE relative_path = %s AND user_id = (SELECT id FROM users WHERE username = %s);"
    values = (relativePath, username)
    result = db.singleSelect(query, values)
    return result["K"]

  def SelectUserDatasetByStoreName(self, username, storeName):
    db = DatabaseHandler()
    query = "SELECT * FROM user_datasets WHERE user_id = (SELECT id FROM users WHERE username = %s) AND store_name = %s;"
    values = (username, storeName)
    result = db.singleSelect(query, values)
    return result

  def GetUserDatasets(self, username, relativePath="", fromIndex=0, amount=10):
    db = DatabaseHandler()
    if (amount is None):
      query = "SELECT * FROM user_datasets WHERE relative_path = %s AND user_id = (SELECT id FROM users WHERE username = %s);"
      values = (relativePath, username)
    else:
      query = "SELECT * FROM user_datasets WHERE relative_path = %s AND user_id = (SELECT id FROM users WHERE username = %s) LIMIT %s, %s;"
      values = (relativePath, username, fromIndex, amount)
    result = db.select(query, values)
    return result

  def IsUserDatasetBelongToUser(self, username, datasetName):
    db = DatabaseHandler()
    query = "SELECT * FROM user_datasets WHERE user_id = (SELECT id FROM users WHERE username = %s) AND store_name = %s;"
    values = (username, datasetName)
    result = db.singleSelect(query, values)
    if (result is None):
      return False
    else:
      return True

  def AddUserDataset(self, username, datasetName, storeName, relativePath, isFile):
    db = DatabaseHandler()
    query = "INSERT INTO user_datasets (user_id, name, store_name, relative_path, is_file) VALUES ((SELECT id FROM users WHERE username = %s), %s, %s, %s, %s);"
    values = (username, datasetName, storeName, relativePath, isFile)
    lastRowID, rowCount, statement = db.insert(query, values)
    if (rowCount == 1):
      return {"success": True, "message": "Dataset is added successfully."}
    else:
      return {"success": False, "message": "Could not add new dataset."}

  def DeleteUserDataset(self, username, id):
    db = DatabaseHandler()
    query = "DELETE FROM user_datasets WHERE user_id = (SELECT id FROM users WHERE username = %s) AND id = %s;"
    values = (username, id)
    lastRowID, rowCount, statement = db.delete(query, values)
    if (rowCount == 1):
      return {"success": True, "message": "Dataset is deleted successfully."}
    else:
      return {"success": False, "message": "Could not delete dataset."}

  def GetUniqueWSIFilename(self, extension, includeExtension=True):
    return self.GenerateUniqueFilename("user_wsis", extension, includeExtension=includeExtension)

  def CountUserWSIs(self, username, searchFor="DZI"):
    db = DatabaseHandler()
    query = "SELECT COUNT(*) as K FROM user_wsis WHERE type = %s AND user_id = (SELECT id FROM users WHERE username = %s);"
    values = (searchFor, username)
    result = db.singleSelect(query, values)
    return result["K"]

  def SelectUserWSIByStoreName(self, username, storeName, searchFor="DZI"):
    db = DatabaseHandler()
    query = "SELECT * FROM user_wsis WHERE type = %s AND user_id = (SELECT id FROM users WHERE username = %s) AND store_name = %s;"
    values = (searchFor, username, storeName)
    result = db.singleSelect(query, values)
    return result

  def GetUserWSIs(self, username, fromIndex=0, amount=10, searchFor="DZI"):
    db = DatabaseHandler()
    if (amount is None):
      query = "SELECT * FROM user_wsis WHERE type = %s AND user_id = (SELECT id FROM users WHERE username = %s);"
      values = (searchFor, username)
    else:
      query = "SELECT * FROM user_wsis WHERE type = %s AND user_id = (SELECT id FROM users WHERE username = %s) LIMIT %s, %s;"
      values = (searchFor, username, fromIndex, amount)
    result = db.select(query, values)
    return result

  def IsUserWSIBelongToUser(self, username, wsiName):
    db = DatabaseHandler()
    query = "SELECT * FROM user_wsis WHERE user_id = (SELECT id FROM users WHERE username = %s) AND store_name = %s;"
    values = (username, wsiName)
    result = db.singleSelect(query, values)
    if (result is None):
      return False
    else:
      return True

  def AddUserWSI(self, username, wsiName, storeName, type):
    db = DatabaseHandler()
    query = "INSERT INTO user_wsis (user_id, name, store_name, type) VALUES ((SELECT id FROM users WHERE username = %s), %s, %s, %s);"
    values = (username, wsiName, storeName, type)
    lastRowID, rowCount, statement = db.insert(query, values)
    if (rowCount == 1):
      return {"success": True, "message": f"The {type} file is added successfully."}
    else:
      return {"success": False, "message": f"Could not add the new {type} file."}

  def DeleteUserWSI(self, username, id):
    db = DatabaseHandler()
    query = "DELETE FROM user_wsis WHERE user_id = (SELECT id FROM users WHERE username = %s) AND id = %s;"
    values = (username, id)
    lastRowID, rowCount, statement = db.delete(query, values)
    if (rowCount == 1):
      return {"success": True, "message": f"{type} is deleted successfully."}
    else:
      return {"success": False, "message": f"Could not delete {type}."}

  def GetUniqueAugmentationFilename(self, extension, includeExtension=True):
    return self.GenerateUniqueFilename("user_augmentations", extension, includeExtension=includeExtension)

  def AddUserAugmentation(self, username, originalName, storeName, configurations):
    import json

    db = DatabaseHandler()
    query = "INSERT INTO user_augmentations (user_id, name, store_name, configurations) VALUES ((SELECT id FROM users WHERE username = %s), %s, %s, %s);"
    values = (username, originalName, storeName, json.dumps(configurations))
    lastRowID, rowCount, statement = db.insert(query, values)
    if (rowCount == 1):
      return {"success": True, "message": f"The image file is added successfully."}
    else:
      return {"success": False, "message": f"Could not add the new image file."}

  def GetUserClassifications(self, username, fromIndex=0, amount=10):
    db = DatabaseHandler()
    query = "SELECT * FROM user_classifications WHERE post_configurations IS NOT NULL AND user_id = (SELECT id FROM users WHERE username = %s) LIMIT %s, %s;"
    values = (username, fromIndex, amount)
    result = db.select(query, values)
    return result

  def GetUniqueClassificationFilename(self, extension, includeExtension=True):
    return self.GenerateUniqueFilename("user_classifications", extension, includeExtension=includeExtension)

  def CountUserClassifications(self, username):
    db = DatabaseHandler()
    query = "SELECT COUNT(*) as K FROM user_classifications WHERE post_configurations IS NOT NULL AND user_id = (SELECT id FROM users WHERE username = %s);"
    values = (username,)
    result = db.singleSelect(query, values)
    return result["K"]

  def AddUserClassification(self, username, name, storeName, relativePath, configurations):
    import json

    db = DatabaseHandler()
    query = "INSERT INTO user_classifications (user_id, name, store_name, relative_path, configurations, is_success) VALUES ((SELECT id FROM users WHERE username = %s), %s, %s, %s, %s, 0);"
    values = (username, name, storeName, relativePath, json.dumps(configurations))
    lastRowID, rowCount, statement = db.insert(query, values)
    if (rowCount == 1):
      return {"success": True, "message": f"The classification is added successfully."}
    else:
      return {"success": False, "message": f"Could not add the new classification."}

  def UpdateUserClassificationSuccess(self, username, name, storeName, isSuccess, message):
    db = DatabaseHandler()
    query = "UPDATE user_classifications SET is_success = %s, message = %s WHERE user_id = (SELECT id FROM users WHERE username = %s) AND name = %s AND store_name = %s;"
    values = (isSuccess, message, username, name, storeName)
    lastRowID, rowCount, statement = db.update(query, values)
    if (rowCount == 1):
      return {"success": True, "message": f"The classification is updated successfully."}
    else:
      return {"success": False, "message": f"Could not update the classification."}

  def UpdateUserClassificationPostConfigurations(self, username, name, storeName, configurations):
    import json

    db = DatabaseHandler()
    query = "UPDATE user_classifications SET post_configurations = %s WHERE user_id = (SELECT id FROM users WHERE username = %s) AND name = %s AND store_name = %s;"
    values = (json.dumps(configurations), username, name, storeName)
    lastRowID, rowCount, statement = db.update(query, values)
    if (rowCount == 1):
      return {"success": True, "message": f"The classification is updated successfully."}
    else:
      return {"success": False, "message": f"Could not update the classification."}

  def DeleteUserClassification(self, username, id):
    db = DatabaseHandler()
    query = "DELETE FROM user_classifications WHERE user_id = (SELECT id FROM users WHERE username = %s) AND id = %s;"
    values = (username, id)
    lastRowID, rowCount, statement = db.delete(query, values)
    if (rowCount == 1):
      return {"success": True, "message": f"The classification record is deleted successfully."}
    else:
      return {"success": False, "message": f"Could not delete the classification record."}

  def DeleteUserClassificationByStoreName(self, username, storeName):
    db = DatabaseHandler()
    query = "DELETE FROM user_classifications WHERE user_id = (SELECT id FROM users WHERE username = %s) AND store_name = %s;"
    values = (username, storeName)
    lastRowID, rowCount, statement = db.delete(query, values)
    if (rowCount == 1):
      return {"success": True, "message": f"The classification record is deleted successfully."}
    else:
      return {"success": False, "message": f"Could not delete the classification record."}

  def IsUserClassificationBelongToUser(self, username, storeName):
    db = DatabaseHandler()
    query = "SELECT * FROM user_classifications WHERE user_id = (SELECT id FROM users WHERE username = %s) AND store_name = %s;"
    values = (username, storeName)
    result = db.singleSelect(query, values)
    if (result is None):
      return False
    else:
      return True

  def GetUserClassificationByStoreName(self, username, storeName):
    db = DatabaseHandler()
    query = "SELECT * FROM user_classifications WHERE user_id = (SELECT id FROM users WHERE username = %s) AND store_name = %s;"
    values = (username, storeName)
    result = db.singleSelect(query, values)
    return result

  def GetUniqueSegmentationFilename(self, extension, includeExtension=True):
    return self.GenerateUniqueFilename("user_segmentations", extension, includeExtension=includeExtension)

  def AddUserSegmentation(self, username, name, storeName, relativePath, configurations):
    import json

    db = DatabaseHandler()
    query = "INSERT INTO user_segmentations (user_id, name, store_name, relative_path, configurations, is_success) VALUES ((SELECT id FROM users WHERE username = %s), %s, %s, %s, %s, 0);"
    values = (username, name, storeName, relativePath, json.dumps(configurations))
    lastRowID, rowCount, statement = db.insert(query, values)
    if (rowCount == 1):
      return {"success": True, "message": f"The segmentation is added successfully."}
    else:
      return {"success": False, "message": f"Could not add the new segmentation."}

  def UpdateUserSegmentationSuccess(self, username, name, storeName, isSuccess, message):
    db = DatabaseHandler()
    query = "UPDATE user_segmentations SET is_success = %s, message = %s WHERE user_id = (SELECT id FROM users WHERE username = %s) AND name = %s AND store_name = %s;"
    values = (isSuccess, message, username, name, storeName)
    lastRowID, rowCount, statement = db.update(query, values)
    if (rowCount == 1):
      return {"success": True, "message": f"The segmentation is updated successfully."}
    else:
      return {"success": False, "message": f"Could not update the segmentation."}

  def UpdateUserSegmentationPostConfigurations(self, username, name, storeName, configurations):
    import json

    db = DatabaseHandler()
    query = "UPDATE user_segmentations SET post_configurations = %s WHERE user_id = (SELECT id FROM users WHERE username = %s) AND name = %s AND store_name = %s;"
    values = (json.dumps(configurations), username, name, storeName)
    lastRowID, rowCount, statement = db.update(query, values)
    if (rowCount == 1):
      return {"success": True, "message": f"The segmentation is updated successfully."}
    else:
      return {"success": False, "message": f"Could not update the segmentation."}

  def DeleteUserSegmentation(self, username, id):
    db = DatabaseHandler()
    query = "DELETE FROM user_segmentations WHERE user_id = (SELECT id FROM users WHERE username = %s) AND id = %s;"
    values = (username, id)
    lastRowID, rowCount, statement = db.delete(query, values)
    if (rowCount == 1):
      return {"success": True, "message": f"The segmentation record is deleted successfully."}
    else:
      return {"success": False, "message": f"Could not delete the segmentation record."}

  def DeleteUserSegmentationByStoreName(self, username, storeName):
    db = DatabaseHandler()
    query = "DELETE FROM user_segmentations WHERE user_id = (SELECT id FROM users WHERE username = %s) AND store_name = %s;"
    values = (username, storeName)
    lastRowID, rowCount, statement = db.delete(query, values)
    if (rowCount == 1):
      return {"success": True, "message": f"The segmentation record is deleted successfully."}
    else:
      return {"success": False, "message": f"Could not delete the segmentation record."}

  def IsUserSegmentationBelongToUser(self, username, storeName):
    db = DatabaseHandler()
    query = "SELECT * FROM user_segmentations WHERE user_id = (SELECT id FROM users WHERE username = %s) AND store_name = %s;"
    values = (username, storeName)
    result = db.singleSelect(query, values)
    if (result is None):
      return False
    else:
      return True

  def GetUserSegmentationByStoreName(self, username, storeName):
    db = DatabaseHandler()
    query = "SELECT * FROM user_segmentations WHERE user_id = (SELECT id FROM users WHERE username = %s) AND store_name = %s;"
    values = (username, storeName)
    result = db.singleSelect(query, values)
    return result

  def GetUserSegmentations(self, username, fromIndex=0, amount=10):
    db = DatabaseHandler()
    if (amount is None):
      query = "SELECT * FROM user_segmentations WHERE post_configurations IS NOT NULL AND user_id = (SELECT id FROM users WHERE username = %s);"
      values = (username,)
    else:
      query = "SELECT * FROM user_segmentations WHERE post_configurations IS NOT NULL AND user_id = (SELECT id FROM users WHERE username = %s) LIMIT %s, %s;"
      values = (username, fromIndex, amount)
    result = db.select(query, values)
    return result

  def CountUserSegmentations(self, username):
    db = DatabaseHandler()
    query = "SELECT COUNT(*) as K FROM user_segmentations WHERE post_configurations IS NOT NULL AND user_id = (SELECT id FROM users WHERE username = %s);"
    values = (username,)
    result = db.singleSelect(query, values)
    return result["K"]