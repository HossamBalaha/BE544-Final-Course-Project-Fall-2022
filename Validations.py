import re


class ValidationRules(object):
  EMAIL_REGEX = re.compile(r"[^@]+@[^@]+\.[^@]+")
  USERNAME_REGEX = re.compile(r"^[a-z.A-Z0-9_]+$")
  NUMBER_REGEX = re.compile(r"^[0-9]+(\.[0-9]+)?$")
  INTEGER_REGEX = re.compile(r"^[0-9]+$")
  PHONE_REGEX = re.compile(r"^\d{3}-\d{3}-\d{4}$")
  DATE_REGEX = re.compile(r"^\d{1,2}/\d{1,2}/\d{4}$")
  TIME_REGEX = re.compile(r"^\d{1,2}:\d{2} (AM|PM)$")
  DATE_TIME_REGEX = re.compile(r"^\d{1,2}/\d{1,2}/\d{4} \d{1,2}:\d{2} (AM|PM)$")
  URL_REGEX = re.compile(r"^https?://(www\.)?(\w+)(\.\w+)+(/.*)?$")

  def Required(self, value):
    value = value.strip()
    if (value is None or value == ''):
      return False
    return True

  def Regex(self, value, regex):
    if (not re.match(regex, value)):
      return False
    return True

  def Identical(self, value, confirm):
    value = value.strip()
    confirm = confirm.strip()
    if (value != confirm):
      return False
    return True

  def MaxLength(self, value, maxLength=100):
    value = value.strip()
    if (len(value) > maxLength):
      return False
    return True

  def MinLength(self, value, minLength=100):
    value = value.strip()
    if (len(value) < minLength):
      return False
    return True

  def MaxValue(self, value, maxValue=100):
    value = value.strip()
    numberFlag = self.IsNumber(value)
    if (not numberFlag):
      return False
    if (float(value) > maxValue):
      return False
    return True

  def MinValue(self, value, minValue=100):
    value = value.strip()
    numberFlag = self.IsNumber(value)
    if (not numberFlag):
      return False
    if (float(value) < minValue):
      return False
    return True

  def IsNumber(self, value):
    value = value.strip()
    return self.Regex(value, self.NUMBER_REGEX)

  def IsInteger(self, value):
    value = value.strip()
    value = value.strip()
    return self.Regex(value, self.INTEGER_REGEX)

  def IsPhoneNumber(self, value):
    value = value.strip()
    return self.Regex(value, self.PHONE_REGEX)

  def IsDate(self, value):
    value = value.strip()
    return self.Regex(value, self.DATE_REGEX)

  def IsTime(self, value):
    value = value.strip()
    return self.Regex(value, self.TIME_REGEX)

  def IsDateTime(self, value):
    value = value.strip()
    return self.Regex(value, self.DATE_TIME_REGEX)

  def IsURL(self, value):
    value = value.strip()
    return self.Regex(value, self.URL_REGEX)

  def In(self, value, options):
    value = value.strip()
    if (value not in options):
      return False
    return True


class ValidateFields(object):
  def __init__(self):
    self.formInputValidator = ValidationRules()

  def Username(self, value, minLength=6, maxLength=100):
    result = self.formInputValidator.Required(value)
    if (not result):
      return False, "This username field is required."
    result = self.formInputValidator.MinLength(value, minLength)
    if (not result):
      return False, "Username must be at least " + str(minLength) + " characters."
    result = self.formInputValidator.MaxLength(value, maxLength)
    if (not result):
      return False, "Username cannot be more than " + str(maxLength) + " characters."
    result = self.formInputValidator.Regex(value, self.formInputValidator.USERNAME_REGEX)
    if (not result):
      return False, "Please enter a valid username."
    return True, ""

  def Password(self, value, minLength=6, maxLength=100):
    result = self.formInputValidator.Required(value)
    if (not result):
      return False, "This password field is required."
    result = self.formInputValidator.MinLength(value, minLength)
    if (not result):
      return False, "Password must be at least " + str(minLength) + " characters."
    result = self.formInputValidator.MaxLength(value, maxLength)
    if (not result):
      return False, "Password cannot be more than " + str(maxLength) + " characters."
    return True, ""

  def ConfirmPassword(self, value, password):
    result = self.formInputValidator.Identical(value, password)
    if (not result):
      return False, "Password and confirm password must be identical."
    return True, ""

  def Title(self, value, minLength=6, maxLength=100):
    result = self.formInputValidator.Required(value)
    if (not result):
      return False, "This title field is required."
    result = self.formInputValidator.MinLength(value, minLength)
    if (not result):
      return False, "Title must be at least " + str(minLength) + " characters."
    result = self.formInputValidator.MaxLength(value, maxLength)
    if (not result):
      return False, "Title cannot be more than " + str(maxLength) + " characters."
    return True, ""

  def Overlap(self, value, minValue=0, maxValue=10):
    result = self.formInputValidator.Required(value)
    if (not result):
      return False, "This overlap field is required."
    result = self.formInputValidator.MinValue(value, minValue)
    if (not result):
      return False, "Overlap must be at least " + str(minValue) + " characters."
    result = self.formInputValidator.MaxValue(value, maxValue)
    if (not result):
      return False, "Overlap cannot be more than " + str(maxValue) + " characters."
    return True, ""

  def Size(self, value, minValue=0, maxValue=2048):
    result = self.formInputValidator.Required(value)
    if (not result):
      return False, "This size field is required."
    result = self.formInputValidator.MinValue(value, minValue)
    if (not result):
      return False, "Size must be at least " + str(minValue) + " characters."
    result = self.formInputValidator.MaxValue(value, maxValue)
    if (not result):
      return False, "Size cannot be more than " + str(maxValue) + " characters."
    return True, ""

  def Rotation(self, value, minValue=0, maxValue=360):
    result = self.formInputValidator.Required(value)
    if (not result):
      return False, "This rotation field is required."
    result = self.formInputValidator.MinValue(value, minValue)
    if (not result):
      return False, "Rotation must be at least " + str(minValue) + " characters."
    result = self.formInputValidator.MaxValue(value, maxValue)
    if (not result):
      return False, "Rotation cannot be more than " + str(maxValue) + " characters."
    return True, ""

  def Zoom(self, value, minValue=-1, maxValue=1):
    result = self.formInputValidator.Required(value)
    if (not result):
      return False, "This zoom field is required."
    result = self.formInputValidator.MinValue(value, minValue)
    if (not result):
      return False, "Zoom must be at least " + str(minValue) + " characters."
    result = self.formInputValidator.MaxValue(value, maxValue)
    if (not result):
      return False, "Zoom cannot be more than " + str(maxValue) + " characters."
    return True, ""

  def WidthShift(self, value, minValue=-1, maxValue=1):
    result = self.formInputValidator.Required(value)
    if (not result):
      return False, "This width shift field is required."
    result = self.formInputValidator.MinValue(value, minValue)
    if (not result):
      return False, "Width shift must be at least " + str(minValue) + " characters."
    result = self.formInputValidator.MaxValue(value, maxValue)
    if (not result):
      return False, "Width shift cannot be more than " + str(maxValue) + " characters."
    return True, ""

  def HeightShift(self, value, minValue=-1, maxValue=1):
    result = self.formInputValidator.Required(value)
    if (not result):
      return False, "This height shift field is required."
    result = self.formInputValidator.MinValue(value, minValue)
    if (not result):
      return False, "Height shift must be at least " + str(minValue) + " characters."
    result = self.formInputValidator.MaxValue(value, maxValue)
    if (not result):
      return False, "Height shift cannot be more than " + str(maxValue) + " characters."
    return True, ""

  def HorizontalFlip(self, value):
    result = self.formInputValidator.Required(value)
    if (not result):
      return False, "This horizontal flip field is required."
    return True, ""

  def VerticalFlip(self, value):
    result = self.formInputValidator.Required(value)
    if (not result):
      return False, "This vertical flip field is required."
    return True, ""

  def Shear(self, value, minValue=-1, maxValue=1):
    result = self.formInputValidator.Required(value)
    if (not result):
      return False, "This shear field is required."
    result = self.formInputValidator.MinValue(value, minValue)
    if (not result):
      return False, "Shear must be at least " + str(minValue) + " characters."
    result = self.formInputValidator.MaxValue(value, maxValue)
    if (not result):
      return False, "Shear cannot be more than " + str(maxValue) + " characters."
    return True, ""

  def Brightness(self, value, minValue=0, maxValue=100):
    result = self.formInputValidator.Required(value)
    if (not result):
      return False, "This brightness field is required."
    result = self.formInputValidator.MinValue(value, minValue)
    if (not result):
      return False, "Brightness must be at least " + str(minValue) + " characters."
    result = self.formInputValidator.MaxValue(value, maxValue)
    if (not result):
      return False, "Brightness cannot be more than " + str(maxValue) + " characters."
    return True, ""

  def Saturation(self, value, minValue=0, maxValue=100):
    result = self.formInputValidator.Required(value)
    if (not result):
      return False, "This saturation field is required."
    result = self.formInputValidator.MinValue(value, minValue)
    if (not result):
      return False, "Saturation must be at least " + str(minValue) + " characters."
    result = self.formInputValidator.MaxValue(value, maxValue)
    if (not result):
      return False, "Saturation cannot be more than " + str(maxValue) + " characters."
    return True, ""

  def Hue(self, value, minValue=0, maxValue=100):
    result = self.formInputValidator.Required(value)
    if (not result):
      return False, "This hue field is required."
    result = self.formInputValidator.MinValue(value, minValue)
    if (not result):
      return False, "Hue must be at least " + str(minValue) + " characters."
    result = self.formInputValidator.MaxValue(value, maxValue)
    if (not result):
      return False, "Hue cannot be more than " + str(maxValue) + " characters."
    return True, ""

  def SpecificRandom(self, value):
    result = self.formInputValidator.Required(value)
    if (not result):
      return False, "This specific random field is required."
    return True, ""

  def SingleMultiple(self, value):
    result = self.formInputValidator.Required(value)
    if (not result):
      return False, "This single multiple field is required."
    return True, ""

  def In(self, keyword, value, listOfValues):
    result = self.formInputValidator.Required(value)
    if (not result):
      return False, f"The {keyword} field is required."
    result = self.formInputValidator.In(value, listOfValues)
    if (not result):
      return False, f"The {keyword} field must be one of the following: " + str(listOfValues)
    return True, ""

  def Shape(self, shape):
    result = self.formInputValidator.Required(shape)
    if (not result):
      return False, "The shape field is required."
    parts = shape.split(",")
    for part in parts:
      if (not self.formInputValidator.IsInteger(part)):
        return False, "The shape field must be a comma separated list of integers."
    return True, ""

  def Trainable(self, shape):
    result = self.formInputValidator.Required(shape)
    if (not result):
      return False, "The Is Trainable? field is required."
    result = self.formInputValidator.In(shape, ["1", "0"])
    if (not result):
      return False, "The Is Trainable? field must be one of the following: Yes, No"
    return True, ""

  def TLTrainingRatio(self, shape):
    result = self.formInputValidator.Required(shape)
    if (not result):
      return False, "The TL Training Ratio field is required."
    result = self.formInputValidator.IsNumber(shape)
    if (not result):
      return False, "The TL Training Ratio field must be a number."
    result = self.formInputValidator.MinValue(shape, 0)
    if (not result):
      return False, "The TL Training Ratio field must be at least 0."
    result = self.formInputValidator.MaxValue(shape, 1)
    if (not result):
      return False, "The TL Training Ratio field cannot be more than 1."
    return True, ""

  def BatchSize(self, shape):
    result = self.formInputValidator.Required(shape)
    if (not result):
      return False, "The Batch Size field is required."
    result = self.formInputValidator.IsInteger(shape)
    if (not result):
      return False, "The Batch Size field must be an integer."
    result = self.formInputValidator.MinValue(shape, 1)
    if (not result):
      return False, "The Batch Size field must be at least 1."
    return True, ""

  def TrainingRatio(self, shape):
    result = self.formInputValidator.Required(shape)
    if (not result):
      return False, "The Training Ratio field is required."
    result = self.formInputValidator.IsNumber(shape)
    if (not result):
      return False, "The Training Ratio field must be a number."
    result = self.formInputValidator.MinValue(shape, 0)
    if (not result):
      return False, "The Training Ratio field must be at least 0."
    result = self.formInputValidator.MaxValue(shape, 1)
    if (not result):
      return False, "The Training Ratio field cannot be more than 1."
    return True, ""

  def Epochs(self, shape):
    result = self.formInputValidator.Required(shape)
    if (not result):
      return False, "The Epochs field is required."
    result = self.formInputValidator.IsInteger(shape)
    if (not result):
      return False, "The Epochs field must be an integer."
    result = self.formInputValidator.MinValue(shape, 1)
    if (not result):
      return False, "The Epochs field must be at least 1."
    return True, ""

  def Classes(self, shape):
    result = self.formInputValidator.Required(shape)
    if (not result):
      return False, "The Classes field is required."
    parts = shape.split(",")
    for part in parts:
      if (not self.formInputValidator.Required(part)):
        return False, "The Classes field must be a comma separated list of string classes."
    return True, ""

  def Annotations(self, shape):
    result = self.formInputValidator.Required(shape)
    if (not result):
      return False, "The Annotations field is required."
    parts = shape.split(",")
    for part in parts:
      if (not self.formInputValidator.Required(part)):
        return False, "The Annotations field must be a comma separated list of string classes."
    return True, ""

  def Shuffle(self, value):
    result = self.formInputValidator.Required(value)
    if (not result):
      return False, "This Shuffle field is required."
    return True, ""