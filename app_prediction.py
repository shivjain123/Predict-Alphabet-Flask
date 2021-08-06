import numpy as np
import pandas as pd
import PIL.ImageOps
import os
import ssl
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from PIL import Image

if (not os.environ.get('PYTHONHTTPSVERIFIED', '') and getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

X = np.load('D:\(4) WhiteHatJr\Third Module\Flask\Alphabet Prediction\image.npz')['arr_0']
Y = pd.read_csv("D:\(4) WhiteHatJr\Third Module\Flask\Alphabet Prediction\labels.csv")["labels"]

x_train, x_test, y_train, y_test = tts(X, Y, train_size=3500, test_size=500, random_state=9)

x_train_scaled = x_train/255.0
x_test_scaled = x_test/255.0

lr = LogisticRegression(solver='saga', multi_class='multinomial')

lr.fit(x_train_scaled, y_train)

def get_prediction(image):
    iam_pil = Image.fromarray(image)
    image_bw = iam_pil.convert('L').resize((22, 30), Image.ANTIALIAS)
    image_inverted = PIL.ImageOps.invert(image_bw)
    pixel_filter = 20
    min_pixel = np.percentile(image_inverted, pixel_filter)
    image_scaled = np.clip(image_inverted - min_pixel, 0, 255)
    max_pixel = np.max(image_inverted)
    image_scaled = np.asarray(image_scaled)/max_pixel
    test_sample = np.array(image_scaled).reshape(1, 660)
    test_pred = lr.predict(test_sample)
    print("Predicted class is: ", test_pred)