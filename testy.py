from flask import Flask
from flask import request, jsonify
import cv2 as cv
import pickle
import time
import numpy as np

app = Flask(__name__)

@app.route('/read', methods = ['POST'])
def home():
    scf = pickle.load(open('net.sav', 'rb'))
    lb = pickle.load(open('label-encoder', 'rb'))
    image = request.files['image']
    api_key = request.form['api_key']
    image = image.read()
    if len(image) == 0:
        return jsonify({'ImageError': 'image is not valid'})
    
    file = open('image.png', 'wb')
    file.write(image)
    file.close()
    time.sleep(2)
    img = cv.imread('image.png')
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.threshold(img, 0, 255, cv.THRESH_BINARY)
    img_erode = cv.erode(thresh, np.ones((3, 3), np.uint8), iterations=1)

    contours, hierarchy = cv.findContours(img_erode, cv.RETR_TREE, cv.CHAIN_APPROX_NONE)

    letters = []
    for idx, contour in enumerate(contours):
        (x, y, w, h) = cv.boundingRect(contour)
        if hierarchy[0][idx][3] == 0:
            letter_crop = img[y:y + h, x:x + w]

            size_max = max(w, h)
            letter_square = 255 * np.ones(shape=[size_max, size_max], dtype=np.uint8)
            if w > h:

                y_pos = size_max//2 - h//2
                letter_square[y_pos:y_pos + h, 0:w] = letter_crop
            elif w < h:

                x_pos = size_max//2 - w//2
                letter_square[0:h, x_pos:x_pos + w] = letter_crop
            else:
                letter_square = letter_crop

            letters.append((x, w, cv.resize(letter_square, (28, 28), interpolation=cv.INTER_AREA)))
    letters.sort(key=lambda x: x[0], reverse=False)

    predict = ''
    for letter in letters:
        img = letter[2]

        img = cv.resize(img, (128, 128))
        img = np.array(img)
        img = img.reshape(-1, 128*128)

        pred = scf.predict(img)
        pred = lb.inverse_transform(pred)
        predict = predict+pred[0]
    return jsonify({'text': predict})

app.run()
