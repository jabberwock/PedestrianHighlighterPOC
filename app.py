#!/usr/bin/env python3
from flask import Flask, render_template, Response
import cv2
import numpy as np
import os


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True  # Automatically reload templates
person_cascade_src = os.path.join('assets', 'hogcascade_pedestrians.xml')
cap = cv2.VideoCapture(0)  # Capture from first camera found on system
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

"""
Feel free to set the width and height as desired:
"""
width = 1024  # in pixels
height = 768  # in pixels
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, height)


def gen_frames():
    while True:
        success, image_arr_color = cap.read()

        if not success:
            print("Error capturing image. Video feed ended?")
            break

        # Greyscale
        grey = cv2.cvtColor(image_arr_color, cv2.COLOR_BGR2GRAY)

        # Gaussian blur:
        blur = cv2.GaussianBlur(grey, (5, 5), 0)

        # Dilation
        dilated = cv2.dilate(blur, np.ones((3, 3)))

        """
        Now we will perform a Morphology transformation with the kernel.
        Here we are using a morphology-Ex technique that tells the function on
        which image processing operations need to be done. The second argument
        is about what operations must be done, and you may need
        elliptical/circular shaped kernels.  To implement the morphology-Ex
        method using OpenCV we will be using the get
        structuring element method.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        closing = cv2.morphologyEx(dilated, cv2.MORPH_CLOSE, kernel)

        # Detecting persons using person cascade
        persons, _ = hog.detectMultiScale(
                            closing,
                            winStride=(4, 4),
                            padding=(4, 4),
                            scale=1.05
                        )

        # Reset person counter
        person_counter = 0

        for (x, y, w, h) in persons:
            # increment the person counter
            person_counter = person_counter + 1
            """
            Loop through identified persons and draw rectangles around them:
            Color is specified below:
            """
            highlight_color = (0, 255, 0)  # B, G, R
            cv2.rectangle(
                image_arr_color,
                (x, y), (x+w, y+h),
                highlight_color, 2
            )

            # setup font
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottom_left_text_corner = (10, 380)
            font_scale = 1
            font_color = (255, 255, 255)
            thickness = 3
            line_type = 2

            # place text on rendered image
            cv2.putText(image_arr_color,
                        f'{person_counter} person(s) detected.',
                        bottom_left_text_corner,
                        font,
                        font_scale,
                        font_color,
                        thickness,
                        line_type)

        # convert the numpy array to jpg
        success, buffer = cv2.imencode('.jpg', image_arr_color)
        if not success:
            print("Error converting capture to jpg")
            break

        frame = buffer.tobytes()
        # concat frame one by one and show result
        yield (
            b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
            + frame
            + b'\r\n'
        )


@app.route('/')
def index():
    return render_template('index.html', width=width, height=height)


@app.route('/video_feed')
def video_feed():
    return Response(
        gen_frames(),
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )


if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
