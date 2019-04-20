import tensorflow as tf
import os
import utils
import sys
import numpy as np
import pandas as pd
import cv2
from mtcnn.mtcnn import MTCNN
import utils
from keras import backend as K
from keras.models import load_model, Model
from datetime import datetime
import time
people_dir = sys.argv[1]
dic = {

}
i=0
for dir in os.listdir(people_dir):
    dic.update({i: dir})
    i +=1
print(dic)
with tf.Graph().as_default():
    with tf.Session() as sess:
        print('Loading feature extraction model')
        #load embedding face
        utils.load_model(sys.argv[2])
        images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
        embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
        phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
        embedding_size = embeddings.get_shape()[1]
        #load model
        model = load_model('CDCN.h5')
        #open camera
        video_capture = cv2.VideoCapture(0)
        idx = 0
        last_pred = ''
        num_pred = 1
        num_fps = 0
        detector = MTCNN()
        while (True):
            ret, frame = video_capture.read()
            #num_rows, num_cols = frame.shape[:2]
            #rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2),90, 1)
            #frame = cv2.warpAffine(frame, rotation_matrix, (num_cols, num_rows))
            #Resize an image
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            #convert BGR to RGB
            rgb_small_frame = small_frame[:, :, ::-1]
            face_locations = detector.detect_faces(rgb_small_frame)
            for i, face_location in enumerate(face_locations):

                # Print the location of each face in this image
                x, y, w, h = face_location['box']
                top = y*2
                right = (x+w)*2
                bottom = (y+h)*2
                left = x *2
                print("A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}".format(
                    top, left, bottom, right))
                image = utils.load_image(frame[top:bottom, left:right])
                ## Predict face
                feed_dict = { images_placeholder:image, phase_train_placeholder:False }
                embedding_array = sess.run(embeddings, feed_dict=feed_dict)
                pred = model.predict(embedding_array)
                acc = np.max(pred,axis=1)
                print(acc)
                pred = np.argmax(pred, axis=1)
                name = dic[pred[0]]

                if (pred == last_pred):
                    num_pred +=1
                elif(pred != last_pred):
                    last_pred = pred
                    num_pred = 1
                #print(pred)
                #print(last_pred)
                #print(num_pred)
                #print('=======')
                #print('======')

                # if accurancy >0.9 mean true people
                if (acc >= 0.9 and num_pred > 2):
                    #save image to folder 
                    s = str(datetime.now())
                    second = float('0.' + s.split('.')[-1])
                    tmp = time.mktime(datetime.strptime(
                        s, "%Y-%m-%d %H:%M:%S.%f").timetuple()) + second
                    cv2.imwrite(os.path.join("people", name, "{}.png".format(
                        tmp)), frame[top:bottom, left:right])
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top), (right, bottom), (124,252,0), 2)

                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 35),
                                (right, bottom), (124,252,0), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6),
                                font, 1.0, (255, 255, 255), 1)
            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.imshow('Video', frame)
# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
