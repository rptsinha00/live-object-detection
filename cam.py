# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 11:34:50 2018

@author: Arpit
"""
import cv2
import numpy as np
from PIL import Image
		
from keras import backend as K
from keras.models import load_model
import tensorflow as tf
		
# The below provided fucntions will be used from yolo_utils.py
from yolo_utils import read_classes, read_anchors, generate_colors, draw_boxes
		
# The below functions from the yad2k library will be used
from yad2k.models.keras_yolo import yolo_head, yolo_eval

#Loading the classes and the anchor boxes that are provided in the madel_data folder
class_names = read_classes("model_data/coco_classes.txt")
anchors = read_anchors("model_data/yolo_anchors.txt")
		
#Load the pretrained model. Please refer the README file to get info on how to obtain the yolo.h5 file
yolo_model = load_model("model_data/yolo.h5")
		
#Convert final layer features to bounding box parameters
yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
with tf.device('/device:GPU:0'):
    cv2.namedWindow('preview')
    vc = cv2.VideoCapture(0)
    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
    i = 0
    while rval:
        i = i + 1
        cv2.imshow("preview", frame)
        rval, frame = vc.read()
    #path = 'C:/Users/hp/car detection/images'
    #string = 'pic'
    #string+=str(i)
    #cv2.imwrite(os.path.join(path,string+'.jpg'),frame)
	
		
    #Provide the name of the image that you saved in the images folder to be fed through the network
    #input_image_name = string+'.jpg'
	
    #Obtaining the dimensions of the input image
    #input_image = Image.open("images/" + input_image_name)
        image = Image.fromarray(frame)
        if i == 1:
            width, height = image.size
            width = np.array(width, dtype=float)
            height = np.array(height, dtype=float)
	
    #Assign the shape of the input image to image_shapr variable
            image_shape = (height, width)
	
    #Now yolo_eval function selects the best boxes using filtering and non-max suppression techniques.
    # If you want to dive in more to see how this works, refer keras_yolo.py file in yad2k/models
            boxes, scores, classes = yolo_eval(yolo_outputs, image_shape)
	
		
    # Initiate a session
            sess = K.get_session()
	
	
    #Preprocess the input image before feeding into the convolutional network
    #image, image_data = preprocess_image("images/" + input_image_name, model_image_size = (416, 416))
            model_image_size = (416,416)
            resized_image = image.resize(tuple(reversed(model_image_size)), Image.BICUBIC)
            image_data = np.array(resized_image, dtype='float32')
            image_data /= 255.
            image_data = np.expand_dims(image_data, 0)  # Add batch dimension
	
    #Run the session
            out_scores, out_boxes, out_classes = sess.run([scores, boxes, classes],feed_dict={yolo_model.input:image_data,K.learning_phase(): 0})
	
	
    #Print the results
            print('Found {} boxes'.format(len(out_boxes)))
    #Produce the colors for the bounding boxs
            colors = generate_colors(class_names)
        elif i == 10:
            i = 0
    #Draw the bounding boxes
        draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    #Apply the predicted bounding boxes to the image and save it
        arr = np.array(image)
        cv2.imshow('out',arr)
    #image.save(os.path.join("out", input_image_name), quality=90)
    #output_image = scipy.misc.imread(os.path.join("out", input_image_name))
    #cv2.imshow('out',output_image)

        key = cv2.waitKey(20)
        if key == 27: # exit on ESC
            break
    cv2.destroyWindow('preview')
    cv2.destroyWindow('out')
