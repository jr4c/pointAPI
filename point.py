
# coding: utf-8

# In[1]:


# %load point.py
"""
This module serves as the API provider for point detection.
"""

import io
import json
import cv2

import numpy as np
import tensorflow as tf
from collections import OrderedDict


# In[2]:


CABLE_CKPT = 'cables_inference_graph.pb'
POLES_CKPT = 'poles_inference_graph.pb'
TOPBOTTOM_CKPT = 'top_bottom_inference_graph.pb'


# In[3]:


def get_graph(path_):
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    return detection_graph


# In[4]:


POINT_GRAPH = get_graph(CABLE_CKPT)
POLES_GRAPH = get_graph(POLES_CKPT)
TOPBOTTOM_GRAPH = get_graph(TOPBOTTOM_CKPT)


# In[8]:


def topMiddleBottom(size,middleBoxes):
    yt, xt = size
    points = OrderedDict({})
    
    #yi, xi, yh ,xh = topBottomBoxes[0]
    #xm = (xi+xh)/2.00
    
    #pbx = int(xt*xm)
    #pbyi = int(yt*yi)
    #pbyh = int(yt*yh)
    
    
    #points["top"] = [pbx,pbyi]
    
    for i,box in enumerate(middleBoxes):
        yi, xi, yh ,xh = box
        xm = (xi+xh)/2.00
        ym = (yi+yh)/2.00
    
        px = int(xt*xm)
        py = int(yt*ym)
        
        points["point{}".format(i+1)] = [px,py]

    #points["bottom"] = [pbx,pbyh]
    
    return points


# In[5]:


def namedPoints(size,top,middle,bottom):
    yt, xt = size
    points = OrderedDict({})
    
    for i,box in enumerate(top):
        yi, xi, yh ,xh = box
        xm = (xi+xh)/2.00
        ym = (yi+yh)/2.00
        px = int(xt*xm)
        py = int(yt*ym)
        points["top{}".format(i+1)] = [px,py]
        
    for i,box in enumerate(middle):
        yi, xi, yh ,xh = box
        xm = (xi+xh)/2.00
        ym = (yi+yh)/2.00
        px = int(xt*xm)
        py = int(yt*ym)
        points["point{}".format(i+1)] = [px,py]
        
    for i,box in enumerate(bottom):
        yi, xi, yh ,xh = box
        xm = (xi+xh)/2.00
        ym = (yi+yh)/2.00
        px = int(xt*xm)
        py = int(yt*ym)
        points["bottom{}".format(i+1)] = [px,py]

    return points


# In[7]:


def extraction(image,point_result,pole_result,topbottom_result):
    
    #poles_boxes = np.average(pole_result['detection_boxes'],weights=pole_result['detection_scores'],axis=0).reshape([1,4])
    
    objects_boxes = list(point_result['detection_boxes'])
    bottom_boxes = [list(topbottom_result['detection_boxes'][i]) for i,x in enumerate(topbottom_result['detection_classes']) if x == 2]
    top_boxes = [list(topbottom_result['detection_boxes'][i]) for i,x in enumerate(topbottom_result['detection_classes']) if x == 1]
        
    result = namedPoints(image.shape[:2],top_boxes,objects_boxes,bottom_boxes)
    
    return result   


# In[14]:


def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session() as sess:
            # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}

            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores', 'detection_classes',
                        'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
            
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            
            output_dict['detection_classes'] = output_dict['detection_classes'][:output_dict['num_detections']] 
            output_dict['detection_boxes'] = output_dict['detection_boxes'][:output_dict['num_detections']]
            output_dict['detection_scores'] = output_dict['detection_scores'][:output_dict['num_detections']]
            #print(output_dict)

            return output_dict


# In[11]:


def post_image(file):
    """
    Given a posted image, classify it using the pretrained model.

    This will take 'any size' image, and scale it down to 28x28 like our MNIST
    training data -- and convert to grayscale.

    Parameters
    ----------
    file:
        Bytestring contents of the uploaded file. This will be in an image file format.
    """

    img_array = np.asarray(bytearray(file.read()), dtype=np.uint8)
    img = cv2.imdecode(img_array,cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  
    image_np = img.copy()

    poles_result = run_inference_for_single_image(image_np, POLES_GRAPH)
    point_result = run_inference_for_single_image(image_np, POINT_GRAPH)
    topbottom_result = run_inference_for_single_image(image_np, TOPBOTTOM_GRAPH)

    
    result = extraction(img,point_result,poles_result,topbottom_result)

    return json.dumps(result)

