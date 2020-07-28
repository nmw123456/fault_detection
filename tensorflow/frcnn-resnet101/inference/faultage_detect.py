# -*- coding: UTF-8 -*-
import os
import cv2
import sys
import time
import getopt
import numpy as np
import tensorflow as tf

g_base_path            = ''
g_default_image_path   = os.path.join(g_base_path, 'images')
g_default_result_path  = os.path.join(g_base_path, 'results')
g_default_model_file   = os.path.join(g_base_path, 'models/frozen_inference_graph.pb')
g_score_thresh         = 0.9
g_use_gpu              = False
g_max_label_id         = 1
g_line_type            = '0'
g_x_len_thresh         = 140
g_y_len_thresh         = 40
    
# load pb file
def load_pb_file(modelFile):
  graph = tf.Graph()
  with graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(modelFile, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')
  return graph

# create session
def create_session(modelFile):
  graph  = load_pb_file(modelFile)
  config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
  os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
  if not g_use_gpu:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
  else:
    config.gpu_options.allow_growth = True
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1'
  sess = tf.Session(graph=graph, config=config)
  return sess, graph

# 获取列表的元素
def takeFirst(elem):
  return elem[0]
  
def takeSecond(elem):
  return elem[1]
    
def get_best_faultage_points(basePoint, pointList, yLenThresh):
  pointListDraw = []
  lineLen       = 0
  for point in pointList:
    if abs(point[1] - basePoint[1]) < yLenThresh:
      pointListDraw.append(point)
    
  # x轴排序
  pointListDraw.sort(key=takeFirst)
  if len(pointListDraw) > 1:
    beginPoint = pointListDraw[0]
    for point in pointListDraw:
      lineLen += point[0] - beginPoint[0]
      beginPoint = point
  
  return pointListDraw, lineLen
  
def get_faultage_points(imgSize, boxes, scores, classes, threshold, xLenThresh, yLenThresh):
  pointList      = []
  for i, box in enumerate(boxes[0]):
    if (scores[0][i] > threshold and classes[0][i] <= g_max_label_id):
      box[0] = int(box[0] * imgSize[0]) # 左上角 y坐标
      box[1] = int(box[1] * imgSize[1]) # 左上角 x坐标
      box[2] = int(box[2] * imgSize[0]) # 右下角 y坐标
      box[3] = int(box[3] * imgSize[1]) # 右下角 x坐标
      pointl = (box[1], (box[0] + box[2]) / 2)
      pointr = (box[3], (box[0] + box[2]) / 2)
      if ((pointr[0] - pointl[0]) < xLenThresh):
        continue
      pointList.append(pointl)
      pointList.append(pointr)
  pointList.sort(key=takeSecond)
  
  if len(pointList) <= 1:
    return []
    
  len1 = len3 = len5 = 0
  basePoint = pointList[0] # 第一个点为基准点
  pointListDraw1, len1 = get_best_faultage_points(basePoint, pointList, yLenThresh)
  
  if len(pointList) > 2:
    basePoint = pointList[2] # 第三个点为基准点
    pointListDraw3, len3 = get_best_faultage_points(basePoint, pointList, yLenThresh)
    
  if len(pointList) > 4:
    basePoint = pointList[4] # 第五个点为基准点
    pointListDraw5, len5 = get_best_faultage_points(basePoint, pointList, yLenThresh)
  
  if len5 > len3 and len5 > len1:
    return pointListDraw5
  elif len3 > len1:
    return pointListDraw3
  else:
    return pointListDraw1
    
def draw_faultage_line(image, imgSize, pointList, lineType):
  listLen = len(pointList)
  if lineType == '1': # 画直线
    ySum = 0
    for point in pointList:
      ySum += point[1]
    yAverage = ySum / listLen
    cv2.line(image, (0, int(yAverage)), (imgSize[1], int(yAverage)), (0, 0, 0), 3, 4)
  else: # 画折线
    pointList.append((imgSize[1], pointList[listLen-1][1])) # 最右边的点
    beginPoint = (0, int(pointList[0][1]))
    for point in pointList:
      endPoint = (int(point[0]), int(point[1]))
      cv2.line(image, beginPoint, endPoint, (0, 0, 0), 3, 4)
      beginPoint = endPoint
    
def detect_faultage(imageFile, outputPath, modelFile, threshold, lineType, xLenThresh, yLenThresh):
  imageFileList = []
  try:
    threshold = float(threshold)
  except:
    return 'threshold is not number!'
  if threshold > 1 or threshold < 0:
    return 'threshold should be between 0 and 1!'
  
  if os.path.exists(modelFile) == False:
    return 'model file is not exist!'  
  
  if os.path.isfile(outputPath):
    return 'output path is not dir!'
  elif os.path.exists(outputPath) == False:
    os.mkdir(outputPath)
     
  if os.path.exists(imageFile) == False:
    return 'input file is not exist!'
  
  if lineType != '0' and lineType != '1':
    return 'line type is error!'
  
  if os.path.isfile(imageFile):
    imageFileList.append(imageFile)
  else:
    imageFileList = [ os.path.join(imageFile, i) for i in os.listdir(imageFile) ]
  
  try:
    sess, model = create_session(modelFile)
  except:
    return 'create session failed'
  
  for imageFile in imageFileList:
    imagenp  = cv2.imread(imageFile)
    imagenp  = cv2.cvtColor(imagenp, cv2.COLOR_BGR2RGB)
    imgSize  = imagenp.shape
    try:
      imgExpanded   = np.expand_dims(imagenp, axis=0)
      imgTensor     = model.get_tensor_by_name('image_tensor:0')
      boxes         = model.get_tensor_by_name('detection_boxes:0')
      scores        = model.get_tensor_by_name('detection_scores:0')
      classes       = model.get_tensor_by_name('detection_classes:0')
      numDetections = model.get_tensor_by_name('num_detections:0')
      
      # inference
      beginExecTime = time.time()
      (boxes, scores, classes, numDetections) = sess.run([boxes, scores, classes, numDetections], feed_dict={imgTensor: imgExpanded}) 
      endExecTime   = time.time()
      print(imageFile, 'exec duration:', endExecTime-beginExecTime)
      '''
      # debug for result
      for i in range(int(numDetections[0])):
        if scores[0][i] > threshold:
          cv2.rectangle(imagenp, (int(boxes[0][i][1] * imgSize[1]), int(boxes[0][i][0] * imgSize[0])), (int(boxes[0][i][3] * imgSize[1]), int(boxes[0][i][2] * imgSize[0])), (0, 0, 0), 3)
      '''   
      pointList = get_faultage_points(imgSize, boxes, scores, classes, threshold, xLenThresh, yLenThresh)
      if len(pointList) > 1:
        draw_faultage_line(imagenp, imgSize, pointList, lineType)
        imagenp   = cv2.cvtColor(imagenp, cv2.COLOR_RGB2BGR)
        imageName = os.path.basename(imageFile)
        cv2.imwrite(os.path.join(outputPath, imageName), imagenp)        
    except:
      return 'session run failed'
  return ''
  
def analysis_faultage(options):
  if os.path.splitext(options[0])[1] == '.py':
    programName = 'python ' + options[0]
  else:
    programName = options[0]
  cmdStr = 'Usage: {0} [OPTION]...\nOptions:\n\
       -i, --input      input path or image file, default is {1}\n\
       -o, --output     images saving path, default is {2}\n\
       -m, --model      model file, default is {3}\n\
       -t, --threshold  threshold[0~1], default is {4}\n\
       -l, --line       line type, {5}[default]: polyline, 1: straight line'.format(programName,
       g_default_image_path, g_default_result_path, g_default_model_file, g_score_thresh, g_line_type)
       
  try:
    opts, args = getopt.getopt(options[1:], 'hi:o:m:t:l:x:y:', ['input=', 'ouput=', 'model=', 'threshold=', 'line=', 'xthresh=', 'ythresh='])
  except:
    print(cmdStr)
    return 'options is error!'
  
  inputFile  = g_default_image_path
  outputPath = g_default_result_path
  modelFile  = g_default_model_file
  threshold  = g_score_thresh
  lineType   = g_line_type
  xLenThresh = g_x_len_thresh
  yLenThresh = g_y_len_thresh
  for opt, arg in opts:
    if opt == '-h':
      print(cmdStr)
      return ''
    elif opt in ('-i', '--input'):
      inputFile = arg
    elif opt in ('-o', '--output'):
      outputPath = arg
    elif opt in ('-m', '--model'):
      modelFile = arg
    elif opt in ('-t', '--threshold'):
      threshold = arg
    elif opt in ('-l', '--line'):
      lineType = arg
    elif opt in ('-x', '--xthresh'):
      xLenThresh = int(arg)
    elif opt in ('-y', '--ythresh'):
      yLenThresh = int(arg)
  
  result = detect_faultage(inputFile, outputPath, modelFile, threshold, lineType, xLenThresh, yLenThresh)
  return result  

'''
if __name__ == '__main__':
  result = analysis_faultage(sys.argv)
  if result != '':
    print('Error:', result)
  else:
    print('Execute success!')
'''