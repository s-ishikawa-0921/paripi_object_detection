# -*- coding: utf-8 -*-
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import cv2
import base64
import uuid
import gc

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image

from utils import label_map_util
from utils import visualization_utils as vis_util

# ---------------------------------------------------------------------------------
# チェック
# ---------------------------------------------------------------------------------
graph = {}

def load_graph(path, name):

    global graph

    key = path + str(name)

    if key not in graph:

        graph[key] = tf.Graph()
        with graph[key].as_default():
          od_graph_def = tf.GraphDef()
          #with tf.gfile.GFile(path, 'rb') as fid:
          with open(path, "rb") as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name=name)
        print('load graph ' + path)

    return graph[key]

# ---------------------------------------------------------------------------------
# チェック
# ---------------------------------------------------------------------------------
def read_tensor_from_image_file(file_name,
                                input_height=299,
                                input_width=299,
                                input_mean=0,
                                input_std=255):
  input_name = "file_reader"
  output_name = "normalized"
  file_reader = tf.read_file(file_name, input_name)
  if file_name.endswith(".png"):
    image_reader = tf.image.decode_png(
        file_reader, channels=3, name="png_reader")
  elif file_name.endswith(".gif"):
    image_reader = tf.squeeze(
        tf.image.decode_gif(file_reader, name="gif_reader"))
  elif file_name.endswith(".bmp"):
    image_reader = tf.image.decode_bmp(file_reader, name="bmp_reader")
  else:
    image_reader = tf.image.decode_jpeg(
        file_reader, channels=3, name="jpeg_reader")
  float_caster = tf.cast(image_reader, tf.float32)
  dims_expander = tf.expand_dims(float_caster, 0)
  resized = tf.image.resize_bilinear(dims_expander, [input_height, input_width])
  normalized = tf.divide(tf.subtract(resized, [input_mean]), [input_std])
  sess = tf.Session()
  result = sess.run(normalized)

  return result

def load_labels(label_file):
  label = []
  proto_as_ascii_lines = tf.gfile.GFile(label_file).readlines()
  for l in proto_as_ascii_lines:
    label.append(l.rstrip())
  return label

def check(file_path, model_file, label_file):
    input_height = 299
    input_width = 299
    input_mean = 0
    input_std = 255
    input_layer = "Placeholder"
    output_layer = "final_result"

    graph = load_graph(model_file, None)
    t = read_tensor_from_image_file(
        file_path,
        input_height=input_height,
        input_width=input_width,
        input_mean=input_mean,
        input_std=input_std)
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)

    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0], {
            input_operation.outputs[0]: t
        })
    results = np.squeeze(results)

    top_k = results.argsort()[-5:][::-1]
    labels = load_labels(label_file)

    scores = {}
    for i in top_k:
        scores[labels[i]] = float(results[i])

    return scores


# ---------------------------------------------------------------------------------
# モザイク
# ---------------------------------------------------------------------------------
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# imgは元画像、rectはモザイクをかける座標
def add_image(img, rect, overlaid_image_path):

    # モザイクをかける座標を取得、左上(x1,y1),右下(x2,y2)
    (x1, y1, x2, y2) = rect

    # モザイクをかける幅と高さ
    w = x2 - x1
    h = y2 - y1

    #img_bottle = cv2.imread('/content/paripi_object_detection/data/bottle.png', -1)
    img_bottle = cv2.imread(overlaid_image_path, -1)

    bw = round(w)
    # bh = round(h / 2)
    bh = round(h)

    img_bottle = cv2.resize(img_bottle, (bw, bh), interpolation=cv2.INTER_LANCZOS4)
    height, width = img_bottle.shape[:2]

    img2 = img.copy()

    #img2y1 = y1 + round(height / 2)
    img2y1 = y1;
    #img2x1 = x1 + round(width / 2)
    img2x1 = x1

    #透明部分が0、不透明部分が1のマスクを作る
    alpha_mask = np.ones((height, width)) - np.clip(cv2.split(img_bottle)[3],0,1)
    #貼り付ける位置の背景部分
    target_background = img2[img2y1:img2y1+height, img2x1:img2x1+width]

    #各BRGチャンネルにalpha_maskを掛けて、前景の不透明部分が[0, 0, 0]のnew_backgroundを作る
    new_background = cv2.merge(list(map(lambda x:x * alpha_mask,cv2.split(target_background))))
    #BGRAをBGRに変換した画像とnew_backgroundを足すと合成できる
    img2[img2y1:img2y1+height, img2x1:img2x1+width] = cv2.merge(cv2.split(img_bottle)[:3]) + new_background

    return img2

# imgは元画像、rectはモザイクをかける座標
def put_mosaic(img, rect):

    # 縮小するサイズ
    size = (10, 10)
    # モザイクをかける座標を取得、左上(x1,y1),右下(x2,y2)
    (x1, y1, x2, y2) = rect
    # モザイクをかける幅と高さ
    w = x2 - x1
    h = y2 - y1
    # モザイクをかける部分を元画像から切り取り
    area = img[y1:y2, x1:x2]
    # 縮小
    small = cv2.resize(area, size)
    # 縮小した画像を拡大,zoomにはモザイク画像が入る
    zoom = cv2.resize(small, (w, h), interpolation=cv2.INTER_AREA)
    # 元の画像へモザイク画像をコピー
    img2 = img.copy()
    img2[y1:y2, x1:x2] = zoom

    return img2

def run_inference_for_single_image(image, graph):
  with graph.as_default():
    with tf.Session() as sess:
      # Get handles to input and output tensors
      ops = tf.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
          'num_detections', 'detection_boxes', 'detection_scores',
          'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
          tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
              tensor_name)
      if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
        detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
        detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
        real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
        detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])

        print(detection_boxes);

        detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
        detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
            detection_masks, detection_boxes, image.shape[0], image.shape[1])
        detection_masks_reframed = tf.cast(
            tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # Follow the convention by adding back the batch dimension
        tensor_dict['detection_masks'] = tf.expand_dims(
            detection_masks_reframed, 0)
      image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                             feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
      output_dict['num_detections'] = int(output_dict['num_detections'][0])
      output_dict['detection_classes'] = output_dict[
          'detection_classes'][0].astype(np.uint8)
      output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
      output_dict['detection_scores'] = output_dict['detection_scores'][0]
      if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
  return output_dict


def mosaic(image_path, pb_path, pb_label_path, overlaid_image_path):

    detection_graph = load_graph(pb_path, '')
    category_index = label_map_util.create_category_index_from_labelmap(pb_label_path, use_display_name=True)

    image_np = load_image_into_numpy_array(Image.open(image_path))
    #image_np_expanded = np.expand_dims(image_np, axis=0)
    image_dict = run_inference_for_single_image(image_np, detection_graph)

    image_out = cv2.imread(image_path)

    for i in range(len(image_dict['detection_scores'])):

        if image_dict['detection_scores'][i] < 0.5:
          continue;

        if(image_dict['detection_classes'][i] != 1):
          continue

        xmin = int(round(image_dict['detection_boxes'][i][1] * image_np.shape[1]))
        xmax = int(round(image_dict['detection_boxes'][i][3] * image_np.shape[1]))
        ymin = int(round(image_dict['detection_boxes'][i][0] * image_np.shape[0]))
        ymax = int(round(image_dict['detection_boxes'][i][2] * image_np.shape[0]))

        #ボトルの重ね合わせ
        image_out = add_image(image_out, (xmin,ymin,xmax,ymax), overlaid_image_path)
        #モザイク
        image_out = put_mosaic(image_out, (xmin,ymin,xmax,ymax))

    #cv2.imwrite(image_path, out_img)
    #b64 = base64.encodestring(open(image_path, 'rb').read())

    return image_out

    # return {
    #     "result":True,
    #     "binary": 'data:image/png;base64,' + b64.decode('utf8')
    # }


def trim(org_dir_path, dest_dir_path, pb_path, pb_label_path):

    detection_graph = load_graph(pb_path, '')
    category_index = label_map_util.create_category_index_from_labelmap(pb_label_path, use_display_name=True)

    if not os.path.exists(dest_dir_path):
        os.makedirs(dest_dir_path)

    for file_name in os.listdir(org_dir_path):

        image_path = org_dir_path + file_name
        image_root, image_ext = os.path.splitext(image_path)

        print(image_path)

        try:
            image_np = load_image_into_numpy_array(Image.open(image_path))
            image_dict = run_inference_for_single_image(image_np, detection_graph)

            image = cv2.imread(image_path)

            for i in range(len(image_dict['detection_scores'])):

                if image_dict['detection_scores'][i] < 0.5:
                  continue;

                if(category_index[image_dict['detection_classes'][i]]['name'] != 'person'):
                  continue

                xmin = int(round(image_dict['detection_boxes'][i][1] * image_np.shape[1]))
                xmax = int(round(image_dict['detection_boxes'][i][3] * image_np.shape[1]))
                ymin = int(round(image_dict['detection_boxes'][i][0] * image_np.shape[0]))
                ymax = int(round(image_dict['detection_boxes'][i][2] * image_np.shape[0]))

                # trimして保存
                cv2.imwrite(dest_dir_path + str(uuid.uuid4()) + image_ext, image[ymin:ymax,xmin:xmax])
        except:
            print('file error : ' + image_path)
            pass
