#!/usr/bin/env python3

# Copyright 2020 coyote009
#  Based on the tensorflow lite example code
#  The original code is licensed and copyrighted as follows

# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Example using TF Lite to detect objects with the Raspberry Pi camera."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import re
import time

import numpy as np
import cv2
import picamera
import picamera.array

from tflite_runtime.interpreter import Interpreter

def load_labels(path):
  """Loads the labels file. Supports files with or without index numbers."""
  with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()
    labels = {}
    for row_number, content in enumerate(lines):
      pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
      if len(pair) == 2 and pair[0].strip().isdigit():
        labels[int(pair[0])] = pair[1].strip()
      else:
        labels[row_number] = pair[0].strip()
  return labels


def set_input_tensor(interpreter, image):
  """Sets the input tensor."""
  tensor_index = interpreter.get_input_details()[0]['index']
  input_tensor = interpreter.tensor(tensor_index)()[0]
  input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
  """Returns the output tensor at the given index."""
  output_details = interpreter.get_output_details()[index]
  tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
  return tensor


def detect_objects(interpreter, image, threshold):
  """Returns a list of detection results, each a dictionary of object info."""
  set_input_tensor(interpreter, image)
  interpreter.invoke()

  # Get all output details
  boxes = get_output_tensor(interpreter, 0)
  classes = get_output_tensor(interpreter, 1)
  scores = get_output_tensor(interpreter, 2)
  count = int(get_output_tensor(interpreter, 3))

  results = []
  for i in range(count):
    if scores[i] >= threshold:
      result = {
          'bounding_box': boxes[i],
          'class_id': classes[i],
          'score': scores[i]
      }
      results.append(result)
  return results


def annotate_objects(image, results, labels):
  """Draws the bounding box and label for each object in the results."""
  for obj in results:
    # Convert the bounding box figures from relative coordinates
    # to absolute coordinates based on the original resolution
    ymin, xmin, ymax, xmax = obj['bounding_box']
    xmin = int(xmin * image.shape[1])
    xmax = int(xmax * image.shape[1])
    ymin = int(ymin * image.shape[0])
    ymax = int(ymax * image.shape[0])

    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0))

    ## Overlay the box, label, and score on the camera preview
    image = cv2.putText(image, '%s(%.2f)' % (labels[obj['class_id']], obj['score']),
                        (xmin, ymin+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0))

  return image

def main():
  parser = argparse.ArgumentParser(
      formatter_class=argparse.ArgumentDefaultsHelpFormatter)
  parser.add_argument(
      '--model', help='File path of .tflite file.', required=True)
  parser.add_argument(
      '--labels', help='File path of labels file.', required=True)
  parser.add_argument(
      '--threshold',
      help='Score threshold for detected objects.',
      required=False,
      type=float,
      default=0.4)
  args = parser.parse_args()

  labels = load_labels(args.labels)
  interpreter = Interpreter(args.model)
  interpreter.allocate_tensors()
  _, input_height, input_width, _ = interpreter.get_input_details()[0]['shape']

  image_size = (480, 480)
  camera = picamera.PiCamera(resolution=image_size,
                             framerate=30)
  buf = picamera.array.PiRGBArray(camera, size=image_size)

  for frame in camera.capture_continuous(buf, format='bgr',
                                         use_video_port=True):

    image_out = frame.array.copy()

    image = cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (input_width, input_height))

    start_time = time.monotonic()
    results = detect_objects(interpreter, image, args.threshold)
    elapsed_ms = (time.monotonic() - start_time) * 1000

    image_out = annotate_objects(image_out, results, labels)

    cv2.imshow("img", image_out)
    if cv2.waitKey(1) == 27:
        break

    buf.truncate(0)

if __name__ == '__main__':
  main()
