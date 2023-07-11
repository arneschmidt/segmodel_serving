from __future__ import print_function
import requests
import json
import cv2

addr = 'http://localhost:6500'
test_url = addr + '/pc_segmentation'

image = './example_images/images/slide005_core063.png'


img = cv2.imread(image)
# encode image as jpeg
_, img_encoded = cv2.imencode('.png', img)
# send http request with image and receive response
response = requests.post(test_url, data=img_encoded.tostring())
# decode response
print(json.loads(response.text))