---
title: COCO数据集
tags: 
       - 数据集
       - COCO
grammar_cjkRuby: true
---



&ensp;&ensp;&ensp;&ensp;深度学习能够快速发展，大规模数据集的建立是一个不可忽视的先决条件，现在有很多能够满足不同任务的数据集，比如imagenet[1]，COCO[2]，LVIS[3]等等，今天主要记录一下之前COCO数据集一些理解的误区。
<!--more-->
COCO数据集现在主要是指的COCO2017数据集，其共包含train/118k，val/5k，test-dev/20k，test-challenge/40k这四个部分，其包含目标检测、keypoint检测、实例分割、全景分割等任务，关于其数据集的解释可以看其官网[COCO](https://cocodataset.org/)，也可以看知乎上一篇比较好的介绍[COCO数据集的标注格式](https://zhuanlan.zhihu.com/p/29393415)。今天主要是记录一下自己关于annotation部分的一些误解。
1.COCO的类别数量
   在不同的代码中经常会看到COCO的类别数量有时候会设置为80，而有的时候却会设置为90，这是怎么一回事呢，其实看了下面这个COCO类别映射的定义就知道了。
```javascript
	coco_id_name_map={1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
                   6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
                   11: 'fire hydrant', 13: 'stop sign', 14: 'parking meter', 15: 'bench',
                   16: 'bird', 17: 'cat', 18: 'dog', 19: 'horse', 20: 'sheep', 21: 'cow',
                   22: 'elephant', 23: 'bear', 24: 'zebra', 25: 'giraffe', 27: 'backpack',
                   28: 'umbrella', 31: 'handbag', 32: 'tie', 33: 'suitcase', 34: 'frisbee',
                   35: 'skis', 36: 'snowboard', 37: 'sports ball', 38: 'kite', 39: 'baseball bat',
                   40: 'baseball glove', 41: 'skateboard', 42: 'surfboard', 43: 'tennis racket',
                   44: 'bottle', 46: 'wine glass', 47: 'cup', 48: 'fork', 49: 'knife', 50: 'spoon',
                   51: 'bowl', 52: 'banana', 53: 'apple', 54: 'sandwich', 55: 'orange',
                   56: 'broccoli', 57: 'carrot', 58: 'hot dog', 59: 'pizza', 60: 'donut',
                   61: 'cake', 62: 'chair', 63: 'couch', 64: 'potted plant', 65: 'bed', 67: 'dining table',
                   70: 'toilet', 72: 'tv', 73: 'laptop', 74: 'mouse', 75: 'remote', 76: 'keyboard',
                   77: 'cell phone', 78: 'microwave', 79: 'oven', 80: 'toaster', 81: 'sink',
                   82: 'refrigerator', 84: 'book', 85: 'clock', 86: 'vase', 87: 'scissors',
                   88: 'teddy bear', 89: 'hair drier', 90: 'toothbrush'}
```
从上面可以看出COCO数据集总共有80类，但是编号是到90，其中一些编号是没有的，比如12。
2.annotation的‘id’字段
   在COCO数据集的annotation部分共有三个和id有关的字段，‘image_id’，‘category_id’和‘id’，其中‘image_id’就是指这张图片的编号，‘category_id’指的是对应类别的编号，‘id’我之前一直理解的是一张图片中有很多个实例，也包含很多类别，比如一张图片中有三个人还有两只狗，以前误以为‘id’就是示‘person1_id’，‘person2_id’，‘person3_id’，‘dog1_id’，‘dog2_id’，不同图片中是可以重复的，今天查看之后发现并不是，其实是每一个实例都会分配一个唯一的‘id’，而且这个‘id’的命名现在看来毫无规律，我猜应该也是从0到某一个数，当然这是COCO数据集中所有实例，但是现在我们不能得到test部分的标签所以暂时不知道。
```javascript
      from pycocotools.coco import COCO
      annfile = "/Users/zhangwenchao/Documents/dataset/coco/annotations/instances_train2017.json"
      coco = COCO(annfile)
	  annos_ids = []
	  annos_image_ids = []
	  for anno in annos:
	      id = anno['id']
		  image_id = anno['image_id']
		  annos_ids.append(id)
		  annos_image_ids.append(image_id)
	  print(len(annos_ids))
	  print(len(set(annos_ids)))
	  print(len(annos_image_ids))
	  print(len(set(annos_image_ids)))
	  
```
3.annotation的‘area’字段
   之前以为‘area’字段指的是目标box的面积，其实不是，这个指的是每个实例的分割mask所占的面积，其计算可参考一下代码。
   ```javascript
   from pycocotools.coco import COCO
   annfile = "/Users/zhangwenchao/Documents/dataset/coco/annotations/instances_train2017.json"
   coco = COCO(annfile)
   import pycocotools.mask as maskUtils
   def annToRLE(ann, i_w, i_h):
		h, w = i_h, i_w
		segm = ann['segmentation']
		if type(segm) == list:
			# polygon -- a single object might consist of multiple parts
			# we merge all parts into one mask rle code
			rles = maskUtils.frPyObjects(segm, h, w)
			rle = maskUtils.merge(rles)
		elif type(segm['counts']) == list:
			# uncompressed RLE
			rle = maskUtils.frPyObjects(segm, h, w)
		else:
			# rle
			rle = ann['segmentation']
		return rle
  imgIds = 558840
  annIds = coco.getAnnIds(imgIds=imgIds)
  anns = coco.loadAnns(annIds)
  print(anns[0]['area'])
  annos_imgs = coco.dataset["images"]
  for annos_img in annos_imgs:
	  if annos_img["id"] == imgIds:
			img_w = annos_img['width']
			img_h = annos_img['height']
			rle = annToRLE(anns[0], img_w, img_h)
			area = float(maskUtils.area(rle))
			print(area)
   ```
参考：
   &ensp;http://www.image-net.org/
  &ensp;https://cocodataset.org/
  &ensp;https://www.lvisdataset.org/
 **注**：此博客内容为原创，转载请说明出处