import torch
cl = 5
bbox = torch.tensor([[ 823.0825,  422.6323,  884.4810,  445.4470,  825.1765,  424.3098,
          882.1190,  445.5706,  825.4967,  424.4359,  883.8351,  445.8607,
          825.0917,  423.1759,  885.2707,  445.9502,  824.9314,  424.7397,
          884.5775,  445.5822],
        [1177.8347,  450.7871, 1230.6790,  475.7169, 1176.9585,  451.9310,
         1230.2957,  474.7883, 1175.9114,  452.3015, 1232.7776,  475.2092,
         1176.5146,  451.3338, 1232.7363,  475.4039, 1175.0518,  452.9634,
         1232.4177,  474.2813],
        [ 331.3047,  414.8334,  380.4967,  436.3731,  330.4767,  416.3872,
          379.2007,  435.5522,  331.6607,  415.5224,  379.5897,  436.3015,
          329.6631,  414.7757,  381.1109,  436.5462,  330.9028,  416.1662,
          380.2472,  436.3661]])
scores = torch.tensor([[1.1710e-01, 1.3347e-05, 1.1557e-04, 8.8255e-01, 2.2150e-04],
        [3.3483e-02, 2.4148e-06, 7.6900e-07, 9.6651e-01, 3.6592e-06],
        [1.6371e-01, 1.2196e-03, 9.8198e-05, 5.5764e-04, 8.3441e-01]])
base_label = torch.ones((list(scores.shape)[0],5),dtype=int)
label_index = torch.tensor((0,1,2,3,4),dtype=int)
labels = base_label * label_index
print(labels)
score_list = []
label_list = []
bbox_list = []
self.detection_thresh
			print(cls_inds)
			if not cls_inds.any():
				continue
for i in range(scores.size(0)):
    print(scores[i])
    cls_inds = scores[i] > 0.5
    print(cls_inds)
    cls_score=scores[i][cls_inds].tolist()
    score_list.append(cls_score[0])

    cls_label=labels[i][cls_inds].tolist()
    label_list.append(cls_label[0])

    cls_box = []
    for b in range(cl):
      box = bbox[i][b*4:(b+1)*4].tolist()
      cls_box.append(box)
    cls_box = torch.tensor(cls_box)[cls_inds].tolist()
    bbox_list.append(cls_box[0])  

cls_scores = torch.tensor(score_list)
cls_labels = torch.tensor(label_list)
cls_bboxes = torch.tensor(bbox_list)

print(cls_scores, cls_labels, cls_bboxes)

