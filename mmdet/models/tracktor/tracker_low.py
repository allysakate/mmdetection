import os
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from torch.autograd import Variable

import cv2
from mmdet.ops.nms import nms_wrapper
from mmdet.models.ocr import get_text

from .utils import bbox_overlaps, bbox_transform_inv, clip_boxes

class Tracker_Low():
	"""The main tracking file, here is where magic happens."""

	def __init__(self, obj_detect, reid_network, tracker_cfg, test_cfg, single):
		self.obj_detect = obj_detect
		self.reid_network = reid_network
		self.detection_thresh = tracker_cfg.detection_thresh
		self.regression_thresh = tracker_cfg.regression_thresh
		self.detection_nms_thresh = tracker_cfg.detection_nms_thresh
		self.regression_nms_thresh = tracker_cfg.regression_nms_thresh
		self.public_detections = tracker_cfg.public_detections
		self.inactive_patience = tracker_cfg.inactive_patience
		self.max_features_num = tracker_cfg.max_features_num
		self.reid_sim_threshold = tracker_cfg.reid_sim_threshold
		self.reid_iou_threshold = tracker_cfg.reid_iou_threshold
		self.do_align = tracker_cfg.do_align
		self.motion_model = tracker_cfg.motion_model

		self.warp_mode = eval(tracker_cfg.warp_mode)
		self.number_of_iterations = tracker_cfg.number_of_iterations
		self.termination_eps = tracker_cfg.termination_eps
		if not single:
			self.nms_cfg = test_cfg.rcnn.nms
		else:
			self.nms_cfg = test_cfg.nms
		self.img_scale = tracker_cfg.img_scale
		self.cl = 5

		self.reset()

	def reset(self, hard=True):
		print(f'Reset')
		self.tracks = []
		self.inactive_tracks = []

		if hard:
			self.track_num = 0
			self.results = {}
			self.im_index = 0

	def tracks_to_inactive(self, tracks):
		
		self.tracks = [t for t in self.tracks if t not in tracks]
		for t in tracks:
			t.pos = t.last_pos
			print(f'Tracks_to_Inactive: id={t.id} | pos={t.pos} ')
		self.inactive_tracks += tracks

	def add_lowframe(self, img, new_det_pos, new_det_scores, new_det_features, new_det_labels, ocr_model, cfg_ocr, ocr_converter):
		"""Initializes new Track objects and saves them for lowframe."""
		num_new = new_det_pos.size(0)
		for i in range(num_new):
			bbox = new_det_pos[i].view(1,-1)[0]
			if bbox[1] > 450:
				plate_string= self.get_ocr(img, bbox, ocr_model, cfg_ocr, ocr_converter)
			else:
				plate_string = "None"
			track_input = Track_Low(new_det_pos[i].view(1,-1), new_det_scores[i], new_det_labels[i], self.track_num + i, plate_string,
												new_det_features[i].view(1,-1),self.inactive_patience, self.max_features_num)
			self.tracks.append(track_input)
		self.track_num += num_new
		#print(f'LOW_add_num_new: {track_input}')

	def get_active_tracks(self):
		"""Get the positions of all active tracks."""
		if len(self.tracks) == 1:
			pos = [self.tracks[0].pos]
			sc  = [self.tracks[0].score]
			lbl = [self.tracks[0].label.cuda()]
		elif len(self.tracks) > 1:
			pos = []
			sc  = []
			lbl = []
			for t in self.tracks:
				p = t.pos.tolist()
				pos.append(p[0])
				s = t.score
				sc.append(s[0])
				l = t.label
				lbl.append(l)
			lbl = torch.tensor(lbl).cuda()
			pos = torch.tensor(pos)
			sc = torch.tensor(sc)
		else:
			pos = torch.zeros(0).cuda()
			sc  = torch.tensor([0]).cuda()
			lbl = torch.tensor([0]).cuda()
		print(f'Get Pos: Bbox={pos} \n \t Label={lbl}')
		return pos, sc, lbl

	def get_features(self):
		"""Get the features of all active tracks."""
		if len(self.tracks) == 1:
			features = self.tracks[0].features
		elif len(self.tracks) > 1:
			features = torch.cat([t.features for t in self.tracks],0)
		else:
			features = torch.zeros(0).cuda()
		return features

	def get_inactive_features(self):
		"""Get the features of all inactive tracks."""
		if len(self.inactive_tracks) == 1:
			features = self.inactive_tracks[0].features
		elif len(self.inactive_tracks) > 1:
			features = torch.cat([t.features for t in self.inactive_tracks],0)
		else:
			features = torch.zeros(0).cuda()
		return features

	def regress_tracks(self, blob):
		"""Regress the position of the tracks and also checks their scores."""
		print(f'REGRESS TRACKS')
		reg_score = True
		pos, lbl = self.get_active_tracks()
		# regress
		with torch.no_grad():
			bboxes, scores = self.obj_detect(return_loss=False, proposals=pos, track=True, regress=True, **blob)

		base_label = torch.ones((list(scores.shape)[0],4),dtype=int)
		label_index = torch.tensor((1,2,3,4),dtype=int)
		labels = base_label * label_index

		score_list = []
		label_list = []
		bbox_list = []
		s = []
		for i in range(scores.size(0)):
			cls_inds = scores[i][1:5] > self.detection_thresh
			if not cls_inds.any():
				continue
			cls_score=scores[i][1:5][cls_inds].tolist()
			score_list.append(cls_score[0])

			cls_label=labels[i][cls_inds].tolist()
			label_list.append(cls_label[0])

			cls_box = []
			for b in range(1,self.cl):
				box = bboxes[i][b*4:(b+1)*4].tolist()
				cls_box.append(box)
			cls_box = torch.tensor(cls_box)[cls_inds].tolist()
			bbox_list.append(cls_box[0])  

		cls_scores = torch.tensor(score_list).cuda()
		cls_labels = torch.tensor(label_list).cuda()
		cls_bboxes = torch.tensor(bbox_list).cuda()

		print(f'Prediction: Bbox={cls_bboxes} \n \t Score={cls_scores} \n  \t Label={cls_labels}')
	
		return cls_bboxes, cls_scores, cls_labels

	def get_ocr(self, img, bbox, ocr_model, cfg_ocr, ocr_converter):
		"""Initializes new Track objects and saves them for lowframe."""
		x_min = int(bbox[0])
		y_min = int(bbox[1])
		x_max = int(bbox[2])
		y_max = int(bbox[3])
		plate_img = img[y_min:y_max,x_min:x_max] 
		plate_string = get_text(plate_img, ocr_model, cfg_ocr, ocr_converter)
		if len(plate_string) == 0:
			plate_string = 'NAN'
		return plate_string


	def reid(self, blob, frame, new_det_pos, new_det_scores, new_det_labels, do_reid, ocr_model, cfg_ocr, ocr_converter):
		"""Tries to ReID with provided detections."""

		img_meta     = blob['img_meta'][0]
		scale_factor = img_meta[0]['scale_factor']
		new_det_features = self.reid_network.test_rois(img_meta[0]['reid_img'], new_det_pos/scale_factor).data		
	
		if do_reid:
			# calculate appearance distances
			dist_mat  = []
			pos = []
			num_track = 0
			track_list = []
			for t in self.tracks:
				track_list.append(num_track)
				num_track += 1
				print(f'New_Feat={new_det_features.size(0)}')
				if new_det_features.size(0) == 1:
					for feat in new_det_features:
						dist = t.test_features(feat.view(1,-1))
						dist_mat.append(dist)
				else:
					dist_mat.append(torch.cat([t.test_features(feat.view(1,-1)) for feat in new_det_features], 0))
					print(f'dist_mat: {dist_mat}')
				pos.append(t.pos)
			if len(dist_mat) > 1:
				dist_mat = torch.cat(dist_mat, 0)
				pos = torch.cat(pos,0)
			else:
				dist_mat = dist_mat[0]
				pos = pos[0]
			print(f'Dist_mat={dist_mat} \n \t pos={pos}')

			# calculate IoU distances
			iou = bbox_overlaps(pos, new_det_pos)
			iou_mask = torch.ge(iou, self.reid_iou_threshold)
			iou_neg_mask = ~iou_mask
			print(f'iou={iou} \n \t mask={iou_mask} \n \t neg_mask={iou_neg_mask}')
			# make all impossible assignemnts to the same add big value
			try:
				dist_mat = dist_mat * iou_mask.float() + iou_neg_mask.float()*1000
			except:
				mat_shape = iou_mask.size()
				dist_mat = dist_mat.resize_(mat_shape) * iou_mask.float() + iou_neg_mask.float()*1000
			dist_mat = dist_mat.cpu().numpy()
			print(f'dist_mat={dist_mat}')

			row_ind, col_ind = linear_sum_assignment(dist_mat)
			print(len(col_ind))
			assigned = []
			row_list = []
			inactive = []
			for r,c in zip(row_ind, col_ind):
				print(f'r={r} \t c={c}')
				print(f'dist_mat: {dist_mat[r,c]}')
				row_list.append(r)
			if len(col_ind) == len(self.tracks):
				idx_R = True
			else:
				idx_R = False
			less_Track = False
			for r,c in zip(row_ind, col_ind):
				if idx_R:
					idx = r
					print(f'R={idx}')
				else:
					idx = c
					print(f'C={idx}')
					inactive = list(set(track_list)-set(row_list))
				if dist_mat[r,c] <= self.reid_sim_threshold:
					t = self.tracks[r]
					t.pos = new_det_pos[idx].view(1,-1)
					print(t.pos, t.pos[0][1])
					if t.pos[0][1] > 450:
						t.ocr = self.get_ocr(frame, t.pos[0], ocr_model, cfg_ocr, ocr_converter)
					t.score = new_det_scores[idx]
					t.label = new_det_labels[idx]
					t.add_features(new_det_features[idx].view(1,-1))
					assigned.append(idx)
				else:
					inactive.append(idx)
			print(f'Active: {assigned} /n Inactive: {inactive}')
			if len(inactive) > 1:
				inactive = sorted(inactive, reverse=True)
			for i in inactive:
				print(i)
				t = self.tracks[i]
				self.tracks_to_inactive([t])
				print(f'Regression To Inactive: Bbox={t.pos} \n \t Score={t.score} \n \t Label={t.label}')

			keep = torch.Tensor([i for i in range(new_det_pos.size(0)) if i not in assigned]).long().cuda()
			if keep.nelement() > 0:
				new_det_pos = new_det_pos[keep]
				new_det_scores = new_det_scores[keep]
				new_det_features = new_det_features[keep]
			else:
				new_det_pos = torch.zeros(0).cuda()
				new_det_scores = torch.zeros(0).cuda()
				new_det_features = torch.zeros(0).cuda()
		print(f'reid: pos: {new_det_pos} \n sc: {new_det_scores} \n lbl: {new_det_labels}')
		return new_det_pos, new_det_scores, new_det_features, new_det_labels

	def clear_inactive(self):
		"""Checks if inactive tracks should be removed."""
		to_remove = []
		for t in self.inactive_tracks:
			print(f'Inactive Tracks: {t.pos} \n Purge? {t.is_to_purge()}')
			if t.is_to_purge():
				to_remove.append(t)
		for t in to_remove:
			self.inactive_tracks.remove(t)
		print(f'Clear Inactive: {to_remove}')
		
	def get_appearances(self, blob):
		"""Uses the siamese CNN to get the features for all active tracks."""
		print(f'Get Appearances')
		img_meta = blob['img_meta'][0]
		reid_img = img_meta[0]['reid_img']
		scale_factor = img_meta[0]['scale_factor']
		pos, sc, lbl = self.get_active_tracks()
		new_features = self.reid_network.test_rois(reid_img, pos[0] / scale_factor).data
		return new_features

	def add_features(self, new_features):
		"""Adds new appearance features to active tracks."""
		for t,f in zip(self.tracks, new_features):
			t.add_features(f.view(1,-1))

	def align(self, blob):
		"""Aligns the positions of active and inactive tracks depending on camera motion."""
		if self.im_index > 0:
			im1 = self.last_image.cpu().numpy().transpose(1, 2, 0)
			im2 = blob['img'][0].cpu().numpy().transpose(1, 2, 0)
			im1_gray = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
			im2_gray = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)
			sz = im1.shape
			warp_mode = self.warp_mode
			warp_matrix = np.eye(2, 3, dtype=np.float32)
			#number_of_iterations = 5000
			number_of_iterations = self.number_of_iterations
			termination_eps = self.termination_eps
			criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations,  termination_eps)
			(cc, warp_matrix) = cv2.findTransformECC (im1_gray,im2_gray,warp_matrix, warp_mode, criteria, None,1)
			warp_matrix = torch.from_numpy(warp_matrix)
			pos = []
			for t in self.tracks:
				p = t.pos[0]
				p1 = torch.Tensor([p[0], p[1], 1]).view(3,1)
				p2 = torch.Tensor([p[2], p[3], 1]).view(3,1)

				p1_n = torch.mm(warp_matrix, p1).view(1,2)
				p2_n = torch.mm(warp_matrix, p2).view(1,2)
				pos = torch.cat((p1_n, p2_n), 1).cuda()

				t.pos = pos.view(1,-1)
				#t.pos = clip_boxes(Variable(pos), blob['im_info'][0][:2]).data

			if self.do_reid:
				for t in self.inactive_tracks:
					p = t.pos[0]
					p1 = torch.Tensor([p[0], p[1], 1]).view(3,1)
					p2 = torch.Tensor([p[2], p[3], 1]).view(3,1)
					p1_n = torch.mm(warp_matrix, p1).view(1,2)
					p2_n = torch.mm(warp_matrix, p2).view(1,2)
					pos = torch.cat((p1_n, p2_n), 1).cuda()
					t.pos = pos.view(1,-1)

			if self.motion_model:
				for t in self.tracks:
					if t.last_pos.nelement() > 0:
						p = t.last_pos[0]
						p1 = torch.Tensor([p[0], p[1], 1]).view(3,1)
						p2 = torch.Tensor([p[2], p[3], 1]).view(3,1)

						p1_n = torch.mm(warp_matrix, p1).view(1,2)
						p2_n = torch.mm(warp_matrix, p2).view(1,2)
						pos = torch.cat((p1_n, p2_n), 1).cuda()

						t.last_pos = pos.view(1,-1)

	def motion(self):
		"""Applies a simple linear motion model that only consideres the positions at t-1 and t-2."""
		for t in self.tracks:
			# last_pos = t.pos.clone()
			# t.last_pos = last_pos
			# if t.last_pos.nelement() > 0:
				# extract center coordinates of last pos

			x1l = t.last_pos[0,0]
			y1l = t.last_pos[0,1]
			x2l = t.last_pos[0,2]
			y2l = t.last_pos[0,3]
			cxl = (x2l + x1l)/2
			cyl = (y2l + y1l)/2

			# extract coordinates of current pos
			x1p = t.pos[0,0]
			y1p = t.pos[0,1]
			x2p = t.pos[0,2]
			y2p = t.pos[0,3]
			cxp = (x2p + x1p)/2
			cyp = (y2p + y1p)/2
			wp = x2p - x1p
			hp = y2p - y1p

			# v = cp - cl, x_new = v + cp = 2cp - cl
			cxp_new = 2*cxp - cxl
			cyp_new = 2*cyp - cyl

			t.pos[0,0] = cxp_new - wp/2
			t.pos[0,1] = cyp_new - hp/2
			t.pos[0,2] = cxp_new + wp/2
			t.pos[0,3] = cyp_new + hp/2

			t.last_v = torch.Tensor([cxp - cxl, cyp - cyl]).cuda()

		if self.do_reid:
			for t in self.inactive_tracks:
				if t.last_v.nelement() > 0:
					# extract coordinates of current pos
					x1p = t.pos[0, 0]
					y1p = t.pos[0, 1]
					x2p = t.pos[0, 2]
					y2p = t.pos[0, 3]
					cxp = (x2p + x1p)/2
					cyp = (y2p + y1p)/2
					wp = x2p - x1p
					hp = y2p - y1p

					cxp_new = cxp + t.last_v[0]
					cyp_new = cyp + t.last_v[1]

					t.pos[0,0] = cxp_new - wp/2
					t.pos[0,1] = cyp_new - hp/2
					t.pos[0,2] = cxp_new + wp/2
					t.pos[0,3] = cyp_new + hp/2

	def step(self, blob, frame, ocr_model, cfg_ocr, ocr_converter):
		"""This function should be called every timestep to perform tracking with a blob
		containing the image information.
		Input:
		blob: {'img_meta': [DataContainer([[{'filename': 'data/CVAT_track/test/images/ch01_07-12_10.35-5.jpg', 
		'ori_shape': (1080, 1920, 3), 
		'img_shape': (750, 1333, 3), 
		'pad_shape': (768, 1344, 3), 
		'scale_factor': 0.6942708333333333, 
		'flip': False, 
		'img_norm_cfg': {'mean': array([123.675, 116.28 , 103.53 ], dtype=float32), 
		'std': array([58.395, 57.12 , 57.375], dtype=float32), 'to_rgb': True},
		'reid_img': array(..)
		"""
		print(f'DETECTIONS')
		# for t in self.tracks:
		# 	t.last_pos = t.pos.clone()
		# 	print(f'Last Get Pos: {t.last_pos}')
		
		###########################
		# Look for new detections #
		###########################
		with torch.no_grad():
			det_bboxes, det_labels = self.obj_detect(return_loss=False, track=True, regress=False, **blob)
			det_scores = det_bboxes[:, 4]
			det_bboxes = det_bboxes[:,:4]
		print(f'Detection: Bbox={det_bboxes} \n \t Score={det_scores} \n \t Labels {det_labels}')

		##################
		# Predict tracks #
		##################
		# num_tracks = 0
		# nms_inp_reg = torch.zeros(0).cuda()
		print(f'Track Length: {len(self.tracks)} \n')
		# align
		if self.do_align:
			self.align(blob)
		# apply motion model
		if self.motion_model:
			self.motion()
							# create nms input
				# new_features = self.get_appearances(blob)
				# nms here if tracks overlap
		if det_bboxes.nelement() > 0:
			new_det_pos    = det_bboxes
			new_det_scores = det_scores
			new_det_labels = det_labels
		else:
			new_det_pos    = torch.zeros(0).cuda()
			new_det_scores = torch.zeros(0).cuda()
			new_det_labels = torch.zeros(0).cuda()		
		
		if len(self.tracks):
			print('Do Reid')
			do_reid = True
		else:
			do_reid = False
		if new_det_pos.nelement() > 0: 
			new_det_pos, new_det_scores, new_det_features, new_det_labels = self.reid(blob, frame, new_det_pos, new_det_scores, new_det_labels, do_reid, ocr_model, cfg_ocr, ocr_converter)
			print(f'New: Bbox={new_det_pos} \n \t Score={new_det_scores} \n \t Label={new_det_labels}')

		print('CREATE NEW TRACKS')
		if new_det_pos.nelement() > 0:
			self.add_lowframe(frame, new_det_pos, new_det_scores, new_det_features, new_det_labels, ocr_model, cfg_ocr, ocr_converter)

		####################
		# Generate Results #
		####################

		for t in self.tracks:
			track_ind = int(t.id)
			if track_ind not in self.results.keys():
				self.results[track_ind] = {}
			img_meta = blob['img_meta'][0]
			scale_factor = img_meta[0]['scale_factor']
			try:
				pos = t.pos[0] / scale_factor
			except:
				pos = t.pos[0] / t.pos[0].new_tensor(scale_factor)
			sc = t.score
			lb = t.label
			ocr = t.ocr
			print(f'Tracks: Index={track_ind} \n \t Pos={pos} \n \t label={lb}  \n \t label={ocr} ')
			self.results[track_ind][self.im_index] = np.concatenate([pos.cpu().numpy(), np.array([sc]), np.array([lb]),  np.array([ocr])])

		self.im_index += 1
		self.last_image = blob['img'][0]
	#	print(f'last image: {self.last_image}')

		self.clear_inactive()

	def get_results(self):
		return self.results

class Track_Low(object):
	"""This class contains all necessary for every individual track."""

	def __init__(self, pos, score, label, track_id, ocr, features, inactive_patience, max_features_num):
		self.id = track_id
		self.pos = pos
		self.score = score
		self.label = label
		self.ocr = ocr
		self.features = deque([features])
		self.ims = deque([])
		self.count_inactive = 0
		self.inactive_patience = inactive_patience
		self.max_features_num = max_features_num
		self.last_pos = torch.Tensor([])
		self.last_v = torch.Tensor([])
		self.gt_id = None

	def is_to_purge(self):
		"""Tests if the object has been too long inactive and is to remove."""
		self.count_inactive += 1
		self.last_pos = torch.Tensor([])
		if self.count_inactive > self.inactive_patience:
			return True
		else:
			return False

	def add_features(self, features):
		"""Adds new appearance features to the object."""
		print(f'Add Features')
		self.features.append(features)
		if len(self.features) > self.max_features_num:
			self.features.popleft()

	def test_features(self, test_features):
		"""Compares test_features to features of this Track object"""
		print(f'Test Features')
		feat_list = []
		if len(self.features) > 1:
			feat_list = list(self.features)
			features = torch.cat(feat_list, 0)
		else:
			features = self.features[0]
		features = features.mean(0, keepdim=True)
		dist = F.pairwise_distance(features, test_features)
		return dist
