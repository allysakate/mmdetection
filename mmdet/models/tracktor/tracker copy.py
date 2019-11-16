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

from .utils import bbox_overlaps, bbox_transform_inv, clip_boxes

class Tracker():
	"""The main tracking file, here is where magic happens."""
	# only track pedestrian

	def __init__(self, obj_detect, reid_network, tracker_cfg, test_cfg, rcnn=True):
		self.obj_detect = obj_detect
		self.reid_network = reid_network
		self.vid_framerate = 30
		self.detection_thresh = tracker_cfg.detection_thresh
		self.regression_thresh = tracker_cfg.regression_thresh
		self.detection_nms_thresh = tracker_cfg.detection_nms_thresh
		self.regression_nms_thresh = tracker_cfg.regression_nms_thresh
		self.public_detections = tracker_cfg.public_detections
		self.inactive_patience = tracker_cfg.inactive_patience
		self.do_reid = tracker_cfg.do_reid
		self.max_features_num = tracker_cfg.max_features_num
		self.reid_sim_threshold = tracker_cfg.reid_sim_threshold
		self.reid_iou_threshold = tracker_cfg.reid_iou_threshold
		self.do_align = tracker_cfg.do_align
		self.motion_model = tracker_cfg.motion_model

		self.warp_mode = eval(tracker_cfg.warp_mode)
		self.number_of_iterations = tracker_cfg.number_of_iterations
		self.termination_eps = tracker_cfg.termination_eps
		if rcnn:
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

	def add_lowframe(self, new_det_pos, new_det_scores, new_det_features, new_det_labels):
		"""Initializes new Track objects and saves them for lowframe."""
		num_new = new_det_pos.size(0)
		for i in range(num_new):
			#pos, score, label, track_id, features, inactive_patience, max_features_num):
			track_input = Track(new_det_pos[i].view(1,-1), new_det_scores[i], new_det_labels[i], self.track_num + i, new_det_features[i].view(1,-1),
																	self.inactive_patience, self.max_features_num)
			self.tracks.append(track_input)
		self.track_num += num_new
		#print(f'LOW_add_num_new: {track_input}')

	def add_highframe(self, new_det_pos, new_det_scores, new_det_labels):
		"""Initializes new Track objects and saves them for highframe."""
		num_new = new_det_pos.size(0)
		for i in range(num_new):
			#pos, score, label, track_id, features, inactive_patience, max_features_num):
			track_label = new_det_labels[i]
			if isinstance(track_label, int):
				track_label = torch.tensor([track_label]).cuda()
				
			track_input = Track(new_det_pos[i].view(1,-1), new_det_scores[i], track_label, self.track_num + i, None,
																	self.inactive_patience, None)
			self.tracks.append(track_input)
		self.track_num += num_new
		#print(f'HIGH_add_num_new: {track_input.pos} | {track_input.label}')

	def regress_tracks(self, blob):
		"""Regress the position of the tracks and also checks their scores."""
		print(f'REGRESS TRACKS')
		reg_score = True
		pos, lbl = self.get_pos()
		# regress
		with torch.no_grad():
			pred_bbox_score, pred_labels = self.obj_detect(return_loss=False, proposals=pos, track=True, regress=True, **blob)
		pred_scores = pred_bbox_score[:, 4]
		pos_idx     = torch.gt(pred_scores,self.detection_thresh).nonzero().view(-1)
		pred_scores = pred_scores[pos_idx]
		pred_bboxes = pred_bbox_score[:,:4]
		pred_bboxes = pred_bboxes[pos_idx]
		pred_labels = pred_labels[pos_idx]
		print(f'Prediction: Bbox={pred_bboxes} \n \t Score={pred_scores} \n  \t Label={pred_labels}')
		s = []
		pred_len = len(pred_bboxes)
		track_len = len(self.tracks)
		if track_len >= pred_len:
			range_len = track_len
		else:
			range_len = pred_len
		for i in range(track_len-1,-1,-1):
			#print(f'i: {i} | {len(self.tracks)-1,-1,-1}')
			try:
				t = self.tracks[i]
				t.score = pred_scores[i]
				if t.score <= self.regression_thresh:
					self.tracks_to_inactive([t])
					print(f'Regression To Inactive: Bbox={t.pos} \n \t Score={t.score} \n \t Label={t.label}')
				else:
					s.append(t.score)
					t.pos = pred_bboxes[i].view(1,-1)
					t.label = torch.tensor(pred_labels[i]).cuda()
				print(f'Regression MultiClass: Bbox={t.pos} \n \t Score={t.score} \n \t Label={t.label} \n')
			except:
				t = self.tracks[i]
				self.tracks_to_inactive([t])
				print(f'ERROR: Reg To Inactive: Bbox={t.pos} \n \t Score={t.score} \n \t Label={t.label}')
	
		return torch.Tensor(s[::-1]).cuda()

	def get_pos(self):
		"""Get the positions of all active tracks."""
		if len(self.tracks) == 1:
			pos = [self.tracks[0].pos]
			lbl = [self.tracks[0].label]
		elif len(self.tracks) > 1:
			pos = torch.stack([t.pos for t in self.tracks])
			for t in self.tracks:
				print(t.label)
			lbl = torch.stack([t.label for t in self.tracks])
		else:
			pos = torch.zeros(0).cuda()
			lbl = torch.tensor([0])
		print(f'Get Pos: Bbox={pos} \n \t Label={lbl}')
		return pos, lbl

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

	def reid(self, blob, new_det_pos, new_det_scores, new_det_labels):
		"""Tries to ReID inactive tracks with provided detections."""

		img_meta = blob['img_meta'][0].data[0]
		new_det_features = self.reid_network.test_rois(img_meta[0]['reid_img'], new_det_pos / img_meta[0]['scale_factor']).data		
		if len(self.inactive_tracks) >= 1 and self.do_reid:
			# calculate appearance distances
			dist_mat = []
			pos = []
			for t in self.inactive_tracks:
				dist_mat.append(torch.cat([t.test_features(feat.view(1,-1)) for feat in new_det_features], 1))
				pos.append(t.pos)
			if len(dist_mat) > 1:
				dist_mat = torch.cat(dist_mat, 0)
				pos = torch.cat(pos,0)
			else:
				dist_mat = dist_mat[0]
				pos = pos[0]

			# calculate IoU distances
			iou = bbox_overlaps(pos, new_det_pos)
			iou_mask = torch.ge(iou, self.reid_iou_threshold)
			iou_neg_mask = ~iou_mask
			# make all impossible assignemnts to the same add big value
			dist_mat = dist_mat * iou_mask.float() + iou_neg_mask.float()*1000
			dist_mat = dist_mat.cpu().numpy()

			row_ind, col_ind = linear_sum_assignment(dist_mat)

			assigned = []
			remove_inactive = []
			for r,c in zip(row_ind, col_ind):
				if dist_mat[r,c] <= self.reid_sim_threshold:
					t = self.inactive_tracks[r]
					self.tracks.append(t)
					t.count_inactive = 0
					t.last_v = torch.Tensor([])
					t.pos = new_det_pos[c].view(1,-1)
					t.add_features(new_det_features[c].view(1,-1))
					assigned.append(c)
					remove_inactive.append(t)

			for t in remove_inactive:
				self.inactive_tracks.remove(t)

			keep = torch.Tensor([i for i in range(new_det_pos.size(0)) if i not in assigned]).long().cuda()
			if keep.nelement() > 0:
				new_det_pos = new_det_pos[keep]
				new_det_scores = new_det_scores[keep]
				new_det_features = new_det_features[keep]
			else:
				new_det_pos = torch.zeros(0).cuda()
				new_det_scores = torch.zeros(0).cuda()
				new_det_features = torch.zeros(0).cuda()
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
		img_meta = blob['img_meta'][0].data[0]
		reid_img = img_meta[0]['reid_img']
		scale_factor = img_meta[0]['scale_factor']
		pos, lbl = self.get_pos()
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
			im2 = blob['img'][0].data[0].cpu().numpy().transpose(1, 2, 0)
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

	def step(self, blob):
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
		for t in self.tracks:
			t.last_pos = t.pos.clone()
			print(f'Last Get Pos: {t.last_pos}')
		
		###########################
		# Look for new detections #
		###########################
		with torch.no_grad():
			det_bboxes, det_labels = self.obj_detect(return_loss=False, track=True, regress=False, **blob)

		print(f'Detection: Bbox: {det_bboxes} \n \t Labels {det_labels}')

		##################
		# Predict tracks #
		##################
		num_tracks = 0
		nms_inp_reg = torch.zeros(0).cuda()
		print(f'Track Length: {len(self.tracks)} \n')
		if len(self.tracks):
			# align
			if self.do_align:
				self.align(blob)
			# apply motion model
			if self.motion_model:
				self.motion()
			#regress
			regress_scores = self.regress_tracks(blob)
			#print(f'regress_scores: {regress_scores}')

			if len(self.tracks):

				# create nms input
				# new_features = self.get_appearances(blob)

				# nms here if tracks overlap
				print('TRACKS OVERLAP')
				predict_bbox, predict_label = self.get_pos()
				#print(f'inp_reg: {predict_bbox} | {regress_scores} | {regress_scores.view(-1,1)}')
				if len(predict_bbox) == 1: 
					nms_inp_reg = torch.cat((predict_bbox[0], regress_scores.add_(3).view(-1 ,1)), 1)
				else:
					predict_bbox = torch.tensor(predict_bbox.squeeze().tolist()).add_(3).cuda()
					nms_inp_reg = torch.cat((predict_bbox, regress_scores.view(-1,1)), 1)
				nms_type = self.nms_cfg.pop('type', 'nms')
				nms_op = getattr(nms_wrapper, nms_type)
				keep = nms_op(nms_inp_reg, self.regression_nms_thresh)
				print(f'Tracks to Keep: {keep}')
				self.tracks_to_inactive([self.tracks[i]
				                         for i in list(range(len(self.tracks)))
				                         if i not in keep[1]])

				if keep[1].nelement() > 0:
					predict_bbox, predict_label = self.get_pos()
					if len(predict_bbox) == 1: 
						nms_inp_reg = torch.cat((predict_bbox[0], torch.ones(predict_bbox[0].size(0)).add_(3).view(-1,1).cuda()),1)
						predict_label = torch.tensor(predict_label).cuda()
					else:
						predict_bbox = torch.tensor(predict_bbox.squeeze().tolist()).cuda()
						predict_label = torch.tensor(predict_label.squeeze().tolist()).cuda()
						nms_inp_reg = torch.cat((predict_bbox, torch.ones(predict_bbox.size(0)).add_(3).view(-1,1).cuda()),1)
					
					if self.vid_framerate < 24:
						new_features = self.get_appearances(blob)
						self.add_features(new_features)
					else:
						num_tracks = nms_inp_reg.size(0)
				else:
					nms_inp_reg = torch.zeros(0).cuda()
					num_tracks = 0
					
				print(f'Concat: Bbox={nms_inp_reg} \n \t Label={predict_label} \n')

		#####################
		# Create new tracks #
		#####################

		# !!! Here NMS is used to filter out detections that are already covered by tracks. This is
		# !!! done by iterating through the active tracks one by one, assigning them a bigger score
		# !!! than 1 (maximum score for detections) and then filtering the detections with NMS.
		# !!! In the paper this is done by calculating the overlap with existing tracks, but the
		# !!! result stays the same.
		print('CREATE NEW TRACKS')
		if det_bboxes.nelement() > 0:
			new_track_bbox  = det_bboxes
			new_track_label = det_labels
		else:
			new_track_bbox = torch.zeros(0).cuda()
			new_track_label  = torch.zeros(0).cuda()

		#print(f'new_track_bbox: {new_track_bbox, new_track_label}')

		if new_track_bbox.nelement() > 0:
			nms_type = self.nms_cfg.pop('type', 'nms')
			nms_op = getattr(nms_wrapper, nms_type)
			keep = nms_op(new_track_bbox, self.detection_nms_thresh)[1]
			#print(f'new_track_nms: {nms_type,nms_op} \n {nms_op(new_track_bbox, self.detection_nms_thresh)} \n {keep}')
			new_track_bbox = new_track_bbox[keep]
			new_track_label = new_track_label[keep]
			print(f'To Keep: Bbox={new_track_bbox} \n \t Label to Keep={new_track_label}')
			
			# check with every track in a single run (problem if tracks delete each other)
			#print(f'num_tracks = {num_tracks}')
			for i in range(num_tracks):
				nms_pos = torch.cat((nms_inp_reg[i].view(1,-1), new_track_bbox), 0)
				nms_lbl = torch.cat((predict_label[i].view(1), new_track_label))
				#print(f'nms_inp[i]:{i} | {nms_pos} | {nms_lbl}')
				nms_keep = nms_op(nms_pos, self.detection_nms_thresh)
				print(f'Reg-Det To Keep: {nms_keep}')
				try:
					scores_keep = nms_keep[0][:, 4]
					keep = torch.ge(scores_keep)
				except:
					keep = nms_keep[1]
					keep = keep[torch.ge(keep,1)]
				print(f'Filter: {keep}')
				if keep.nelement() == 0:
					new_track_bbox = new_track_bbox.new(0)
					new_track_label = new_track_label.new(0)
					break
				new_track_bbox = nms_pos[keep]
				new_track_label= nms_lbl[keep]
				#print(f'new_track: {new_track_bbox} | {new_track_label}')

		if new_track_bbox.nelement() > 0:
			new_det_pos    = new_track_bbox[:,:4]
			new_det_scores = new_track_bbox[:, 4]
			new_det_labels = new_track_label
		
			print(f'New Pos: Bbox={new_det_pos} \n \t Score={new_det_scores} \n \t Label={new_det_labels}')

			# try to redientify tracks
			if self.vid_framerate < 24:
				new_det_pos, new_det_scores, new_det_features, new_det_labels = self.reid(blob, new_det_pos, new_det_scores, new_det_labels)
				print(f'reid: {new_det_pos} | {new_det_scores}')

	
			# add new
			# if new_det_pos.nelement() > 0:
			# 	self.add(new_det_pos, new_det_scores, new_det_features, new_det_labels)
			
			if new_det_pos.nelement() > 0:
				if self.vid_framerate < 24:
					self.add_lowframe(new_det_pos, new_det_scores, new_det_features, new_det_labels)
				else:
					self.add_highframe(new_det_pos, new_det_scores, new_det_labels)

		####################
		# Generate Results #
		####################

		for t in self.tracks:
			track_ind = int(t.id)
			if track_ind not in self.results.keys():
				self.results[track_ind] = {}
			img_meta = blob['img_meta'][0].data[0]
			scale_factor = img_meta[0]['scale_factor']
			try:
				pos = t.pos[0] / scale_factor
			except:
				pos = t.pos[0] / t.pos[0].new_tensor(scale_factor)
			sc = t.score
			lb = t.label
			print(f'Tracks: Index={track_ind} \n \t Pos={pos} \n \t label={lb} ')
			self.results[track_ind][self.im_index] = np.concatenate([pos.cpu().numpy(), np.array([sc]), np.array([lb])])

		self.im_index += 1
		self.last_image = blob['img'][0].data[0]
	#	print(f'last image: {self.last_image}')

		self.clear_inactive()

	def get_results(self):
		return self.results

class Track(object):
	"""This class contains all necessary for every individual track."""

	def __init__(self, pos, score, label, track_id, features, inactive_patience, max_features_num):
		self.id = track_id
		self.pos = pos
		self.score = score
		self.label = label
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
		print('Test Features')
		if len(self.features) > 1:
			features = torch.cat(self.features, 0)
		else:
			features = self.features[0]
		features = features.mean(0, keepdim=True)
		dist = F.pairwise_distance(features, test_features)
		return dist
