__author__ = 'andreasveit'
__version__ = '1.2'

# Interface for evaluating with the COCO-Text dataset.

# COCO-Text is a large dataset designed for text detection and recognition.
# This is a Python API that assists in evaluating text detection and recognition results 
# on COCO-Text. The format of the COCO-Text annotations is described on 
# the project website http://vision.cornell.edu/se3/coco-text/. In addition to this evaluation API, please download 
# the COCO-Text tool API, both the COCO images and annotations.
# This dataset is based on Microsoft COCO. Please visit http://mscoco.org/
# for more information on COCO, including for the image data, object annotatins
# and caption annotations. 

# The following functions are defined:
#  getDetections  - Compute TP, FN and FP
#  evaluateAttribute  - Evaluates accuracy for classifying text attributes
#  evaluateTranscription  - Evaluates accuracy of transcriptions
#  area, intersect, iou_score, decode, inter  -  small helper functions
#  printDetailedResults   - Prints detailed results as reported in COCO-Text paper

# COCO-Text Evaluation Toolbox.        Version 1.2
# Data, Data API and paper available at:  http://vision.cornell.edu/se3/coco-text/
# Code written by Andreas Veit, 2016.
# Licensed under the Simplified BSD License [see bsd.txt]

import editdistance
import copy
import re

# Compute detections
def getDetections(groundtruth, evaluation, imgIds = [], annIds = [], detection_threshold = 0.5):
	"""
	A box is a match iff the intersection of union score is >= 0.5.
	Params
	------
	Input dicts have the format of annotation dictionaries
	"""
	#parameters

	detectRes = {}
	# results are lists of dicts {gt_id: xxx, eval_id: yyy}
	detectRes['true_positives'] = []
	detectRes['false_negatives'] = []
	detectRes['false_positives'] = []

	imgIds = imgIds if len(imgIds)>0 else inter(groundtruth.imgToAnns.keys(), evaluation.imgToAnns.keys())

	for cocoid in imgIds:
		gt_bboxes = groundtruth.imgToAnns[cocoid]
		eval_bboxes = copy.copy(evaluation.imgToAnns[cocoid])

		for gt_box_id in gt_bboxes:
			gt_box = groundtruth.anns[gt_box_id]['bbox']
			max_iou = detection_threshold
			match = None
			for eval_box_id in eval_bboxes:
				eval_box = evaluation.anns[eval_box_id]['bbox']
				iou = iou_score(gt_box,eval_box)
				if iou > max_iou:
					match = eval_box_id
			if match:
				detectRes['true_positives'].append({'gt_id': gt_box_id, 'eval_id': eval_box_id})
				eval_bboxes.remove(eval_box_id)
			else:
				detectRes['false_negatives'].append({'gt_id': gt_box_id})
		if len(eval_bboxes)>0:
			detectRes['false_positives'].append({'eval_id': eval_box_id for eval_box_id in eval_bboxes})

	return detectRes


def evaluateAttribute(groundtruth, evaluation, resultDict, attributes):
	'''
	Input:
	groundtruth_Dict: dict, AnnFile format
	evalDict: dict, AnnFile format
	resultDict: dict, output from getDetections
	attributes : list of strings, attribute categories
	-----
	Output:

	'''
	assert 'utf8_string' not in attributes, 'there is a separate function for utf8_string'
	res = {}
	for attribute in attributes:
		correct = []
		incorrect = []
		for detection in resultDict['true_positives']:
			gt_val = groundtruth.anns[detection['gt_id']][attribute]
			eval_val = evaluation.anns[detection['eval_id']][attribute]
			if gt_val==eval_val:
				correct.append(detection)
			else:
				if gt_val!='na':
					incorrect.append(detection)
		res[attribute] = {'attribute': attribute, 'correct':len(correct), 'incorrect':len(incorrect), 'accuracy':len(correct)*1.0/len(correct+incorrect)}
	return res

def evaluateEndToEnd(groundtruth, evaluation, imgIds = [], annIds = [], detection_threshold = 0.5):
	"""
	A box is a match iff the intersection of union score is >= 0.5.
	Params
	------
	Input dicts have the format of annotation dictionaries
	"""
	#parameters

	detectRes = {}
	# results are lists of dicts {gt_id: xxx, eval_id: yyy}
	detectRes['true_positives'] = []
	detectRes['false_negatives'] = []
	detectRes['false_positives'] = []

	imgIds = imgIds if len(imgIds)>0 else inter(groundtruth.imgToAnns.keys(), evaluation.imgToAnns.keys())

	for cocoid in imgIds:
		gt_bboxes = groundtruth.imgToAnns[cocoid]
		eval_bboxes = copy.copy(evaluation.imgToAnns[cocoid])

		for gt_box_id in gt_bboxes:

			gt_box = groundtruth.anns[gt_box_id]['bbox']
			if 'utf8_string' not in groundtruth.anns[gt_box_id]:
				continue
			gt_val = decode(groundtruth.anns[gt_box_id]['utf8_string'])

			max_iou = detection_threshold

			match = None
			for eval_box_id in eval_bboxes:
				eval_box = evaluation.anns[eval_box_id]['bbox']
				iou = iou_score(gt_box,eval_box)

				if iou > max_iou:
					match = eval_box_id
					if 'utf8_string' in evaluation.anns[eval_box_id]:
						eval_val = decode(evaluation.anns[eval_box_id]['utf8_string'])
						if editdistance.eval(gt_val, eval_val)==0:
							break
			if match is not None:
				detectRes['true_positives'].append({'gt_id': gt_box_id, 'eval_id': eval_box_id})
				eval_bboxes.remove(eval_box_id)
			else:
				detectRes['false_negatives'].append({'gt_id': gt_box_id})
		if len(eval_bboxes)>0:
			detectRes['false_positives'].append({'eval_id': eval_box_id for eval_box_id in eval_bboxes})

	resultDict = detectRes

	res = {}
	for setting, threshold in zip(['exact', 'distance1'],[0,1]):
		correct = []
		incorrect = []
		ignore = []
		for detection in resultDict['true_positives']:
			if 'utf8_string' not in groundtruth.anns[detection['gt_id']]:
				ignore.append(detection)
				continue

			gt_val = decode(groundtruth.anns[detection['gt_id']]['utf8_string'])
			if len(gt_val)<3:
				ignore.append(detection)
				continue

			if 'utf8_string' not in evaluation.anns[detection['eval_id']]:
				incorrect.append(detection)
				continue

			eval_val = decode(evaluation.anns[detection['eval_id']]['utf8_string'])

			detection['gt_string'] = gt_val
			detection['eval_string'] = eval_val
			if editdistance.eval(gt_val, eval_val)<=threshold:
				correct.append(detection)
			else:
				incorrect.append(detection)

		res[setting] = {'setting': setting, 'correct':correct, 'incorrect':incorrect, 'ignore':ignore, 'accuracy':len(correct)*1.0/len(correct+incorrect)}
	return res

def area(bbox):
	return bbox[2] * 1.0 * bbox[3] # width * height

def intersect(bboxA, bboxB):
	"""Return a new bounding box that contains the intersection of
	'self' and 'other', or None if there is no intersection
	"""
	new_top = max(bboxA[1], bboxB[1])
	new_left = max(bboxA[0], bboxB[0])
	new_right = min(bboxA[0]+bboxA[2], bboxB[0]+bboxB[2])
	new_bottom = min(bboxA[1]+bboxA[3], bboxB[1]+bboxB[3])
	if new_top < new_bottom and new_left < new_right:
		return [new_left, new_top, new_right - new_left, new_bottom - new_top]
	return None

def iou_score(bboxA, bboxB):
	"""Returns the Intersection-over-Union score, defined as the area of
	the intersection divided by the intersection over the union of
	the two bounding boxes. This measure is symmetric.
	"""
	if intersect(bboxA, bboxB):
		intersection_area = area(intersect(bboxA, bboxB))
	else:
		intersection_area = 0
	union_area = area(bboxA) + area(bboxB) - intersection_area
	if union_area > 0:
		return float(intersection_area) / float(union_area)
	else:
		return 0

def decode(trans):
	trans = trans.encode("ascii" ,'ignore')
	trans = trans.replace('\n', ' ')
	trans2 = re.sub('[^a-zA-Z0-9!?@\_\-\+\*\:\&\/ \.]', '', trans)
	return trans2.lower()

def inter(list1, list2):
	return list(set(list1).intersection(set(list2)))

def printDetailedResults(c_text, detection_results, transcription_results, name):
	print name
	#detected coco-text annids
	found = [x['gt_id'] for x in detection_results['true_positives']]
	n_found = [x['gt_id'] for x in detection_results['false_negatives']]
	fp = [x['eval_id'] for x in detection_results['false_positives']]

	leg_eng_mp = c_text.getAnnIds(imgIds=[], catIds=[('legibility','legible'),('language','english'),('class','machine printed')], areaRng=[])
	leg_eng_hw = c_text.getAnnIds(imgIds=[], catIds=[('legibility','legible'),('language','english'),('class','handwritten')], areaRng=[])
	leg_mp  = c_text.getAnnIds(imgIds=[], catIds=[('legibility','legible'),('class','machine printed')], areaRng=[])
	ileg_mp = c_text.getAnnIds(imgIds=[], catIds=[('legibility','illegible'),('class','machine printed')], areaRng=[])
	leg_hw  = c_text.getAnnIds(imgIds=[], catIds=[('legibility','legible'),('class','handwritten')], areaRng=[])
	ileg_hw = c_text.getAnnIds(imgIds=[], catIds=[('legibility','illegible'),('class','handwritten')], areaRng=[])
	leg_ot  = c_text.getAnnIds(imgIds=[], catIds=[('legibility','legible'),('class','others')], areaRng=[])
	ileg_ot = c_text.getAnnIds(imgIds=[], catIds=[('legibility','illegible'),('class','others')], areaRng=[])

	#Detection 
	print 
	print "Detection"
	print "Recall"

	if (len(inter(found+n_found, leg_mp)))>0:
		lm = "%.2f"%(100*len(inter(found, leg_mp))*1.0/(len(inter(found+n_found, leg_mp))))
	else:
		lm = 0
	print 'legible & machine printed: ', lm 

	if (len(inter(found+n_found, leg_hw)))>0:
		lh = "%.2f"%(100*len(inter(found, leg_hw))*1.0/(len(inter(found+n_found, leg_hw))))
	else:
		lh = 0
	print 'legible & handwritten: ', lh 

	if (len(inter(found+n_found, leg_ot)))>0:
		lo = "%.2f"%(100*len(inter(found, leg_ot))*1.0/(len(inter(found+n_found, leg_ot))))
	else:
		lo = 0
	# print 'legible & others: ', lo

	if (len(inter(found+n_found, leg_mp+leg_hw)))>0:
		lto = "%.2f"%(100*len(inter(found, leg_mp+leg_hw))*1.0/(len(inter(found+n_found, leg_mp+leg_hw))))
	else:
		lto = 0
	print 'legible overall: ', lto  

	if (len(inter(found+n_found, ileg_mp)))>0:
		ilm = "%.2f"%(100*len(inter(found, ileg_mp))*1.0/(len(inter(found+n_found, ileg_mp))))
	else:
		ilm = 0
	print 'illegible & machine printed: ', ilm 

	if (len(inter(found+n_found, ileg_hw)))>0:
		ilh = "%.2f"%(100*len(inter(found, ileg_hw))*1.0/(len(inter(found+n_found, ileg_hw))))
	else:
		ilh = 0
	print 'illegible & handwritten: ', ilh 

	if (len(inter(found+n_found, ileg_ot)))>0:
		ilo = "%.2f"%(100*len(inter(found, ileg_ot))*1.0/(len(inter(found+n_found, ileg_ot))))
	else:
		ilo = 0
	# print 'illegible & others: ', ilo 

	if (len(inter(found+n_found, ileg_mp+ileg_hw)))>0:
		ilto = "%.2f"%(100*len(inter(found, ileg_mp+ileg_hw))*1.0/(len(inter(found+n_found, ileg_mp+ileg_hw))))
	else:
		ilto = 0
	print 'illegible overall: ', ilto 

	#total = "%.1f"%(100*len(found)*1.0/(len(found)+len(n_found)))
	t_recall = 100*len(found)*1.0/(len(inter(found+n_found, leg_mp+leg_hw+ileg_mp+ileg_hw)))
	total = "%.1f"%(t_recall)
	print 'total recall: ', total

	print "Precision"

	t_precision = 100*len(found)*1.0/(len(found+fp))
	precision = "%.2f"%(t_precision)
	print 'total precision: ', precision

	print "f-score"

	f_score = "%.2f"%(2 * t_recall * t_precision / (t_recall + t_precision)) if (t_recall + t_precision)>0 else 0
	print 'f-score localization: ', f_score

	print 
	print "Transcription"
	transAcc = "%.2f"%(100*transcription_results['exact']['accuracy'])
	transAcc1 = "%.2f"%(100*transcription_results['distance1']['accuracy'])
	print 'accuracy for exact matches: ', transAcc
	print 'accuracy for matches with edit distance<=1: ', transAcc1

	print
	print 'End-to-end'
	TP_new = len(inter(found, leg_eng_mp+leg_eng_hw)) * transcription_results['exact']['accuracy']
	FP_new = len(fp) + len(inter(found, leg_eng_mp+leg_eng_hw))*(1-transcription_results['exact']['accuracy'])
	FN_new = len(inter(n_found, leg_eng_mp+leg_eng_hw)) + len(inter(found, leg_eng_mp+leg_eng_hw))*(1-transcription_results['exact']['accuracy'])
	t_recall_new = 100 * TP_new / (TP_new + FN_new)
	t_precision_new = 100 * TP_new / (TP_new + FP_new) if (TP_new + FP_new)>0 else 0
	fscore = "%.2f"%(2 * t_recall_new * t_precision_new / (t_recall_new + t_precision_new)) if (t_recall_new + t_precision_new)>0 else 0

	recall_new = "%.2f"%(t_recall_new)
	precision_new = "%.2f"%(t_precision_new)
	print 'recall: ', recall_new, 
	print 'precision: ', precision_new
	print 'End-to-end f-score: ', fscore

	print
   	#print lm, ' & ', lh, ' & ', lto, ' & ', ilm,  ' & ', ilh, ' & ', ilto, '&', total,  ' & ', precision,  ' & ', transAcc,  ' & ', transAcc1, ' & ', fscore  	
   	print lm, ' & ', lh, ' & ', ilm,  ' & ', ilh, '&', total,  ' & ', precision,  ' & ', f_score,  ' & ', transAcc,  ' & ', recall_new,  ' & ', precision_new,  ' & ', fscore
   	print











