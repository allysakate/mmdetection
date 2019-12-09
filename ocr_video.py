'''
Usage:
python tools/test_track.py configs/faster_rcnn_r101_fpn_1x_mot.py checkpoints/faster_crnn_r101_fpn_1x_mot_50ep_101019-a3b5c112.pth  --out results/results_track.pkl
'''
import time
import os
import os.path as osp
import string

import mmcv
import torch
import torch.distributed as dist
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmcv.parallel import collate, scatter
from mmdet.apis import init_detector, show_tracks_result
from mmdet.datasets.pipelines import Compose
from mmdet.core import wrap_fp16_model
from mmdet.models import build_detector
from mmdet.models.tracktor import resnet50, Tracker, Tracker_Low, plot_sequence
from mmdet.models.ocr import Model, CTCLabelConverter, AttnLabelConverter
from tools import exec_time
import cv2
import pandas as pd
from collections import Counter

class LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
        else:
            results['filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        results['reid_img']  = img
        return results

def single_gpu_test(detect_model, video_name, show, skip, single, tracktor_cfg, test_cfg, 
                    ocr_model, cfg_ocr, ocr_converter, 
                    scale=(1333, 800), keep_ratio=True):
    output_dir = tracktor_cfg.output_dir 
    print(f'Output_DIR: {output_dir}')
    if not osp.exists(output_dir):
        os.makedirs(output_dir)  

    cfg_detect = detect_model.cfg
    device = next(detect_model.parameters()).device  # model device

    reid_load_start = time.time()
    reid_network = resnet50(pretrained=False, output_dim=128)
    reid_network.load_state_dict(torch.load(tracktor_cfg.reid_weights))
    reid_network.eval()
    reid_network.cuda()
    print(f'Load ReID: {exec_time.get_proctime(reid_load_start)}')

    cap = cv2.VideoCapture(video_name)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = int(cap.get(cv2.CAP_PROP_FPS))
    prog_bar = mmcv.ProgressBar(n_frames)
    out = cv2.VideoWriter('iou.avi',cv2.VideoWriter_fourcc('M','J','P','G'), int(30/skip), (1920,1080))
    print(f'CAP={n_frames, fps}')

    proc_frame_rate = int(30/skip)
    if proc_frame_rate > 24:
        tracker = Tracker(detect_model, reid_network, tracktor_cfg.tracker, test_cfg, single)
        print('IOU')
        time_csv = f'{output_dir}/iou_time_{skip}.csv'
        ocr_csv  = f'{output_dir}/iou_ocr_{skip}.csv'
        max_csv  = f'{output_dir}/iou_max_{skip}.csv'
        mot_txt  = f'{output_dir}/iou_mot_{skip}.txt'

    else:
        tracker = Tracker_Low(detect_model, reid_network, tracktor_cfg.tracker, test_cfg, single)
        print('ReId')
        time_csv = f'{output_dir}/reid_time_{skip}.csv'
        ocr_csv  = f'{output_dir}/reid_ocr_{skip}.csv'
        max_csv  = f'{output_dir}/reid_max_{skip}.csv'
        mot_txt  = f'{output_dir}/reid_mot_{skip}.txt'

    mot_result   = []
    track_result = []
    time_result  = []
    frame_cnt    = 0

    start_vid = time.time()
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if frame_cnt % skip == 0:

                # build the data pipeline
                test_pipeline = [LoadImage()] + cfg_detect.data.test.pipeline[1:]
                test_pipeline = Compose(test_pipeline)
                data = dict(img=frame)
                data = test_pipeline(data)
                data = scatter(collate([data], samples_per_gpu=1), [device])[0]
                
                tracker.step(data, frame, ocr_model, cfg_ocr, ocr_converter, frame_cnt)
                result, step_time = tracker.get_results()
                time_result.append(step_time)
                img, mot_result, track_result = show_tracks_result(frame, result, detect_model.CLASSES, frame_cnt, skip, mot_result, track_result)
                cv2.imshow('Tracking',cv2.resize(img,(1540,860)))
                out.write(img)
                #cv2.imwrite(f'{output_dir}/{frame_cnt}.jpg',img)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord('q') or ch == ord('Q'):
                    break
            frame_cnt += 1
        else:
            cap.release()
            break 
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    end_vid = exec_time.get_proctime(time.time()-start_vid)
    print(f'Run Vid: {end_vid}')

    track_column = ['framecnt', 'id', 'xmin', 'ymin', 'xmax', 'ymax', 'class', 'ocr']
    track_df = pd.DataFrame(track_result, columns=track_column)
    track_df.to_csv((ocr_csv), index=None)
    print('Track to CSV Done')

    time_column = ['detect_time', 'regress_time', 'ocr_time', 'track_time', 'plate_count', 'frame_cnt']
    time_df = pd.DataFrame(time_result, columns=time_column)
    time_df.to_csv((time_csv), index=None)
    print('Time to CSV Done')

    text_track = mot_txt
    print(text_track)
    if osp.exists(text_track):
        os.remove(text_track)
    with open(text_track,'a') as textfile:
        for res in mot_result:
            res = [str(r)  for r in res]
            str_res = ','.join(res)
            textfile.writelines(str_res + '\n')
    print('Mot Eval')

    ocr_max_list = []
    for t_id,t in result.items():
        ocr_list = []
        for i in t.keys():
            ocr_list.append(t[i][6])
        ocr_cnt = Counter(ocr_list)
        ocrstr = max(ocr_cnt,key=len)
        ocr_max = [t_id, ocrstr]
        ocr_max_list.append(ocr_max)
    max_column = ['ID', 'OCR']
    max_df = pd.DataFrame(ocr_max_list, columns=max_column)
    max_df.to_csv((max_csv), index=None)
    print('Max to CSV Done')

def main():
    
    #config = 'configs/ssd_vgg16_mot.py'
    config = 'configs/faster_rcnn_r101_fpn_1x_mot.py'

    cfg = mmcv.Config.fromfile(config)

    skip = cfg.skip
    show = cfg.show
    single = cfg.single
    checkpoint = cfg.checkpoint
    video_name = cfg.video_name
    
    #OCR Model configuration
    cfg_ocr = cfg.ocr
    if cfg_ocr.sensitive:
        cfg_ocr.character = string.printable[:-6]

    if 'CTC' in cfg_ocr.Prediction:
        ocr_converter = CTCLabelConverter(cfg_ocr.character)
    else:
        ocr_converter = AttnLabelConverter(cfg_ocr.character)
    cfg_ocr.num_class = len(ocr_converter.character)

    if cfg_ocr.rgb:
        cfg_ocr.input_channel = 3

    ocr_model_start = time.time()
    ocr_model = Model(cfg_ocr)
    print('model input parameters', cfg_ocr.imgH, cfg_ocr.imgW, cfg_ocr.num_fiducial, cfg_ocr.input_channel, cfg_ocr.output_channel,
          cfg_ocr.hidden_size, cfg_ocr.num_class, cfg_ocr.batch_max_length, cfg_ocr.Transformation, cfg_ocr.FeatureExtraction,
          cfg_ocr.SequenceModeling, cfg_ocr.Prediction)
    device='cuda:0'
    ocr_model = torch.nn.DataParallel(ocr_model).to(device)

    # Load OCR Model
    print('loading pretrained model from %s' % cfg_ocr.saved_model)
    ocr_model.load_state_dict(torch.load(cfg_ocr.saved_model, map_location=device))
    ocr_model.eval()
    print(f'Load OCR: {exec_time.get_proctime(ocr_model_start)}')

    detect_model_start = time.time()
    detect_model = init_detector(config, checkpoint, device=torch.device('cuda', 0))
    print(f'Load Detect: {exec_time.get_proctime(detect_model_start)}')

    single_gpu_test(detect_model, video_name, show, skip, 
                            single, tracktor_cfg=cfg.tracktor, test_cfg=cfg.test_cfg, 
                            ocr_model=ocr_model, cfg_ocr=cfg_ocr, ocr_converter=ocr_converter)

if __name__ == '__main__':
    main()