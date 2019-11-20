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
from tools.exec_time import Timer
import cv2

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
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    
    cfg_detect = detect_model.cfg
    device = next(detect_model.parameters()).device  # model device

    results = []
    
    reid_network = resnet50(pretrained=False, output_dim=128)
    reid_network.load_state_dict(torch.load(tracktor_cfg.reid_weights))
    reid_network.eval()
    reid_network.cuda()


    cap = cv2.VideoCapture(video_name)

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps      = int(cap.get(cv2.CAP_PROP_FPS))
    prog_bar = mmcv.ProgressBar(n_frames)

    print(f'CAP={n_frames, fps}')

    proc_frame_rate = int(30/skip)
    # if proc_frame_rate > 24:
    #     tracker = Tracker(detect_model, reid_network, tracktor_cfg.tracker, test_cfg, single)
    #     print('IOU')
    # else:
    tracker = Tracker_Low(detect_model, reid_network, tracktor_cfg.tracker, test_cfg, single)
    print('ReId')

    frame_cnt = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == True:
            if frame_cnt % skip == 0:

                # build the data pipeline
                test_pipeline = [LoadImage()] + cfg_detect.data.test.pipeline[1:]
                test_pipeline = Compose(test_pipeline)
                # prepare data
                data = dict(img=frame)
                data = test_pipeline(data)
                data = scatter(collate([data], samples_per_gpu=1), [device])[0]
                
                tracker.step(data, frame, ocr_model, cfg_ocr, ocr_converter)
                result = tracker.get_results()
                img = show_tracks_result(frame, result, detect_model.CLASSES, frame_cnt)
                cv2.imshow('Tracking',cv2.resize(img,(1540,860)))
                cv2.imwrite(f'{output_dir}/{frame_cnt}.jpg',img)
                ch = cv2.waitKey(1)
                if ch == 27 or ch == ord('q') or ch == ord('Q'):
                    break

            frame_cnt += 1
    cap.release()
    cv2.destroyAllWindows()

def main():
    skip = 1
    show = False
    single = False
    checkpoint = 'checkpoints/faster_crnn_r101_fpn_1x_mot_50ep_101019-a3b5c112.pth'
    video_name = '/media/allysakatebrillantes/MyPassport/DATASET/catchall-dataset/cvat/test/videos/ch01_07-12_10.35.mp4'
    start_time = time.time()
    config = 'configs/faster_rcnn_r101_fpn_1x_mot.py'
    
    cfg = mmcv.Config.fromfile(config)

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
    print()
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


    detect_model = init_detector(config, checkpoint, device=torch.device('cuda', 0))
    outputs = single_gpu_test(detect_model, video_name, show, skip, 
                            single, tracktor_cfg=cfg.tracktor, test_cfg=cfg.test_cfg, 
                            ocr_model=ocr_model, cfg_ocr=cfg_ocr, ocr_converter=ocr_converter)

    print(time.time()-start_time)

if __name__ == '__main__':
    with Timer("Time for Tracking"):
        main()
