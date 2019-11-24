import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from .dataset import image_transform
from .model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import time

def get_text(plate_img, ocr_model, cfg_ocr, ocr_converter):
    ocr_time = time.time()
    image_tensor = image_transform(plate_img, imgH=cfg_ocr.imgH, imgW=cfg_ocr.imgW, keep_ratio_with_pad=cfg_ocr.PAD)

    # predict
    ocr_model.eval()
    with torch.no_grad():
        batch_size = image_tensor.size(0)
        image = image_tensor.to(device)
        # For max length prediction
        length_for_pred = torch.IntTensor([cfg_ocr.batch_max_length] * batch_size).to(device)
        text_for_pred = torch.LongTensor(batch_size, cfg_ocr.batch_max_length + 1).fill_(0).to(device)

        if 'CTC' in cfg_ocr.Prediction:
            preds = ocr_model(image, text_for_pred).log_softmax(2)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.max(2)
            preds_index = preds_index.view(-1)
            preds_str = ocr_converter.decode(preds_index.data, preds_size.data)
            preds_str = preds_str[0]
        else:
            preds = ocr_model(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = ocr_converter.decode(preds_index, length_for_pred)

        preds_prob = F.softmax(preds, dim=2)
        pred_max_prob, _ = preds_prob.max(dim=2)

        if 'Attn' in cfg_ocr.Prediction:
            pred_EOS = preds_str[0].find('[s]')
            preds_str = preds_str[0][:pred_EOS]  # prune after "end of sentence" token ([s])
            pred_max_prob = pred_max_prob[:pred_EOS]

        # calculate confidence score (= multiply of pred_max_prob)
        confidence_score = pred_max_prob.cumprod(dim=0)[-1]
        #print(preds_str, confidence_score)
        return preds_str, ocr_time