import torch
import torch.nn as nn

from mmdet import ops
from ..registry import RECOG_HEADS
from ..registry import ROI_EXTRACTORS


def totensor(data, cuda=True):
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    if isinstance(data, torch.Tensor):
        tensor = data.detach()
    if cuda:
        tensor = tensor.cuda()
    return tensor

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output

@RECOG_HEADS.register_module
class CRNN(nn.Module):

    def __init__(self, roi_layer, feat_strides, abc_len, rnn_hid_size, rnn_n_layer):
        super(CRNN, self).__init__()
        self.roi_layer = roi_layer
        self.feat_strides =    feat_strides    
        self.cnn = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 1)),
                nn.Conv2d(256, 256, kernel_size=(2, 2), stride=(1, 1)),nn.ReLU(inplace=True))
        self.rnn = nn.Sequential(
            BidirectionalLSTM(256, rnn_hid_size, rnn_hid_size),
            BidirectionalLSTM(rnn_hid_size, rnn_hid_size, abc_len))
        
        self.linear = nn.Linear(rnn_hid_size * 2, abc_len + 1)
        self.softmax = nn.Softmax(dim=2)

    def build_roi_layers(self, layer_cfg, feat_strides):
        cfg = layer_cfg.copy()
        layer_type = cfg.pop('type')
        assert hasattr(ops, layer_type)
        layer_cls = getattr(ops, layer_type)
        roi_layers = nn.ModuleList(
            [layer_cls(spatial_scale=1 / s, **cfg) for s in feat_strides])
        return roi_layers

    def forward(self, x, rois):
        roi_layers = self.build_roi_layers(self.roi_layer,self.feat_strides)
        pool = roi_layers[0](x[0], rois)
        conv = self.cnn(pool)
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]


        # rnn features
        output = self.rnn(conv)
        #output = self.linear(output)             
        output = self.softmax(output)
        print(output.size())
        return seq

    def get_target(self, gt_texts,
                   rcnn_train_cfg):
        pos_proposals = [res.pos_bboxes for res in sampling_results]
        neg_proposals = [res.neg_bboxes for res in sampling_results]
        pos_gt_bboxes = [res.pos_gt_bboxes for res in sampling_results]
        pos_gt_labels = [res.pos_gt_labels for res in sampling_results]
        reg_classes = 1 if self.reg_class_agnostic else self.num_classes
        cls_reg_targets = bbox_target(
            pos_proposals,
            neg_proposals,
            pos_gt_bboxes,
            pos_gt_labels,
            rcnn_train_cfg,
            reg_classes,
            target_means=self.target_means,
            target_stds=self.target_stds)
        return cls_reg_targets