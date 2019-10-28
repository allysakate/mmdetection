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

class BiRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, num_classes):
        super(BiRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=False, dropout=dropout, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)  # 2 for bidirection
    
    def forward(self, x):
        # Set initial states
        h0 = torch.zeros(self.num_layers*2, x.size(1), self.hidden_size).cuda() # 2 for bidirection 
        c0 = torch.zeros(self.num_layers*2, x.size(1), self.hidden_size).cuda()
        print(f'init_state: {h0.size()} | {c0.size()}') #([4, 19, 256])
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_size*2)
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out

class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.fc = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.fc(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output


@RECOG_HEADS.register_module
class CRNN(nn.Module):

    def __init__(self, abc, rnn_hid_size, rnn_n_layer, rnn_dropout, decode):
        super(CRNN, self).__init__()
        self.abc = abc
        self.num_classes = len(self.abc)
        self.decode = decode
        
        self.cnn = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)), nn.ReLU(inplace=True),
                nn.MaxPool2d((2, 1)),
                nn.Conv2d(256, 256, kernel_size=(2, 2), stride=(1, 1)),nn.ReLU(inplace=True))
        
        self.rnn_hidden_size = rnn_hid_size
        self.rnn_num_layers = rnn_n_layer
        # self.rnn = nn.GRU(256,
        #                   rnn_hid_size, rnn_n_layer,
        #                   batch_first=False,
        #                   dropout=rnn_dropout, bidirectional=True)
        # self.rnn = nn.Sequential(
        #     BidirectionalLSTM(256, rnn_hid_size, rnn_hid_size),
        #     BidirectionalLSTM(rnn_hid_size, rnn_hid_size, self.num_classes))
        self.rnn = BiRNN(256, rnn_hid_size, rnn_n_layer, rnn_dropout, self.num_classes)
        self.linear = nn.Linear(rnn_hid_size * 2, self.num_classes + 1)
        self.softmax = nn.Softmax(dim=2)

    def forward(self, x, rois):
        conv = self.cnn(rois)
        print(f'conv: {conv.size()} | rois: {rois.size()}')
        features = self.features_to_sequence(conv)
        print(f'feat_seq: {features.size()}')

        hidden = self.init_hidden(rois.size(0), next(self.parameters()).is_cuda)
        #print(f'hid: {hidden.size()}| hid {hidden.size()}')
        #seq, hidden = self.rnn(features, hidden)
        seq = self.rnn(features)

        #print(f'rnn: {seq.size()}| hid {hidden.size()}')
        seq = self.linear(seq)
        #print(f'lin: {seq.size()}')
        seq = self.softmax(seq)
        #print(f'soft: {seq.size()}')
        if self.decode:
            seq = self.decode(seq)
            #print(f'decode: {seq.size()}')
        return seq

    def init_hidden(self, batch_size, gpu=False):
        h0 = torch.zeros( self.rnn_num_layers * 2,
                                   batch_size,
                                   self.rnn_hidden_size)
        if gpu:
            h0 = h0.cuda()
        return h0

    def features_to_sequence(self, features):
        b, c, h, w = features.size() #([4, 512, 1, 10])
        assert h == 1, "the height of out must be 1"
        # if not self.fully_conv:
        #     features = features.permute(0, 3, 2, 1) #([4, 10, 1, 512])
        #     print(f'feat: {features.size()}') #([1024, 19, 1, 256])
        #     features = self.proj(features) #([4, 20, 1, 512])
        #     print(f'feat: {features.size()}')
        #     features = features.permute(1, 0, 2, 3) #([20, 4, 1, 512])
        # else:
        features = features.permute(3, 0, 2, 1)
        features = features.squeeze(2)  #([20, 4, 512])
        return features

    def pred_to_string(self, pred):
        seq = []
        for i in range(pred.shape[0]):
            label = np.argmax(pred[i])
            seq.append(label - 1)
        out = []
        for i in range(len(seq)):
            if len(out) == 0:
                if seq[i] != -1:
                    out.append(seq[i])
            else:
                if seq[i] != -1 and seq[i] != seq[i - 1]:
                    out.append(seq[i])
        out = ''.join(self.abc[i] for i in out)
        return out

    def decode(self, pred):
        pred = pred.permute(1, 0, 2).cpu().data.numpy()
        seq = []
        for i in range(pred.shape[0]):
            seq.append(self.pred_to_string(pred[i]))
        return seq