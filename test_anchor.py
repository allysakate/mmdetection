import torch
import numpy as np

class AnchorGenerator(object):

    def __init__(self, base_size, scales, ratios, scale_major=True, ctr=None):
        self.base_size = base_size
        self.scales = torch.Tensor(scales)
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.ctr = ctr
        self.base_anchors = self.gen_base_anchors()

    @property
    def num_base_anchors(self):
        return self.base_anchors.size(0)

    def gen_base_anchors(self):
        w = self.base_size
        h = self.base_size
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1)
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr

        h_ratios = torch.sqrt(self.ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * self.scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * self.scales[None, :]).view(-1)
        else:
            ws = (w * self.scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * self.scales[:, None] * h_ratios[None, :]).view(-1)

        # yapf: disable
        base_anchors = torch.stack(
            [
                x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
            ],
            dim=-1).round()
        # yapf: enable

        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_size, stride=16, device='cuda'):
        base_anchors = self.base_anchors.to(device)

        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride
        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        return all_anchors

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:, None].expand(
            valid.size(0), self.num_base_anchors).contiguous().view(-1)
        return valid
'''
Get Pos: Bbox=[tensor([[316.2104, 287.4065, 337.5739, 301.2364]], device='cuda:0')] 
 	 Label=[tensor([2], device='cuda:0')]
get_bbox: 7 
 size:torch.Size([1, 20, 64, 64]) 
 -2:torch.Size([64, 64])
'''

pos = torch.tensor([[316.2104, 287.4065, 337.5739, 301.2364]])
input_size=512
in_channels=[512, 1024, 512, 256, 256, 256, 256]
anchor_strides=[8, 16, 32, 64, 128, 256, 512]
basesize_ratio_range=[0.15, 0.9]

anchor_ratios=([2], [2, 3], [2, 3], [2, 3], [2, 3], [2], [2])

target_means=(.0, .0, .0, .0)
target_stds=(1.0, 1.0, 1.0, 1.0)
min_ratio, max_ratio = basesize_ratio_range
min_ratio = int(min_ratio * 100)
max_ratio = int(max_ratio * 100)
step = int(np.floor(max_ratio - min_ratio) / (len(in_channels) - 2))
min_sizes = []
max_sizes = []
for r in range(int(min_ratio), int(max_ratio) + 1, step):
    min_sizes.append(int(input_size * r / 100))
    max_sizes.append(int(input_size * (r + step) / 100))

if basesize_ratio_range[0] == 0.1:  # SSD512 COCO
    min_sizes.insert(0, int(input_size * 4 / 100))
    max_sizes.insert(0, int(input_size * 10 / 100))
elif basesize_ratio_range[0] == 0.15:  # SSD512 VOC
    min_sizes.insert(0, int(input_size * 7 / 100))
    max_sizes.insert(0, int(input_size * 15 / 100))
print(min_sizes,max_sizes)
pos = [element.item() for element in pos.flatten()]
anchor_generators = []
for k in range(len(anchor_strides)):
    base_size = min_sizes[k]
    stride = anchor_strides[k]
    ctr = ((pos[2] - pos[0]) / 2., (pos[3] - pos[1]) / 2.)

    ctr = ((stride - 1) / 2., (stride - 1) / 2.)
    scales = [1., np.sqrt(max_sizes[k] / min_sizes[k])]
    ratios = [1.]
    for r in anchor_ratios[k]:
        ratios += [1 / r, r]  # 4 or 6 ratio
    print(k,stride)
    print(base_size, scales, ratios, ctr)
    anchor_generator = AnchorGenerator(
        base_size, scales, ratios, scale_major=False, ctr=ctr)
    indices = list(range(len(ratios)))
    indices.insert(1, len(indices))
    anchor_generator.base_anchors = torch.index_select(
        anchor_generator.base_anchors, 0, torch.LongTensor(indices))
    anchor_generators.append(anchor_generator)

mlvl_anchors = [anchor_generators[i].grid_anchors(torch.tensor([64,64]),anchor_strides[i]) for i in range(7)]
print(f'mlvl:{mlvl_anchors, len(mlvl_anchors)}')