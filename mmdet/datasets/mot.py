import os
import pickle

from .registry import DATASETS
#from .xml_style import XMLDataset
from .custom import CustomDataset

@DATASETS.register_module
class MOTDataset(CustomDataset):

    def load_annotations(self, ann_file, ann_dict):
        img_infos = []

        anno_name, ext =  os.path.splitext(ann_file)
        if ext == '.pkl':
            annos = pickle.load(open(ann_file, 'rb'))
        else:
            print('This only works if annotation file is .pkl')
        
        for anno in annos:
            filename = anno['filename']
            vidname = anno['videoname']
            frameid = int(anno['frameid'])
            width = int(anno['width'])
            height = int(anno['height'])
            img_infos.append(dict(id=frameid, filename=filename, width=width, height=height))
            ann_dict[frameid] = anno['ann']
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann =  self.ann_dict[img_id]
        return ann