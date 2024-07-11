import numpy as np
from torch.utils.data import Dataset
import mmcv
import os.path as osp
import torch
from pipelines import Compose
import copy


class Feeder(Dataset):
    def __init__(self, ann_file='', data_prefix='', split='', test_mode=False, num_classes=None, 
                 start_index=1, modality="RGB", pipeline=None, multimodal=False, memcached=False, dataset="ntu"):
    

        self.data_prefix = data_prefix
        self.ann_file = ann_file
        self.split = split
        self.test_mode = test_mode
        self.num_classes = num_classes
        self.start_index = start_index
        self.modality = modality
        self.cli = None
        self.memcached = memcached
        self.multimodal=multimodal
        self.dataset=dataset

        self.pipeline = Compose(pipeline)


        self.load_data()


    def load_data(self):
        
        data = mmcv.load(self.ann_file)

        if self.split:
            split, data = data['split'], data['annotations']
            identifier = 'filename' if 'filename' in data[0] else 'frame_dir'
            split = set(split[self.split])
            data = [x for x in data if x[identifier] in split]
            
        for item in data:
            # Sometimes we may need to load anno from the file
            if 'filename' in item:
                item['filename'] = osp.join(self.data_prefix, item['filename'])
            if 'frame_dir' in item:
                item['frame_dir'] = osp.join(self.data_prefix, item['frame_dir'])

        self.data = data
        print(len(self.data))

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def prepare_train_frames(self, idx):
        """Prepare the frames for training given the index."""
        results = copy.deepcopy(self.data[idx])
        if self.memcached and 'key' in results:
            from pymemcache import serde
            from pymemcache.client.base import Client

            if self.cli is None:
                self.cli = Client(self.mc_cfg, serde=serde.pickle_serde)
            key = results.pop('key')
            try:
                pack = self.cli.get(key)
            except:
                self.cli = Client(self.mc_cfg, serde=serde.pickle_serde)
                pack = self.cli.get(key)
            if not isinstance(pack, dict):
                raw_file = results['raw_file']
                data = mmcv.load(raw_file)
                pack = data[key]
                for k in data:
                    try:
                        self.cli.set(k, data[k])
                    except:
                        self.cli = Client(self.mc_cfg, serde=serde.pickle_serde)
                        self.cli.set(k, data[k])
            for k in pack:
                results[k] = pack[k]

        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results['test_mode'] = self.test_mode
        if(self.dataset=="ntu"):
            results['img_shape'] = (1080, 1920)
        if(self.multimodal==True): 
            results['multimodal'] = True
        else:
            results['multimodal'] = False

        return self.pipeline(results)

    def prepare_test_frames(self, idx):
        """Prepare the frames for testing given the index."""
        results = copy.deepcopy(self.data[idx])
        if self.memcached and 'key' in results:
            from pymemcache import serde
            from pymemcache.client.base import Client

            if self.cli is None:
                self.cli = Client(self.mc_cfg, serde=serde.pickle_serde)
            key = results.pop('key')
            try:
                pack = self.cli.get(key)
            except:
                self.cli = Client(self.mc_cfg, serde=serde.pickle_serde)
                pack = self.cli.get(key)
            if not isinstance(pack, dict):
                raw_file = results['raw_file']
                data = mmcv.load(raw_file)
                pack = data[key]
                for k in data:
                    try:
                        self.cli.set(k, data[k])
                    except:
                        self.cli = Client(self.mc_cfg, serde=serde.pickle_serde)
                        self.cli.set(k, data[k])
            for k in pack:
                results[k] = pack[k]

        results['modality'] = self.modality
        results['start_index'] = self.start_index
        results['test_mode'] = self.test_mode
        if(self.dataset=="ntu"):
            results['img_shape'] = (1080, 1920)
        if(self.multimodal==True): 
            results['multimodal'] = True
        else:
            results['multimodal'] = False

        return self.pipeline(results)

    def __getitem__(self, index):
       
        s=self.prepare_test_frames(index) if self.test_mode else self.prepare_train_frames(index)
       
        if(self.multimodal):

            if(s['keypoint'].shape[0]==1):
                kps_new=torch.zeros((2, s['keypoint'].shape[1],s['keypoint'].shape[2],s['keypoint'].shape[3]))
                kps_new[0,:,:,:]=s['keypoint']
                return (s['imgs'],kps_new, s['label'])

            elif(s['keypoint'].shape[0]>2):
                kps_new=torch.zeros((2, s['keypoint'].shape[1],s['keypoint'].shape[2],s['keypoint'].shape[3]))
                kps_new=s['keypoint'][:2,:,:,:]
                return (s['imgs'],kps_new, s['label'])
            else:
                
                return (s['imgs'],s['keypoint'], s['label'])
        
        else:
            return ((s['imgs'], 0, s['label']))
