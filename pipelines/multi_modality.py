import numpy as np
from torch.nn.modules.utils import _pair

from .loading import DecordDecode, DecordInit
from .sampling import UniformSampleFrames
from .pose_related import PoseDecode


from .builder import PIPELINES
import matplotlib.pyplot as plt
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights, SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large
from torchvision.transforms import ToPILImage
import os.path as osp
import pickle

@PIPELINES.register_module()
class MMUniformSampleFrames(UniformSampleFrames):

    def __call__(self, results):
        num_frames = results['total_frames']
        modalities = []
        for modality, clip_len in self.clip_len.items():
            if results['test_mode']:
                inds = self._get_test_clips(num_frames, clip_len)
            else:
                inds = self._get_train_clips(num_frames, clip_len)
            inds = np.mod(inds, num_frames)
            results[f'{modality}_inds'] = inds.astype(int)
            modalities.append(modality)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = None
        results['num_clips'] = self.num_clips
        if not isinstance(results['modality'], list):
            # should override
            results['modality'] = modalities
        return results


@PIPELINES.register_module()
class MMDecode(DecordInit, DecordDecode, PoseDecode):
    def __init__(self, io_backend='disk', **kwargs):
        DecordInit.__init__(self, io_backend=io_backend, **kwargs)
        DecordDecode.__init__(self)
        self.io_backend = io_backend
        self.kwargs = kwargs
        self.file_client = None

    def __call__(self, results):
        for mod in results['modality']:
            if results[f'{mod}_inds'].ndim != 1:
                results[f'{mod}_inds'] = np.squeeze(results[f'{mod}_inds'])
            frame_inds = results[f'{mod}_inds']
            if mod == 'RGB':
                if 'filename' not in results:
                    results['filename'] = results['frame_dir'] + '.mp4'
                video_reader = self._get_videoreader(results['filename'])
                imgs = self._decord_load_frames(video_reader, frame_inds)
                del video_reader
                results['imgs'] = imgs
            elif mod == 'Pose':
                #if 'keypoint' not in results and 'raw_file' in results:
                #    skele= osp.join(self.skeleton_data_prefix, results['frame_dir'].split("/")[-1])+".pkl"
                #    with open(skele, 'rb') as f:
                #        kps = pickle.load(f)
                #    results['keypoint']=np.expand_dims(kps['keypoint'][:,:,:2],axis=0)
                
                assert 'keypoint' in results
                if 'keypoint_score' not in results:
                    keypoint_score = [
                        np.ones(keypoint.shape[:-1], dtype=np.float32)
                        for keypoint in results['keypoint']
                    ]
                    results['keypoint_score'] = keypoint_score
                
                results['keypoint'] = self._load_kp(results['keypoint'], frame_inds)
                #results['keypoint_score'] = self._load_kpscore(results['keypoint_score'], frame_inds)
            else:
                raise NotImplementedError(f'MMDecode: Modality {mod} not supported')

        # We need to scale human keypoints to the new image size
        if 'imgs' in results:
            real_img_shape = results['imgs'][0].shape[:2]
            if real_img_shape != results['img_shape']:
                oh, ow = results['img_shape']
                nh, nw = real_img_shape

                assert results['keypoint'].shape[-1] in [2, 3]
                results['keypoint'][..., 0] *= (nw / ow)
                results['keypoint'][..., 1] *= (nh / oh)

                results['img_shape'] = real_img_shape
                results['original_shape'] = real_img_shape
        
        return results


@PIPELINES.register_module()
class MMCompact:

    def __init__(self, padding=0.25, threshold=10, hw_ratio=1, allow_imgpad=True):

        self.padding = padding
        self.threshold = threshold
        if hw_ratio is not None:
            hw_ratio = _pair(hw_ratio)
        self.hw_ratio = hw_ratio
        self.allow_imgpad = allow_imgpad
        assert self.padding >= 0

    
    def _get_box_bb(self,bboxes, img_shape):

        h, w = img_shape
        global_min_x, global_min_y, global_max_x, global_max_y = float('inf'), float('inf'), float('-inf'), float('-inf')

        for bb in bboxes:
            top_people_boxes = [bb]

            # Update global min and max coordinates
            for box in top_people_boxes:
                min_x, min_y, max_x, max_y, _ = box
                if(min_x!=0):
                    global_min_x = min(global_min_x, min_x)
                if(min_y!=0):
                    global_min_y = min(global_min_y, min_y)
                if(max_x!=0):
                    global_max_x = max(global_max_x, max_x)
                if(max_y!=0):
                    global_max_y = max(global_max_y, max_y)

        min_x,max_x, min_y, max_y= global_min_x, global_max_x, global_min_y, global_max_y

        # The compact area is too small
        if max_x - min_x < self.threshold or max_y - min_y < self.threshold or (min_x==float('inf') or max_x==float('-inf') or min_y==float('inf') or max_y==float('-inf')):
            return (0, 0, w, h)

        center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
        half_width = (max_x - min_x) / 2 * (1 + self.padding)
        half_height = (max_y - min_y) / 2 * (1 + self.padding)

        if self.hw_ratio is not None:
            half_height = max(self.hw_ratio[0] * half_width, half_height)
            half_width = max(1 / self.hw_ratio[1] * half_height, half_width)

        min_x, max_x = center[0] - half_width, center[0] + half_width
        min_y, max_y = center[1] - half_height, center[1] + half_height

        # hot update
        if not self.allow_imgpad:
            min_x, min_y = int(max(0, min_x)), int(max(0, min_y))
            max_x, max_y = int(min(w, max_x)), int(min(h, max_y))
        else:
            min_x, min_y = int(min_x), int(min_y)
            max_x, max_y = int(max_x), int(max_y)
        return (min_x, min_y, max_x, max_y)
       
    def _get_box(self, keypoint, img_shape):
        # will return x1, y1, x2, y2
        h, w = img_shape

        kp_x = keypoint[..., 0]
        kp_y = keypoint[..., 1]

        min_x = np.min(kp_x[kp_x != 0], initial=np.Inf)
        min_y = np.min(kp_y[kp_y != 0], initial=np.Inf)
        max_x = np.max(kp_x[kp_x != 0], initial=-np.Inf)
        max_y = np.max(kp_y[kp_y != 0], initial=-np.Inf)

        # The compact area is too small
        if max_x - min_x < self.threshold or max_y - min_y < self.threshold:
            return (0, 0, w, h)

        center = ((max_x + min_x) / 2, (max_y + min_y) / 2)
        half_width = (max_x - min_x) / 2 * (1 + self.padding)
        half_height = (max_y - min_y) / 2 * (1 + self.padding)

        if self.hw_ratio is not None:
            half_height = max(self.hw_ratio[0] * half_width, half_height)
            half_width = max(1 / self.hw_ratio[1] * half_height, half_width)

        min_x, max_x = center[0] - half_width, center[0] + half_width
        min_y, max_y = center[1] - half_height, center[1] + half_height

        # hot update
        if not self.allow_imgpad:
            min_x, min_y = int(max(0, min_x)), int(max(0, min_y))
            max_x, max_y = int(min(w, max_x)), int(min(h, max_y))
        else:
            min_x, min_y = int(min_x), int(min_y)
            max_x, max_y = int(max_x), int(max_y)
    
        return (min_x, min_y, max_x, max_y)

    def _compact_images(self, imgs, img_shape, box):
        h, w = img_shape
        min_x, min_y, max_x, max_y = box
        pad_l, pad_u, pad_r, pad_d = 0, 0, 0, 0
        if min_x < 0:
            pad_l = -min_x
            min_x, max_x = 0, max_x + pad_l
            w += pad_l
        if min_y < 0:
            pad_u = -min_y
            min_y, max_y = 0, max_y + pad_u
            h += pad_u
        if max_x > w:
            pad_r = max_x - w
            w = max_x
        if max_y > h:
            pad_d = max_y - h
            h = max_y

        if pad_l > 0 or pad_r > 0 or pad_u > 0 or pad_d > 0:
            imgs = [
                np.pad(img, ((pad_u, pad_d), (pad_l, pad_r), (0, 0))) for img in imgs
            ]
        imgs = [img[min_y: max_y, min_x: max_x] for img in imgs]
        return imgs

    def __call__(self, results):
        img_shape = results['img_shape']
        h, w = img_shape
        kp = results['keypoint']
        # Make NaN zero
        kp[np.isnan(kp)] = 0.

        if(results['multimodal']):
            min_x, min_y, max_x, max_y = self._get_box(kp, img_shape) #if we use skeleton extract bboxes from skeleton
        else:
            min_x, min_y, max_x, max_y = self._get_box_bb(results['bboxes'], img_shape) #else use preestimated bboxes

        kp_x, kp_y = kp[..., 0], kp[..., 1]
        kp_x[kp_x != 0] -= min_x
        kp_y[kp_y != 0] -= min_y

        new_shape = (max_y - min_y, max_x - min_x)
        results['img_shape'] = new_shape        
        results['imgs'] = self._compact_images(results['imgs'], img_shape, (min_x, min_y, max_x, max_y))

        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}(padding={self.padding}, '
                    f'threshold={self.threshold}, '
                    f'hw_ratio={self.hw_ratio}, '
                    f'allow_imgpad={self.allow_imgpad})')
        return repr_str