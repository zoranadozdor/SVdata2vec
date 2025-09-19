import multiprocessing as mp
import os
import subprocess
import os.path as osp


def ls(dirname='.', full=True, match=''):
    if not full or dirname == '.':
        ans = os.listdir(dirname)
    ans = [osp.join(dirname, x) for x in os.listdir(dirname)]
    ans = [x for x in ans if match in x]
    return ans

def get_shape(vid):
    cmd = 'ffprobe -v error -select_streams v:0 -show_entries stream=width,height -of csv=s=x:p=0 \"{}\"'.format(vid)
    w, h = subprocess.check_output(cmd, shell=True).decode('utf-8').split('x')
    return int(w), int(h)


def compress(src, dest, shape=None, target_size=540, fps=-1):
    if shape is None:
        shape = get_shape(src)
    w, h = shape
    scale_str = f'-vf scale=-2:{target_size}' if w >= h else f'-vf scale={target_size}:-2'
    fps_str = f'-r {fps}' if fps > 0 else ''
    quality_str = '-q:v 1'
    vcodec_str = '-c:v libx264'
    cmd = f'ffmpeg -y -loglevel error -i {src} -threads 1 {quality_str} {scale_str} {fps_str} {vcodec_str} {dest}'
    os.system(cmd)


def compress_nturgbd(name):
    src = name
    dest = src.replace('nturgb+d_rgb', 'nturgb+d_videos_c').replace('_rgb.avi', '.mp4')
    shape = (1920, 1080)
    compress(src, dest, shape)

os.makedirs('SVDATA2VEC/data/ntu/RGB_videos/nturgb+d_videos_c', exist_ok=True)
files = ls('SVDATA2VEC/data/ntu/RGB_videos/nturgb+d_rgb', match='.avi')
pool = mp.Pool(32)
pool.map(compress_nturgbd, files)