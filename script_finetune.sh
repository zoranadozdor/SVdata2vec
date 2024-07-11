export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=6,7

# NTU-60 xsub finetune rgb only
python torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_finetune.py \
--config ./config/ntu60/finetune_svdata2vec_rgb_only.yaml \
--output_dir ./output_dir/ntu60/finetune_svdata2vec_rgb_only_xsub \
--log_dir ./output_dir/ntu60/finetune_svdata2vec_rgb_only_xsub \
--finetune ./output_dir/ntu60/pretrain_svdata2vec_mask_0.7_0.9_xsub/checkpoint-399.pth \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5

# NTU-60 xsub finetune both modalities
python torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_finetune.py \
--config ./config/ntu60/finetune_svdata2vec.yaml \
--output_dir ./output_dir/ntu60/finetune_svdata2vec_xsub \
--log_dir ./output_dir/ntu60/finetune_svdata2vec_xsub \
--finetune ./output_dir/ntu60/pretrain_svdata2vec_mask_0.7_0.9_xsub/checkpoint-399.pth \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5


