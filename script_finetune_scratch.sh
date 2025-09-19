export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=2,3

# NTU-60 xsub finetune scratch rgb only
python torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_finetune.py \
--config ./config/ntu60/finetune_svdata2vec_rgb_only.yaml \
--output_dir ./output_dir/ntu60/finetune_scratch_rgb_only_xsub \
--log_dir ./output_dir/ntu60/finetune_scratch_rgb_only_xsub \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5 \
--layer_decay 1.0

# NTU-60 xsub finetune scratch rgb only
python torch.distributed.launch --nproc_per_node=2 --master_port 12345 main_finetune.py \
--config ./config/ntu60/finetune_svdata2vec.yaml \
--output_dir ./output_dir/ntu60/finetune_scratch_xsub \
--log_dir ./output_dir/ntu60/finetune_scratch_xsub \
--dist_eval \
--lr 3e-4 \
--min_lr 1e-5 \
--layer_decay 1.0

