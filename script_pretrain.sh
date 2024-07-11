export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=4,5,6,7

# NTU60 xsub
python torch.distributed.launch --nproc_per_node=4 --master_port 11234 main_pretrain.py \
--config ./config/ntu60/pretrain_svdata2vec.yaml \
--output_dir ./output_dir/ntu60/pretrain_svdata2vec_mask_0.7_0.9_xsub \
--log_dir ./output_dir/ntu60/pretrain_svdata2vec_mask_0.7_0.9_xsub