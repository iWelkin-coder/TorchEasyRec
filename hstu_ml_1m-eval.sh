# if [ -d "experiments/hstu_ml_1m_match_2025-03-28-16-03/export" ]; then
#     rm -rf experiments/hstu_ml_1m_match_2025-03-28-16-03/export
# fi
CUDA_VISIBLE_DEVICES=0 QUANT_EMB=0 torchrun --master_addr=localhost --master_port=32557 \
    --nnodes=1 --nproc-per-node=1 --node_rank=0 \
    -m tzrec.export \
    --pipeline_config_path experiments/hstu_ml_1m_match_2025-03-28-16-03/pipeline.config \
    --export_dir experiments/hstu_ml_1m_match_2025-03-28-16-03/export

CUDA_VISIBLE_DEVICES=0 torchrun --master_addr=localhost --master_port=32557 \
    --nnodes=1 --nproc-per-node=1 --node_rank=0 \
    -m tzrec.predict \
    --scripted_model_path experiments/hstu_ml_1m_match_2025-03-28-16-03/export/item \
    --predict_input_path "odps://pai_test_xjp/tables/ML_1M_Eval_ITEM_Flatten" \
    --predict_output_path 'odps://pai_test_xjp/tables/ml_1m_itemid_emb_flatten' \
    --reserved_columns item_id \
    --output_columns item_tower_emb

CUDA_VISIBLE_DEVICES=0 torchrun --master_addr=localhost --master_port=32557 \
    --nnodes=1 --nproc-per-node=1 --node_rank=0 \
    -m tzrec.predict \
    --scripted_model_path experiments/hstu_ml_1m_match_2025-03-28-16-03/export/user \
    --predict_input_path "odps://pai_test_xjp/tables/ml_1m_eval_flatten" \
    --predict_output_path "odps://pai_test_xjp/tables/ml_1m_userid_emb_flatten" \
    --reserved_columns user_id,item_id \
    --output_columns user_tower_emb

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=16 torchrun --master_addr=localhost --master_port=32557 \
    --nnodes=1 --nproc-per-node=1 --node_rank=0 \
    -m tzrec.tools.hitrate \
    --user_gt_input 'odps://pai_test_xjp/tables/ml_1m_userid_emb_flatten' \
    --item_embedding_input 'odps://pai_test_xjp/tables/ml_1m_itemid_emb_flatten' \
    --item_id_field item_id --request_id_field user_id \
    --gt_items_field item_id --odps_data_quota_name 's' \
    --hitrate_details_output "odps://pai_test_xjp/tables/ml_1m_hitrate_detail" \
    --total_hitrate_output "odps://pai_test_xjp/tables/ml_1m_hitrate_total_output"\
    --top_k 1 --ivf_nprobe 1000
