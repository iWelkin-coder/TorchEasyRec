for i in {1..1}; do
    rm -f graphlearn.iZ*
    rm -f graphlearn.INFO
    rm -f graphlearn.WARNING
    # rm -rf experiments/hstu_ml_20m_match

    timestamp=$(date +"%Y-%m-%d-%H-%M")
    # --model_dir "experiments/hstu_ml_20m_match_${timestamp}" \

    OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1,3 ODPS_CONFIG_FILE_PATH=odps_conf \
    torchrun --master_addr=localhost --master_port=32556 \
    --nnodes=1 --nproc-per-node=2 --node_rank=0 \
    -m tzrec.train_eval \
    --pipeline_config_path "hstu_ml_20m_match.config" \
    --model_dir "experiments/hstu_ml_20m_match_${timestamp}" \
    --continue_train false
done
