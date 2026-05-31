#!/bin/bash

MODEL_PATH=Qwen/Qwen2.5-VL-3B-Instruct
# MODEL_PATH=Qwen/Qwen2.5-VL-7B-Instruct
EVAL=False

TRAIN_FILE=${TRAIN_FILE:-./data/anol_processed_mllm_data/train_full.parquet}
VAL_FILE=${VAL_FILE:-./data/anol_processed_mllm_data/test_full.parquet}

if [ "$EVAL" = "True" ]; then
    VAL_ONLY=True
else
    VAL_ONLY=False
fi

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$TRAIN_FILE \
    data.val_files=$VAL_FILE \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.image_key=images \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='anomseer' \
    trainer.experiment_name='anomseer_timerpo' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.save_freq=500 \
    trainer.test_freq=10 \
    trainer.val_only=$VAL_ONLY \
    trainer.val_before_train=True \
    trainer.total_epochs=10 \
    ts.use_sem_orth=True \
    ts.adv_mix=0.3 \
    ts.similarity_method=ot \
    ts.ot_eps=0.08 \
    ts.ot_n_iter=50 $@
