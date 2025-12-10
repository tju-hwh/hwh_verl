rm -rf /openbayes/home/verl/hwh_test/tp_allreduce.log

export pop_outputs_config="4"
export stream_rollout_and_train="1"
nsys profile -w true \
  --trace=cuda,nvtx,osrt,cudnn,cublas \
    python3 -m verl.trainer.main_ppo \
 data.train_files=/openbayes/home/data/gsm8k/train.parquet \
 data.val_files=/openbayes/home/data/gsm8k/test.parquet \
 data.train_batch_size=64 \
 data.max_prompt_length=256 \
 data.max_response_length=3 \
 actor_rollout_ref.model.path=/openbayes/home/models/Qwen2.5-0.5B \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=/openbayes/home/models/Qwen2.5-0.5B \
 critic.ppo_micro_batch_size_per_gpu=4 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=['console'] \
 trainer.val_before_train=False \
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=2 \
 trainer.nnodes=1 \
 trainer.total_training_steps=5 \
 trainer.save_freq=0 \
 trainer.test_freq=0 \
 trainer.total_epochs=1 2>&1 | tee hwh_test/1121_10_test.log