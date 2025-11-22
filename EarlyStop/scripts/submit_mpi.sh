#!/bin/bash

source $1
set -x

ray stop --force

ray start --head --disable-usage-stats
for i in $(seq 1 $(($MLP_WORKER_NUM - 1))); do
  # 使用eval命令来获取变量的值
  var_name="MLP_WORKER_${i}_HOST"
  host_i=$(eval echo \$$var_name)
  ssh $host_i "ray start --address=${MLP_WORKER_0_HOST}:6379"
done

ray status
eval $RUN_SCRIPT