docker run \
           --name jh_mlperf_hpu0 -td               \
           -e HABANA_VISIBLE_DEVICES=2 \
           -v $PWD:/root/llama2-70b-99.9 \
           -v /scratch-1/models:/mnt/weka/data/pytorch/llama2 \
           --user root --workdir=/root           \
           --entrypoint /bin/bash                      \
           --ulimit memlock=-1:-1 mlperf4-docker-1.15.1:latest
        #--net=host           
      #   -v /dev:/dev                                     \
      #   --device=/dev:/dev                               \
      # --privileged --security-opt seccomp=unconfined   \
      # -p 9494:80 \
      # --runtime=habana \
      # -v /tmp:/tmp                                     \
      # -v /sys/kernel/debug:/sys/kernel/debug           \
      # --cap-add=sys_nice --cap-add=SYS_PTRACE          \
