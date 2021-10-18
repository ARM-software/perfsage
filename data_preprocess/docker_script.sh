CNAME=yuji_tflite_jupyter

docker kill $CNAME
docker rm $CNAME

docker run --privileged=true --gpus '"device=all"' -itd \
    -v /home/yujichai/mlrsh/yuezha01/latency_data_gnn:/workspace/datasets/efficientnet_cifar10_quantization \
    -v /hddb/yuezha01/latency_data_gnn:/workspace/datasets \
    -v /hddb/chuzho01/kws_chu_july19:/workspace/datasets/kws_chu_july19 \
    -v /hddb/yujichai/sr_data:/workspace/datasets/super \
    -v /hddb/yujichai/emnist_data/hactar/28x28:/workspace/datasets/emnist_28 \
    -v /hddb/yujichai/emnist_data/hactar/64x64:/workspace/datasets/emnist_64 \
    -v /hddb/gtuzi/sampling_exploration/full_nas_ssdlite80:/workspace/datasets/ssd_80 \
    -v /hddb/gtuzi/sampling_exploration/full_nas_ssdlite300:/workspace/datasets/ssd_300 \
    -v /home/yujichai/mlrsh/yujichai/PerfSAGE:/workspace/PerfSAGE \
    -p 8001:8001 \
    -w /workspace \
    --name $CNAME \
    rmatas/bbb:tf2-nightly-6.20 \
    bash

docker attach $CNAME
