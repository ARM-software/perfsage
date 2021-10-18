EPOCHS=200
WEIGHT_DECAY=0.0

#python -u ./perfsage/supervised_train.py --train_prefix ./data/efficientnet_cifar10_quantization/ --model graphsage_mean --sigmoid --epochs $EPOCHS --weight_decay $WEIGHT_DECAY
python -u ./perfsage/supervised_train.py --train_prefix ./data/cifar10_july8/ --model graphsage_mean --sigmoid --epochs $EPOCHS --weight_decay $WEIGHT_DECAY
#python -u ./perfsage/supervised_train.py --train_prefix ./data/imagenet_may26/ --model graphsage_mean --sigmoid --epochs $EPOCHS --weight_decay $WEIGHT_DECAY
