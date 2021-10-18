EPOCHS=200
WEIGHT_DECAY=0.0

#python -u ./perfsage/supervised_train.py --train_prefix ./data/efficientnet_cifar10_quantization/ --model graphsage_mean --sigmoid --epochs $EPOCHS --weight_decay $WEIGHT_DECAY
#python -u ./perfsage/supervised_train.py --train_prefix ./data/cifar10_july8/ --model graphsage_mean --sigmoid --epochs $EPOCHS --weight_decay $WEIGHT_DECAY
#python -u ./perfsage/supervised_train.py --train_prefix ./data/cifar10_july8_mask/ --train_conditional True --model graphsage_mean --sigmoid --epochs $EPOCHS --weight_decay $WEIGHT_DECAY
#python -u ./perfsage/supervised_train.py --train_prefix ./data/imagenet_may26/ --model graphsage_mean --sigmoid --epochs $EPOCHS --weight_decay $WEIGHT_DECAY
#python -u ./perfsage/supervised_train.py --train_prefix ./data/kws_chu_july19/ --model graphsage_mean --sigmoid --epochs $EPOCHS --weight_decay $WEIGHT_DECAY
#python -u ./perfsage/supervised_train.py --train_prefix ./data/EC_EI_KWS/ --train_combined True --model graphsage_mean --sigmoid --epochs $EPOCHS --weight_decay $WEIGHT_DECAY
#python -u ./perfsage/supervised_train.py --train_prefix ./data/EC_EI_KWS_SR_EM28_EM64/ --train_conditional True --train_combined True --model graphsage_mean --sigmoid --epochs $EPOCHS --weight_decay $WEIGHT_DECAY
#python -u ./perfsage/supervised_train.py --train_prefix ./data/EC_EI_KWS_SR_EM28_EM64_v3/ --train_conditional False --train_combined True --model graphsage_mean --sigmoid --epochs $EPOCHS --weight_decay $WEIGHT_DECAY
python -u ./perfsage/supervised_train.py --train_prefix ./data/emnist_64/ --train_conditional False --train_combined False --model graphsage_mean --sigmoid --epochs $EPOCHS --weight_decay $WEIGHT_DECAY