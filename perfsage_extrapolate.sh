#python -u ./perfsage/extrapolate.py --train_prefix ./data/efficientnet_cifar10_quantization/ --load_model_step 100000 --extrapolate_prefix ./data/efficientnet_cifar10_quantization/
#python -u ./perfsage/extrapolate.py --train_prefix ./data/efficientnet_cifar10_quantization/ --load_model_step 100000 --extrapolate_prefix ./data/cifar10_july8/
#python -u ./perfsage/extrapolate.py --train_prefix ./data/efficientnet_cifar10_quantization/ --load_model_step 100000 --extrapolate_prefix ./data/imagenet_may26/

EC_STEP=190500
#python -u ./perfsage/extrapolate.py --train_prefix ./data/cifar10_july8/ --load_model_step $EC_STEP --extrapolate_prefix ./data/efficientnet_cifar10_quantization/ 
#python -u ./perfsage/extrapolate.py --train_prefix ./data/cifar10_july8/ --load_model_step $EC_STEP --extrapolate_prefix ./data/cifar10_july8/ 
#python -u ./perfsage/extrapolate.py --train_prefix ./data/cifar10_july8/ --load_model_step $EC_STEP --extrapolate_prefix ./data/imagenet_may26/ 
#python -u ./perfsage/extrapolate.py --train_prefix ./data/cifar10_july8/ --load_model_step $EC_STEP --extrapolate_prefix ./data/kws_chu_july19/ 

ECC_STEP=100000
#python -u ./perfsage/extrapolate.py --train_prefix ./data/cifar10_july8/ --load_model_step $ECC_STEP --extrapolate_prefix ./data/efficientnet_cifar10_quantization/ --train_conditional True
#python -u ./perfsage/extrapolate.py --train_prefix ./data/cifar10_july8_mask/ --load_model_step $ECC_STEP --extrapolate_prefix ./data/cifar10_july8_mask/ --train_conditional True
#python -u ./perfsage/extrapolate.py --train_prefix ./data/cifar10_july8_mask/ --load_model_step $ECC_STEP --extrapolate_prefix ./data/imagenet_may26/ --train_conditional True
#python -u ./perfsage/extrapolate.py --train_prefix ./data/cifar10_july8_mask/ --load_model_step $ECC_STEP --extrapolate_prefix ./data/kws_chu_july19/ --train_conditional True

EI_STEP=100000
#python -u ./perfsage/extrapolate.py --train_prefix ./data/imagenet_may26/ --load_model_step $EI_STEP --extrapolate_prefix ./data/efficientnet_cifar10_quantization/ 
#python -u ./perfsage/extrapolate.py --train_prefix ./data/imagenet_may26/ --load_model_step $EI_STEP --extrapolate_prefix ./data/cifar10_july8/ 
#python -u ./perfsage/extrapolate.py --train_prefix ./data/imagenet_may26/ --load_model_step $EI_STEP --extrapolate_prefix ./data/imagenet_may26/ 
#python -u ./perfsage/extrapolate.py --train_prefix ./data/imagenet_may26/ --load_model_step $EI_STEP --extrapolate_prefix ./data/kws_chu_july19/ 

EC_EI_KWS_STEP=195250
#python -u ./perfsage/extrapolate.py --train_prefix ./data/EC_EI_KWS/ --load_model_step $EC_EI_KWS_STEP --extrapolate_prefix ./data/efficientnet_cifar10_quantization/ 
#python -u ./perfsage/extrapolate.py --train_prefix ./data/EC_EI_KWS/ --load_model_step $EC_EI_KWS_STEP --extrapolate_prefix ./data/cifar10_july8/ 
#python -u ./perfsage/extrapolate.py --train_prefix ./data/EC_EI_KWS/ --load_model_step $EC_EI_KWS_STEP --extrapolate_prefix ./data/imagenet_may26/ 
#python -u ./perfsage/extrapolate.py --train_prefix ./data/EC_EI_KWS/ --load_model_step $EC_EI_KWS_STEP --extrapolate_prefix ./data/kws_chu_july19/ 

EC_EI_KWS_SR_EM28_EM64_STEP=544200
TCF=False
#python -u ./perfsage/extrapolate.py --train_prefix ./data/EC_EI_KWS_SR_EM28_EM64/ --load_model_step $EC_EI_KWS_SR_EM28_EM64_STEP --extrapolate_prefix ./data/efficientnet_cifar10_quantization/ --train_conditional $TCF
python -u ./perfsage/extrapolate.py --train_prefix ./data/EC_EI_KWS_SR_EM28_EM64/ --load_model_step $EC_EI_KWS_SR_EM28_EM64_STEP --extrapolate_prefix ./data/cifar10_july8/ --train_conditional $TCF
python -u ./perfsage/extrapolate.py --train_prefix ./data/EC_EI_KWS_SR_EM28_EM64/ --load_model_step $EC_EI_KWS_SR_EM28_EM64_STEP --extrapolate_prefix ./data/imagenet_may26/ --train_conditional $TCF
python -u ./perfsage/extrapolate.py --train_prefix ./data/EC_EI_KWS_SR_EM28_EM64/ --load_model_step $EC_EI_KWS_SR_EM28_EM64_STEP --extrapolate_prefix ./data/kws_chu_july19/ --train_conditional $TCF
python -u ./perfsage/extrapolate.py --train_prefix ./data/EC_EI_KWS_SR_EM28_EM64/ --load_model_step $EC_EI_KWS_SR_EM28_EM64_STEP --extrapolate_prefix ./data/super/ --train_conditional $TCF
python -u ./perfsage/extrapolate.py --train_prefix ./data/EC_EI_KWS_SR_EM28_EM64/ --load_model_step $EC_EI_KWS_SR_EM28_EM64_STEP --extrapolate_prefix ./data/emnist_28/ --train_conditional $TCF
python -u ./perfsage/extrapolate.py --train_prefix ./data/EC_EI_KWS_SR_EM28_EM64/ --load_model_step $EC_EI_KWS_SR_EM28_EM64_STEP --extrapolate_prefix ./data/emnist_64/ --train_conditional $TCF
