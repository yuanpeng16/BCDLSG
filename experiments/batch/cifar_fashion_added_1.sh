ID=$(basename "$0" | sed "s/.sh$//g")
ABS_PATH=$(readlink -f $0)
cd $(dirname $(dirname $(dirname ${ABS_PATH})))
GPU_ID=1

sh experiments/cifar_fashion_added/cifar_fashion_added_0_7.sh 4 ${GPU_ID}; sleep 300
sh experiments/cifar_fashion_added/cifar_fashion_added_1_6.sh 4 ${GPU_ID}; sleep 300
sh experiments/cifar_fashion_added/cifar_fashion_added_2_5.sh 4 ${GPU_ID}; sleep 300
sh experiments/cifar_fashion_added/cifar_fashion_added_3_4.sh 4 ${GPU_ID}; sleep 300
sh experiments/cifar_fashion_added/cifar_fashion_added_4_3.sh 4 ${GPU_ID}; sleep 300
sh experiments/cifar_fashion_added/cifar_fashion_added_5_2.sh 4 ${GPU_ID}; sleep 300
sh experiments/cifar_fashion_added/cifar_fashion_added_6_1.sh 4 ${GPU_ID}; sleep 300
sh experiments/cifar_fashion_added/cifar_fashion_added_7_0.sh 4 ${GPU_ID}; sleep 300
sh experiments/cifar_fashion_added/cifar_fashion_added_0_7.sh 5 ${GPU_ID}; sleep 300
sh experiments/cifar_fashion_added/cifar_fashion_added_1_6.sh 5 ${GPU_ID}; sleep 300
sh experiments/cifar_fashion_added/cifar_fashion_added_2_5.sh 5 ${GPU_ID}; sleep 300
sh experiments/cifar_fashion_added/cifar_fashion_added_3_4.sh 5 ${GPU_ID}; sleep 300
sh experiments/cifar_fashion_added/cifar_fashion_added_4_3.sh 5 ${GPU_ID}; sleep 300
sh experiments/cifar_fashion_added/cifar_fashion_added_5_2.sh 5 ${GPU_ID}; sleep 300
sh experiments/cifar_fashion_added/cifar_fashion_added_6_1.sh 5 ${GPU_ID}; sleep 300
sh experiments/cifar_fashion_added/cifar_fashion_added_7_0.sh 5 ${GPU_ID}; sleep 300

