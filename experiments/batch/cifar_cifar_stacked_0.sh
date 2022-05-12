ID=$(basename "$0" | sed "s/.sh$//g")
ABS_PATH=$(readlink -f $0)
cd $(dirname $(dirname $(dirname ${ABS_PATH})))

sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_3_4.sh 1; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_4_3.sh 1; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_5_2.sh 1; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_6_1.sh 1; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_0_7.sh 2; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_1_6.sh 2; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_2_5.sh 2; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_3_4.sh 2; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_4_3.sh 2; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_5_2.sh 2; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_6_1.sh 2; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_0_7.sh 3; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_1_6.sh 3; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_2_5.sh 3; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_3_4.sh 3; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_4_3.sh 3; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_5_2.sh 3; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_6_1.sh 3; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_0_7.sh 4; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_1_6.sh 4; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_2_5.sh 4; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_3_4.sh 4; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_4_3.sh 4; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_5_2.sh 4; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_6_1.sh 4; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_0_7.sh 5; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_1_6.sh 5; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_2_5.sh 5; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_3_4.sh 5; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_4_3.sh 5; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_5_2.sh 5; sleep 60
sh experiments/cifar_cifar_stacked/cifar_cifar_stacked_6_1.sh 5;
