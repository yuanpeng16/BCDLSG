ID=$(basename "$0" | sed "s/.sh$//g")
ABS_PATH=$(readlink -f $0)
cd $(dirname $(dirname $(dirname ${ABS_PATH})))

sh experiments/cifar_fashion_added/cifar_fashion_added_7_0.sh 1; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_7_0.sh 2; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_7_0.sh 3; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_7_0.sh 4; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_7_0.sh 5; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_0_7.sh 1; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_1_6.sh 1; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_2_5.sh 1; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_3_4.sh 1; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_4_3.sh 1; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_5_2.sh 1; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_6_1.sh 1; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_0_7.sh 2; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_1_6.sh 2; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_2_5.sh 2; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_3_4.sh 2; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_4_3.sh 2; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_5_2.sh 2; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_6_1.sh 2; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_0_7.sh 3; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_1_6.sh 3; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_2_5.sh 3; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_3_4.sh 3; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_4_3.sh 3; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_5_2.sh 3; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_6_1.sh 3; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_0_7.sh 4; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_1_6.sh 4; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_2_5.sh 4; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_3_4.sh 4; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_4_3.sh 4; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_5_2.sh 4; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_6_1.sh 4; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_0_7.sh 5; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_1_6.sh 5; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_2_5.sh 5; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_3_4.sh 5; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_4_3.sh 5; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_5_2.sh 5; sleep 60
sh experiments/cifar_fashion_added/cifar_fashion_added_6_1.sh 5; sleep 60
