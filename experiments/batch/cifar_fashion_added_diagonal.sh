ID=$(basename "$0" | sed "s/.sh$//g")
ABS_PATH=$(readlink -f $0)
cd $(dirname $(dirname $(dirname ${ABS_PATH})))

sh experiments/cifar_fashion_added_diagonal/cifar_fashion_added_diagonal_2_5.sh 1; sleep 60
sh experiments/cifar_fashion_added_diagonal/cifar_fashion_added_diagonal_5_2.sh 1; sleep 60
sh experiments/cifar_fashion_added_diagonal/cifar_fashion_added_diagonal_7_0.sh 3; sleep 60
