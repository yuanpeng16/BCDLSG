ID=$(basename "$0" | sed "s/.sh$//g")
ABS_PATH=$(readlink -f $0)
cd $(dirname $(dirname $(dirname ${ABS_PATH})))

sh experiments/cifar_fashion_added_diagonal_cnn_early/cifar_fashion_added_diagonal_cnn_early_7_0.sh 1; sleep 10
sh experiments/cifar_fashion_added_diagonal_cnn_early/cifar_fashion_added_diagonal_cnn_early_7_0.sh 2; sleep 10
sh experiments/cifar_fashion_added_diagonal_cnn_early/cifar_fashion_added_diagonal_cnn_early_7_0.sh 3; sleep 10
sh experiments/cifar_fashion_added_diagonal_cnn_early/cifar_fashion_added_diagonal_cnn_early_7_0.sh 4; sleep 10
sh experiments/cifar_fashion_added_diagonal_cnn_early/cifar_fashion_added_diagonal_cnn_early_7_0.sh 5; sleep 10
