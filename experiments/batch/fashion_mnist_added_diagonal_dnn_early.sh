ID=$(basename "$0" | sed "s/.sh$//g")
ABS_PATH=$(readlink -f $0)
cd $(dirname $(dirname $(dirname ${ABS_PATH})))

sh experiments/fashion_mnist_added_diagonal_dnn_early/fashion_mnist_added_diagonal_dnn_early_7_0.sh 1 1; sleep 60
sh experiments/fashion_mnist_added_diagonal_dnn_early/fashion_mnist_added_diagonal_dnn_early_7_0.sh 2 1; sleep 60
sh experiments/fashion_mnist_added_diagonal_dnn_early/fashion_mnist_added_diagonal_dnn_early_7_0.sh 3 1; sleep 60
sh experiments/fashion_mnist_added_diagonal_dnn_early/fashion_mnist_added_diagonal_dnn_early_7_0.sh 4 1; sleep 60
sh experiments/fashion_mnist_added_diagonal_dnn_early/fashion_mnist_added_diagonal_dnn_early_7_0.sh 5 1; sleep 60
