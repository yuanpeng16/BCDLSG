# Gradient Descent Resists Compositionality

## Experiments
    sh experiments/batch/fashion_mnist_added_diagonal.sh

## Plot results
    # DNN and others
    python3 plot_results.py --experiment_id fashion_mnist_added_diagonal
    # ResNet
    python3 plot_results.py --experiment_id cifar_fashion_added_diagonal_resnet --depth 5

## Plot training process
    python3 plot_results.py --experiment_id fashion_mnist_added_diagonal_dnn_early --experiment_type steps
    python3 plot_results.py --experiment_id cifar_fashion_added_diagonal_cnn_early --experiment_type steps