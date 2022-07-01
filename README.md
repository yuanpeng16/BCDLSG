# Gradient Descent Resists Compositionality

## Experiments
    sh experiments/batch/fashion_mnist_added_diagonal.sh

## Additional experiments
#### Equally difficult factors
    # DNN
    sh experiments/batch/fashion_fashion_added_diagonal_dnn.sh
    # CNN
    sh experiments/batch/fashion_fashion_added_diagonal_cnn.sh

#### Label combinations
    # Tile
    sh experiments/batch/fashion_mnist_added_tile.sh
    # One-shot
    sh experiments/batch/fashion_mnist_added_oneshot.sh

## Plot results
    # DNN and others
    python3 plot_results.py --experiment_id fashion_mnist_added_diagonal --show_legend
    # ResNet
    python3 plot_results.py --experiment_id cifar_fashion_added_diagonal_resnet --depth 5

## Plot training process
    python3 plot_results.py --experiment_id fashion_mnist_added_diagonal_dnn_early --experiment_type steps --show_legend
    python3 plot_results.py --experiment_id cifar_fashion_added_diagonal_cnn_early --experiment_type steps