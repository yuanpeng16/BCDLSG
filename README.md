# Gradient Descent Resists Compositionality

## Experiments
    #DNN
    sh experiments/batch/fashion_mnist_added_diagonal.sh
    # LSTM horizontal
    sh experiments/batch/reuters_reuters_diagonal_lstm_horizontal.sh

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
    # LSTM horizontal
    python3 plot_results.py --experiment_id reuters_reuters_diagonal_lstm_horizontal --depth 2

## Plot training process
    python3 plot_results.py --experiment_id fashion_mnist_added_diagonal_dnn_early --experiment_type steps --show_legend
    python3 plot_results.py --experiment_id cifar_fashion_added_diagonal_cnn_early --experiment_type steps

## Zero-shot learning
Please download data and put to corresponding folder.

    # Preprocess
    python3 preprocess_data.py
