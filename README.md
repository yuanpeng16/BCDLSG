# Gradient Descent Resists Compositionality

## Image Classification
    # Repeat for A, B, C, D, E
    sh experiments/mnist_paired_first/mnist_paired_first_A.sh
    sh experiments/mnist_paired_second/mnist_paired_second_A.sh

    # Plot results
    python3 plot_results.py --experiment_type mnist_paired

## Language Learning
    git clone https://github.com/brendenlake/SCAN.git
    
    # Repeat for A, B, C, D, E
    sh experiments/scan_first_large/scan_first_large_A.sh
    sh experiments/scan_second_large/scan_second_large_A.sh
    
    # Plot results
    python3 plot_results.py --experiment_type scan

## Visualization
    git clone https://github.com/tensorflow/playground.git
    cp dataset.ts playground/src
    cd playground
    npm i
    npm run build
    npm run serve

Please refer to playground/README.md for more information.
