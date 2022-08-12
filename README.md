# On a Built-in Conflict between Deep Learning and Systematic Generalization

## Main experiments

    # Scripts in scripts/main
    sh scripts/batch.sh scripts/main/main_dnn.sh

## More experiments

### Zero-shot learning

Please download data and put to corresponding folder.

    # Preprocess
    python3 preprocess_data.py --dataest cub
    python3 preprocess_data.py --dataest sun

    # Experiments
    # Scripts in scripts/zeroshot
    sh scripts/batch.sh scripts/zeroshot/zeroshot_apy.sh

### Training process

    # Scripts in scripts/partition-t_main
    sh scripts/batch.sh scripts/partition-t_main/partition-t_dnn.sh

### New classes

    # Scripts in scripts/single
    sh scripts/batch.sh scripts/single/single_dnn.sh

### Equally difficult factors

    # Scripts in scripts/equal
    sh scripts/batch.sh scripts/equal/equal_dnn.sh

### Label combinations

    # Scripts in scripts/label
    sh scripts/batch.sh scripts/label/label_tile.sh

## Plot results

    python3 plot_results.py --experiment_id main_dnn --show_legend

### training process

    python3 plot_analysis.py --experiment_id partition-t_main_dnn

## Visualized examples

Please visit https://gitee.com/yuanpeng16/multilabel_playground
