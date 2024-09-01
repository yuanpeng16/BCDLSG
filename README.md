# On a Built-in Conflict between Deep Learning and Systematic Generalization

## Main experiments

    # Scripts in scripts/main
    sh scripts/batch.sh scripts/main/main_dnn.sh

## More experiments

### Natural inputs

Please download data and put to corresponding folder.

NICO++ data: Please download data from https://nicochallenge.com/dataset and put to ../../data/nico.

Amazon review data:
Please download 5-core data for the following five categories from https://nijianmo.github.io/amazon/index.html and
extract to ../../data/amazon.

- Books_5
- Clothing_Shoes_and_Jewelry_5
- Home_and_Kitchen_5
- Electronics_5
- Movies_and_TV_5

    # Preprocess
    # Images
    python3 preprocess_data.py --dataset nico
    # Texts: repeat for the five categories.
    python3 amazon_review_dataset.py --category Movies_and_TV_5

    # Experiments
    # Scripts in scripts/natural
    sh scripts/batch.sh scripts/natural/natural_dnn.sh

    # Ablation
    # Scripts in scripts/ablation
    sh scripts/batch.sh scripts/ablation/ablation_dnn-64.sh

### Zero-shot learning

Please download data and put to corresponding folder.

- apy: https://vision.cs.uiuc.edu/attributes/
- awa2: https://cvml.ist.ac.at/AwA2/
- cub: http://www.vision.caltech.edu/datasets/cub_200_2011/
- sun: https://cs.brown.edu/~gmpatter/sunattributes.html

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

### Linear experiment

    sh linear_scripts/linear_16.sh

### Same outputs

    sh same_outputs.sh

### Initialization experiments

    python3 initialization_experiments.py

## Plot results

    python3 plot_results.py --experiment_id main_dnn --show_legend

### Summarize results

    python3 summarize_results.py

### Training process

    python3 plot_analysis.py --experiment_id partition-t_main_dnn
