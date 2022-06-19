ID=$(basename "$0" | sed "s/.sh$//g")
ABS_PATH=$(readlink -f $0)
cd $(dirname $(dirname $(dirname ${ABS_PATH})))

sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_1_1.sh 1 1; sleep 120
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_1_1.sh 2 1; sleep 120
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_1_1.sh 3 1; sleep 120
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_1_1.sh 4 1; sleep 120
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_1_1.sh 5 1; sleep 120
