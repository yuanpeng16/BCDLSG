ID=$(basename "$0" | sed "s/.sh$//g")
ABS_PATH=$(readlink -f $0)
cd $(dirname $(dirname $(dirname ${ABS_PATH})))

sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_1_0.sh 1 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_2_0.sh 1 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_3_0.sh 1 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_4_0.sh 1 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_5_0.sh 1 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_6_0.sh 1 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_1_0.sh 2 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_2_0.sh 2 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_3_0.sh 2 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_4_0.sh 2 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_5_0.sh 2 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_6_0.sh 2 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_1_0.sh 3 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_2_0.sh 3 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_3_0.sh 3 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_4_0.sh 3 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_5_0.sh 3 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_6_0.sh 3 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_1_0.sh 4 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_2_0.sh 4 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_3_0.sh 4 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_4_0.sh 4 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_5_0.sh 4 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_6_0.sh 4 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_1_0.sh 5 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_2_0.sh 5 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_3_0.sh 5 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_4_0.sh 5 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_5_0.sh 5 1; sleep 60
sh experiments/imdb_imdb_diagonal_lstm/imdb_imdb_diagonal_lstm_6_0.sh 5 1; sleep 60
