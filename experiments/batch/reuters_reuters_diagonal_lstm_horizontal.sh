ID=$(basename "$0" | sed "s/.sh$//g")
ABS_PATH=$(readlink -f $0)
cd $(dirname $(dirname $(dirname ${ABS_PATH})))

sh experiments/reuters_reuters_diagonal_lstm_horizontal/reuters_reuters_diagonal_lstm_horizontal_0_2.sh 1 1; sleep 20
sh experiments/reuters_reuters_diagonal_lstm_horizontal/reuters_reuters_diagonal_lstm_horizontal_1_1.sh 1 1; sleep 20
sh experiments/reuters_reuters_diagonal_lstm_horizontal/reuters_reuters_diagonal_lstm_horizontal_2_0.sh 1 1; sleep 20
sh experiments/reuters_reuters_diagonal_lstm_horizontal/reuters_reuters_diagonal_lstm_horizontal_0_2.sh 2 1; sleep 20
sh experiments/reuters_reuters_diagonal_lstm_horizontal/reuters_reuters_diagonal_lstm_horizontal_1_1.sh 2 1; sleep 20
sh experiments/reuters_reuters_diagonal_lstm_horizontal/reuters_reuters_diagonal_lstm_horizontal_2_0.sh 2 1; sleep 20
sh experiments/reuters_reuters_diagonal_lstm_horizontal/reuters_reuters_diagonal_lstm_horizontal_0_2.sh 3 1; sleep 20
sh experiments/reuters_reuters_diagonal_lstm_horizontal/reuters_reuters_diagonal_lstm_horizontal_1_1.sh 3 1; sleep 20
sh experiments/reuters_reuters_diagonal_lstm_horizontal/reuters_reuters_diagonal_lstm_horizontal_2_0.sh 3 1; sleep 20
sh experiments/reuters_reuters_diagonal_lstm_horizontal/reuters_reuters_diagonal_lstm_horizontal_0_2.sh 4 1; sleep 20
sh experiments/reuters_reuters_diagonal_lstm_horizontal/reuters_reuters_diagonal_lstm_horizontal_1_1.sh 4 1; sleep 20
sh experiments/reuters_reuters_diagonal_lstm_horizontal/reuters_reuters_diagonal_lstm_horizontal_2_0.sh 4 1; sleep 20
sh experiments/reuters_reuters_diagonal_lstm_horizontal/reuters_reuters_diagonal_lstm_horizontal_0_2.sh 5 1; sleep 20
sh experiments/reuters_reuters_diagonal_lstm_horizontal/reuters_reuters_diagonal_lstm_horizontal_1_1.sh 5 1; sleep 20
sh experiments/reuters_reuters_diagonal_lstm_horizontal/reuters_reuters_diagonal_lstm_horizontal_2_0.sh 5 1; sleep 20
