for i in 6 7
do
  python3 same_outputs.py --depth "${i}"
done

python3 plot_same.py