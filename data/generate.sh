python -m generate --n_sentences=10 --out_file_name="./datasets/mini_dataset.jsonl" &
python -m generate --n_sentences=100 --out_file_name="./datasets/small_dataset.jsonl" &
python -m generate --n_sentences=1000 --out_file_name="./datasets/big_dataset.jsonl"&
python -m generate --n_sentences=10000 --out_file_name="./datasets/huge_dataset.jsonl"&
python -m generate --n_sentences=1000000 --out_file_name="./datasets/full_dataset.jsonl"&
python -m generate --len_filter_args='(0,10)' --n_sentences=10 --out_file_name="./datasets/short_mini_dataset.jsonl"&
python -m generate --len_filter_args='(0,10)' --n_sentences=100 --out_file_name="./datasets/short_small_dataset.jsonl"&
python -m generate --len_filter_args='(0,10)' --n_sentences=1000 --out_file_name="./datasets/short_big_dataset.jsonl"&
python -m generate --len_filter_args='(10,50)' --n_sentences=10 --out_file_name="./datasets/long_mini_dataset.jsonl"&
python -m generate --len_filter_args='(10,50)' --n_sentences=100 --out_file_name="./datasets/long_small_dataset.jsonl"&
python -m generate --len_filter_args='(10,50)' --n_sentences=1000 --out_file_name="./datasets/long_big_dataset.jsonl"&
python -m generate --len_filter_args='(50,5000000)' --n_sentences=10 --out_file_name="./datasets/very_long_mini_dataset.jsonl"&
python -m generate --len_filter_args='(50,5000000)' --n_sentences=100 --out_file_name="./datasets/very_long_small_dataset.jsonl"&
python -m generate --len_filter_args='(50,5000000)' --n_sentences=1000 --out_file_name="./datasets/very_long_big_dataset.jsonl"&

wait
