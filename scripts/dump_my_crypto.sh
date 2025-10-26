name_lst=("15min" "30min" "60min" "240min" "720min" "1d")
freq_lst=("15min" "30min" "60min" "240min" "720min" "day")

# Loop over each string
for i in "${!name_lst[@]}"; do
  name="${name_lst[$i]}"
  freq="${freq_lst[$i]}"
  echo "Processing: $name, $freq"

  python ./scripts/dump_bin.py dump_all \
    --data_path /capstor/scratch/cscs/ljiayong/datasets/qlib/my_crypto/resampled/$name \
    --qlib_dir /capstor/scratch/cscs/ljiayong/datasets/qlib/my_crypto/bin/$name \
    --freq $freq --exclude_fields date,symbol

  # Copy the file
  cp "/capstor/scratch/cscs/ljiayong/datasets/qlib/my_crypto/resampled/$name/instruments/my_universe.txt" \
    "/capstor/scratch/cscs/ljiayong/datasets/qlib/my_crypto/bin/$name/instruments/my_universe.txt"
done
