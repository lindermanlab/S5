# Make a directory to dump the raw data into.
rm -rf ./raw_datasets
mkdir ./raw_datasets

./bin/download_lra.sh
./bin/download_aan.sh
./bin/download_sc35.sh

