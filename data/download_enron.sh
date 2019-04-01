
checkCmd() {
  command -v $1 >/dev/null 2>&1 \
    || exit "$1 command not found. Please install from your package manager."
}

checkCmd wget
checkCmd unzip

###### Download emotions dataset

rm -rf enron
mkdir -p enron

cd enron
wget https://sci2s.ugr.es/keel/dataset/data/multilabel/enron.zip


unzip enron.zip

cd ..
python preprocess_enron.py

