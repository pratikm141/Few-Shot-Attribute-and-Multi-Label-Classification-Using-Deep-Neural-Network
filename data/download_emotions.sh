
checkCmd() {
  command -v $1 >/dev/null 2>&1 \
    || exit "$1 command not found. Please install from your package manager."
}

checkCmd wget
checkCmd unzip

###### Download emotions dataset

rm -rf emotions
mkdir -p emotions

cd emotions
wget https://sci2s.ugr.es/keel/dataset/data/multilabel/emotions.zip


unzip emotions.zip

cd ..
python preprocess_emotions.py

