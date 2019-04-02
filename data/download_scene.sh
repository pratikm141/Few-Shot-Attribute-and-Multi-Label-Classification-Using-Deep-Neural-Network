
checkCmd() {
  command -v $1 >/dev/null 2>&1 \
    || exit "$1 command not found. Please install from your package manager."
}

checkCmd wget
checkCmd unzip

###### Download scene dataset

rm -rf scene
mkdir -p scene

cd scene
wget https://sci2s.ugr.es/keel/dataset/data/multilabel/scene.zip


unzip scene.zip

cd ..
python preprocess_scene.py
