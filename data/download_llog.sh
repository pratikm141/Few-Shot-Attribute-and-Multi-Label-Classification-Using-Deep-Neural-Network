
checkCmd() {
  command -v $1 >/dev/null 2>&1 \
    || exit "$1 command not found. Please install from your package manager."
}

checkCmd wget

###### Download llog dataset

rm -rf llog
mkdir -p llog

cd llog
wget https://liquidtelecom.dl.sourceforge.net/project/meka/Datasets/LLOG-F.arff


cd ..
python preprocess_llog.py
