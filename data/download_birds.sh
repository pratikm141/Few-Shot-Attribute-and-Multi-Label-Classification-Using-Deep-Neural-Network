
checkCmd() {
  command -v $1 >/dev/null 2>&1 \
    || exit "$1 command not found. Please install from your package manager."
}

checkCmd wget
checkCmd unrar

###### Download birds dataset

rm -rf birds
mkdir -p birds

cd birds
wget http://sourceforge.net/projects/mulan/files/datasets/birds.rar


unrar x birds.rar

cd ..
python preprocess_birds.py
