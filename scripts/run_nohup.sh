m=${1}
d=${2}
g=${3}

nohup python train.py -m ${m} -d ${d} -g ${g} > /dev/null 2>&1 &
