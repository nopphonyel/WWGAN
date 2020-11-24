# Just exec this file to start training
# NOTE: Python expect to have pytorch installed
export PYTHONPATH=$(dirname "$(realpath "$0")");
python training/semi_supervised/training_code.py