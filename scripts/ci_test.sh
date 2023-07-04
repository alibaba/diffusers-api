#================================================================
#   Copyright (C) 2022 Alibaba Ltd. All rights reserved.
#
#================================================================

cd ${1}

# pre-commit run --all-files
# if [ $? -ne 0 ]; then
#     echo "linter test failed, please run 'pre-commit run --all-files' to check"
#     exit -1
# fi

mkdir -p logs

# coverage UT and report
PYTHONPATH=. coverage run tests/run.py | tee logs/ci_test.log
PYTHONPATH=. coverage report  | tee logs/ci_report.log
# PYTHONPATH=. coverage html


# please add key requirements you think is worth record

echo "" | tee >> logs/ci_report.log
echo "" | tee >> logs/ci_report.log
echo "Requirements                  Version
-----------------------------------------------------------" | tee >> logs/ci_report.log
key_requirements=("^torch"  "easycv"  "easynlp"  "easyretrieval"\
  "blade"  "mmcv"  "tokenizer")
pip list >> logs/envlistcache.txt

for val1 in ${key_requirements[*]}; do
    grep $val1 logs/envlistcache.txt | tee >> logs/ci_report.log
done
echo "-----------------------------------------------------------" | tee >> logs/ci_report.log

rm logs/envlistcache.txt
