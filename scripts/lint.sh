yapf -r -i ${1}
isort -rc ${1}
flake8 ${1}
