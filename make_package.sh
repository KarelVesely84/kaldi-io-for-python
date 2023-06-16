#!/bin/bash

exit 0

# Use conda for releasing:
# `unset PYTHONHOME; conda create --prefix /mnt/matylda5/iveselyk/CONDA_ENVS/kaldi-io-for-python_release python=3.9`
# `unset PYTHONHOME; conda activate /mnt/matylda5/iveselyk/CONDA_ENVS/kaldi-io-for-python_release`

# Step 0: upgrade python packages:
#python2 -m pip install --upgrade pip
#python2 -m pip install --user --upgrade setuptools wheel twine
python3 -m pip install --upgrade pip
python3 -m pip install --user --upgrade setuptools wheel twine

# Step 1: 'increase' version in 'setup.py'.
# commit, push

# Step 1a: add tag with the 'increased' version:
#
# `git tag v0.9.6` # create the tag
# `git push origin v0.9.6` # export it to remote
#
# Adding tag ex-post:
# `git tag v0.9.5 <sha>`
#
# Read the <sha> from the tag:
# `git show v0.9.6`
#
# Remove tag:
# `git push --delete origin v0.9.6` (remote)
# `git tag -d v0.9.6` (local)
#
# see: https://git-scm.com/book/en/v2/Git-Basics-Tagging


# Step 2: make packages,
rm dist/*
python2 setup.py bdist_wheel # create python2 package,
python3 setup.py sdist bdist_wheel # create python3 package,

ll dist/
# -rw-r--r-- 1 iveselyk speech 13839 Feb 26 13:55 kaldi_io-0.9.2-py3-none-any.whl
# -rw-r--r-- 1 iveselyk speech  9220 Feb 26 13:55 kaldi_io-0.9.2.tar.gz

# Step 2.1: check the README.md format,
python3 -m twine check dist/*


### SANDBOX DEPLOYMENT ###
# Hint: skip to 'Step 8' to skip sandboxing on 'test.pypi.org'
{
  # TEST_DEPLOYMENT_ON test.pypi.org,
  # Step 3: upload the packages (test site),
  python3 -m twine upload --verbose --repository-url https://test.pypi.org/legacy/ dist/*

  # Step 4: see webpage,
  # https://test.pypi.org/project/kaldi_io_vesis84

  # Step 5: try installing it locally,
  python3 -m pip install --user --index-url https://test.pypi.org/simple/ kaldi_io_vesis84

  # Stepy 6: try to install it,
  python3
  <import kaldi_io
  <print(kaldi_io)

  # Step 7: remove the package,
  python3 -m pip uninstall kaldi_io_vesis84
}

# Step 8: Put the packages to 'production' pypi,
python3 -m twine upload --verbose dist/* # (login,pwd) from https://pypi.org/account/login/
python3 -m pip install --user kaldi_io
python3 -m pip uninstall kaldi_io

