kaldi-io-for-python
===================
'Glue' code connecting kaldi data and python.
-----------------------------------------------

#### Supported data types
- vector (integer)
- Vector (float, double)
- Matrix (float, double)
- Posterior (posteriors, nnet1 training targets, confusion networks, ...)

#### Examples

###### Reading feature scp example:

```python
import kaldi_io
for key,mat in kaldi_io.read_mat_scp(file):
  ...
```

###### Writing feature ark to file/stream:
```python
import kaldi_io
with open(ark_file,'wb') as f:
  for key,mat in dict.iteritems():
    kaldi_io.write_mat(f, mat, key=key)
```

###### Writing features as 'ark,scp' by pipeline with 'copy-feats':
```python
import kaldi_io
ark_scp_output='ark:| copy-feats --compress=true ark:- ark,scp:data/feats2.ark,data/feats2.scp'
with kaldi_io.open_or_fd(ark_scp_output,'wb') as f:
  for key,mat in dict.iteritems():
    kaldi_io.write_mat(f, mat, key=key)
```

#### Install
- from pypi:
```
pip install kaldi_io`
```

- from sources:
```
git clone https://github.com/vesis84/kaldi-io-for-python.git <kaldi-io-dir>`
pip install -r requirements.txt
pip install --editable .
```

Note: it is recommended to set `export KALDI_ROOT=<some_kaldi_dir>` environment variable.
The I/O based on pipes can then contain kaldi binaries.

#### Unit tests

(note: these are not included in pypi package)

Unit tests are started this way:

`./run_tests.sh`

or by:

`python3 -m unittest discover -s tests -t .`
`python2 -m unittest discover -s tests -t .`


#### License
Apache License, Version 2.0 ('LICENSE-2.0.txt')

#### Community
- accepting pull requests with extensions on GitHub
- accepting feedback via GitHub 'Issues' in the repo


