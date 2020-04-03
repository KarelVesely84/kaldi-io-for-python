#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2014-2020 Brno University of Technology (author: Karel Vesely)
# Licensed under the Apache License, Version 2.0 (the "License")

from __future__ import print_function
from __future__ import division

import numpy as np
import sys, os, re, gzip, struct, io

#################################################
# Adding 'kaldi binaries' to shell path,

# Select kaldi,
if not 'KALDI_ROOT' in os.environ:
    # Default! To change run python with 'export KALDI_ROOT=/some_dir python'
    os.environ['KALDI_ROOT']='/mnt/matylda5/iveselyk/Tools/kaldi-trunk'

# See if the path exists,
if not os.path.exists(os.environ['KALDI_ROOT']):
    print(80*"#", file=sys.stderr)
    print("### WARNING, path does not exist: KALDI_ROOT=%s" % os.environ['KALDI_ROOT'], file=sys.stderr)
    print("###          (please add 'export KALDI_ROOT=<your_path>' in your $HOME/.profile)", file=sys.stderr)
    print("###          (or run as: KALDI_ROOT=<your_path> python <your_script>.py)", file=sys.stderr)
    print(80*"#"+"\n", file=sys.stderr)

# Add 'kaldi binaries' to shell path,
try:
    path = os.popen('echo $KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin:$KALDI_ROOT/src/nnet3bin:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/ivectorbin/:$KALDI_ROOT/src/lmbin/')
    os.environ['PATH'] = path.readline().strip() + ':' + os.environ['PATH']
    path.close()
except:
    print(80*"#", file=sys.stderr)
    print("### WARNING: could not modify $PATH, (and add the kaldi binaries...)", file=sys.stderr)
    print(80*"#"+"\n", file=sys.stderr)


#################################################
# Define all 'kaldi_io' exceptions,
class UnsupportedDataType(Exception): pass
class UnknownVectorHeader(Exception): pass
class UnknownMatrixHeader(Exception): pass

class BadSampleSize(Exception): pass
class BadInputFormat(Exception): pass

class SubprocessFailed(Exception): pass

#################################################
# Data-type independent helper functions,

def open_or_fd(file, mode='rb'):
    """ fd = open_or_fd(file)
     Open file, gzipped file, pipe, or forward the file-descriptor.
     Eventually seeks in the 'file' argument contains ':offset' suffix.
    """
    offset = None
    try:
        # strip 'ark:' prefix from r{x,w}filename (optional),
        if re.search('^(ark|scp)(,scp|,b|,t|,n?f|,n?p|,b?o|,n?s|,n?cs)*:', file):
            (prefix,file) = file.split(':',1)
        # separate offset from filename (optional),
        if re.search(':[0-9]+$', file):
            (file,offset) = file.rsplit(':',1)
        # input pipe?
        if file[-1] == '|':
            fd = popen(file[:-1], 'rb') # custom,
        # output pipe?
        elif file[0] == '|':
            fd = popen(file[1:], 'wb') # custom,
        # is it gzipped?
        elif file.split('.')[-1] == 'gz':
            fd = gzip.open(file, mode)
        # a normal file...
        else:
            fd = open(file, mode)
    except TypeError:
        # 'file' is opened file descriptor,
        fd = file
    # Eventually seek to offset,
    if offset != None: fd.seek(int(offset))
    return fd

# based on '/usr/local/lib/python3.6/os.py'
def popen(cmd, mode="rb"):
    if not isinstance(cmd, str):
        raise TypeError("invalid cmd type (%s, expected string)" % type(cmd))

    import subprocess, io, threading

    # cleanup function for subprocesses,
    def cleanup(proc, cmd):
        ret = proc.wait()
        if ret > 0:
            raise SubprocessFailed('cmd %s returned %d !' % (cmd,ret))
        return

    # text-mode,
    if mode == "r":
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=sys.stderr)
        threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
        return io.TextIOWrapper(proc.stdout)
    elif mode == "w":
        proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stderr=sys.stderr)
        threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
        return io.TextIOWrapper(proc.stdin)
    # binary,
    elif mode == "rb":
        proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=sys.stderr)
        threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
        return proc.stdout
    elif mode == "wb":
        proc = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE, stderr=sys.stderr)
        threading.Thread(target=cleanup,args=(proc,cmd)).start() # clean-up thread,
        return proc.stdin
    # sanity,
    else:
        raise ValueError("invalid mode %s" % mode)


def read_key(fd):
    """ [key] = read_key(fd)
     Read the utterance-key from the opened ark/stream descriptor 'fd'.
    """
    assert('b' in fd.mode), "Error: 'fd' was opened in text mode (in python3 use sys.stdin.buffer)"

    key = ''
    while 1:
        char = fd.read(1).decode("latin1")
        if char == '' : break
        if char == ' ' : break
        key += char
    key = key.strip()
    if key == '': return None # end of file,
    assert(re.match('^\S+$',key) != None) # check format (no whitespace!)
    return key


#################################################
# Integer vectors (alignments, ...),

def read_ali_ark(file_or_fd):
    """ Alias to 'read_vec_int_ark()' """
    return read_vec_int_ark(file_or_fd)

def read_vec_int_ark(file_or_fd):
    """ generator(key,vec) = read_vec_int_ark(file_or_fd)
     Create generator of (key,vector<int>) tuples, which reads from the ark file/stream.
     file_or_fd : ark, gzipped ark, pipe or opened file descriptor.

     Read ark to a 'dictionary':
     d = { u:d for u,d in kaldi_io.read_vec_int_ark(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            ali = read_vec_int(fd)
            yield key, ali
            key = read_key(fd)
    finally:
        if fd is not file_or_fd: fd.close()

def read_vec_int(file_or_fd):
    """ [int-vec] = read_vec_int(file_or_fd)
     Read kaldi integer vector, ascii or binary input,
    """
    fd = open_or_fd(file_or_fd)
    binary = fd.read(2).decode()
    if binary == '\0B': # binary flag
        assert(fd.read(1).decode() == '\4'); # int-size
        vec_size = np.frombuffer(fd.read(4), dtype='int32', count=1)[0] # vector dim
        if vec_size == 0:
            return np.array([], dtype='int32')
        # Elements from int32 vector are sored in tuples: (sizeof(int32), value),
        vec = np.frombuffer(fd.read(vec_size*5), dtype=[('size','int8'),('value','int32')], count=vec_size)
        assert(vec[0]['size'] == 4) # int32 size,
        ans = vec[:]['value'] # values are in 2nd column,
    else: # ascii,
        arr = (binary + fd.readline().decode()).strip().split()
        try:
            arr.remove('['); arr.remove(']') # optionally
        except ValueError:
            pass
        ans = np.array(arr, dtype=int)
    if fd is not file_or_fd : fd.close() # cleanup
    return ans

# Writing,
def write_vec_int(file_or_fd, v, key=''):
    """ write_vec_int(f, v, key='')
     Write a binary kaldi integer vector to filename or stream.
     Arguments:
     file_or_fd : filename or opened file descriptor for writing,
     v : the vector to be stored,
     key (optional) : used for writing ark-file, the utterance-id gets written before the vector.

     Example of writing single vector:
     kaldi_io.write_vec_int(filename, vec)

     Example of writing arkfile:
     with open(ark_file,'w') as f:
         for key,vec in dict.iteritems():
             kaldi_io.write_vec_flt(f, vec, key=key)
    """
    assert(isinstance(v, np.ndarray))
    assert(v.dtype == np.int32)
    fd = open_or_fd(file_or_fd, mode='wb')
    if sys.version_info[0] == 3: assert(fd.mode == 'wb')
    try:
        if key != '' : fd.write((key+' ').encode("latin1")) # ark-files have keys (utterance-id),
        fd.write('\0B'.encode()) # we write binary!
        # dim,
        fd.write('\4'.encode()) # int32 type,
        fd.write(struct.pack(np.dtype('int32').char, v.shape[0]))
        # data,
        for i in range(len(v)):
            fd.write('\4'.encode()) # int32 type,
            fd.write(struct.pack(np.dtype('int32').char, v[i])) # binary,
    finally:
        if fd is not file_or_fd : fd.close()


#################################################
# Float vectors (confidences, ivectors, ...),

# Reading,
def read_vec_flt_scp(file_or_fd):
    """ generator(key,mat) = read_vec_flt_scp(file_or_fd)
     Returns generator of (key,vector) tuples, read according to kaldi scp.
     file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

     Iterate the scp:
     for key,vec in kaldi_io.read_vec_flt_scp(file):
         ...

     Read scp to a 'dictionary':
     d = { key:mat for key,mat in kaldi_io.read_mat_scp(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        for line in fd:
            (key,rxfile) = line.decode().split(' ')
            vec = read_vec_flt(rxfile)
            yield key, vec
    finally:
        if fd is not file_or_fd : fd.close()

def read_vec_flt_ark(file_or_fd):
    """ generator(key,vec) = read_vec_flt_ark(file_or_fd)
     Create generator of (key,vector<float>) tuples, reading from an ark file/stream.
     file_or_fd : ark, gzipped ark, pipe or opened file descriptor.

     Read ark to a 'dictionary':
     d = { u:d for u,d in kaldi_io.read_vec_flt_ark(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            ali = read_vec_flt(fd)
            yield key, ali
            key = read_key(fd)
    finally:
        if fd is not file_or_fd : fd.close()

def read_vec_flt(file_or_fd):
    """ [flt-vec] = read_vec_flt(file_or_fd)
     Read kaldi float vector, ascii or binary input,
    """
    fd = open_or_fd(file_or_fd)
    binary = fd.read(2).decode()
    if binary == '\0B': # binary flag
        ans = _read_vec_flt_binary(fd)
    else:    # ascii,
        arr = (binary + fd.readline().decode()).strip().split()
        try:
            arr.remove('['); arr.remove(']') # optionally
        except ValueError:
            pass
        ans = np.array(arr, dtype=float)
    if fd is not file_or_fd : fd.close() # cleanup
    return ans

def _read_vec_flt_binary(fd):
    header = fd.read(3).decode()
    if header == 'FV ' : sample_size = 4 # floats
    elif header == 'DV ' : sample_size = 8 # doubles
    else : raise UnknownVectorHeader("The header contained '%s'" % header)
    assert (sample_size > 0)
    # Dimension,
    assert (fd.read(1).decode() == '\4'); # int-size
    vec_size = np.frombuffer(fd.read(4), dtype='int32', count=1)[0] # vector dim
    if vec_size == 0:
        return np.array([], dtype='float32')
    # Read whole vector,
    buf = fd.read(vec_size * sample_size)
    if sample_size == 4 : ans = np.frombuffer(buf, dtype='float32')
    elif sample_size == 8 : ans = np.frombuffer(buf, dtype='float64')
    else : raise BadSampleSize
    return ans


# Writing,
def write_vec_flt(file_or_fd, v, key=''):
    """ write_vec_flt(f, v, key='')
     Write a binary kaldi vector to filename or stream. Supports 32bit and 64bit floats.
     Arguments:
     file_or_fd : filename or opened file descriptor for writing,
     v : the vector to be stored,
     key (optional) : used for writing ark-file, the utterance-id gets written before the vector.

     Example of writing single vector:
     kaldi_io.write_vec_flt(filename, vec)

     Example of writing arkfile:
     with open(ark_file,'w') as f:
         for key,vec in dict.iteritems():
             kaldi_io.write_vec_flt(f, vec, key=key)
    """
    assert(isinstance(v, np.ndarray))
    fd = open_or_fd(file_or_fd, mode='wb')
    if sys.version_info[0] == 3: assert(fd.mode == 'wb')
    try:
        if key != '' : fd.write((key+' ').encode("latin1")) # ark-files have keys (utterance-id),
        fd.write('\0B'.encode()) # we write binary!
        # Data-type,
        if v.dtype == 'float32': fd.write('FV '.encode())
        elif v.dtype == 'float64': fd.write('DV '.encode())
        else: raise UnsupportedDataType("'%s', please use 'float32' or 'float64'" % v.dtype)
        # Dim,
        fd.write('\04'.encode())
        fd.write(struct.pack(np.dtype('uint32').char, v.shape[0])) # dim
        # Data,
        fd.write(v.tobytes())
    finally:
        if fd is not file_or_fd : fd.close()


#################################################
# Float matrices (features, transformations, ...),

# Reading,
def read_mat_scp(file_or_fd):
    """ generator(key,mat) = read_mat_scp(file_or_fd)
     Returns generator of (key,matrix) tuples, read according to kaldi scp.
     file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

     Iterate the scp:
     for key,mat in kaldi_io.read_mat_scp(file):
         ...

     Read scp to a 'dictionary':
     d = { key:mat for key,mat in kaldi_io.read_mat_scp(file) }

    The scp can also be in a list of the form
    ["AMI_ES2011a_H00_FEE041_0003714_0003915_slice2 tests/data/feats.ark:14913[:,7:13]",
     "AMI_ES2011a_H00_FEE041_0003714_0003915_slice2 tests/data/feats.ark:14913[20:30,7:13]"]

    """
    if isinstance(file_or_fd, list): fd = file_or_fd
    else: fd = open_or_fd(file_or_fd)
    
    try:
        for line in fd:
            
            if isinstance(line, str): (key, rxfile) = line.split(' ')
            else: (key, rxfile) = line.decode().split(' ')
            
            (rxfile, range_slice) = _strip_mat_range(rxfile)

            if range_slice is not None:
                if ( (range_slice[0].step != None) or (len(range_slice)==2 and (range_slice[1].step != None)) ):
                    raise NotImplementedError("Step other than 1 in slices is currently not supported.")
                mat = read_mat(rxfile, range_slice)
            else:
                mat = read_mat(rxfile)
                
            yield key, mat
    finally:
        if fd is not file_or_fd : fd.close()

def read_mat_ark(file_or_fd):
    """ generator(key,mat) = read_mat_ark(file_or_fd)
     Returns generator of (key,matrix) tuples, read from ark file/stream.
     file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

     Iterate the ark:
     for key,mat in kaldi_io.read_mat_ark(file):
         ...

     Read ark to a 'dictionary':
     d = { key:mat for key,mat in kaldi_io.read_mat_ark(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            mat = read_mat(fd)
            yield key, mat
            key = read_key(fd)
    finally:
        if fd is not file_or_fd : fd.close()

def _strip_mat_range(rxfile_with_range):
    """ (stripped_rxfile, range) = _strip_mat_range(rxfile)

    Returns (rxfile, None) if rxfile does not contain a matrix range.
    Otherwise a tuple containing the stripped rxfile and a tuple of slice objects is returned.

    "/some/dir/feats.ark:0" -> ("/some/dir/feats.ark:0", None)
    "/some/dir/feats.ark:0[10:19]" -> ("/some/dir/feats.ark:0", slice(10,19))
    "/some/dir/feats.ark:0[10:19,0:12]" -> ("/some/dir/feats.ark:0", (slice(10,19),slice(0,12)))
    "/some/dir/feats.ark:0[:,0:12]" -> ("/some/dir/feats.ark:0", (slice(None,None),slice(0,12)))

    rxfile: file descriptor for an ark file that optionally contains an offset or/and a matrix range.

    For info see: "Table I/O (with ranges)" in https://kaldi-asr.org/doc/io_tut.html
    """

    # search for the form: ...rxfile...[...range...]
    search_res = re.search('(.+)\[(.+)\]', rxfile_with_range)

    if search_res == None:
        # 'rxfile_with_range' HAD NO RANGE "[...]" !!!
        return (rxfile_with_range, None)

    assert(len(search_res.groups()) == 2)
    rxfile, range_str = search_res.groups() # rxfile = "/some/dir/feats.ark:0", range_str = "10:19,0:12"

    slice_arr = []
    for r in range_str.split(','): # "10:19,0:12" -> ['10:19', '0:12']
        s1, s2 = r.split(':') # ':' -> ['', '']
        i1 = int(s1) if s1 else None
        i2 = int(s2) if s2 else None
        slice_arr.append(slice(i1,i2))

    assert(len(slice_arr) > 0)
    return (rxfile, tuple(slice_arr))



def read_mat(file_or_fd, range_slice=None):
    """ [mat] = read_mat(file_or_fd)
     Reads single kaldi matrix, supports ascii and binary.
     file_or_fd : file, gzipped file, pipe or opened file descriptor.
    """
    fd = open_or_fd(file_or_fd)
    try:
        binary = fd.read(2).decode()
        if binary == '\0B' :
            mat = _read_mat_binary(fd, range_slice)
        else:
            mat = _read_mat_ascii(fd)
            if range_slice is not None: mat = (mat[range_slice]).copy()
            
    finally:
        if fd is not file_or_fd: fd.close()
    return mat


def _read_mat_binary(fd, range_slice=None):
    # Data type
    header = fd.read(3).decode()
    # 'CM', 'CM2', 'CM3' are possible values,
    if header.startswith('CM'): return _read_compressed_mat(fd, header, range_slice)
    elif header == 'FM ': floatX ='float32' # floats
    elif header == 'DM ': floatX = 'float64' # doubles
    else: raise UnknownMatrixHeader("The header contained '%s'" % header)

    # Dimensions
    s1, rows, s2, cols = np.frombuffer(fd.read(10), dtype='int8,int32,int8,int32', count=1)[0]
    return _read_range_slice(fd, rows, cols, floatX, range_slice=range_slice)


def _read_mat_ascii(fd):
    rows = []
    while 1:
        line = fd.readline().decode()
        if (len(line) == 0) : raise BadInputFormat # eof, should not happen!
        if len(line.strip()) == 0 : continue # skip empty line
        arr = line.strip().split()
        if arr[-1] != ']':
            rows.append(np.array(arr,dtype='float32')) # not last line
        else:
            rows.append(np.array(arr[:-1],dtype='float32')) # last line
            mat = np.vstack(rows)
            return mat


def _read_compressed_mat(fd, format, range_slice):
    """ Read a compressed matrix,
        see: https://github.com/kaldi-asr/kaldi/blob/master/src/matrix/compressed-matrix.h
        methods: CompressedMatrix::Read(...), CompressedMatrix::CopyToMat(...),
    """
    assert(format == 'CM ') # The formats CM2, CM3 are not supported...
    
    # Format of header 'struct',
    global_header = np.dtype([('minvalue','float32'),('range','float32'),('num_rows','int32'),('num_cols','int32')]) # member '.format' is not written,
    per_col_header = np.dtype([('percentile_0','uint16'),('percentile_25','uint16'),('percentile_75','uint16'),('percentile_100','uint16')])

    # Read global header,
    globmin, globrange, rows, cols = np.frombuffer(fd.read(16), dtype=global_header, count=1)[0]

    # Standardize range_slice
    if range_slice is None: range_slice = (slice(None,None,None),slice(None,None,None))
    elif len(range_slice) ==1 : range_slice = (range_slice[0],slice(None,None,None))
    
    # The data is structed as [Colheader, ... , Colheader, Data, Data , .... ]
    #                                                 {                     cols                     }{         size                 }
    col_headers = np.frombuffer(fd.read(cols*8), dtype=per_col_header, count=cols)[range_slice[1]]
    col_headers = np.array([np.array([x for x in y]) * globrange * 1.52590218966964e-05 + globmin for y in col_headers], dtype=np.float32)

    # Note that, contrary to standard matrices, the compressed matrices are column-major
    # so we need to flip rows and colums when using the below function for reading.
    data = _read_range_slice(fd, cols, rows, 'uint8', range_slice=(range_slice[1],range_slice[0]))
    
    mat = np.zeros_like(data, dtype='float32')
    p0 = col_headers[:, 0].reshape(-1, 1)
    p25 = col_headers[:, 1].reshape(-1, 1)
    p75 = col_headers[:, 2].reshape(-1, 1)
    p100 = col_headers[:, 3].reshape(-1, 1)

    mask_0_64 = (data <= 64)
    mask_193_255 = (data > 192)
    mask_65_192 = (~(mask_0_64 | mask_193_255))

    mat += (p0    + (p25 - p0) / 64. * data) * mask_0_64.astype(np.float32)
    mat += (p25 + (p75 - p25) / 128. * (data - 64)) * mask_65_192.astype(np.float32)
    mat += (p75 + (p100 - p75) / 63. * (data - 192)) * mask_193_255.astype(np.float32)

    return mat.T # transpose! col-major -> row-major,


def _read_range_slice(fd, rows, cols, dtype, range_slice=None):

    if (dtype == 'float32'): sample_size=4
    elif (dtype == 'float64'): sample_size=8
    elif (dtype == 'uint8'): sample_size=1
    else: raise UnsupportedDataType("Data type was %s" % str(dtype))
    
    # Find the start and end indices etc.
    if range_slice is None: range_slice = (slice(None,None,None),slice(None,None,None))
    #
    start_row    = 0 if range_slice[0].start is None else range_slice[0].start
    end_row      = rows if range_slice[0].stop is None else range_slice[0].stop
    rows_to_read = end_row - start_row
    #
    if (len(range_slice)==2):
        start_col = 0 if range_slice[1].start is None else range_slice[1].start
        end_col   = cols if range_slice[1].stop is None else range_slice[1].stop
    else:
        start_col = 0
        end_col   = cols
        
    # We want to read the data using as few seek as possible. So the procedure will    
    # be different depending on the properties of range_slice    
    if (start_col == 0 and end_col == cols):
        # In this case we can read consequtively
        if fd.seekable():                                           # Comment 1: We only only read slices on seekable input. This should be      
            header_offset = fd.tell()                               # every case except piped input, right? And there is no way that piped 
            fd.seek(header_offset + start_row*cols*sample_size )    # could come with slice information since slice info is provided in scp. 
        else:                                                       
            # In this case the input is pipe and there should be no offset
            assert (start_row ==0), ("Start row is %s but should be 0 for non-seekable data" %str(start_row))
            
        buf = fd.read(rows_to_read * cols * sample_size)
        vec = np.frombuffer(buf, dtype=dtype)
        mat = np.reshape(vec,(rows_to_read,cols))
    else:
        # In this case we need to read at different places
        assert fd.seekable(), ("fd %str(fd)is not seekable" % str(fd) )  # Again, this should not happend for pipes
        header_offset = fd.tell()
        cols_to_read = end_col - start_col
        mat = np.zeros((rows_to_read, cols_to_read), dtype=dtype)
        for r in range(start_row, end_row):
            fd.seek( header_offset + (r*cols + start_col)*sample_size )
            d = fd.read(cols_to_read*sample_size)
            mat[r-start_row,:] = (np.frombuffer(d, dtype=dtype, count=cols_to_read))

    # Comment 2: Currently it is not supported to provide slice info via "read_mat_ark"
    # If we want to extend it so it takes slice info as input i.e.
    # read_mat_ark(file_or_fd, list_of_row_slices, list_of_col_slices) where "list_of_row_slices"
    # contains one slice per key in the ark file we have to add something like this here:
    # # Seek to the next key
    #    fd.seek( header_offset + (end_row*cols + start_col)*sample_size )
    # to make sure that we are at the start of the next key after data reading is done.
            
    return mat

    
# Writing,
def write_mat(file_or_fd, m, key=''):
    """ write_mat(f, m, key='')
    Write a binary kaldi matrix to filename or stream. Supports 32bit and 64bit floats.
    Arguments:
     file_or_fd : filename of opened file descriptor for writing,
     m : the matrix to be stored,
     key (optional) : used for writing ark-file, the utterance-id gets written before the matrix.

     Example of writing single matrix:
     kaldi_io.write_mat(filename, mat)

     Example of writing arkfile:
     with open(ark_file,'w') as f:
         for key,mat in dict.iteritems():
             kaldi_io.write_mat(f, mat, key=key)
    """
    assert(isinstance(m, np.ndarray))
    assert(len(m.shape) == 2), "'m' has to be 2d matrix!"
    fd = open_or_fd(file_or_fd, mode='wb')
    if sys.version_info[0] == 3: assert(fd.mode == 'wb')
    try:
        if key != '' : fd.write((key+' ').encode("latin1")) # ark-files have keys (utterance-id),
        fd.write('\0B'.encode()) # we write binary!
        # Data-type,
        if m.dtype == 'float32': fd.write('FM '.encode())
        elif m.dtype == 'float64': fd.write('DM '.encode())
        else: raise UnsupportedDataType("'%s', please use 'float32' or 'float64'" % m.dtype)
        # Dims,
        fd.write('\04'.encode())
        fd.write(struct.pack(np.dtype('uint32').char, m.shape[0])) # rows
        fd.write('\04'.encode())
        fd.write(struct.pack(np.dtype('uint32').char, m.shape[1])) # cols
        # Data,
        fd.write(m.tobytes())
    finally:
        if fd is not file_or_fd : fd.close()


#################################################
# 'Posterior' kaldi type (posteriors, confusion network, nnet1 training targets, ...)
# Corresponds to: vector<vector<tuple<int,float> > >
# - outer vector: time axis
# - inner vector: records at the time
# - tuple: int = index, float = value
#

def read_cnet_ark(file_or_fd):
    """ Alias of function 'read_post_ark()', 'cnet' = confusion network """
    return read_post_ark(file_or_fd)

def read_post_rxspec(file_):
    """ adaptor to read both 'ark:...' and 'scp:...' inputs of posteriors,
    """
    if file_.startswith("ark:"):
        return read_post_ark(file_)
    elif file_.startswith("scp:"):
        return read_post_scp(file_)
    else:
        print("unsupported intput type: %s" % file_)
        print("it should begint with 'ark:' or 'scp:'")
        sys.exit(1)

def read_post_scp(file_or_fd):
    """ generator(key,post) = read_post_scp(file_or_fd)
     Returns generator of (key,post) tuples, read according to kaldi scp.
     file_or_fd : scp, gzipped scp, pipe or opened file descriptor.

     Iterate the scp:
     for key,post in kaldi_io.read_post_scp(file):
         ...

     Read scp to a 'dictionary':
     d = { key:post for key,post in kaldi_io.read_post_scp(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        for line in fd:
            (key,rxfile) = line.decode().split(' ')
            post = read_post(rxfile)
            yield key, post
    finally:
        if fd is not file_or_fd : fd.close()

def read_post_ark(file_or_fd):
    """ generator(key,vec<vec<int,float>>) = read_post_ark(file)
     Returns generator of (key,posterior) tuples, read from ark file.
     file_or_fd : ark, gzipped ark, pipe or opened file descriptor.

     Iterate the ark:
     for key,post in kaldi_io.read_post_ark(file):
         ...

     Read ark to a 'dictionary':
     d = { key:post for key,post in kaldi_io.read_post_ark(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            post = read_post(fd)
            yield key, post
            key = read_key(fd)
    finally:
        if fd is not file_or_fd: fd.close()

def read_post(file_or_fd):
    """ [post] = read_post(file_or_fd)
     Reads single kaldi 'Posterior' in binary format.

     The 'Posterior' is C++ type 'vector<vector<tuple<int,float> > >',
     the outer-vector is usually time axis, inner-vector are the records
     at given time,    and the tuple is composed of an 'index' (integer)
     and a 'float-value'. The 'float-value' can represent a probability
     or any other numeric value.

     Returns vector of vectors of tuples.
    """
    fd = open_or_fd(file_or_fd)
    ans=[]
    binary = fd.read(2).decode(); assert(binary == '\0B'); # binary flag
    assert(fd.read(1).decode() == '\4'); # int-size
    outer_vec_size = np.frombuffer(fd.read(4), dtype='int32', count=1)[0] # number of frames (or bins)

    # Loop over 'outer-vector',
    for i in range(outer_vec_size):
        assert(fd.read(1).decode() == '\4'); # int-size
        inner_vec_size = np.frombuffer(fd.read(4), dtype='int32', count=1)[0] # number of records for frame (or bin)
        data = np.frombuffer(fd.read(inner_vec_size*10), dtype=[('size_idx','int8'),('idx','int32'),('size_post','int8'),('post','float32')], count=inner_vec_size)
        assert(data[0]['size_idx'] == 4)
        assert(data[0]['size_post'] == 4)
        ans.append(data[['idx','post']].tolist())

    if fd is not file_or_fd: fd.close()
    return ans

def write_post(file_or_fd, p, key=''):
    """ write_post(f, p, key='')
     Write a binary kaldi integer vector to filename or stream.
     Arguments:
     file_or_fd : filename or opened file descriptor for writing,
     p : the posterior to be stored,
     key (optional) : used for writing ark-file, the utterance-id gets written before the posterior.

     Example of writing single vector:
     kaldi_io.write_post(filename, post)

     Example of writing arkfile:
     with open(ark_file,'w') as f:
         for key, post in dict.iteritems():
             kaldi_io.write_post(f, post, key=key)

     Write single kaldi 'Posterior' in binary format.

     The 'Posterior' is C++ type 'vector<vector<tuple<int,float> > >',
     the outer-vector is usually time axis, inner-vector are the records
     at given time,    and the tuple is composed of an 'index' (integer)
     and a 'float-value'. The 'float-value' can represent a probability
     or any other numeric value.

    """
    assert(isinstance(p, list)), str(type(p))
    fd = open_or_fd(file_or_fd, mode='wb')
    if sys.version_info[0] == 3:
        assert(fd.mode == 'wb')
    try:
        if key != '':
            fd.write((key+' ').encode("latin1"))  # ark-files have keys (utterance-id),
        fd.write('\0B'.encode())  # we write binary!
        fd.write('\4'.encode())  # int32 type,
        fd.write(struct.pack(np.dtype('int32').char, len(p)))  # outer vec size
        inner_vec_size = None
        for inner_arr in p:
            inner_arr_len = len(inner_arr)
            inner_vec_size = inner_arr_len if inner_vec_size is None else inner_vec_size
            assert inner_arr_len == inner_vec_size, str((key, inner_arr_len, inner_vec_size))
            fd.write('\4'.encode())  # int32 type,
            fd.write(struct.pack(np.dtype('int32').char, inner_vec_size))  # inner vec size
            for idx, pv in inner_arr:
                fd.write(struct.pack(np.dtype('int8').char, 4))  # size_idx
                fd.write(struct.pack(np.dtype('int32').char, idx))
                fd.write(struct.pack(np.dtype('int8').char, 4))  # size_post
                fd.write(struct.pack(np.dtype('float32').char, pv))
    finally:
        if fd is not file_or_fd:
            fd.close()

#################################################
# Kaldi Confusion Network bin begin/end times,
# (kaldi stores CNs time info separately from the Posterior).
#

def read_cntime_ark(file_or_fd):
    """ generator(key,vec<tuple<float,float>>) = read_cntime_ark(file_or_fd)
     Returns generator of (key,cntime) tuples, read from ark file.
     file_or_fd : file, gzipped file, pipe or opened file descriptor.

     Iterate the ark:
     for key,time in kaldi_io.read_cntime_ark(file):
         ...

     Read ark to a 'dictionary':
     d = { key:time for key,time in kaldi_io.read_post_ark(file) }
    """
    fd = open_or_fd(file_or_fd)
    try:
        key = read_key(fd)
        while key:
            cntime = read_cntime(fd)
            yield key, cntime
            key = read_key(fd)
    finally:
        if fd is not file_or_fd : fd.close()

def read_cntime(file_or_fd):
    """ [cntime] = read_cntime(file_or_fd)
     Reads single kaldi 'Confusion Network time info', in binary format:
     C++ type: vector<tuple<float,float> >.
     (begin/end times of bins at the confusion network).

     Binary layout is '<num-bins> <beg1> <end1> <beg2> <end2> ...'

     file_or_fd : file, gzipped file, pipe or opened file descriptor.

     Returns vector of tuples.
    """
    fd = open_or_fd(file_or_fd)
    binary = fd.read(2).decode(); assert(binary == '\0B'); # assuming it's binary

    assert(fd.read(1).decode() == '\4'); # int-size
    vec_size = np.frombuffer(fd.read(4), dtype='int32', count=1)[0] # number of frames (or bins)

    data = np.frombuffer(fd.read(vec_size*10), dtype=[('size_beg','int8'),('t_beg','float32'),('size_end','int8'),('t_end','float32')], count=vec_size)
    assert(data[0]['size_beg'] == 4)
    assert(data[0]['size_end'] == 4)
    ans = data[['t_beg','t_end']].tolist() # Return vector of tuples (t_beg,t_end),

    if fd is not file_or_fd : fd.close()
    return ans


#################################################
# Segments related,
#

# Segments as 'Bool vectors' can be handy,
# - for 'superposing' the segmentations,
# - for frame-selection in Speaker-ID experiments,
def read_segments_as_bool_vec(segments_file, return_key=False):
    """ Synopsis:
     bool_vec = read_segments_as_bool_vec(segments_file)
     bool_vec, key = read_segments_as_bool_vec(segments_file, return_key=True)

     Loads: kaldi 'segments' file for 1 wav, format : '<utt> <rec> <t-beg> <t-end>'
     - t-beg, t-end is in seconds,
     - assumed 100 frames/second,

     Returns: np.array(dtype=bool), key is string.
    """
    segs = np.loadtxt(segments_file, dtype='object,object,f,f', ndmin=1)
    # Sanity checks,
    assert(len(segs) > 0) # empty segmentation is an error,
    assert(len(np.unique([rec[1] for rec in segs ])) == 1) # segments with only 1 wav-file,
    # Convert time to frame-indexes,
    start = np.rint([100 * rec[2] for rec in segs]).astype(int)
    end = np.rint([100 * rec[3] for rec in segs]).astype(int)
    # Taken from 'read_lab_to_bool_vec', htk.py,
    frms = np.repeat(np.r_[np.tile([False,True], len(end)), False],
                     np.r_[np.c_[start - np.r_[0, end[:-1]], end-start].flat, 0])
    assert np.sum(end-start) == np.sum(frms)

    if return_key:
        key = segs[0][1]
        return frms, key
    else:
        return frms

