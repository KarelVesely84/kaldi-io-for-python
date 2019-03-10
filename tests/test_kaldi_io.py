#!/usr/bin/env python

import unittest
import sys

import numpy as np

import kaldi_io
print("### KALDI_IO_PATH: %s ###" % str(kaldi_io))

class KaldiIoTest(unittest.TestCase):
    def testInt32VectorReadWrite(self):
        """
        Test read/write for int32 vectors.
        """
        # read,
        i32_vec = { k:v for k,v in kaldi_io.read_vec_int_ark('tests/data/ali.ark') } # binary,
        i32_vec2 = { k:v for k,v in kaldi_io.read_vec_int_ark('tests/data/ali_ascii.ark') } # ascii,
        # re-save the data,
        with kaldi_io.open_or_fd('tests/data_re-saved/ali.ark','wb') as f:
            for k,v in i32_vec.items(): kaldi_io.write_vec_int(f, v, k)
        # read and make sure it is the same,
        for k,v in kaldi_io.read_vec_int_ark('tests/data_re-saved/ali.ark'):
            self.assertTrue(np.array_equal(v,i32_vec[k]), msg="int32 vector same after re-saving")

    def testFloatVectorReadWrite(self):
        """
        Test read/write for float vectors.
        """
        # read,
        flt_vec = { k:v for k,v in kaldi_io.read_vec_flt_scp('tests/data/conf.scp') } # scp,
        return

        flt_vec2 = { k:v for k,v in kaldi_io.read_vec_flt_ark('tests/data/conf.ark') } # binary-ark,
        flt_vec3 = { k:v for k,v in kaldi_io.read_vec_flt_ark('tests/data/conf_ascii.ark') } # ascii-ark,
        # store,
        with kaldi_io.open_or_fd('tests/data_re-saved/conf.ark','wb') as f:
            for k,v in flt_vec.items(): kaldi_io.write_vec_flt(f, v, k)
        # read and compare,
        for k,v in kaldi_io.read_vec_flt_ark('tests/data_re-saved/conf.ark'):
            self.assertTrue(np.array_equal(v,flt_vec[k]), msg="flt. vector same after re-saving")

    def testMatrixReadWrite(self):
        """
        Test read/write for float matrices.
        """
        # read,
        flt_mat = { k:m for k,m in kaldi_io.read_mat_scp('tests/data/feats_ascii.scp') } # ascii-scp,
        flt_mat2 = { k:m for k,m in kaldi_io.read_mat_ark('tests/data/feats_ascii.ark') } # ascii-ark,
        flt_mat3 = { k:m for k,m in kaldi_io.read_mat_ark('tests/data/feats.ark') } # ascii-ark,
        # store,
        with kaldi_io.open_or_fd('tests/data_re-saved/mat.ark','wb') as f:
            for k,m in flt_mat3.items(): kaldi_io.write_mat(f, m, k)
        # read and compare,
        for k,m in kaldi_io.read_mat_ark('tests/data_re-saved/mat.ark'):
            self.assertTrue(np.array_equal(m, flt_mat3[k]), msg="flt. matrix same after re-saving")

    def testPipeReadWrite(self):
        """
        Test read/write for pipes.

        Note: make sure the "os.environ['KALDI_ROOT']" in "kaldi_io/kaldi_io.py" is correct.
        """
        # the following line disables 'stderr' forwarding, comment it for DEBUG,
        with open("/dev/null","w") as sys.stderr:
            # read,
            flt_mat4 = { k:m for k,m in kaldi_io.read_mat_ark('ark:copy-feats ark:tests/data/feats.ark ark:- |') }
            # write to pipe,
            with kaldi_io.open_or_fd('ark:| copy-feats ark:- ark:tests/data_re-saved/mat_pipe.ark','wb') as f:
                for k,m in flt_mat4.items(): kaldi_io.write_mat(f, m, k)
            # read it again and compare,
            for k,m in kaldi_io.read_mat_ark('tests/data_re-saved/mat_pipe.ark'):
                self.assertTrue(np.array_equal(m, flt_mat4[k]),"flt. matrix same after read/write via pipe")

            # read some other formats from pipe,
            i32_vec3 = { k:v for k,v in kaldi_io.read_vec_int_ark('ark:copy-int-vector ark:tests/data/ali.ark ark:- |') }
            flt_vec4 = { k:v for k,v in kaldi_io.read_vec_flt_ark('ark:copy-vector ark:tests/data/conf.ark ark:- |') }

    def testReadRawSCP(self):
        """
        Test read for raw scp strings
        """
        correct_vec = kaldi_io.read_mat('tests/data/feats_ascii.ark:39')
        eg_text = 'AMI_ES2011a_H00_FEE041_0003427_0003714 tests/data/feats_ascii.ark:39'
        key, mat = kaldi_io.read_scp_raw(eg_text)
        assert mat.all() == correct_vec.all()


# if stand-alone, run this...
if __name__ == '__main__':
    unittest.main()

