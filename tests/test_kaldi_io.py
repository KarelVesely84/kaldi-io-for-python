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
        flt_mat3 = { k:m for k,m in kaldi_io.read_mat_scp('tests/data/feats.scp') } # scp for binary,
        flt_mat4 = { k:m for k,m in kaldi_io.read_mat_ark('tests/data/feats.ark') } # binary-ark,
        flt_mat5 = { k:m for k,m in kaldi_io.read_mat_scp('tests/data/feats_compressed.scp') } # scp for compressed binary,
        flt_mat6 = { k:m for k,m in kaldi_io.read_mat_ark('tests/data/feats_compressed.ark') } # compressed binary-ark,

        # store,
        with kaldi_io.open_or_fd('tests/data_re-saved/mat.ark','wb') as f:
            for k,m in flt_mat6.items(): kaldi_io.write_mat(f, m, k)
        # read and compare,
        for k,m in kaldi_io.read_mat_ark('tests/data_re-saved/mat.ark'):
            self.assertTrue(np.array_equal(m, flt_mat6[k]), msg="flt. matrix same after re-saving")

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


    def testListScp(self):
        print ("list scp")
        scp = ["AMI_ES2011a_H00_FEE041_0003427_0003714 tests/data/feats.ark:39",
               "AMI_ES2011a_H00_FEE041_0003714_0003915 tests/data/feats.ark:14913",
               "AMI_ES2011a_H00_FEE041_0003714_0003915_slice1 tests/data/feats.ark:14913[:]",
               "AMI_ES2011a_H00_FEE041_0003714_0003915_slice2 tests/data/feats.ark:14913[0:5]",
               "AMI_ES2011a_H00_FEE041_0003714_0003915_slice3 tests/data/feats.ark:14913[:,7:13]",
               "AMI_ES2011a_H00_FEE041_0003714_0003915_slice4 tests/data/feats.ark:14913[20:30,7:13]"]

        flt_mat = { k:m for k,m in kaldi_io.read_mat_scp(scp) } # ascii-scp,

        # store,
        with kaldi_io.open_or_fd('tests/data_re-saved/mat_listScp.ark','wb') as f:
            for k,m in flt_mat.items(): kaldi_io.write_mat(f, m, k)
        # read and compare,
        for k,m in kaldi_io.read_mat_ark('tests/data_re-saved/mat_listScp.ark'):
            self.assertTrue(np.array_equal(m, flt_mat[k]), msg="flt. matrix same after re-saving")

class PosteriorIOTest(unittest.TestCase):
    def testWriteReadPosteriors(self):
        data = [[(0, 0.0), (1, 0.1), (2, 0.2)],
                [(0, 0.00), (1, 0.11), (2, 0.22)],
                [(0, 0.000), (1, 0.111), (3, 0.333)]]
        key = 'posterior_test1'
        with kaldi_io.open_or_fd('tests/data_re-saved/posterior_tests.ark','wb') as w:
            kaldi_io.write_post(w, data, key=key)

        with kaldi_io.open_or_fd('tests/data_re-saved/posterior_tests.ark', 'rb') as r:
            posts = [(k, posteriors) for k, posteriors in kaldi_io.read_post_ark(r)]
            self.assertEqual(len(posts), 1)
            self.assertEqual(posts[0][0], key)
            rdata = posts[0][1]
            self.assertEqual(len(rdata), len(data))
            for a1, a2 in zip(rdata, data):
                self.assertEqual(len(a1), len(a2))
                for ((idx1, p1), (idx, p)) in zip(a1, a2):
                    self.assertEqual(idx1, idx)
                    self.assertAlmostEqual(p1, p)


# if stand-alone, run this...
if __name__ == '__main__':
    unittest.main()

