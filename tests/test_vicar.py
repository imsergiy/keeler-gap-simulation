"""Synthetic binary records isolate layout from archive metadata."""
import hashlib
from pathlib import Path
import tempfile
import unittest
import numpy as np
from data.reader import read_vicar
from data.fetch import verify


class VicarTests(unittest.TestCase):
    def test_binary_header_negative_noise_and_exact_samples(self):
        header = "LBLSIZE=512 FORMAT='REAL' REALFMT='RIEEE' ORG='BSQ' NB=1 NBB=0 RECSIZE=8 EOL=0 NLB=1 NL=2 NS=2 "
        values = np.array([[-.25, 2], [3, np.nan]], dtype='<f4')
        raw = header.encode().ljust(512, b' ')+b'\xff'*8+values.tobytes()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'sample.img'
            path.write_bytes(raw)
            image, info = read_vicar(path)
            np.testing.assert_array_equal(image, values)
            self.assertEqual(info['offset_bytes'], 520)
            self.assertEqual(path.read_bytes(), raw)
            for changed in (raw[:-1], raw+b'\0', raw.replace(b'RIEEE', b'IEEE '), raw.replace(b'NL=2', b'NL=0'), raw.replace(b'NLB=1', b'NLB=-1')):
                path.write_bytes(changed)
                with self.assertRaises(ValueError):
                    read_vicar(path)

    def test_cache_checksum_rejects_changed_bytes(self):
        raw = b'archive bytes'
        entry = dict(name='test.img', byte_length=len(raw), sha256=hashlib.sha256(raw).hexdigest())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/entry['name']
            path.write_bytes(raw)
            verify(path, entry)
            path.write_bytes(b'Archive bytes')
            with self.assertRaises(ValueError):
                verify(path, entry)
