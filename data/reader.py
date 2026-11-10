"""Restricted VICAR reader; returns unmodified native calibrated samples."""
from pathlib import Path
import re, hashlib
import numpy as np

def read_vicar(path):
    raw=Path(path).read_bytes()
    match=re.match(rb'LBLSIZE\s*=\s*(\d+)',raw)
    if not match: raise ValueError('Missing VICAR LBLSIZE')
    size=int(match[1])
    if not 0 < size <= len(raw): raise ValueError('Invalid VICAR label size')
    header=raw[:size].decode('ascii')
    fields=dict(re.findall(r"(\w+)\s*=\s*('(?:[^']*)'|[-+\d.]+)",header))
    required={'LBLSIZE','FORMAT','REALFMT','ORG','NB','NBB','RECSIZE','EOL','NLB','NL','NS'}
    if not required <= fields.keys(): raise ValueError('Incomplete VICAR header')
    def n(k): return int(fields[k])
    def s(k): return fields[k].strip("'")
    if (s('FORMAT'),s('REALFMT'),s('ORG'),n('NB'),n('NBB')) != ('REAL','RIEEE','BSQ',1,0):
        raise ValueError('Unsupported representation')
    if n('RECSIZE') != n('NS')*4 or n('EOL') != 0: raise ValueError('Unsupported records')
    if n('NL') < 1 or n('NS') < 1 or n('NLB') < 0: raise ValueError('Invalid VICAR dimensions')
    offset=size+n('NLB')*n('RECSIZE')
    expected=offset+n('NL')*n('RECSIZE')
    if len(raw)!=expected: raise ValueError('File length mismatch')
    return np.frombuffer(raw,dtype='<f4',offset=offset).reshape(n('NL'),n('NS')).copy(),{'offset_bytes':offset,'file_bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'header_fields':{k:fields[k] for k in ['LBLSIZE','FORMAT','REALFMT','ORG','RECSIZE','NLB','NBB','NL','NS','NB']}}
