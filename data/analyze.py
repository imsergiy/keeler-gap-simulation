"""Descriptive calibrated edge-strip sampling; no navigation or physical inversion."""
from pathlib import Path
import argparse,json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parent
from reader import read_vicar
from fetch import verify, products
SRC=ROOT/'cache'


def trace_boundary(a,frame):
    # Search windows follow the visible gap; the descending gradient selects its ring-facing boundary.
    if frame=='2009':
        along=np.arange(280,621); guess=np.full(len(along),559.); axis=0
    else:
        along=np.arange(100,561); guess=690+0.5*along;axis=1
    edge=[]
    for j,g in zip(along,guess):
        lo,hi=int(g)-18,int(g)+19
        v=a[lo:hi,j] if axis==0 else a[j,lo:hi]
        sm=np.convolve(v,np.ones(3)/3,mode='valid')
        edge.append(lo+1+np.argmin(np.diff(sm)))
    return along,np.array(edge),axis

def analyze(frame,name):
    a,info=read_vicar(SRC/name)
    finite=np.isfinite(a);info.update(frame=frame,units='I/F',calibration='CISSCAL 4.0beta stage 10',invalid_nonfinite=int((~finite).sum()),negative_finite=int(((a<0)&finite).sum()),negative_policy='Retained; background subtraction can produce negative calibrated values',quality_mask='Finiteness only; no independent detector quality mask established')
    along,edge,axis=trace_boundary(a,frame)
    excluded=(along>=278)&(along<=286) if frame=='2009' else (along>=413)&(along<=431)
    info['excluded_scan_coordinates']=along[excluded].tolist()
    info['exclusion_reason']='Visible narrow bright streak near x=282' if frame=='2009' else 'Visible Daphnis point source and adjacent trace disruption near y=422'
    along,edge=along[~excluded],edge[~excluded]
    source_metadata=next(p for p in products() if p['files'][0]['name']==name)
    info['source_geometry']=source_metadata['geometry']
    info['source_daphnis_geometry']=source_metadata['daphnis_geometry']
    info['source_association']='Candidate association inferred from date/geometry/morphology; not official cross-reference'
    # Strips remain on the bright ring side of the detected ring-to-gap boundary.
    bins=np.array_split(np.arange(len(along)),6)
    output=[];profile=[];allmasks={}
    for width in [3,5,8]:
      for offset in [18,30,45]:
        em=np.zeros(a.shape,bool);bm=em.copy()
        for t,(j,e) in enumerate(zip(along,edge)):
            if axis==0: em[e-width-2:e-2,j]=True;bm[e-offset-width:e-offset,j]=True
            else: em[j,e-width-2:e-2]=True;bm[j,e-offset-width:e-offset]=True
        allmasks[f'edge_w{width}_o{offset}']=em;allmasks[f'background_w{width}_o{offset}']=bm
        for b,ix in enumerate(bins):
            ev=[];bv=[];er=[];br=[]
            for t in ix:
                j,e=along[t],edge[t]
                v=a[e-width-2:e-2,j] if axis==0 else a[j,e-width-2:e-2]
                w=a[e-offset-width:e-offset,j] if axis==0 else a[j,e-offset-width:e-offset]
                ev.extend(v[np.isfinite(v)]);bv.extend(w[np.isfinite(w)])
                er.append(np.nanmean(v));br.append(np.nanmean(w))
            ratio=np.mean(ev)/np.mean(bv); paired=np.array(er)/np.array(br)
            output.append(dict(frame=frame,bin=b,width_px=width,background_offset_px=offset,along_min=int(along[ix[0]]),along_max=int(along[ix[-1]]),edge_mean_if=float(np.mean(ev)),background_mean_if=float(np.mean(bv)),ratio=float(ratio),edge_n=len(ev),background_n=len(bv),paired_scan_ratio_sd=float(np.std(paired,ddof=1)),paired_scan_ratio_p16=float(np.percentile(paired,16)),paired_scan_ratio_p84=float(np.percentile(paired,84))))
        if width==5 and offset==30:
          for j,e in zip(along,edge):
            for d in range(-50,16):
              y,x=(e+d,j) if axis==0 else (j,e+d)
              profile.append(dict(frame=frame,along_px=int(j),boundary_px=int(e),cross_offset_px=d,x=int(x),y=int(y),if_value=float(a[y,x])))
    np.savez_compressed(ROOT/f'{frame}-masks.npz',**allmasks)
    info.update(along_axis='x' if axis==0 else 'y',trace_rule='3-pixel mean then minimum descending gradient within fixed visible-gap search window',search_guess='y=559 +/-18' if axis==0 else 'x=690+0.5*y +/-18',strip_convention='edge cross offsets [-width-2,-2); background [-offset-width,-offset)',widths_px=[3,5,8],background_offsets_px=[18,30,45],along_bins=6,navigation='native pixel coordinates only; no ring-plane deprojection',calibration_uncertainty=None,sampling_uncertainty='Spatial SD and 16/84 percentiles of paired scan means; correlated spatial samples, not standard error or confidence interval',mask_sensitivity='All 9 width/offset choices retained, not optimized for ratio')
    fig,ax=plt.subplots(figsize=(12,6));lo,hi=np.percentile(a[finite],[1,99]);ax.imshow(a,cmap='gray',vmin=lo,vmax=hi)
    em=allmasks['edge_w5_o30'];bm=allmasks['background_w5_o30'];overlay=np.zeros((*a.shape,4));overlay[em]=[1,0,0,.7];overlay[bm]=[0,.8,1,.7];ax.imshow(overlay)
    for b,ix in enumerate(bins):
      t=ix[len(ix)//2];j,e=along[t],edge[t];ax.text(j if axis==0 else e-60,e-60 if axis==0 else j,str(b),color='yellow')
    if axis==0:ax.set_xlim(250,650);ax.set_ylim(590,480)
    else:ax.set_xlim(680,1000);ax.set_ylim(580,80)
    ax.set_title(f'{frame}: red edge strip, cyan background; native pixels; display stretch only')
    fig.savefig(ROOT/f'{frame}-rois.png',dpi=160);plt.close(fig)
    for file,rows in [(f'{frame}-comparisons.csv',output),(f'{frame}-profiles.csv',profile)]:
      with (ROOT/file).open('w') as f:
       w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    return info,output

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,default=SRC)
    parser.add_argument('--out',type=Path,default=ROOT/'results')
    args=parser.parse_args()
    SRC=args.cache
    ROOT=args.out
    ROOT.mkdir(parents=True,exist_ok=True)
    for product in products():
        for entry in product['files']: verify(SRC/entry['name'],entry)
    results=[]
    for frame,name in [('2009','N1621851000_1_CALIB.IMG'),('2006','N1540685777_1_CALIB.IMG')]:
        info,rows=analyze(frame,name);results.append(info)
        print(frame,'ratio range all bins/choices',min(r['ratio'] for r in rows),max(r['ratio'] for r in rows))
    (ROOT/'analysis-metadata.json').write_text(json.dumps(results,indent=2))
