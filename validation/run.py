"""SI normal signed contact reference; run(case, resolution=32) returns JSON data."""
import copy, hashlib, itertools, json, math, pathlib, resource, time
import numpy as np
import scipy
from scipy.integrate import solve_ivp
ROOT=pathlib.Path(__file__).resolve().parent
T=.01
# Declared before execution: 1 ppm of 0.1 m/s; 1 ppm of ~1 J; 0.1 ppm of T.
LIMITS=dict(velocity=1e-7, energy=1e-6, time=1e-9, momentum=1e-7, angular=1e-7)

def coefficients(mi,mj,e):
    mu=mi*mj/(mi+mj); g=-math.log(e)/T
    return mu*((math.pi/T)**2+g*g),2*mu*g

def run(case,resolution=32):
    x=np.array(case['x'],float); v=np.array(case['v'],float)
    r=np.array(case['r']); m=4*np.pi*r**3*500/3; n=len(r)
    if resolution not in (16,32): raise ValueError('resolution must be 16 or 32')
    if x.shape!=(n,3) or v.shape!=(n,3) or n<2 or not np.all(np.isfinite(np.r_[x.ravel(),v.ravel(),r])) or np.any(r<=0) or not 0<case['e']<=1 or case.get('end',.1)<=0:
        raise ValueError('Invalid case input')
    pairs=list(itertools.combinations(range(n),2)); active=set(); events=[]
    if any(np.linalg.norm(x[j]-x[i])<=r[i]+r[j] for i,j in pairs): raise ValueError('Initial states must be strictly separated')
    pars={p:coefficients(m[p[0]],m[p[1]],case['e']) for p in pairs}
    y=np.r_[x.ravel(),v.ravel(),0.]; t=0.; end=case.get('end',.1)
    def unpack(z): return z[:3*n].reshape(n,3),z[3*n:6*n].reshape(n,3)
    def measures(z):
        a,b=unpack(z); U=0.
        for p in pairs:
            i,j=p; d=max(0.,r[i]+r[j]-np.linalg.norm(a[j]-a[i])); U+=.5*pars[p][0]*d*d
        return dict(K=float(.5*np.sum(m[:,None]*b*b)),U=U,Q=float(z[-1]),P=np.sum(m[:,None]*b,axis=0),L=np.sum(np.cross(a,m[:,None]*b),axis=0))
    initial=measures(y); failures=[]; evaluations=0
    while t<end:
        if not active:
            a,b=unpack(y); arrivals=[]
            for p in pairs:
                i,j=p; d=a[j]-a[i]; w=b[j]-b[i]
                aa=float(w@w); bb=float(d@w); cc=float(d@d-(r[i]+r[j])**2)
                disc=bb*bb-aa*cc
                if aa>0 and bb<0 and disc>0:
                    dt=cc/(-bb+math.sqrt(disc))
                    if dt>=-1e-12 and t+max(0.,dt)<=end: arrivals.append((max(0.,dt),p))
            dt=min((z[0] for z in arrivals),default=end-t)
            y[:3*n]+=(b*dt).ravel(); t+=dt
            if not arrivals: break
            changed=[]
            for arrival,p in arrivals:
                if abs(arrival-dt)<=1e-12:
                    active.add(p); changed.append(dict(pair=list(p),kind='entry',gap=0.,normal_speed=None))
            events.append(dict(t=t,changes=changed))
        start=t
        def rhs(_,z):
            a,b=unpack(z); acc=np.zeros_like(a); q=0.
            for p in active:
                i,j=p; d=a[j]-a[i]; dist=np.linalg.norm(d)
                normal=d/dist; overlap=r[i]+r[j]-dist; rate=-np.dot(b[j]-b[i],normal)
                k,c=pars[p]; force=(k*overlap+c*rate)*normal
                acc[i]-=force/m[i]; acc[j]+=force/m[j]; q+=c*rate*rate
            return np.r_[b.ravel(),acc.ravel(),q]
        funcs=[]
        for p in pairs:
            def event(tt,z,p=p):
                a,_=unpack(z); i,j=p
                gap=np.linalg.norm(a[j]-a[i])-r[i]-r[j]
                # At a restart the state is on the root; choose its known side.
                if tt==start and abs(gap)<1e-12: return -1e-14 if p in active else 1e-14
                return gap
            event.direction=1 if p in active else -1; event.terminal=True; funcs.append(event)
        sol=solve_ivp(rhs,(t,end),y,method='DOP853',events=funcs,max_step=T/resolution,rtol=1e-10 if resolution==16 else 1e-12,atol=1e-13 if resolution==16 else 1e-15)
        evaluations+=sol.nfev; t=float(sol.t[-1]); y=sol.y[:,-1]
        if not sol.success: failures.append(sol.message); break
        if sol.status==0: break
        a,b=unpack(y); changed=[]
        triggered={pairs[z] for z,ts in enumerate(sol.t_events) if len(ts)}
        for p in pairs:
            i,j=p; d=a[j]-a[i]; gap=np.linalg.norm(d)-r[i]-r[j]; speed=np.dot(b[j]-b[i],d/np.linalg.norm(d))
            # Group coincident roots only at machine-scale separation.
            if p in triggered or (abs(gap)<1e-12 and ((p in active and speed>0) or (p not in active and speed<0))):
                entering=p not in active
                if entering: active.add(p)
                else: active.remove(p)
                changed.append(dict(pair=list(p),kind='entry' if entering else 'exit',gap=float(gap),normal_speed=float(speed)))
        events.append(dict(t=t,changes=changed))
        if len(events)>100 or t<=start: failures.append('Unresolved event/restart'); break
    final=measures(y); a,b=unpack(y)
    err=dict(energy=abs(final['K']+final['U']+final['Q']-initial['K']-initial['U']),momentum=float(np.linalg.norm(final['P']-initial['P'])),angular=float(np.linalg.norm(final['L']-initial['L'])))
    for key,value in err.items():
        if value>LIMITS[key]: failures.append(key)
    if active: failures.append('Unfinished active contact')
    return dict(case=case,resolution=resolution,t=t,x=a.tolist(),v=b.tolist(),m=m.tolist(),K=final['K'],U=final['U'],Q=final['Q'],errors=err,events=events,failures=failures,nfev=evaluations)

def cases():
    out=[]
    for e in [.5,1.]:
        for phase in [0.,.000137,.000311]:
            for triple in [False,True]:
                d=1.003+.1*phase
                c=dict(name=f'{"triple" if triple else "head"}-e{e}-phase{phase}',e=e,r=[.5]*(3 if triple else 2),x=[[-d,0,0],[0,0,0],[d,0,0]] if triple else [[-d/2,0,0],[d/2,0,0]],v=[[.1,0,0],[0,0,0],[-.1,0,0]] if triple else [[.05,0,0],[-.05,0,0]],oracle='triple' if triple else 'pair',end=.09)
                out.append(c)
    base=dict(name='oblique',e=.5,r=[.5,.5],x=[[-.6,-.3,0],[.6,.3,0]],v=[[1,0,0],[-1,0,0]],end=.3)
    out.append(base)
    unequal=copy.deepcopy(base); unequal.update(name='unequal',r=[.35,.65]); out.append(unequal)
    for offset in [-.00001,.00001,.002]:
        c=copy.deepcopy(out[1]); c.pop('oracle'); c['name']=f'arrival{offset}'; c['x'][2][0]+=.1*offset; out.append(c)
    for impact in [1.,1.000001,.999999]:
        c=copy.deepcopy(base); c.update(name=f'grazing{impact}',x=[[-.1,-impact/2,0],[.1,impact/2,0]],end=.2); out.append(c)
    for transform in ['rotation','boost']:
        c=copy.deepcopy(unequal); c['name']=transform
        if transform=='rotation':
            axis=np.array([1.,2.,3.]); axis/=np.linalg.norm(axis); A=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]]); R=np.eye(3)+math.sin(.73)*A+(1-math.cos(.73))*(A@A); c['x']=(np.array(c['x'])@R.T).tolist(); c['v']=(np.array(c['v'])@R.T).tolist()
        else: c['v']=(np.array(c['v'])+np.array([.3,-.2,.1])).tolist()
        out.append(c)
    repeated=dict(name='repeated-chain',e=1.,r=[.5]*3,x=[[-1.001,0,0],[0,0,0],[1.01,0,0]],v=[[.2,0,0],[0,0,0],[-.1,0,0]],end=.15)
    out.append(repeated)
    return out

def main():
    start=time.perf_counter(); results=[]
    for case in cases():
        for res in [16,32]:
            z=run(case,res)
            if 'oracle' in case:
                m=z['m'][0]; k,c=coefficients(m,m,case['e']); effective=m if case['oracle']=='triple' else m/2
                gamma=c/(2*effective); omega=math.sqrt(k/effective-gamma*gamma); duration=math.pi/omega
                incoming=.1 if case['oracle']=='triple' else .05; outgoing=incoming*math.exp(-gamma*duration)
                expected=np.array(case['v'])*(-outgoing/incoming)
                K=m*outgoing*outgoing; initial=m*incoming*incoming
                durations=[]
                for pair in ([0,1],[1,2]) if case['oracle']=='triple' else ([0,1],):
                    times=[ev['t'] for ev in z['events'] for ch in ev['changes'] if ch['pair']==pair]
                    durations.append(times[1]-times[0] if len(times)==2 else float('inf'))
                z['oracle_errors']=dict(velocity=float(np.max(np.abs(np.array(z['v'])-expected))),K=abs(z['K']-K),Q=abs(z['Q']-(initial-K)),time=max(abs(d-duration) for d in durations))
                for key,error in z['oracle_errors'].items():
                    if error>LIMITS['energy' if key in ['K','Q'] else key]: z['failures'].append('oracle_'+key)
            entries=[ch['pair'] for ev in z['events'] for ch in ev['changes'] if ch['kind']=='entry']
            expected={'repeated-chain':[[0,1],[1,2],[0,1]],'grazing1.0':[],'grazing1.000001':[],'grazing0.999999':[[0,1]]}.get(case['name'])
            if expected is not None and entries!=expected: z['failures'].append('control contact sequence')
            results.append(z)
    comparisons=[]
    for coarse,fine in zip(results[::2],results[1::2]):
        velocity=float(np.max(np.abs(np.array(coarse['v'])-fine['v']))); energy=max(abs(coarse[key]-fine[key]) for key in ['K','Q']); sequence=lambda z:[[(c['pair'],c['kind']) for c in e['changes']] for e in z['events']]
        failures=[]
        if velocity>LIMITS['velocity']: failures.append('velocity convergence')
        if energy>LIMITS['energy']: failures.append('energy partition convergence')
        if sequence(coarse)!=sequence(fine): failures.append('contact sequence convergence')
        comparisons.append(dict(name=coarse['case']['name'],velocity=velocity,energy=energy,failures=failures))
    transforms=[]
    lookup={(z['case']['name'],z['resolution']):z for z in results}
    for resolution in [16,32]:
        ref=lookup['unequal',resolution]
        for name in ['rotation','boost']:
            z=lookup[name,resolution]; weights=np.array(z['m']); vv=np.array(z['v']); rv=np.array(ref['v'])
            if name=='boost': vv-=np.array([.3,-.2,.1])
            else:
                axis=np.array([1.,2.,3.]); axis/=np.linalg.norm(axis); A=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]]); R=np.eye(3)+math.sin(.73)*A+(1-math.cos(.73))*(A@A); vv=vv@R
            velocity=float(np.max(np.abs(vv-rv)))
            com=lambda v: .5*np.sum(weights[:,None]*(v-np.average(v,axis=0,weights=weights))**2)
            energy=max(abs(com(vv)-com(rv)),abs(z['Q']-ref['Q']))
            transforms.append(dict(name=name,resolution=resolution,velocity=velocity,COM_K_or_Q=energy,failures=[] if velocity<=LIMITS['velocity'] and energy<=LIMITS['energy'] else ['frame covariance']))
    summary=dict(cases=len(results)//2,runs=len(results),limits=LIMITS,failures=[dict(name=z['case']['name'],resolution=z['resolution'],failures=z['failures']) for z in results if z['failures']],comparison_failures=[z for z in comparisons if z['failures']],transform_checks=transforms,max_oracle={key:max(z['oracle_errors'][key] for z in results if 'oracle_errors' in z) for key in ['velocity','K','Q','time']},seconds=time.perf_counter()-start,maxrss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,versions=dict(scipy=scipy.__version__,numpy=np.__version__),sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
    (ROOT/'results.json').write_text(json.dumps(dict(results=results,comparisons=comparisons,transforms=transforms),indent=2)); (ROOT/'summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
    if summary['failures'] or summary['comparison_failures'] or any(z['failures'] for z in transforms):
        raise AssertionError('Contact reference checks failed; inspect summary.json')

if __name__=='__main__': main()
