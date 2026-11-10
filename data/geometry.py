"""Offline geometric states at two image mid-exposure UTCs; no pixel navigation."""
from pathlib import Path
import hashlib,json,re,math
import numpy as np


def derive(directory=None):
    P=Path(directory) if directory is not None else Path(__file__).with_name('epochs')
    states={}
    manifest=json.loads((P/'queries.json').read_text())
    for entry in manifest:
     raw=(P/(entry['name']+'.json')).read_bytes()
     if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('Horizons response checksum mismatch')
    for name in ['daphnis','cassini','sun']:
     response=json.loads((P/f'{name}.json').read_text())
     text=response['result']
     required=['Center body name: Saturn (699)','Output units    : KM-S','Output type     : GEOMETRIC cartesian states','Reference frame : IAU_SATURN body equator and node of date']
     if not all(token in text for token in required):raise ValueError('Unsupported Horizons frame, units or correction convention')
     rows=text.split('$$SOE')[1].split('$$EOE')[0].strip().splitlines()
     parsed=[]
     for i in range(0,len(rows),3):
      vals=[float(v) for v in re.findall(r'=\s*([-+\d.E]+)', '\n'.join(rows[i+1:i+3]))]
      if len(vals)!=6:raise ValueError('Expected 6 state components')
      parsed.append(dict(epoch_utc=rows[i].split('A.D. ')[1].strip(),position_km=vals[:3],velocity_km_s=vals[3:]))
     if len(parsed)!=2:raise ValueError('Expected the two declared UTC epochs')
     states[name]=parsed
    if not all([row['epoch_utc'] for row in value]==[row['epoch_utc'] for row in states['daphnis']] for value in states.values()):raise ValueError('Mixed Horizons epochs')
    out=[]
    for i in range(2):
     d=np.array(states['daphnis'][i]['position_km']);c=np.array(states['cassini'][i]['position_km']);s=np.array(states['sun'][i]['position_km']);rho=np.linalg.norm(d[:2]);er=np.array([d[0]/rho,d[1]/rho,0]);et=np.array([-er[1],er[0],0]);ez=np.array([0,0,1]);basis=np.array([er,et,ez]);f=np.array([d[0],d[1],0]);
     def unit(v):return v/np.linalg.norm(v)
     def geom(p):
      su=unit(s-p);ob=unit(c-p);sl=basis@su;ol=basis@ob
      return dict(sun_unit_radial_prograde_north=sl.tolist(),observer_unit_radial_prograde_north=ol.tolist(),sun_elevation_deg=math.degrees(math.asin(sl[2])),observer_elevation_deg=math.degrees(math.asin(ol[2])),phase_deg=math.degrees(math.acos(np.dot(su,ob))),observer_range_km=float(np.linalg.norm(c-p)),cos_phase_scalar_check=float(sl@ol))
     out.append(dict(epoch_utc=states['daphnis'][i]['epoch_utc'],states={k:v[i] for k,v in states.items()},rho_km=rho,z_km=d[2],rho_minus_reference_mean136506_km=rho-136506,position_longitude_from_equator_node_deg=math.degrees(math.atan2(d[1],d[0]))%360,velocity_radial_prograde_north_km_s=(basis@np.array(states['daphnis'][i]['velocity_km_s'])).tolist(),basis_rows_radial_prograde_north=basis.tolist(),footpoint_geometry=geom(f),daphnis_center_geometry=geom(d),outer_radius136522_minus_rho_km=136522-rho,control_radius136526_minus_rho_km=136526-rho))
    return dict(reference='IAU_SATURN body equator and node of date; geometric simultaneous UTC states; no light-time or stellar aberration',
                api_response_version='1.2',position_longitude_definition='Cartesian position longitude from equator/ICRF-plane node; not orbital ascending-node longitude',
                gap_radius_source='https://pds-rings.seti.org/saturn/saturn_tables.html',epochs=out)


if __name__=='__main__':
    print(json.dumps(derive(),indent=2))
