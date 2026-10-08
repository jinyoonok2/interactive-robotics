"""Download and verify the currently released PartInstruct HDF5 demonstrations."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import requests
import h5py


def main():
    p=argparse.ArgumentParser();p.add_argument('--destination',type=Path,required=True);p.add_argument('--report',type=Path,default=Path(__file__).resolve().parents[2]/'results/data/full_dataset_download/manifest.json');a=p.parse_args()
    token_path=Path.home()/'.cache/huggingface/token'
    token=token_path.read_text().strip()
    session=requests.Session();session.headers['Authorization']='Bearer '+token
    response=session.get('https://huggingface.co/api/datasets/SCAI-JHU/PartInstruct/tree/main/demos',timeout=60);response.raise_for_status()
    listing=[x for x in response.json() if x['path'].endswith('.hdf5')]
    a.destination.joinpath('demos').mkdir(parents=True,exist_ok=True);a.report.parent.mkdir(parents=True,exist_ok=True)
    manifest=dict(status='in_progress',files=[],total_bytes=sum(x['size'] for x in listing))
    def save():a.report.write_text(json.dumps(manifest,indent=2)+'\n')
    save()
    for item in listing:
        path=a.destination/item['path'];expected=item.get('lfs',{}).get('oid','');valid_hash=len(expected)==64 and set(expected)<=set('0123456789abcdef')
        downloaded=False
        if not path.exists():
            temp=path.with_name(path.name+'.part')
            for attempt in range(3):
                try:
                    print('Downloading',item['path'],f"{item['size']/1e9:.2f} GB",flush=True)
                    resp=session.get('https://huggingface.co/datasets/SCAI-JHU/PartInstruct/resolve/main/'+item['path'],stream=True,timeout=(60,120));resp.raise_for_status()
                    with resp, temp.open('wb') as output:
                        for chunk in resp.iter_content(8*1024*1024):output.write(chunk)
                    if temp.stat().st_size!=item['size']:raise RuntimeError('Downloaded size mismatch')
                    path_to_verify=temp;downloaded=True;break
                except Exception:
                    if attempt==2:raise RuntimeError('Download failed for '+item['path']) from None
                    time.sleep(5)
        else:path_to_verify=path
        print('Verifying',item['path'],flush=True)
        assert path_to_verify.stat().st_size==item['size'], 'Size mismatch: '+item['path']
        digest=hashlib.sha256()
        with path_to_verify.open('rb') as f:
            while chunk:=f.read(8*1024*1024):digest.update(chunk)
        actual=digest.hexdigest()
        if valid_hash and actual!=expected:raise RuntimeError('SHA256 mismatch: '+item['path'])
        with h5py.File(path_to_verify,'r') as f:
            assert 'data' in f and len(f['data'])>0,'Missing demonstrations'
            count=len(f['data']);first=next(iter(f['data'].values()));assert 'actions' in first and 'obs' in first
        if downloaded:path_to_verify.replace(path)
        manifest['files'].append(dict(path=item['path'],bytes=item['size'],sha256=actual,release_sha256_verified=valid_hash,demos=count,downloaded=downloaded));save()
        print('Verified',item['path'],count,'demos',flush=True)
    manifest['status']='complete';save();print('All released demonstration files verified.',flush=True)

if __name__=='__main__':main()
