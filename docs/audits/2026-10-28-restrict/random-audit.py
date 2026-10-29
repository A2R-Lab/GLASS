import os,subprocess,json
from pathlib import Path
import numpy as np
root=Path('/tmp/glass-random-audit');root.mkdir(exist_ok=True)
results=[]
for bits in (32,64):
 dtype=np.float32 if bits==32 else np.float64
 tol=2e-4 if bits==32 else 1e-10
 for n in (3,7,17):
  for seed in range(10):
   rng=np.random.default_rng(seed+1000*n)
   q,_=np.linalg.qr(rng.normal(size=(n,n)));v,_=np.linalg.qr(rng.normal(size=(n,n)))
   values=np.arange(1,n+1,dtype=float);values[::2]*=-1
   for op in (9,23,24,30):
    A=((q*values)@(q.T if op in (24,30) else v.T)).astype(dtype)
    X=rng.normal(size=(n,2));B=(A@X).astype(dtype)
    A.ravel(order='F').tofile(root/'matrix.bin');B.ravel(order='F').tofile(root/'rhs.bin')
    env=dict(os.environ,AUDIT_MATRIX=str(root/'matrix.bin'),AUDIT_RHS=str(root/'rhs.bin'))
    th=(2,7,33,64)[seed%4]
    try:
     r=subprocess.run(['/tmp/glass-random-restrict',str(op),str(bits),str(n),'0',str(th),'2','0'],env=env,capture_output=True,text=True,timeout=20)
     if r.returncode:raise RuntimeError(r.stderr)
     out=np.fromstring(r.stdout,sep=' ');assert out.size==2*n*n+2*n
     right=out[n*n:2*n*n].reshape(n,n,order='F');sol=out[2*n*n:]
     if op==23:
      err=max(np.linalg.norm(A@right-np.eye(n)),np.linalg.norm(right@A-np.eye(n)))/n
     elif op==30:
      err=max(np.linalg.norm(right.T@right-np.eye(n))/n,np.linalg.norm((right*sol[:n])@right.T-A)/np.linalg.norm(A),np.linalg.norm(np.sort(sol[:n])-np.linalg.eigvalsh(A))/np.linalg.norm(A))
     else:
      cols=2 if op==9 else 1;got=sol[:cols*n].reshape(n,cols,order='F');err=np.linalg.norm(A@got-B[:,:cols])/np.linalg.norm(B[:,:cols])
     assert np.isfinite(err) and err<tol,err
     results.append(dict(op=op,bits=bits,n=n,seed=seed,threads=th,error=float(err),status='pass'))
    except Exception as e:
     np.savez(root/f'failure-{op}-{bits}-{n}-{seed}.npz',A=A,B=B,X=X)
     results.append(dict(op=op,bits=bits,n=n,seed=seed,threads=th,status='fail',error=str(e)))
   (root/'results.json').write_text(json.dumps(results,indent=2)+'\n')
  print('completed',bits,n,'failures',sum(x['status']=='fail' for x in results),flush=True)
print('total',len(results),'failures',[x for x in results if x['status']=='fail'])
