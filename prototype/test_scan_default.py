"""Independent scalar Python oracle and small-volume exhaustive search."""
import argparse
import importlib.util
import json
import random
import subprocess
import tempfile
from pathlib import Path

spec=importlib.util.spec_from_file_location('evaluation',Path(__file__).with_name('evaluate.py'))
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
ap=argparse.ArgumentParser(__doc__)
ap.add_argument('executable')
exe=str(Path(ap.parse_args().executable).resolve())
rng=random.Random(851)
for i in range(128):
    x=rng.randint(-2**31,2**31-1);z=rng.randint(-2**31,2**31-1)
    y=rng.randint(-2000,2000)
    got=list(map(int,subprocess.check_output([exe,'--vector',str(x),str(y),str(z)]).split()))
    want=[e.rotation(x,y+j,z) for j in range(8)]
    assert got==want,(x,y,z,got,want)
with tempfile.TemporaryDirectory() as tmp:
    for north in range(4):
        anchor=[3,-2,-5]
        blocks=[]
        for i in range(12):
            bx,by,bz=rng.randint(-5,5),rng.randint(-3,3),rng.randint(-5,5)
            xx,zz=e.turn(bx,bz,north)
            r=(e.rotation(anchor[0]+xx,anchor[1]+by,anchor[2]+zz)-north)%4
            blocks.append([bx,by,bz,r,0])
        f=Path(tmp)/'formation.txt'
        f.write_text(str(len(blocks))+'\n'+'\n'.join(' '.join(map(str,b)) for b in blocks)+'\n')
        p=subprocess.run([exe,str(f),'-8','8','-8','8','-8','8','2'],text=True,capture_output=True,check=True)
        actual=sorted(tuple(map(int,s.split())) for s in p.stdout.splitlines())
        expected=[]
        for n in range(4):
            turned=[(*e.turn(bx,bz,n),by,(r+n)%4) for bx,by,bz,r,_ in blocks]
            for x in range(-8,9):
                for y in range(-8,9):
                    for z in range(-8,9):
                        if all(e.rotation(x+xx,y+yy,z+zz)==r for xx,zz,yy,r in turned):
                            expected.append((x,y,z,n))
        assert actual==sorted(expected),(actual,expected)
        assert tuple(anchor+[north]) in actual
print('PASS 1024 SIMD hashes across int32 x/z, negative y; four exhaustive all-north searches match independent Python oracle')
