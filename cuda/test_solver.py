import subprocess, tempfile, pathlib, sys
exe=str(pathlib.Path(sys.argv[1]).resolve())
mask=(1<<64)-1
def signed(v,bits):
 v&=(1<<bits)-1
 return v-(1<<bits) if v>>(bits-1) else v
def texture(x,y,z,side):
 l=(signed(x*3129871,32) ^ (z*116129781) ^ y)&mask
 l=(l*l*42317861+l*11)&mask
 seed=((l>>16)^0x5deece66d)*0x5deece66d+11
 r=(seed&((1<<48)-1))>>46
 return r%2 if side else r
def orient(b,n):
 x,y,z,r,s=b
 for _ in range(n):x,z=-z,x
 return x,y,z,(r+n)% (2 if s else 4),s
with tempfile.TemporaryDirectory() as d:
 p=pathlib.Path(d)/"formation.txt"
 def run(data,args,good=True):
  p.write_text(data)
  c=subprocess.run([exe,"--formation",str(p)]+list(map(str,args)),text=True,capture_output=True)
  assert (c.returncode==0)==good,(args,c.stdout,c.stderr)
  return c
 # Dense output verifies no old 100-result truncation, north transform and confidence parser.
 b=(0,0,0,1,0)
 data="# comment\n1 # count\n0 0 0 1 0 confidence=0.9 trailing metadata\n"
 bounds=[-20,20,-1,1,-20,20]
 c=run(data,["--north","all"]+bounds)
 actual=set(tuple(map(int,row.split())) for row in c.stdout.splitlines())
 expected=set()
 for n in range(4):
  bx,by,bz,r,s=orient(b,n)
  for x in range(-20,21):
   for y in range(-1,2):
    for z in range(-20,21):
     if texture(x+bx,y+by,z+bz,s)==r:expected.add((x,y,z,n))
 assert actual==expected and len(actual)>100,(len(actual),len(expected))
 print("PASS dense/all-north hash oracle",len(actual),"matches")
 for origin in [(3538,90,-454),(-100003,-90,99999),(2147483000,0,-2147483000)]:
  for n in range(4):
   blocks=[]
   for i,(x,y,z,s) in enumerate([(0,0,0,0),(2,1,-3,0),(-4,-2,2,1)]):
    tx,tz=x,z
    for _ in range(n):tx,tz=-tz,tx
    r=texture(origin[0]+tx,origin[1]+y,origin[2]+tz,s)
    blocks.append((x,y,z,(r-n)%(2 if s else 4),s))
   data="3\n"+"\n".join(" ".join(map(str,b)) for b in blocks)+"\n"
   args=["--north",n]+[v for a in origin for v in (a-2,a+2)]
   c=run(data,args)
   actual=set(tuple(map(int,row.split())) for row in c.stdout.splitlines())
   expected=set()
   for x in range(origin[0]-2,origin[0]+3):
    for y in range(origin[1]-2,origin[1]+3):
     for z in range(origin[2]-2,origin[2]+3):
      if all(texture(x+bx,y+by,z+bz,s)==r for bx,by,bz,r,s in [orient(b,n) for b in blocks]):expected.add((x,y,z,n))
   assert actual==expected and (*origin,n) in actual,(origin,n,actual,expected)
 print("PASS mixed-height/side/all-north signed-wrap oracle 12 cases")
 for bad in ["","0\n","2\n0 0 0 0 0\n","1\n0 0 0 4 0\n","1\n0 0 0 0 2\n","1\n0 0 0 2 1\n","1\n0 0 0 0 0\n0 0 0 0 0\n","1\n2147483648 0 0 0 0\n"]:
  run(bad,[0,0,0,0,0,0],False)
 for args in [[1,0,0,0,0,0],["--north",4],[0,1],["--north"],[-2147483648,2147483647,-2147483648,2147483647,-2147483648,2147483647]]:
  run("1\n0 0 0 0 0\n",args,False)
 run("1\n1 0 0 0 0\n",[2147483647,2147483647,0,0,0,0],False)
 print("PASS malformed/empty/bounds/overflow validation")
