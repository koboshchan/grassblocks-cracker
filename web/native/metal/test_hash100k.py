import pathlib, subprocess, sys, tempfile
exe=str(pathlib.Path(sys.argv[1]).resolve())
def texture(x,y,z):
    xp=(x*3129871)&0xffffffff
    if xp>=0x80000000:xp-=0x100000000
    l=(xp^(z*116129781)^y)&((1<<64)-1)
    l=(l*l*42317861+l*11)&((1<<64)-1)
    return (((((l>>16)^0x5deece66d)*0x5deece66d+11)&((1<<48)-1))>>46)
with tempfile.TemporaryDirectory() as d:
    p=pathlib.Path(d)/"formation.txt";p.write_text("1\n0 0 0 0 0\n")
    for start,y,z in [(-50000,-90,99999),(2147380000,90,-2147483000)]:
        c=subprocess.run([exe,"--formation",str(p),"--north","all","--autotune","off",str(start),str(start+99999),str(y),str(y),str(z),str(z)],text=True,capture_output=True,check=True)
        actual=[tuple(map(int,line.split())) for line in c.stdout.splitlines()]
        expected={(x,y,z,texture(x,y,z)) for x in range(start,start+100000)}
        assert set(actual)==expected and len(actual)==100000,(len(actual),len(expected),c.stderr)
        print("PASS 100000 independently classified hashes, x",start,"to",start+99999,"y",y,"z",z)
