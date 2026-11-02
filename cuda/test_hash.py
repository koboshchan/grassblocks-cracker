import random,struct,subprocess,sys
rng=random.Random(20261101)
q=[(rng.randint(-2147483648,2147483647),rng.randint(-2147483648,2147483647),rng.randint(-2147483648,2147483647),rng.randrange(2)) for _ in range(100000)]
def expected(x,y,z,side):
 xp=(x*3129871)&0xffffffff
 if xp&0x80000000:xp-=1<<32
 l=(xp^(z*116129781)^y)&((1<<64)-1)
 l=(l*l*42317861+l*11)&((1<<64)-1)
 seed=(((l>>16)^0x5deece66d)*0x5deece66d+11)&((1<<48)-1)
 return (seed>>46)%(2 if side else 4)
payload=struct.pack("<I",len(q))+b"".join(struct.pack("<iiii",*r) for r in q)
r=subprocess.run([sys.argv[1]],input=payload,capture_output=True,check=True)
assert r.stdout==bytes(expected(*v) for v in q),(len(r.stdout),r.stderr)
print("PASS 100000 full-int32 randomized CUDA hashes vs independent Java-semantics Python oracle")
