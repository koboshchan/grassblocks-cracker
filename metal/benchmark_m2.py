import json, pathlib, re, signal, subprocess, time
root=pathlib.Path(__file__).resolve().parent
records=[]
for mode in ("force","auto","off"):
    args=[str(root/"grass_solver"),"--formation",str(root/"fixtures"/"hill22.txt"),"--autotune",mode,
          "--autotune-cache",str(root/"benchmark-autotune.plist"),"-5000","5000","60","128","-5000","5000"]
    start=time.perf_counter()
    process=subprocess.Popen(args,cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    bounded=False
    try:
        out,err=process.communicate(timeout=45)
    except subprocess.TimeoutExpired:
        bounded=True
        process.send_signal(signal.SIGINT)
        try: out,err=process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill();out,err=process.communicate()
    wall=time.perf_counter()-start
    (root/f"BENCH_M2_{mode}.stdout").write_text(out)
    (root/f"BENCH_M2_{mode}.stderr").write_text(err)
    loop=re.search(r"search_loop seconds ([0-9.e+-]+) rate ([0-9.e+-]+)",err)
    total=re.search(r"(complete|cancelled) total (\d+) matches (\d+) seconds ([0-9.e+-]+)",err)
    record=dict(device="Apple M2",mode=mode,args=args,wall_seconds=wall,bounded=bounded,exit_code=process.returncode,
                intended_origins=(10001*10001*69),stdout=out)
    if loop: record.update(search_loop_seconds=float(loop[1]),search_rate=float(loop[2]))
    if total: record.update(status=total[1],tested=int(total[2]),matches=int(total[3]),internal_total_seconds=float(total[4]))
    record["tune"]=[line for line in err.splitlines() if "selected" in line or "cache hit" in line or "autotune off" in line]
    if total: record["wall_rate"]=record["tested"]/wall
    records.append(record)
    (root/"BENCH_M2.json").write_text(json.dumps(records,indent=2)+"\n")
    print(json.dumps(record),flush=True)
    if process.returncode not in (0,130): raise SystemExit("unexpected solver failure")
