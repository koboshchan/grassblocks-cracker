#!/usr/bin/env python3
"""Static-only development server. Never performs detection or search."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import os, argparse
p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=18271);a=p.parse_args()
root=Path(__file__).resolve().parent
class Static(SimpleHTTPRequestHandler):
 def __init__(self,*args,**kwargs): super().__init__(*args,directory=str(root/'dist'),**kwargs)
 def end_headers(self):
  self.send_header('Cross-Origin-Opener-Policy','same-origin')
  self.send_header('Cross-Origin-Embedder-Policy','require-corp')
  self.send_header('Cross-Origin-Resource-Policy','same-origin')
  self.send_header('X-Content-Type-Options','nosniff')
  super().end_headers()
 def guess_type(self,path):
  if path.endswith('.wasm'):return 'application/wasm'
  if path.endswith('.wgsl'):return 'text/plain'
  return super().guess_type(path)
ThreadingHTTPServer(('0.0.0.0',a.port),Static).serve_forever()
