#pragma once
// Host-only deterministic ABI tests. Production uses the real Emscripten header.
#define EMSCRIPTEN_KEEPALIVE
#define EM_JS(ret,name,args,...) ret name args {}
