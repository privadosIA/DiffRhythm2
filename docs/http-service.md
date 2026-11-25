# The HTTP service

`diffrhythm2-service` (project `src/DiffRhythm2.Service`) makes songs over HTTP. It is the same interface as the Python service of the orchestrator (Veritmo), so that application uses it as an **external engine** (a host and a port) with no change on its side, and any other program can use it with `curl` or an HTTP client. Open `http://HOST:PORT/docs` for the interactive page (Swagger UI, bundled, works offline); the machine-readable description is `http://HOST:PORT/openapi.yaml`.

It speaks plain HTTP and is meant for localhost or a trusted network. Put a reverse proxy in front of it if it must be reached from elsewhere, and set a token.

## Start it

```bash
dotnet build -c Release                       # add -p:TorchBackend=cuda-linux for a GPU build
DIFFRHYTHM2_CKPT=~/models/diffrhythm2 \
DIFFRHYTHM2_ESPEAK_LIB=/usr/lib/libespeak-ng.so.1 \
dotnet src/DiffRhythm2.Service/bin/Release/net8.0/diffrhythm2-service.dll
# diffrhythm2 service 1 listening on :8788 (models: /home/me/models/diffrhythm2)
```

The model directory needs the six files of the pipeline; `diffrhythm2 install --model DIR --accept-licenses` builds it ([weights-tools.md](weights-tools.md)). The phonemizer library is found as for the command line ([english-g2p.md](english-g2p.md)).

| variable | default | meaning |
|---|---|---|
| `DIFFRHYTHM2_CKPT` | `/ckpt` | the model directory |
| `DIFFRHYTHM2_TOKEN` | empty | bearer token; empty leaves the API open |
| `DIFFRHYTHM2_PORT` | `8788` | port, on all interfaces |
| `DIFFRHYTHM2_JOBS` | `~/.local/share/diffrhythm2/jobs` (`$XDG_DATA_HOME` if set; the temporary directory when there is no home) | where finished songs are kept (cleaned when the service starts) |
| `DIFFRHYTHM2_VERSION` | `1` | the version string `/health` reports |
| `DIFFRHYTHM2_DEVICE` | `cpu` | `cpu`, `cuda` or `cuda:N` (a CUDA device needs a CUDA build) |
| `DIFFRHYTHM2_ISOLATION` | `process` | `process`: every song in a worker process that gives all its memory back when it ends; `inprocess`: songs inside the service process (see Memory and speed) |
| `DIFFRHYTHM2_ESPEAK_LIB`, `DIFFRHYTHM2_ESPEAK_DATA` | system | the espeak-ng library and its data |

A wrong value stops the service at start with a message that names the variable (exit code 2).

### The start script

`scripts/run-service.sh` starts it for you, from any directory, building it first when it is not built:

```bash
scripts/run-service.sh --model ~/models/diffrhythm2 --port 8788                       # CPU
scripts/run-service.sh --model ~/models/diffrhythm2 --device cuda:0 --token-file ~/.config/diffrhythm2/token
scripts/run-service.sh --help                                                         # every option and variable
```

Every option has the environment variable of the table above (the option wins). A `cuda` device builds the service with the CUDA backend and any other with the CPU one; if the last build used the other backend it builds again (`--build` forces it). `--token-file` reads the token from the first line of a file, so it does not appear in the process list. `--espeak-lib` and `--espeak-data` point to the phonemizer library, and the script adds the library's folder to `LD_LIBRARY_PATH` for its own dependencies. If the model directory lacks files it warns and prints the `diffrhythm2 install` command that fixes it. It runs in the foreground; Ctrl+C or SIGTERM stops it (and cancels a song in progress).

### As a systemd service

`scripts/diffrhythm2-service.service` is a user unit that runs the script and restarts it on failure; its header lists the commands (copy it to `~/.config/systemd/user/`, put the settings in `~/.config/diffrhythm2/service.env`, `systemctl --user enable --now diffrhythm2-service`). Build the service once by hand first, since the first build takes minutes and systemd would give up waiting; `loginctl enable-linger` keeps it running without a login session.

## The calls

All bodies are JSON (UTF-8) except the audio. With a token, add `-H "Authorization: Bearer $TOKEN"` to every call below.

### Is it ready: `GET /health`

```bash
curl -s localhost:8788/health
# {"ready":true,"version":"1","busy":false,"queued":0,"gpu":null,"missing":[]}
```

`ready` is true when the six model files are present, the phonemizer library loads and the device works; otherwise `missing` lists the absent model files. `busy` says whether a song is being made and `queued` how many wait. `gpu` is `null` on CPU; on CUDA it is `{"name", "free_mb", "total_mb"}` read from `nvidia-smi` when it is available (a number that cannot be known is `null`). Health answers at once even while a song is being made.

### Ask for a song: `POST /generate`

```bash
curl -s -X POST localhost:8788/generate -H 'Content-Type: application/json' -d '{
  "lyrics": "[verse]\nHello world tonight\nWe sing along\n[chorus]\nLa la la",
  "style": "Pop, Piano, Female vocals",
  "seed": 1234
}'
# {"id":"3f9a1c07be52"}      (202)
```

| field | meaning | limits | default |
|---|---|---|---|
| `lyrics` | English lyrics, one line per sung line; structure tags `[start] [intro] [verse] [chorus] [bridge] [outro] [inst] [solo] [hook] [break] [stop] [space] [end]` on lines of their own (written in lower case, unknown tags dropped, `[start]` added if missing); at least one line must be text | 20,000 characters | required |
| `style` | comma-separated words, or a list of words | 500 characters | required |
| `max_secs` | the longest the song may be (it ends by itself at the end of the lyrics) | 10 to 210 | 210 |
| `cfg_strength` | guidance strength | 0.5 to 10 | 2.0 |
| `steps` | sampling steps per block (a fraction is truncated) | 4 to 64 | 16 |
| `seed` | seed of the random numbers | whole number, 0 or more | **0** |

A wrong request answers `400` with `{"error": "..."}` naming the field, for example `{"error":"steps must be between 4 and 64"}`, and no job is created. A body larger than 1,000,000 bytes answers `413`.

### Follow it: `GET /jobs/{id}`

```bash
curl -s localhost:8788/jobs/3f9a1c07be52
# {"id":"3f9a1c07be52","state":"running","progress":{"done":3,"total":5},"error":"","seconds":9.0}
```

`state` is `queued`, `running`, `done` or `failed`. `progress` counts blocks of the song sampled (`total` is what the requested length needs; the song may end earlier). `seconds` is the time since the job started running (its total once finished). A failed job has the cause in `error`. Songs are made one at a time, in the order they were accepted.

### Get it: `GET /jobs/{id}/audio`

```bash
curl -s -o song.wav localhost:8788/jobs/3f9a1c07be52/audio
```

A 48 kHz, 16-bit, stereo WAV (`audio/wav`). Before the job is done it answers `409` with `{"error":"the job is running"}`; if the file is no longer there, `410`.

### Forget it: `DELETE /jobs/{id}`

```bash
curl -s -X DELETE localhost:8788/jobs/3f9a1c07be52
# {"ok":true}
```

Removes the job and its audio. If the job is queued it never runs; if it is **running it is cancelled** (its worker process is ended and the next song starts within a second). The service keeps the last 20 finished jobs and forgets older ones.

A client that drops the connection (or aborts a call) is simply no longer served: nothing is logged as an error, and a job it had already submitted is not affected, so you can ask for its audio again.

### Status codes

| code | meaning |
|---|---|
| 200 / 202 | done / accepted |
| 400 | the body or a field is wrong (the error names it) |
| 401 | the service has a token and the call carries none, or another one |
| 404 | unknown job, or a path or method the service does not have |
| 409 | audio asked before the job is done |
| 410 | audio asked but the file is gone |
| 413 | body over 1,000,000 bytes |

## From a program

```python
import json, time, urllib.request

BASE = "http://localhost:8788"

def call(method, path, body=None):
    req = urllib.request.Request(BASE + path, method=method, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})   # add "Authorization": "Bearer ..." with a token
    with urllib.request.urlopen(req) as r:
        return r.read()

job = json.loads(call("POST", "/generate", {"lyrics": "[verse]\nHello world\nWe sing along", "style": "Pop, Piano", "seed": 7}))["id"]
while (state := json.loads(call("GET", f"/jobs/{job}")))["state"] in ("queued", "running"):
    time.sleep(2)
if state["state"] == "done":
    open("song.wav", "wb").write(call("GET", f"/jobs/{job}/audio"))
call("DELETE", f"/jobs/{job}")
```

### As the orchestrator's engine

In the orchestrator's Settings, name the machine and port of this service as the DiffRhythm2 host (and the token, if any). The calls, bodies and status codes are those it already uses. `/health` is what it asks to decide whether the engine is available.

## Seeds and reproducibility

The same request with the same `seed` (default 0) gives the same song on the same machine and device. On CPU in float32 it is the song the reference service makes. On CUDA the network runs in half precision, as the reference does; the same card repeats a song exactly, but different GPU models can diverge on long songs because the sampler amplifies rounding differences.

## Memory and speed

Each song is made by a **worker process** of its own (the service executable started again with `--worker`; it reads the request on its standard input and reports one progress line per block). When the song ends the process ends, so the operating system takes back everything it used, host memory and video memory included. The process that serves the API never loads a model or creates a GPU context, so an idle service is tiny and a song can be stopped at once by ending its worker.

Measured on an RTX 5060 Ti (half precision, a 30-second song):

| moment | video memory in use on the card |
|---|---|
| before the service starts | 13 MiB |
| service up, readiness check answered, no song made | 16 MiB |
| during a song (peak) | 2,479 MiB |
| after the song, idle (twice in a row) | 16 MiB |

On a CPU the idle service uses about 42 MB of memory against a peak of several GB while a song is made. Stopping a running job (`DELETE /jobs/{id}`) ended its worker and started the next song 0.18 s later (3 s in the previous in-process mode, which could only stop between blocks). If the service itself is killed, even with `kill -9`, the worker notices that its input closed, removes its partial file and ends immediately (measured: 0.02 s).

The price is a process start per song: the native libraries are loaded again, on top of the models (which every song already loaded from disk). On the GPU a 30-second song took 10.9 s in-process and 12.7 to 14.1 s with a worker; on a quiet CPU the difference is under a second (timings on a busy machine vary much more than that). If a worker is killed by the system (for example out of memory) the job is `failed` with `the song process ended unexpectedly (exit code N)`, and the service goes on.

`DIFFRHYTHM2_ISOLATION=inprocess` makes the songs inside the service process instead (what earlier versions did): no process start per song, but on CUDA the process then keeps the video memory the allocator reserved (2,479 MiB in the same test) because TorchSharp has no `empty_cache`, and cancelling waits for the end of the block being sampled. A 10-second song takes about 25 s on a CPU (8 steps) and a 60-second song about three minutes; on a GPU it takes seconds. See [song-pipeline.md](song-pipeline.md) for the figures on GPU.

## Documentation endpoints

`GET /docs` (Swagger UI), `GET /openapi.yaml` and the files under `/docs/` need no token. A test keeps `openapi.yaml` true: the same routes, the same status codes, the same defaults and limits as the service applies.
