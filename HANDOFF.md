# Swift Detector — Session Handoff

## What is this?

A Swift rewrite of the Python/DeepStream YOLO11n object detector. It runs on NVIDIA Jetson devices (Orin Nano Super) via WendyOS. The goal is to replace the Python `detector/` service with a native Swift binary that does the same thing: read RTSP camera streams, run YOLO11n inference via TensorRT, track objects, and serve metrics + MJPEG over HTTP.

## Architecture

```
RTSP Camera ──► RTSPFrameReader (FFmpeg subprocess) ──► DetectorEngine (TensorRT actor)
                                                              │
                                                    ┌─────────┴──────────┐
                                                    ▼                    ▼
                                              YOLOPreprocessor    YOLOPostprocessor
                                              (letterbox+CHW)    (decode+NMS)
                                                    │                    │
                                                    └────────┬───────────┘
                                                             ▼
                                                       IOUTracker
                                                             │
                                                    ┌────────┴────────┐
                                                    ▼                 ▼
                                              FrameRenderer     Prometheus Metrics
                                              (bbox+JPEG)       (fps, latency, counts)
                                                    │                 │
                                                    └────────┬────────┘
                                                             ▼
                                                    Hummingbird HTTP :9090
                                                    /metrics /stream /health
```

## Key files

| File | Purpose |
|------|---------|
| `Detector.swift` | `@main` entry point. Loads config, starts engine, spawns per-stream detection loops + HTTP server |
| `DetectorEngine.swift` | `actor` wrapping TensorRT. Single-frame `detect()` and batched `detectBatch()` |
| `RTSPFrameReader.swift` | Spawns `ffmpeg` to decode RTSP → raw RGB frames, exposes `AsyncSequence` |
| `YOLOPreprocessor.swift` | Letterbox resize + normalize + HWC→CHW conversion |
| `YOLOPostprocessor.swift` | Decode [84, 8400] output → bounding boxes, per-class NMS |
| `Tracker.swift` | IoU-based multi-object tracker with confirmed/tentative tracks |
| `FrameRenderer.swift` | Draw bounding boxes on RGB frames, encode to JPEG via TurboJPEG |
| `HTTPServer.swift` | Hummingbird routes: `/metrics`, `/stream` (MJPEG), `/health` |
| `Metrics.swift` | Prometheus metric definitions (gauges, counters, histograms) |

## Dependencies

- **tensorrt-swift** (`wendylabsinc/tensorrt-swift`) — Swift bindings for TensorRT inference
- **Hummingbird 2** — HTTP server
- **swift-argument-parser** — CLI argument parsing
- **swift-log** — Logging
- **CFFmpeg** — System library target wrapping FFmpeg (libavformat, libavcodec, etc.)
- **CTurboJPEG** — System library target wrapping libturbojpeg

## What's been done

### Code written (all files above)
The entire Swift detector was written from scratch, mirroring the Python detector's behavior. All source files compile against each other and the dependency APIs.

### Bugs found and fixed during iterative builds

1. **`enqueueF32` returns `Void`, not `Bool`** — The original code treated the TensorRT inference call as returning a Bool and checked `if !inferenceSucceeded`. The actual `tensorrt-swift` API returns Void and throws on failure. Fixed in both `detect()` and `detectBatch()`.

2. **`enqueueF32` is actor-isolated** — The `ExecutionContext.enqueueF32()` method is actor-isolated in tensorrt-swift, so it requires `try await` not just `try`. Fixed in both methods.

3. **`enqueueF32` takes `[Float]` not `UnsafeBufferPointer`** — The original code wrapped arrays in `withUnsafeBufferPointer`/`withUnsafeMutableBufferPointer` closures. The actual API takes `[Float]` and `inout [Float]` directly. Fixed by passing arrays directly with `&outputBuffer`.

### Dockerfile fixes

4. **`libturbojpeg0` → `libturbojpeg`** — Package name changed between Ubuntu 22.04 and 24.04 (runtime stage).

5. **Swift installation method** — Originally used `swiftly` installer which needs `/dev/tty` (fails in Docker). Replaced with direct tarball download:
   ```
   curl -L https://download.swift.org/swift-6.2.3-release/ubuntu2204-aarch64/swift-6.2.3-RELEASE/swift-6.2.3-RELEASE-ubuntu22.04-aarch64.tar.gz \
       | tar xz --strip-components=2 -C /usr
   ```

6. **Swift runtime lib COPY path** — Updated to match the direct install location (`/usr/lib/swift/linux/*.so`).

## What hasn't been done / known issues

### Build never completed
The Docker build was running under QEMU (x86 host emulating ARM64) and was extremely slow — `swift build` under QEMU never finished compiling. **This is why you're here on a Mac.**

### Untested APIs (marked with TODOs in code)
The code was written against the tensorrt-swift documented API, but since the build never completed, we don't know if these are exactly right:

- `TensorRTRuntime()` constructor
- `runtime.deserializeEngine(from:)`
- `runtime.buildEngine(onnxURL:options:)` with `EngineBuildOptions(precision: [.fp16])`
- `engine.makeExecutionContext()`
- `context.enqueueF32(inputName:input:outputName:output:)` — signature was partially verified (Void return, actor-isolated, takes arrays), but full behavior is untested

### No model file in repo
The TensorRT engine file (`model_b2_gpu0_fp16.engine`) and ONNX model (`yolo11n.onnx`) are NOT in this repo. They need to be on the device at `/app/`. The engine file is device-specific (built for a particular GPU), so you may need to provide the ONNX and let it build on first run.

### Device target
- **Device:** `wendyos-warm-pepper.local` (Jetson Orin Nano Super, JetPack 36.4.4)
- **Dockerfile base:** `nvcr.io/nvidia/l4t-jetpack:r36.2.0` (compatible — runtime libs come from host via CDI)

## How to build and deploy from Mac

```bash
git clone git@github.com:mihai-chiorean/swift-detector.git
cd swift-detector

# Deploy to device (Mac ARM64 = native arm64 Docker build, no QEMU)
wendy run --device wendyos-warm-pepper.local --detach --restart-unless-stopped
```

If the build fails with Swift compilation errors, they'll be from the tensorrt-swift API not matching expectations. Check the TODOs in `DetectorEngine.swift` and compare against the actual tensorrt-swift source at `https://github.com/wendylabsinc/tensorrt-swift`.

## Project config

**wendy.json** — entitlements for GPU access and host networking:
```json
{
    "appId": "sh.wendy.examples.deepstream-detector-swift",
    "version": "1.0.0",
    "language": "swift",
    "entitlements": [
        {"type": "gpu"},
        {"type": "network", "mode": "host"}
    ]
}
```

**Package.swift** — requires `swift-tools-version: 6.2` (Swift 6.2.3)

## HTTP endpoints (once running)

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /metrics` | Prometheus metrics |
| `GET /stream` | MJPEG video stream with bounding boxes |

## Context from the parent project

This detector is one of three services in the `deepstream-vision` stack:
- **detector** (this) — YOLO object detection on RTSP streams
- **vlm** — Vision Language Model for scene descriptions
- **gpu-stats** — GPU utilization metrics

The `monitor.html` dashboard in the parent repo connects to all three. The detector serves on port 9090.
