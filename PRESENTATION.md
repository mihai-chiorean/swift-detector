# Swift on the Edge: Real-Time Object Detection on NVIDIA Jetson

**Presentation for Swift Meetup**

## Abstract

This talk demos a real-time computer vision example in Swift, running on an NVIDIA Jetson edge device. From RTSP stream ingestion and YOLO object detection with TensorRT, to object tracking, live MJPEG streaming, and Prometheus metrics — we'll see how far Swift can go when you point it at a camera and a GPU.

---

## Presentation Structure (25-30 min + Q&A)

### 1. Hook & Demo First (2-3 minutes)

**[Walk on stage, pull up the /stream endpoint on your laptop, show the live video]**

"This is a Jetson Orin Nano—" **[gesture to the device]** "—running Swift. Not Objective-C. Not Python. Swift.

It's reading an RTSP camera feed, running YOLO11n object detection via TensorRT, tracking objects across frames, and streaming it back over HTTP.

**[Point to the screen]** These bounding boxes? Computed on a GPU. In real-time. With type-safe Swift code using actors to manage the GPU context.

Why am I showing you this? Because six months ago, if you'd told me I'd be doing computer vision in Swift on an NVIDIA GPU, I would've laughed. But here we are. And it actually works pretty well.

Let me show you how."

---

### 2. Why Swift on the Edge? (2-3 minutes)

**The challenge**: Computer vision at the edge needs to be fast, efficient, and maintainable

**Why not Python?**
- Mention the original Python DeepStream setup (complexity, performance ceiling)
- GIL limitations for true parallelism
- Runtime errors vs compile-time safety

**Swift's strengths**:
- Type safety catches bugs at compile time
- Swift Concurrency (actors, async/await) for structured concurrency
- C interop makes GPU/TensorRT accessible
- Compiled performance (no interpreter overhead)

**The controversial bit**: "Swift isn't just for iOS anymore"

---

### 3. The Architecture (5-7 minutes)

**Show the flow diagram** (from HANDOFF.md):

```
RTSP Camera → RTSPFrameReader (FFmpeg subprocess) → DetectorEngine (TensorRT actor)
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

**Key technical highlights:**

**Actor-isolated TensorRT engine**
- Show `DetectorEngine.swift:48-72` (how Swift actors protect GPU context)
- The execution context is NOT thread-safe
- Actor serialization prevents concurrent corruption

**Swift Concurrency**
- `AsyncSequence` for frame streaming from FFmpeg
- Natural backpressure handling

**Zero-copy where possible**
- `UnsafeBufferPointer` for image data
- Direct memory mapping for GPU transfers

---

### 4. The Interesting Problems (5-8 minutes)

**Pick 2-3 technical deep-dives:**

#### Option A: TensorRT Swift Bindings

Show `DetectorEngine.swift:203-208` (the enqueueF32 call):

```swift
try await context.enqueueF32(
    inputName:  "images",
    input:      inputData,
    outputName: "output0",
    output:     &outputBuffer
)
```

- Explain how Swift's C interop made this possible
- The bugs we found and fixed:
  - Ambiguous `Logger` type (TensorRT vs swift-log)
  - `deserializeEngine` expects `Data`, not `String` path
- C++ bindings via module maps

#### Option B: RTSP → RGB Pipeline

Show `RTSPFrameReader.swift`:
- Spawning ffmpeg as subprocess
- `AsyncSequence` for frame streaming
- Memory management for raw RGB frames (3 * width * height bytes per frame)

#### Option C: Batched Inference

Show `detectBatch()` method (`DetectorEngine.swift:234-319`):
- How batching multiple camera streams improves GPU utilization
- Preprocessing all frames in parallel
- Single GPU call for entire batch
- The actor serialization prevents context corruption

#### Option D: Deployment via WendyOS

Show `wendy.json`:
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

- Building ARM64 Docker images from Mac (no QEMU!)
- GPU entitlements and host networking
- The entire stack in a container
- Deploy with: `wendy run --device <device> --detach`

---

### 5. The "Aha Moment" Comparison

**Before & After Stats:**

| Metric | Python/DeepStream | Swift/TensorRT |
|--------|-------------------|----------------|
| Lines of code | ~500-800 | ~600 |
| Build time | Instant (interpreted) | ~3 min (compiled) |
| Startup latency | 5-10s | 2-3s |
| Type safety | Runtime | Compile-time |
| Memory overhead | High (GC) | Lower (ARC) |
| GPU interop | ctypes/SWIG | Direct C interop |
| Concurrency model | Threading/asyncio | Actors + async/await |

**The punchline**: Swift gives you Python-like expressiveness with C++-like control, all with type safety that catches bugs at compile time instead of in production.

---

### 6. What You Learned (3-4 minutes)

**Swift can do systems programming**
- Not just apps and iOS
- Viable for embedded/edge computing

**Actors are perfect for GPU state**
- TensorRT context is thread-unsafe
- Actors serialize access naturally
- No manual locks or mutexes needed

**C interop is powerful**
- FFmpeg, TurboJPEG, TensorRT all accessible
- Module maps make it straightforward
- Performance is native (no FFI overhead)

**Cross-compilation works**
- Mac ARM64 → Jetson ARM64 with Docker
- Native ARM64 build (no QEMU emulation)
- Swift 6.2+ has excellent Linux support

---

### 7. The Demo Moments

Throughout the presentation, cycle back to these visuals:

- **`/stream`** - Live MJPEG with bounding boxes (your money shot)
- **`/metrics`** - Prometheus metrics (FPS, latency, detection counts)
- **`/health`** - Simple liveness check
- **Show the code** - 1-2 key files:
  - `DetectorEngine.swift` (actor + TensorRT)
  - `RTSPFrameReader.swift` (FFmpeg + AsyncSequence)

---

### 8. Closing (2 minutes)

**What's next?**
- YOLO11n → larger models (YOLOv8, YOLO11m)
- Multi-GPU support for multi-camera arrays
- Model zoo integration
- Further optimization (quantization, pruning)

**Open questions**
- How far can Swift go in this space?
- Can we build a Swift ML ecosystem for edge devices?
- What other domains benefit from Swift's strengths?

**Resources**
- GitHub repo: [Your repo URL]
- tensorrt-swift: https://github.com/wendylabsinc/tensorrt-swift
- WendyOS: [Wendy Labs info]

---

## Fallback Plan (if demo fails)

Have these ready:

1. **Pre-recorded video** of the stream working (record this RIGHT NOW once it's running)
2. **Screenshots** of metrics dashboard
3. **Architecture diagram** printout or slide
4. **Code walkthrough** becomes the main content instead of live demo

---

## Browser Tabs to Have Ready

On your presentation laptop:

1. `http://wendyos-valiant-coot.local:9090/stream` - Live MJPEG stream
2. `http://wendyos-valiant-coot.local:9090/metrics` - Prometheus metrics
3. `http://wendyos-valiant-coot.local:9090/health` - Health check
4. GitHub repo open to `DetectorEngine.swift`
5. This presentation outline

---

## Key Takeaways for Audience

1. **Swift is not just for apps** - It's a legitimate systems programming language
2. **Actors solve real problems** - GPU state management is a perfect use case
3. **C interop unlocks ecosystems** - You can use CUDA, TensorRT, FFmpeg, etc.
4. **Edge computing needs better tools** - Swift offers compile-time safety that prevents production failures

---

## Q&A Prep

Expected questions:

**"Why not just use Python?"**
- Python works, but type safety matters for production systems
- GIL limits true parallelism
- Swift's compiled performance gives us headroom

**"What about Rust?"**
- Rust is great! But Swift's async/await and actors are more ergonomic
- Swift has better tooling for rapid development
- The C interop story is simpler in Swift

**"Is this production-ready?"**
- It's a proof-of-concept that works
- tensorrt-swift is early (0.0.4)
- But the fundamentals are solid

**"What's the performance compared to Python?"**
- Haven't done rigorous benchmarking yet
- But inference is GPU-bound, so similar
- Swift's lower overhead helps with frame processing

**"Can I use this on Raspberry Pi?"**
- Not yet - TensorRT requires NVIDIA hardware
- But the pattern works for other accelerators (Core ML, ANE, etc.)

---

## Time Allocations

- **Demo**: 2-3 min
- **Why Swift**: 2-3 min
- **Architecture**: 5-7 min
- **Deep-dives**: 5-8 min
- **Aha Moment**: 2-3 min
- **Lessons**: 3-4 min
- **Closing**: 2 min
- **Q&A**: 5-10 min

**Total**: 25-30 minutes + Q&A
