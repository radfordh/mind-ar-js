# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MindAR is a web augmented reality library supporting **Image Tracking** and **Face Tracking**. It's written in pure JavaScript with GPU acceleration via TensorFlow.js WebGL backend and uses Web Workers for performance. The library provides integrations for both Three.js and A-Frame.

**Key Technologies:**
- TensorFlow.js (WebGL backend for GPU operations, NOT for machine learning)
- MediaPipe (face mesh detection)
- MessagePack (binary serialization for `.mind` files)
- Vite (build system)

## Build Commands

### Production Build
```bash
npm run build
```
Outputs to `dist/` folder with minified production builds:
- `mindar-image.prod.js` (ES module)
- `mindar-image-three.prod.js` (ES module)
- `mindar-image-aframe.prod.js` (IIFE)
- `mindar-face.prod.js` (ES module)
- `mindar-face-three.prod.js` (ES module)
- `mindar-face-aframe.prod.js` (IIFE)

### Development Builds

**For Three.js development (auto-rebuild on changes):**
```bash
npm run watch
```
Continuously watches `src/` and builds to `dist-dev/` with inline sourcemaps.

**For A-Frame development (manual rebuild):**
```bash
npm run build-dev
```
The `--watch` parameter doesn't work with A-Frame builds due to IIFE format handling. Run this after each change.

**Run dev server with HTTPS (required for camera access):**
```bash
npm run dev
```
Starts Vite dev server with basic SSL on all network interfaces (`--host`).

### Testing with Examples

Examples in `/examples` folder use development builds from `dist-dev/`. Serve them via:
1. Local web server (examples are static HTML)
2. Chrome extension: Web Server for Chrome
3. `npm run dev` for hot reloading

**Camera access requires:**
- HTTPS (dev server includes basic SSL)
- Physical webcam
- Permission granted in browser

**Mobile testing:**
- Set up localhost sharing to mobile devices
- Alternative: Use ngrok for tunneling (slow with 10MB+ dev builds)

## Architecture Overview

### Dual-System Design

The codebase has **two independent tracking systems**:

1. **Image Target Tracking** (`src/image-target/`)
   - Marker-based AR (track 2D images like posters, cards)
   - SIFT-like feature detection with FREAK binary descriptors
   - 6-DOF pose estimation

2. **Face Target Tracking** (`src/face-target/`)
   - Facial landmark detection (468 points)
   - MediaPipe FaceLandmarker integration
   - Blend shapes for facial expressions

### Image Target Pipeline (Critical Path)

**Offline Compilation** (convert user images → `.mind` files):
```
User Image
  → compiler.js (browser Canvas API)
  → compiler-base.js (orchestration)
    → image-list.js (build multi-scale pyramid)
    → detector/detector.js (SIFT-like feature extraction)
      → detector/freak.js (37-point retina-like pattern)
      → detector/kernels/webgl/ (GPU shaders)
    → matching/hierarchical-clustering.js (K-D tree for fast search)
  → compiler.worker.js (tracking feature extraction)
  → msgpack.encode() → .mind file
```

**Runtime Detection** (find marker in video frame):
```
Video Frame
  → input-loader.js (WebGL grayscale conversion)
  → crop-detector.js (detect moving regions)
  → detector/detector.js (extract features)
  → controller.worker.js (Web Worker for heavy computation)
    → matching/matcher.js (try all keyframe scales)
      → matching/matching.js (K-D tree search + Hamming distance)
      → matching/hough.js (geometric voting)
      → matching/ransacHomography.js (robust homography)
    → estimation/estimator.js (DLT algorithm)
      → estimation/estimate.js (homography decomposition)
  → libs/one-euro-filter.js (smooth pose output)
  → three.js or aframe.js (update AR scene)
```

**Continuous Tracking** (after initial detection):
```
Video Frame
  → tracker/tracker.js (template matching, faster than detection)
  → controller.worker.js
    → estimation/refine-estimate.js (ICP refinement)
  → libs/one-euro-filter.js
  → Update AR scene
```

### Key Data Structures

**Feature Point:**
```javascript
{
  maxima: boolean,        // Light vs dark blob
  x, y: float,           // Pixel coordinates
  scale: float,          // Feature size
  angle: float,          // Orientation (-π to π)
  descriptors: [int]     // 167 integers (666 bits packed)
}
```

**Compiled Keyframe:**
```javascript
{
  maximaPoints: [feature],      // Bright features
  minimaPoints: [feature],      // Dark features
  maximaPointsCluster: {rootNode},  // K-D tree
  minimaPointsCluster: {rootNode},
  width, height, scale
}
```

**.mind File Structure:**
```javascript
{
  v: 2,                    // Version
  dataList: [{
    targetImage: {width, height},
    matchingData: [keyframe],   // Multi-scale (3-10 scales)
    trackingData: [...]         // Simplified (2 scales: 256px, 128px)
  }]
}
```

## Critical Implementation Details

### TensorFlow.js as WebGL Engine

**Not used for machine learning!** TensorFlow.js provides a robust WebGL backend for general-purpose GPU computing. Core computer vision algorithms are implemented as custom TensorFlow.js kernels (like shader programs).

**Custom kernels location:** `src/image-target/detector/kernels/`
- `webgl/` - GPU implementations (faster)
- `cpu/` - JavaScript fallback

**Memory management is manual:**
```javascript
tf.tidy(() => {
  // Tensors auto-disposed at end of scope
});
// Or manually:
tensor.dispose();
```

### Web Worker Architecture

**Main Thread:**
- Video capture
- InputLoader (WebGL preprocessing)
- Detector (feature extraction)
- UI updates
- Three.js/A-Frame rendering

**Worker Thread (controller.worker.js):**
- Matcher (feature matching)
- Estimator (pose computation)
- Prevents UI freezing during heavy computation

**Communication:** `postMessage()` with structured data (no SharedArrayBuffer)

### Performance Characteristics

**Detection Mode (initial match):**
- ~30-60 FPS
- Full feature extraction: ~500 features
- Match against all keyframe scales (3-10 scales)
- K-D tree search: O(log n)
- RANSAC iterations: ~100

**Tracking Mode (continuous):**
- ~60+ FPS
- Template matching: small region (~50×50px)
- Fewer features: ~10-50
- ICP iterations: 3-5
- Falls back to detection if lost

### Coordinate Systems

**World Space:** Target image in normalized units (0-1)
**Camera Space:** 3D coordinates in meters
**Screen Space:** 2D pixel coordinates in video frame

**Transformation pipeline:**
```
World (X,Y,0) → [ModelView 4×4] → Camera (x,y,z) → [Projection 4×4] → Screen (u,v)
```

## Module Entry Points

**Image Tracking:**
- Core: `src/image-target/index.js` → exports `{Controller, Compiler, UI}`
- Three.js: `src/image-target/three.js` → exports `MindARThree`
- A-Frame: `src/image-target/aframe.js` → registers `mindar-image` system

**Face Tracking:**
- Core: `src/face-target/index.js` → exports `{Controller, UI}`
- Three.js: `src/face-target/three.js` → exports `MindARThree`
- A-Frame: `src/face-target/aframe.js` → registers `mindar-face` system

## Vite Build System Quirks

**Multi-stage build process:**
1. Build ES modules for Image/Face × Core/Three.js
2. Build IIFE bundles for A-Frame versions separately
3. Rename `.iife.js` → `.js` for A-Frame builds

**Why separate A-Frame builds?**
A-Frame requires IIFE format with global `MINDAR` variable. Sequential builds prevent module format conflicts.

**Watch mode limitation:**
`npm run watch` works for ES modules but NOT for IIFE (A-Frame). The build script's sequential execution breaks Vite's watch mode for the later stages.

## Computer Vision Algorithms

### Feature Detection (SIFT-like)

1. **Gaussian Pyramid:** 5 octaves max, 2 images per octave
2. **DoG (Difference-of-Gaussian):** Approximates Laplacian
3. **Extrema:** Local max/min in 3×3×3 neighborhood
4. **Pruning:** 10×10 grid buckets, keep top 5 per bucket (max 500 features)
5. **Sub-pixel localization:** Quadratic surface fit
6. **Orientation:** 36-bin histogram of gradients
7. **FREAK descriptors:** 37 sampling points, 666 pairwise comparisons

### Feature Matching

1. **K-D tree search:** O(log n) nearest neighbors
2. **Hamming distance:** Count differing bits (XOR + popcount)
3. **Lowe's ratio test:** Reject if best/second-best > 0.7
4. **Hough voting:** Geometric consistency (scale/rotation/position)
5. **RANSAC homography:** Robust estimation, 3px inlier threshold
6. **Two-pass matching:** Second pass uses homography for guided search

### Pose Estimation

1. **DLT (Direct Linear Transform):** Compute homography from 2D-2D matches
2. **Homography decomposition:** Extract rotation + translation
3. **ICP (Iterative Closest Point):** Minimize reprojection error
4. **One-Euro filter:** Temporal smoothing for jitter reduction

## File Organization Patterns

**Detector components:**
- `detector.js` - Main pipeline orchestration
- `freak.js` - Descriptor pattern definition (static data)
- `kernels/` - WebGL and CPU implementations

**Matching components:**
- `matcher.js` - High-level interface
- `matching.js` - Two-pass matching algorithm
- `hierarchical-clustering.js` - K-D tree builder
- `hamming-distance.js` - Binary descriptor comparison
- `hough.js` - Geometric voting
- `ransacHomography.js` - Robust homography

**Estimation components:**
- `estimator.js` - Wrapper class
- `estimate.js` - DLT + homography decomposition
- `refine-estimate.js` - ICP algorithm
- `utils.js` - Matrix operations

## Common Pitfalls

1. **Memory leaks with TensorFlow.js:** Always dispose tensors or use `tf.tidy()`
2. **Worker data transfer:** Serialize carefully, large transfers are slow
3. **A-Frame watch mode:** Doesn't work, must rebuild manually
4. **Camera permissions:** HTTPS required (dev server has basic SSL)
5. **Mobile testing:** Large dev builds (10MB+) are slow over ngrok
6. **`.mind` file version:** Version 2 is current, older versions incompatible

## Algorithm Complexity Reference

- **Compilation:** O(W×H×scales + features×log(features)) per image
- **Detection:** O(W×H + features×log(compiled_features) + RANSAC_iterations)
- **Tracking:** O(template_size + matches×ICP_iterations)
- **K-D tree search:** O(log n) average, O(n) worst case
- **Hamming distance:** O(descriptor_length) = O(167 integers)

## External Dependencies

- **@tensorflow/tfjs:** WebGL backend (NOT for ML)
- **@mediapipe/tasks-vision:** Face mesh model
- **@msgpack/msgpack:** Binary serialization
- **ml-matrix, svd-js, mathjs:** Matrix operations for pose estimation
- **tinyqueue:** Priority queue for K-D tree backtracking
- **canvas (Node.js only):** Offline compilation

## Debugging Tips

**Enable debug mode in detector:**
```javascript
const detector = new Detector(width, height, true); // debugMode=true
```
Returns intermediate results: pyramids, DoG images, extremas, etc.

**Worker thread debugging:**
Workers run in separate context. Use `console.log` in worker files and check browser's worker inspector.

**Performance profiling:**
- Main thread: Chrome DevTools Performance tab
- Worker thread: Separate worker profile in DevTools
- GPU operations: TensorFlow.js profiler: `tf.profile()`

## Documentation References

- Official docs: https://hiukim.github.io/mind-ar-js-doc
- Compiler tool: https://hiukim.github.io/mind-ar-js-doc/tools/compile
- Examples: https://hiukim.github.io/mind-ar-js-doc/examples/summary
- Computer vision concepts borrowed from ARToolKit5
- Face tracking based on MediaPipe Face Mesh

## Additional Resources in This Repository

- `TIER1_ANNOTATED_WALKTHROUGH.md/pdf` - Detailed code walkthroughs of critical files
- `FLOW_DIAGRAMS.md/pdf` - Visual system architecture and data flow diagrams
