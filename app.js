
import { 
  initRenderer,
  setSpectrogramDimensions
} from './webgpu-renderer.js';

const SIGNAL_LENGTH = 8192;
const NUM_BINS = 512;
const CHUNK_SIZE = 256;
const WORKGROUP_SIZE = 64;
const MULTIPLE = 15.0;
const MIN_FREQ = 100.0;
const MAX_FREQ = 12000.0;
const DEFAULT_SR = 48000;
let sampleRate = DEFAULT_SR;
const CIRCLE_RADIUS_NORMALIZED = 0.25;
const CIRCLE_ALPHA = 0.7;

const statusEl = document.getElementById("status");
const statsEl = document.getElementById("stats");
const canvas = document.getElementById("spectrumCanvas");
const singleChunkCheckbox = document.getElementById("singleChunkMode");
const pixelPerfectCheckbox = document.getElementById("pixelPerfectMode");
const suppressLogsCheckbox = document.getElementById("suppressLogs");
let width = canvas.width;
let height = canvas.height;

// Helper function for conditional logging
function logDebug(...args) {
  if (!suppressLogsCheckbox || !suppressLogsCheckbox.checked) {
    console.debug(...args);
  }
}

function logWarn(...args) {
  if (!suppressLogsCheckbox || !suppressLogsCheckbox.checked) {
    console.warn(...args);
  }
}

function logError(...args) {
  // Always show errors even when logs are suppressed
  console.error(...args);
}

function logInfo(...args) {
  if (!suppressLogsCheckbox || !suppressLogsCheckbox.checked) {
    console.log(...args);
  }
}

const circleUniformArray = new Float32Array(8);
let circleVisible = false;

function updateCircleUniform(enable) {
  circleUniformArray[0] = 0.5;
  circleUniformArray[1] = 0.5;
  circleUniformArray[2] = enable ? CIRCLE_RADIUS_NORMALIZED : 0;
  circleUniformArray[3] = 0;
  circleUniformArray[4] = 1.0;
  circleUniformArray[5] = 0.0;
  circleUniformArray[6] = 0.0;
  circleUniformArray[7] = enable ? CIRCLE_ALPHA : 0;
  if (circleUniformBuffer && queue) {
    queue.writeBuffer(circleUniformBuffer, 0, circleUniformArray);
  }
}

// Initialize the renderer with our logging functions and constants
initRenderer({
  logDebug,
  logWarn,
  logError,
  logInfo,
  magnitudeToColor,
  NUM_BINS,
  CHUNK_SIZE,
  WORKGROUP_SIZE
});

// Spectrogram parameters
let SPECTROGRAM_WIDTH = width;
const SPECTROGRAM_HEIGHT = NUM_BINS;
let spectrogramHistory = [];
let MAX_HISTORY = Math.max(1, Math.floor(width)); // One column per pixel
let spectrogramImageData = null;
let spectrogramCanvas = null;
let spectrogramCtx = null;
// Reusable 1-px column buffer to avoid per-pixel drawing
let columnImageData = null;
// rAF scheduling helpers
let drawPending = false;
let pendingSpectrum = null;
let pendingGlobalMax = 1;

function syncCanvasSize() {
  const rect = canvas.getBoundingClientRect();
  const targetWidth = Math.max(1, Math.floor(rect.width));
  const targetHeight = Math.max(1, Math.floor(rect.height));
  if (pixelPerfectCheckbox && pixelPerfectCheckbox.checked) {
    canvas.style.width = `${targetWidth}px`;
    canvas.style.height = `${targetHeight}px`;
  } else {
    canvas.style.width = "";
    canvas.style.height = "";
  }
  if (canvas.width !== targetWidth || canvas.height !== targetHeight) {
    canvas.width = targetWidth;
    canvas.height = targetHeight;
  }
  width = canvas.width;
  height = canvas.height;
  SPECTROGRAM_WIDTH = width;
  MAX_HISTORY = Math.max(1, Math.floor(width));
}

// Initialize spectrogram
function initSpectrogram() {
  spectrogramHistory = [];
  spectrogramCanvas = document.createElement('canvas');
  spectrogramCanvas.width = SPECTROGRAM_WIDTH;
  spectrogramCanvas.height = SPECTROGRAM_HEIGHT;
  spectrogramCtx = spectrogramCanvas.getContext('2d');
  spectrogramImageData = spectrogramCtx.createImageData(SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT);
  // Fill with black
  for (let i = 0; i < spectrogramImageData.data.length; i += 4) {
    spectrogramImageData.data[i] = 0;     // R
    spectrogramImageData.data[i + 1] = 0; // G
    spectrogramImageData.data[i + 2] = 0; // B
    spectrogramImageData.data[i + 3] = 255; // A
  }
  spectrogramCtx.putImageData(spectrogramImageData, 0, 0);

  // Create a reusable 1-px column buffer for fast updates
  columnImageData = spectrogramCtx.createImageData(1, SPECTROGRAM_HEIGHT);
  for (let i = 0; i < columnImageData.data.length; i += 4) {
    columnImageData.data[i] = 0;
    columnImageData.data[i + 1] = 0;
    columnImageData.data[i + 2] = 0;
    columnImageData.data[i + 3] = 255;
  }
}

// Viridis colormap (simplified)
function viridisColor(t) {
  // t should be between 0 and 1
  const t2 = t * t;
  const t3 = t2 * t;
  // Simplified viridis approximation
  const r = Math.min(255, Math.max(0, Math.floor(255 * (0.2627 + t * 1.8811 - t2 * 2.8294 + t3 * 2.4889))));
  const g = Math.min(255, Math.max(0, Math.floor(255 * (0.1949 + t * 2.0312 - t2 * 3.3897 + t3 * 2.8606))));
  const b = Math.min(255, Math.max(0, Math.floor(255 * (0.3484 + t * 3.5947 - t2 * 6.2286 + t3 * 4.6983))));
  return [r, g, b];
}

// Convert magnitude to color using log scaling
function magnitudeToColor(magnitude, maxMagnitude) {
  if (magnitude <= 0) return [0, 0, 0];
  // Log scale for better dynamic range
  const logVal = Math.log10(magnitude + 1) / Math.log10(maxMagnitude + 1);
  return viridisColor(Math.min(1, logVal));
}

let device;
let queue;
let pipeline;
let bindGroup;
// WebGPU canvas renderer resources
let canvasGPU = null;
let presentationFormat = null;
let renderPipeline = null;
let debugGradientPipeline = null;
let circlePipeline = null;
let circleUniformBuffer = null;
let circleBindGroup = null;
let renderBindGroup = null;
let spectrogramTexture = null;
let spectrogramTextureTmp = null;
let spectrogramSampler = null;
// Staging buffer for a single column (padded to bytesPerRow alignment)
let columnBytesPerRow = 0;
let columnBufferSize = 0;
let columnStagingBuffer = null;
let columnCpuView = null;

// Alternative GPU-driven path: storage buffer + compute shader to write columns
let columnStorageBuffer = null;
let columnParamsBuffer = null; // small uniform buffer: [height, width, x]
let columnWritePipeline = null;

// Scroll compute pipeline resources (shifts texture right by 1 and writes new column)
let scrollPipeline = null;
let scrollParamsBuffer = null;
let scrollBindGroup = null;

// Fill compute pipeline for testing
let fillPipeline = null;
let fillParamsBuffer = null;

// Performance: reuse per-frame resources
let columnPacked = null;              // Uint32Array reused per-frame (one u32 per pixel)
let columnBindGroup = null;           // reused bind group for compute shader (not per-frame)
let columnTextureView = null;         // reused texture view for storage binding
let columnDispatchCount = 0;          // precomputed dispatch count (workgroups)
let columnParamArray = new Uint32Array(3); // reuse small array for params
let scrollParamArray = new Uint32Array(2); // reuse small array for scroll params [width,height]
let scrollDispatchX = 0;
let scrollDispatchY = 0;
let computeFallbackWarned = false; // warn only once if compute path unavailable

// Pre-created bind groups for both textures (to avoid allocations at runtime)
let renderBindGroupA = null;
let renderBindGroupB = null;
let columnBindGroupA = null;
let columnBindGroupB = null;
let scrollBindGroupAB = null; // src=A -> dst=B
let scrollBindGroupBA = null; // src=B -> dst=A
let currentMainIndex = 0; // 0 => A is main, 1 => B is main

let chunkBuffer;
let chunkRealBuffer;
let chunkImagBuffer;
let freqBuffer;
let paramsBuffer;
let stagingBuffer;

let adapter;
let audioCtx;
let audioNode;
let mediaStream;
let running = false;

let pendingChunk = new Float32Array(CHUNK_SIZE);
let pendingCount = 0;
const WINDOW_CHUNKS = SIGNAL_LENGTH / CHUNK_SIZE;
let processedChunks = 0;
let totalSamplesCaptured = 0;
const chunkQueue = [];
let queueProcessing = false;
let currentReal = new Float32Array(NUM_BINS);
let currentImag = new Float32Array(NUM_BINS);
const chunkHistory = Array.from({ length: WINDOW_CHUNKS }, () => ({
  real: new Float32Array(NUM_BINS),
  imag: new Float32Array(NUM_BINS),
}));

function getChunkQueueLimit() {
  if (singleChunkCheckbox && singleChunkCheckbox.checked) {
    return 1;
  }
  return Math.max(8, WINDOW_CHUNKS * 2);
}

window.addEventListener("resize", () => {
  syncCanvasSize();
  initSpectrogram();
  resetHistory();
  createSpectrogramResources();
});

const shaderPath = "shaders/loiacono_stream.wgsl";
const fileInput = document.getElementById("audioFileInput");
fileInput.addEventListener("change", handleFileUpload);
document.getElementById("chirpBtn").addEventListener("click", () => {
  autoplayChirp();
});

async function runTestGradient() {
  try {
    await ensureGpu();
    if (!debugGradientPipeline) {
      logError('Debug gradient pipeline not available, cannot run gradient test.');
      return;
    }
    const encoder = device.createCommandEncoder();
    const swapView = canvasGPU.getCurrentTexture().createView();
    const pass = encoder.beginRenderPass({
      colorAttachments: [{ view: swapView, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 1 }, storeOp: 'store' }],
    });
    pass.setPipeline(debugGradientPipeline);
    pass.draw(6);
    pass.end();
    queue.submit([encoder.finish()]);
    logDebug('Submitted gradient test frame');
  } catch (err) {
    logError('Gradient test failed:', err);
  }
}

// Test gradient button: draw a simple UV gradient into the swapchain to validate rendering
document.getElementById("testGradientBtn").addEventListener("click", runTestGradient);

// New button for purely testing the pipeline
document.getElementById("testRenderPipelineBtn").addEventListener("click", runTestGradient);

// Draw Circle button: toggle drawing a red circle via the WebGPU pass
document.getElementById("drawCircleBtn").addEventListener("click", async () => {
  try {
    await ensureGpu();
    if (!circlePipeline || !circleBindGroup) {
      logWarn('Circle pipeline not available');
      return;
    }
    circleVisible = !circleVisible;
    updateCircleUniform(circleVisible);
    logInfo(`WebGPU circle ${circleVisible ? 'enabled' : 'disabled'}`);
  } catch (err) {
    logError('Circle toggle failed:', err);
  }
});

// Add Fill texture button and handler (yellow) for diagnostics
const fillBtnHtml = '<button id="fillYellowBtn">Fill texture (yellow)</button>';
document.getElementById("testGradientBtn").insertAdjacentHTML('afterend', fillBtnHtml);
document.getElementById("fillYellowBtn").addEventListener("click", async () => {
  try {
    await ensureGpu();
    if (!fillPipeline || !fillParamsBuffer) {
      logWarn('Fill pipeline unavailable');
      return;
    }
    const targetTex = currentMainIndex === 0 ? spectrogramTexture : spectrogramTextureTmp;
    const encoder = device.createCommandEncoder();
    // update params width/height
    queue.writeBuffer(fillParamsBuffer, 0, new Uint32Array([SPECTROGRAM_WIDTH, SPECTROGRAM_HEIGHT]));
    const bindGroup = device.createBindGroup({
      layout: fillPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: targetTex.createView() },
        { binding: 1, resource: { buffer: fillParamsBuffer } },
      ],
    });
    const pass = encoder.beginComputePass();
    pass.setPipeline(fillPipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(SPECTROGRAM_WIDTH / 16), Math.ceil(SPECTROGRAM_HEIGHT / 16));
    pass.end();
    queue.submit([encoder.finish()]);
    logDebug('Submitted fill-yellow compute pass');
  } catch (err) {
    logError('Fill yellow failed:', err);
  }
});

singleChunkCheckbox.addEventListener("change", () => {
  chunkQueue.length = 0;
});
pixelPerfectCheckbox.addEventListener("change", () => {
  syncCanvasSize();
  initSpectrogram();
  resetHistory();
  createSpectrogramResources();
});

document.getElementById("startBtn").addEventListener("click", startStreaming);
document.getElementById("stopBtn").addEventListener("click", stopStreaming);

function resetHistory() {
  currentReal.fill(0);
  currentImag.fill(0);
  chunkHistory.forEach((entry) => {
    entry.real.fill(0);
    entry.imag.fill(0);
  });
  processedChunks = 0;
  totalSamplesCaptured = 0;
  chunkQueue.length = 0;
  queueProcessing = false;
  pendingChunk.fill(0);
  pendingCount = 0;
}

async function startStreaming() {
  if (running) {
    return;
  }
  running = true;
  document.getElementById("startBtn").disabled = true;
  document.getElementById("stopBtn").disabled = false;
  statusEl.textContent = "initializing GPU & microphone...";

  resetHistory();
  try {
    await ensureGpu();
    await ensureAudio();
    statusEl.textContent = "listening to microphone…";
    statsEl.textContent = "collecting first chunk...";
  } catch (error) {
    statusEl.textContent = `error: ${error.message}`;
    running = false;
    document.getElementById("startBtn").disabled = false;
    document.getElementById("stopBtn").disabled = true;
  }
}

async function stopStreaming() {
  running = false;
  document.getElementById("startBtn").disabled = false;
  document.getElementById("stopBtn").disabled = true;
  statusEl.textContent = "stopped";
  statsEl.textContent = "idle";
  if (audioNode) {
    audioNode.disconnect();
    // Clear event handlers
    if (audioNode.onaudioprocess) {
      audioNode.onaudioprocess = null;
    }
    if (audioNode.port && audioNode.port.onmessage) {
      audioNode.port.onmessage = null;
    }
    audioNode = null;
  }
  if (audioCtx) {
    audioCtx.close();
    audioCtx = null;
  }
  if (mediaStream) {
    mediaStream.getTracks().forEach((track) => track.stop());
    mediaStream = null;
  }
  pendingCount = 0;
}

async function handleFileUpload(event) {
  const file = event.target?.files?.[0];
  if (!file) {
    return;
  }
  if (running) {
    stopStreaming();
  }
  try {
    await ensureGpu();
    await decodeAndStreamAudio(await file.arrayBuffer(), file.name);
  } catch (error) {
    statusEl.textContent = `file error: ${error.message}`;
    logError(error);
  }
}

function mixToMono(buffer) {
  const channels = buffer.numberOfChannels;
  const length = buffer.length;
  const output = new Float32Array(length);
  for (let c = 0; c < channels; c++) {
    const channelData = buffer.getChannelData(c);
    for (let i = 0; i < length; i++) {
      output[i] += channelData[i];
    }
  }
  if (channels > 1) {
    const factor = 1 / channels;
    for (let i = 0; i < length; i++) {
      output[i] *= factor;
    }
  }
  return output;
}

async function streamUploadedSamples(samples, options = {}) {
  const { singleChunkMode = false } = options;
  statsEl.textContent = `playing uploaded audio (${samples.length} samples)…`;
  let offset = 0;
  while (offset < samples.length) {
    const end = Math.min(samples.length, offset + CHUNK_SIZE);
    feedSamples(samples.subarray(offset, end));
    offset = end;
    if (singleChunkMode) {
      await waitForQueueIdle();
    } else {
      await new Promise((resolve) => setTimeout(resolve, 0));
    }
  }
  if (pendingCount > 0) {
    const padding = new Float32Array(CHUNK_SIZE - pendingCount);
    feedSamples(padding);
  }
  await waitForQueueIdle();
}

async function decodeAndStreamAudio(arrayBuffer, label) {
  statusEl.textContent = `decoding ${label}…`;
  const decodeCtx = new AudioContext();
  const audioBuffer = await decodeCtx.decodeAudioData(arrayBuffer);
  await decodeCtx.close();
  if (audioBuffer.length === 0) {
    throw new Error("decoded buffer is empty");
  }
  if (audioBuffer.sampleRate !== sampleRate) {
    sampleRate = audioBuffer.sampleRate;
    updateFrequencyBuffer(sampleRate);
  }
  resetHistory();
  await streamUploadedSamples(mixToMono(audioBuffer), {
    singleChunkMode: singleChunkCheckbox?.checked,
  });
  statusEl.textContent = `processed ${label}`;
}

async function ensureGpu() {
  if (device) {
    return;
  }
  if (!navigator.gpu) {
    throw new Error("WebGPU is not supported in this browser. Please use Chrome 113+, Edge 113+, or Safari 17+ with WebGPU enabled.");
  }

  let groupOpen = false;
  try {
    if (!suppressLogsCheckbox || !suppressLogsCheckbox.checked) {
      console.groupCollapsed("WebGPU init");
      groupOpen = true;
    }
    logInfo("navigator.gpu available:", !!navigator.gpu);
    adapter = await navigator.gpu.requestAdapter();
    logInfo("requested adapter:", adapter);
    if (!adapter) {
      throw new Error("Failed to request a GPU adapter. Your system may not have a compatible GPU or GPU drivers.");
    }
    device = await adapter.requestDevice();
    logInfo("device acquired");
    queue = device.queue;

    // Configure canvas for WebGPU presentation
    // Modern browsers use 'webgpu' context
    const rect = canvas.getBoundingClientRect();
    logInfo("canvas metrics", { width: canvas.width, height: canvas.height, rect });
    canvasGPU = canvas.getContext('webgpu');
    logInfo("webgpu context", canvasGPU);
    if (!canvasGPU) {
      // Fallback for older implementations (though unlikely to work)
      canvasGPU = canvas.getContext('gpupresent');
      logInfo("gpupresent context", canvasGPU);
    }
    if (!canvasGPU) {
      throw new Error("Failed to acquire a WebGPU canvas context. The canvas element may not be ready or WebGPU canvas support is not available.");
    }
    presentationFormat = navigator.gpu.getPreferredCanvasFormat();
    canvasGPU.configure({
      device,
      format: presentationFormat,
      alphaMode: 'opaque',
      usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC
    });

    syncCanvasSize();
    initSpectrogram(); // Initialize spectrogram (CPU-side structures)
    await setupPipeline();
    // Create GPU-side spectrogram resources (texture / staging buffer / bind group)
    createSpectrogramResources();

    logInfo("WebGPU initialized successfully");
  } catch (error) {
    // Reset state so future calls will retry
    device = null;
    adapter = null;
    canvasGPU = null;
    throw error;
  } finally {
    if (groupOpen) {
      console.groupEnd();
    }
  }
}

async function setupPipeline() {
  const shaderReq = await fetch(shaderPath);
  if (!shaderReq.ok) {
    throw new Error(`Failed to load shader: ${shaderPath} (${shaderReq.status})`);
  }
  const shaderText = await shaderReq.text();
  const module = device.createShaderModule({ code: shaderText });
  pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

  chunkBuffer = device.createBuffer({
    size: CHUNK_SIZE * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  chunkRealBuffer = device.createBuffer({
    size: NUM_BINS * 4,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  chunkImagBuffer = device.createBuffer({
    size: NUM_BINS * 4,
    usage:
      GPUBufferUsage.STORAGE |
      GPUBufferUsage.COPY_SRC |
      GPUBufferUsage.COPY_DST,
  });
  freqBuffer = device.createBuffer({
    size: NUM_BINS * 4,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });
  paramsBuffer = device.createBuffer({
    size: 16,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  stagingBuffer = device.createBuffer({
    size: NUM_BINS * 4 * 2,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const freqs = buildFrequencyArray(sampleRate);
  queue.writeBuffer(freqBuffer, 0, freqs.buffer, freqs.byteOffset, freqs.byteLength);

  bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: chunkBuffer } },
      { binding: 1, resource: { buffer: chunkRealBuffer } },
      { binding: 2, resource: { buffer: chunkImagBuffer } },
      { binding: 3, resource: { buffer: freqBuffer } },
      { binding: 4, resource: { buffer: paramsBuffer } },
    ],
  });

  // Create a small textured fullscreen render pipeline to draw the spectrogram texture
  const vsCode = `
        struct VSOut {
          @builtin(position) position : vec4<f32>,
          @location(0) uv : vec2<f32>
        };
        @vertex
        fn vs(@builtin(vertex_index) v : u32) -> VSOut {
          var pos = array<vec2<f32>, 6>(
            vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
            vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0)
          );
          var uv = array<vec2<f32>, 6>(
            vec2<f32>(0.0, 0.0), vec2<f32>(1.0, 0.0), vec2<f32>(0.0, 1.0),
            vec2<f32>(0.0, 1.0), vec2<f32>(1.0, 0.0), vec2<f32>(1.0, 1.0)
          );
          var result : VSOut;
          result.position = vec4<f32>(pos[v].x, pos[v].y, 0.0, 1.0);
          result.uv = uv[v];
          return result;
        }
      `;
  const fsCode = `
        @group(0) @binding(0) var samp : sampler;
        @group(0) @binding(1) var tex : texture_2d<f32>;
        @fragment
        fn fs(@builtin(position) fragCoord: vec4<f32>, @location(0) texUv : vec2<f32>) -> @location(0) vec4<f32> {
          // Flip Y to match texture coordinate where 0 is top
          let color = textureSample(tex, samp, vec2<f32>(texUv.x, 1.0 - texUv.y));
          return color;
        }
      `;
  const vsModule = device.createShaderModule({ code: vsCode });
  const fsModule = device.createShaderModule({ code: fsCode });

  try {
    device.pushErrorScope('validation');
    renderPipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: vsModule, entryPoint: 'vs' },
      fragment: {
        module: fsModule,
        entryPoint: 'fs',
        targets: [{ format: presentationFormat }],
      },
      primitive: { topology: 'triangle-list' },
    });
    const renderPipelineError = await device.popErrorScope();
    if (renderPipelineError) {
      logError('Render pipeline creation failed:', renderPipelineError);
    }
  } catch (err) {
    logError('Render pipeline creation failed with exception:', err);
  }


  const circleFsCode = `
        struct Params {
          center : vec2<f32>,
          radius : f32,
          padding : f32,
          color : vec4<f32>,
        };
        @group(0) @binding(0) var<uniform> params : Params;

        @fragment
        fn fs(@builtin(position) fragCoord: vec4<f32>, @location(0) circleUv : vec2<f32>) -> @location(0) vec4<f32> {
          return params.color;
        }
      `;
  const circleModule = device.createShaderModule({ code: circleFsCode });
  try {
    device.pushErrorScope('validation');
    circlePipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: vsModule, entryPoint: 'vs' },
      fragment: {
        module: circleModule,
        entryPoint: 'fs',
        targets: [
          {
            format: presentationFormat,
            blend: {
              color: {
                srcFactor: 'src-alpha',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add',
              },
              alpha: {
                srcFactor: 'one',
                dstFactor: 'one-minus-src-alpha',
                operation: 'add',
              },
            },
          },
        ],
      },
      primitive: { topology: 'triangle-list' },
    });
    const circlePipelineError = await device.popErrorScope();
    if (circlePipelineError) {
      logError('Circle pipeline creation failed (from popErrorScope):', circlePipelineError);
      circlePipeline = null; // Ensure pipeline is null if creation failed
    }
    circleUniformBuffer = device.createBuffer({
      size: circleUniformArray.byteLength,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    // Ensure pipeline is valid before trying to get layout
    if (circlePipeline) {
        circleBindGroup = device.createBindGroup({
            layout: circlePipeline.getBindGroupLayout(0),
            entries: [
                {
                    binding: 0,
                    resource: { buffer: circleUniformBuffer },
                },
            ],
        });
    } else {
        logError('Cannot create circleBindGroup: circlePipeline is null.');
    }
    updateCircleUniform(circleVisible);
    logDebug('Circle pipeline created');
  } catch (err) {
    logError('Circle pipeline creation failed with exception:', err ? err.message || err : 'unknown error');
    circlePipeline = null;
    circleUniformBuffer = null;
    circleBindGroup = null;
  }

  const gradFsCode = `
        @fragment
        fn fs(@builtin(position) fragCoord: vec4<f32>, @location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
          let color = vec3<f32>(uv.x, uv.y, 1.0 - uv.x);
          return vec4<f32>(color, 1.0);
        }
      `;
  const gradFsModule = device.createShaderModule({ code: gradFsCode });

  try {
    device.pushErrorScope('validation');
    debugGradientPipeline = device.createRenderPipeline({
      layout: 'auto',
      vertex: { module: vsModule, entryPoint: 'vs' },
      fragment: { module: gradFsModule, entryPoint: 'fs', targets: [{ format: presentationFormat }] },
      primitive: { topology: 'triangle-list' },
    });
    const debugGradientPipelineError = await device.popErrorScope();
    if (debugGradientPipelineError) {
      logError('Debug gradient pipeline creation failed:', debugGradientPipelineError);
      debugGradientPipeline = null;
    } else {
      logDebug('Debug gradient pipeline created');
    }
  } catch (err) {
    logError('Debug gradient pipeline creation failed with exception:', err ? err.message || err : 'unknown error');
    debugGradientPipeline = null;
  }

  // Compute shader that writes a single column into the spectrogram texture from a storage buffer
  const columnCs = `
        struct Params { height : u32, width : u32, x : u32 };
        @group(0) @binding(0) var outTex : texture_storage_2d<rgba8unorm, write>;
        @group(0) @binding(1) var<storage, read> column : array<u32>;
        @group(0) @binding(2) var<uniform> params : Params;

        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
          let y = gid.x;
          if (y >= params.height) { return; }
          let packed = column[y];
          let r = f32((packed & 0xFFu)) / 255.0;
          let g = f32(((packed >> 8u) & 0xFFu)) / 255.0;
          let b = f32(((packed >> 16u) & 0xFFu)) / 255.0;
          textureStore(outTex, vec2<i32>(i32(params.x), i32(y)), vec4<f32>(r, g, b, 1.0));
        }
      `;

  // Compute shader: shift the spectrogram right by 1 and write the new column into x=0 in a single pass
  const scrollCs = `
        struct Params { width : u32, height : u32 };
        @group(0) @binding(0) var srcTex : texture_2d<f32>;
        @group(0) @binding(1) var dstTex : texture_storage_2d<rgba8unorm, write>;
        @group(0) @binding(2) var<storage, read> column : array<u32>;
        @group(0) @binding(3) var<uniform> params : Params;

        fn unpack_u32_to_color(p : u32) -> vec4<f32> {
          let r = f32((p & 0xFFu)) / 255.0;
          let g = f32(((p >> 8u) & 0xFFu)) / 255.0;
          let b = f32(((p >> 16u) & 0xFFu)) / 255.0;
          return vec4<f32>(r, g, b, 1.0);
        }

        @compute @workgroup_size(16, 16)
        fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
          let x = gid.x;
          let y = gid.y;
          if (x >= params.width || y >= params.height) { return; }

          if (x == 0u) {
            // write new incoming column
            let packed = column[y];
            let c = unpack_u32_to_color(packed);
            textureStore(dstTex, vec2<i32>(0, i32(y)), c);
          } else {
            // read previous column at x-1 and write to x
            let sx = i32(x - 1u);
            let c = textureLoad(srcTex, vec2<i32>(sx, i32(y)), 0);
            textureStore(dstTex, vec2<i32>(i32(x), i32(y)), c);
          }
        }
      `;

  try {
    device.pushErrorScope('validation');
    const csModule = device.createShaderModule({ code: columnCs });
    columnWritePipeline = device.createComputePipeline({ layout: 'auto', compute: { module: csModule, entryPoint: 'main' } });
    const columnWritePipelineError = await device.popErrorScope();
    if (columnWritePipelineError) {
      logError('Column compute pipeline creation failed:', columnWritePipelineError);
      columnWritePipeline = null;
    }
    logDebug('Column compute pipeline created');
  } catch (err) {
    logError('Column compute pipeline creation failed with exception:', err);
    columnWritePipeline = null;
  }

  try {
    device.pushErrorScope('validation');
    const scrollModule = device.createShaderModule({ code: scrollCs });
    scrollPipeline = device.createComputePipeline({ layout: 'auto', compute: { module: scrollModule, entryPoint: 'main' } });
    const scrollPipelineError = await device.popErrorScope();
    if (scrollPipelineError) {
      logError('Scroll compute pipeline creation failed:', scrollPipelineError);
      scrollPipeline = null;
    }
    logDebug('Scroll compute pipeline created');
  } catch (err) {
    logError('Scroll compute pipeline creation failed with exception:', err);
    scrollPipeline = null;
  }

  // Fill compute pipeline: used to write test colors into the spectrogram texture
  const fillCs = `
    struct Params { width : u32, height : u32 };
    @group(0) @binding(0) var dstTex : texture_storage_2d<rgba8unorm, write>;
    @group(0) @binding(1) var<uniform> params : Params;

    @compute @workgroup_size(16, 16)
    fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
      let x = gid.x;
      let y = gid.y;
      if (x >= params.width || y >= params.height) { return; }
      textureStore(dstTex, vec2<i32>(i32(x), i32(y)), vec4<f32>(1.0, 1.0, 0.0, 1.0));
    }
  `;
  try {
    device.pushErrorScope('validation');
    const fillModule = device.createShaderModule({ code: fillCs });
    fillPipeline = device.createComputePipeline({ layout: 'auto', compute: { module: fillModule, entryPoint: 'main' } });
    fillParamsBuffer = device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    const fillPipelineError = await device.popErrorScope();
    if (fillPipelineError) {
      logError('Fill compute pipeline creation failed:', fillPipelineError);
      fillPipeline = null;
      fillParamsBuffer = null;
    }
    logDebug('Fill compute pipeline created');
  } catch (err) {
    logError('Fill compute pipeline creation failed with exception:', err);
    fillPipeline = null;
    fillParamsBuffer = null;
  }
}

// Create GPU-side spectrogram texture and staging buffers. This will be called after pipeline setup and whenever the canvas/width changes.
function createSpectrogramResources() {
  if (!device) return;
  // Destroying textures isn't necessary; just recreate and rebind resources
  // Create spectrogram texture (RGBA8) that we will copy into and sample from
  spectrogramTexture = device.createTexture({
    size: { width: Math.max(1, SPECTROGRAM_WIDTH), height: SPECTROGRAM_HEIGHT },
    format: 'rgba8unorm',
    // Include STORAGE_BINDING so compute shaders can write directly to it.
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  // Create a temporary texture for scrolling
  spectrogramTextureTmp = device.createTexture({
    size: { width: Math.max(1, SPECTROGRAM_WIDTH), height: SPECTROGRAM_HEIGHT },
    format: 'rgba8unorm',
    // Allow compute shaders to write into tmp and it to be sampled / copied
    usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.COPY_SRC,
  });
  spectrogramSampler = device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' });

  // Prepare a padded buffer for buffer->texture copies. bytesPerRow must be a multiple of 256.
  const singleRowBytes = 4 * 1; // rgba8 per pixel * width (1)
  const alignment = 256;
  columnBytesPerRow = Math.ceil(singleRowBytes / alignment) * alignment;
  columnBufferSize = columnBytesPerRow * SPECTROGRAM_HEIGHT;

  // This buffer receives data from the CPU via queue.writeBuffer (needs COPY_DST)
  // and is then used as a source for a copy to a texture (needs COPY_SRC).
  columnStagingBuffer = device.createBuffer({ size: columnBufferSize, usage: GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  columnCpuView = new Uint8Array(columnBufferSize);

  // Create a tight GPU storage buffer for column data (u32 per pixel) and a small uniform buffer for params
  const columnPackedSize = SPECTROGRAM_HEIGHT * 4; // one u32 per row
  columnStorageBuffer = device.createBuffer({ size: columnPackedSize, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  columnParamsBuffer = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  // Initialize params (height, width, x)
  queue.writeBuffer(columnParamsBuffer, 0, new Uint32Array([SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH, 0]));

  // Performance: create reusable packed column array and compute dispatch metrics
  columnPacked = new Uint32Array(SPECTROGRAM_HEIGHT);
  columnDispatchCount = Math.ceil(SPECTROGRAM_HEIGHT / 64);

  // Create persistent texture views for both main and tmp textures
  const viewA = spectrogramTexture.createView();
  const viewB = spectrogramTextureTmp.createView();

  // Create per-texture render bind groups (pre-built) to avoid recreating at runtime
  if (renderPipeline) {
    renderBindGroupA = device.createBindGroup({
      layout: renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: spectrogramSampler },
        { binding: 1, resource: viewA },
      ],
    });
    renderBindGroupB = device.createBindGroup({
      layout: renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: spectrogramSampler },
        { binding: 1, resource: viewB },
      ],
    });
    // Default active render bind group
    renderBindGroup = renderBindGroupA;
  }

  // Create per-texture column bind groups for writing a column into a texture
  if (columnWritePipeline) {
    columnBindGroupA = device.createBindGroup({
      layout: columnWritePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: viewA },
        { binding: 1, resource: { buffer: columnStorageBuffer } },
        { binding: 2, resource: { buffer: columnParamsBuffer } },
      ],
    });
    columnBindGroupB = device.createBindGroup({
      layout: columnWritePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: viewB },
        { binding: 1, resource: { buffer: columnStorageBuffer } },
        { binding: 2, resource: { buffer: columnParamsBuffer } },
      ],
    });
    // Default active column bind group
    columnBindGroup = columnBindGroupA;
  } else {
    columnBindGroup = null;
  }

  // Create scroll compute bind groups & params if supported (both permutations)
  if (scrollPipeline) {
    scrollParamsBuffer = device.createBuffer({ size: 8, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    // initialize scroll params
    scrollParamArray[0] = SPECTROGRAM_WIDTH;
    scrollParamArray[1] = SPECTROGRAM_HEIGHT;
    queue.writeBuffer(scrollParamsBuffer, 0, scrollParamArray);

    scrollBindGroupAB = device.createBindGroup({
      layout: scrollPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: viewA },
        { binding: 1, resource: viewB },
        { binding: 2, resource: { buffer: columnStorageBuffer } },
        { binding: 3, resource: { buffer: scrollParamsBuffer } },
      ],
    });
    scrollBindGroupBA = device.createBindGroup({
      layout: scrollPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: viewB },
        { binding: 1, resource: viewA },
        { binding: 2, resource: { buffer: columnStorageBuffer } },
        { binding: 3, resource: { buffer: scrollParamsBuffer } },
      ],
    });

    // Default active scroll bind group: A -> B
    scrollBindGroup = scrollBindGroupAB;

    scrollDispatchX = Math.ceil(SPECTROGRAM_WIDTH / 16);
    scrollDispatchY = Math.ceil(SPECTROGRAM_HEIGHT / 16);
  } else {
    scrollBindGroup = null;
  }

  // Track which side is currently the 'main' texture (0 => viewA is main, 1 => viewB is main)
  currentMainIndex = 0;
}

async function ensureAudio() {
  audioCtx = new AudioContext();
  sampleRate = audioCtx.sampleRate;
  updateFrequencyBuffer(sampleRate);

  // Try to use AudioWorklet if supported, fall back to ScriptProcessorNode
  if (audioCtx.audioWorklet && typeof AudioWorkletNode === 'function') {
    try {
      // Load and add the audio worklet module
      await audioCtx.audioWorklet.addModule('audio-processor.js');
      mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const source = audioCtx.createMediaStreamSource(mediaStream);
      audioNode = new AudioWorkletNode(audioCtx, 'audio-stream-processor');

      audioNode.port.onmessage = (event) => {
        if (event.data.type === 'audioData') {
          feedSamples(event.data.data);
        }
      };

      source.connect(audioNode);
      audioNode.connect(audioCtx.destination);
      return;
    } catch (error) {
      logWarn('AudioWorklet failed, falling back to ScriptProcessorNode:', error);
      // Fall through to ScriptProcessorNode implementation
    }
  }

  // Fallback to ScriptProcessorNode (deprecated but works)
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const source = audioCtx.createMediaStreamSource(mediaStream);
  audioNode = audioCtx.createScriptProcessor(512, 1, 1);
  source.connect(audioNode);
  audioNode.connect(audioCtx.destination);
  audioNode.onaudioprocess = (event) => {
    const input = event.inputBuffer.getChannelData(0);
    feedSamples(input);
  };
}

function feedSamples(samples) {
  let offset = 0;
  while (offset < samples.length) {
    const need = CHUNK_SIZE - pendingCount;
    const copyLen = Math.min(need, samples.length - offset);
    pendingChunk.set(samples.subarray(offset, offset + copyLen), pendingCount);
    pendingCount += copyLen;
    offset += copyLen;
    totalSamplesCaptured += copyLen;

    if (pendingCount === CHUNK_SIZE) {
      const chunkData = pendingChunk.slice();
      const chunkStart = Math.max(0, totalSamplesCaptured - CHUNK_SIZE);
      chunkQueue.push({ data: chunkData, start: chunkStart });
      logDebug("queued chunk", chunkQueue.length, chunkStart);
      const queueLimit = getChunkQueueLimit();
      if (chunkQueue.length > queueLimit) {
        chunkQueue.shift();
      }
      pendingCount = 0;
      processChunkQueue();
    }
  }
}

function processChunkQueue() {
  if (queueProcessing) {
    return;
  }
  queueProcessing = true;
  (async () => {
    try {
      while (chunkQueue.length > 0) {
        const { data, start } = chunkQueue.shift();
        logDebug("dispatch chunk", start, "queue len", chunkQueue.length);
        await runChunk(data, CHUNK_SIZE, start, processedChunks);
        processedChunks += 1;
      }
    } finally {
      queueProcessing = false;
    }
  })();
}

function waitForQueueIdle() {
  return new Promise((resolve) => {
    const check = () => {
      if (!queueProcessing && chunkQueue.length === 0) {
        resolve();
      } else {
        setTimeout(check, 10);
      }
    };
    check();
  });
}

async function runChunk(chunkData, length, chunkStart, chunkIndex) {
  // Validate that WebGPU resources are initialized
  if (!device || !queue || !chunkBuffer || !paramsBuffer || !pipeline || !bindGroup) {
    logError('WebGPU resources not initialized. Skipping chunk processing.');
    return;
  }

  try {
    queue.writeBuffer(chunkBuffer, 0, chunkData.buffer, 0, CHUNK_SIZE * 4);
    const params = new Uint32Array([chunkStart >>> 0, length, 0, 0]);
    queue.writeBuffer(paramsBuffer, 0, params.buffer, params.byteOffset, params.byteLength);

    const encoder = device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.dispatchWorkgroups(Math.ceil(NUM_BINS / WORKGROUP_SIZE));
    pass.end();

    encoder.copyBufferToBuffer(
      chunkRealBuffer,
      0,
      stagingBuffer,
      0,
      NUM_BINS * 4
    );
    encoder.copyBufferToBuffer(
      chunkImagBuffer,
      0,
      stagingBuffer,
      NUM_BINS * 4,
      NUM_BINS * 4
    );
    queue.submit([encoder.finish()]);

    await stagingBuffer.mapAsync(GPUMapMode.READ);
    const mapped = stagingBuffer.getMappedRange(0, NUM_BINS * 4 * 2);
    const realChunk = new Float32Array(mapped, 0, NUM_BINS);
    const imagChunk = new Float32Array(
      mapped,
      NUM_BINS * 4,
      NUM_BINS
    );
    const realData = new Float32Array(realChunk);
    const imagData = new Float32Array(imagChunk);
    stagingBuffer.unmap();

    const historySlot = chunkHistory[chunkIndex % WINDOW_CHUNKS];
    if (chunkIndex >= WINDOW_CHUNKS) {
      const oldReal = historySlot.real;
      const oldImag = historySlot.imag;
      for (let i = 0; i < NUM_BINS; ++i) {
        currentReal[i] -= oldReal[i];
        currentImag[i] -= oldImag[i];
      }
    }
    historySlot.real.set(realData);
    historySlot.imag.set(imagData);
    for (let i = 0; i < NUM_BINS; ++i) {
      currentReal[i] += realData[i];
      currentImag[i] += imagData[i];
    }

    const magnitudes = new Float32Array(NUM_BINS);
    for (let i = 0; i < NUM_BINS; ++i) {
      magnitudes[i] = Math.sqrt(
        currentReal[i] * currentReal[i] + currentImag[i] * currentImag[i]
      );
    }
    const magnitudeSum = magnitudes.reduce((total, value) => total + value, 0);
    logInfo('compute spectrum sum', magnitudeSum);
    drawSpectrum(magnitudes, chunkIndex);
  } catch (error) {
    logError('Error in runChunk:', error);
    // Don't rethrow to avoid breaking the processing loop
  }
}

function drawSpectrum(data, chunkIndex) {
  // Add new spectrum to history
  spectrogramHistory.push(data);
  if (spectrogramHistory.length > MAX_HISTORY) {
    spectrogramHistory.shift(); // Remove oldest
  }

  // Find global maximum for color mapping
  let globalMax = 0;
  for (const spectrum of spectrogramHistory) {
    const maxInSpectrum = Math.max(...spectrum);
    if (maxInSpectrum > globalMax) {
      globalMax = maxInSpectrum;
    }
  }

  // Store pending spectrum and schedule a draw on the next animation frame.
  pendingSpectrum = spectrogramHistory[spectrogramHistory.length - 1];
  pendingGlobalMax = globalMax || 1;

  if (!drawPending) {
    drawPending = true;
    requestAnimationFrame(function flushSpectrogram() {
      drawPending = false;

      // Diagnostic: log active state
      logDebug(`draw frame; mainIndex=${currentMainIndex}, scroll=${!!scrollPipeline}, columnWrite=${!!columnWritePipeline}`);

      // Build the padded column buffer (RGBA8) directly into the CPU staging view
      if (!columnCpuView) {
        logWarn('columnCpuView not ready; skipping frame');
      } else {
        // Build both the padded CPU staging bytes and reuse the packed u32 column buffer for the compute path
        for (let y = 0; y < SPECTROGRAM_HEIGHT; y++) {
          const binIndex = SPECTROGRAM_HEIGHT - 1 - y;
          const magnitude = pendingSpectrum[binIndex];
          const [r, g, b] = magnitudeToColor(magnitude, pendingGlobalMax);
          const base = y * columnBytesPerRow;
          columnCpuView[base] = r;
          columnCpuView[base + 1] = g;
          columnCpuView[base + 2] = b;
          columnCpuView[base + 3] = 255;
          // zeros for padding bytes are already present or not needed

          // Packed u32: low byte = R, next = G, next = B, high = A
          columnPacked[y] = (r & 0xFF) | ((g & 0xFF) << 8) | ((b & 0xFF) << 16) | (0xFF << 24);
        }

        // Write staging buffer then issue GPU-side copy & render commands in one encoder
        const encoder = device.createCommandEncoder();

        // Shift texture contents right by 1 pixel using a compute pass that writes into tmp, then copy tmp -> original
        if (scrollPipeline && scrollBindGroup) {
          // Update scroll params and packed column (columnPacked already prepared above)
          scrollParamArray[0] = SPECTROGRAM_WIDTH;
          scrollParamArray[1] = SPECTROGRAM_HEIGHT;
          queue.writeBuffer(scrollParamsBuffer, 0, scrollParamArray);
          queue.writeBuffer(columnStorageBuffer, 0, columnPacked);

          const computePass = encoder.beginComputePass();
          computePass.setPipeline(scrollPipeline);
          computePass.setBindGroup(0, scrollBindGroup);
          // Dispatch using current texture dims (in case canvas resized since bind group creation)
          computePass.dispatchWorkgroups(Math.ceil(SPECTROGRAM_WIDTH / 16), Math.ceil(SPECTROGRAM_HEIGHT / 16));
          computePass.end();
          logDebug('Scroll compute pass dispatched');

          // Avoid recreating bind groups — just flip which pre-created bind groups are active
          currentMainIndex = 1 - currentMainIndex;
          if (currentMainIndex === 0) {
            renderBindGroup = renderBindGroupA;
            columnBindGroup = columnBindGroupA;
            scrollBindGroup = scrollBindGroupAB;
          } else {
            renderBindGroup = renderBindGroupB;
            columnBindGroup = columnBindGroupB;
            scrollBindGroup = scrollBindGroupBA;
          }

          logDebug('Switched active bind groups to the other texture (no allocation)');
        } else if (SPECTROGRAM_WIDTH > 1) {
          // fallback to copy-based scroll when width>1
          const copySize = { width: Math.max(0, SPECTROGRAM_WIDTH - 1), height: SPECTROGRAM_HEIGHT, depthOrArrayLayers: 1 };

          // 1. Copy the contents to be scrolled into the temporary texture
          encoder.copyTextureToTexture(
            { texture: spectrogramTexture, origin: { x: 0, y: 0 } },
            { texture: spectrogramTextureTmp, origin: { x: 0, y: 0 } },
            copySize
          );

          // 2. Copy from the temporary texture back to the original, shifted by 1px
          encoder.copyTextureToTexture(
            { texture: spectrogramTextureTmp, origin: { x: 0, y: 0 } },
            { texture: spectrogramTexture, origin: { x: 1, y: 0 } },
            copySize
          );
        } else {
          // width == 1 and no scroll compute available: nothing to shift, column-only write will handle it below
        }

        // If scroll compute handled the full shift + new column, skip this separate column write.
        if (!(scrollPipeline && scrollBindGroup)) {
          // Upload the leftmost column as tightly-packed u32s and dispatch compute to write it into the texture
          // This avoids bytesPerRow/alignment issues and keeps the operation on the GPU
          queue.writeBuffer(columnStorageBuffer, 0, columnPacked);
          // Update params (height, width, x=0)
          queue.writeBuffer(columnParamsBuffer, 0, new Uint32Array([SPECTROGRAM_HEIGHT, SPECTROGRAM_WIDTH, 0]));

          if (columnWritePipeline && columnStorageBuffer && columnBindGroup) {
            // Update params (height, width, x=0) and upload packed data using preallocated array to avoid garbage
            columnParamArray[0] = SPECTROGRAM_HEIGHT;
            columnParamArray[1] = SPECTROGRAM_WIDTH;
            columnParamArray[2] = 0;
            queue.writeBuffer(columnParamsBuffer, 0, columnParamArray);
            queue.writeBuffer(columnStorageBuffer, 0, columnPacked);

            // Dispatch compute pipeline to write the column into the storage texture using the prebuilt bind group
            const computePass = encoder.beginComputePass();
            computePass.setPipeline(columnWritePipeline);
            computePass.setBindGroup(0, columnBindGroup);
            computePass.dispatchWorkgroups(columnDispatchCount);
            computePass.end();
          } else {
            // Fallback: use CPU staging buffer -> texture copy
            if (!computeFallbackWarned) {
              logWarn('Compute pipeline unavailable; falling back to buffer->texture copy');
              computeFallbackWarned = true;
            }
            queue.writeBuffer(columnStagingBuffer, 0, columnCpuView);
            const targetTex = currentMainIndex === 0 ? spectrogramTexture : spectrogramTextureTmp;
            encoder.copyBufferToTexture(
              { buffer: columnStagingBuffer, bytesPerRow: columnBytesPerRow },
              { texture: targetTex, origin: { x: 0, y: 0 } },
              { width: 1, height: SPECTROGRAM_HEIGHT, depthOrArrayLayers: 1 }
            );
          }
        }

        // Render the texture to the canvas via render pass
        const swapView = canvasGPU.getCurrentTexture().createView();
        const pass = encoder.beginRenderPass({
          colorAttachments: [
            {
              view: swapView,
              loadOp: 'load', // avoid an extra clear since we draw full-screen
              storeOp: 'store',
            },
          ],
        });
        pass.setPipeline(renderPipeline);
        pass.setBindGroup(0, renderBindGroup);
        pass.draw(6);
        pass.end();
        if (circlePipeline && circleBindGroup && circleVisible) {
          const circlePass = encoder.beginRenderPass({
            colorAttachments: [
              {
                view: swapView,
                loadOp: 'load',
                storeOp: 'store',
              },
            ],
          });
          circlePass.setPipeline(circlePipeline);
          circlePass.setBindGroup(0, circleBindGroup);
          circlePass.draw(6);
          circlePass.end();
        }

        // Submit all commands
        queue.submit([encoder.finish()]);
        
        logDebug(`render pass using mainIndex=${currentMainIndex}, renderBindGroup=${!!renderBindGroup}`);
      } // end else (columnCpuView available)
    }); // end requestAnimationFrame callback
  } // end if (!drawPending)

  logDebug("queued chunk", chunkIndex, "max", Math.max(...pendingSpectrum));
}

async function autoplayChirp() {
  try {
    statusEl.textContent = "Loading chirp.wav...";
    statsEl.textContent = "Fetching audio file...";
    await ensureGpu();
    const response = await fetch("chirp.wav");
    if (!response.ok) {
      throw new Error(`Failed to fetch chirp.wav: ${response.status} ${response.statusText}`);
    }
    logDebug("chirp fetch status", response.status);
    statusEl.textContent = "Decoding audio...";
    statsEl.textContent = "Processing chirp.wav...";
    await decodeAndStreamAudio(await response.arrayBuffer(), "chirp.wav");
    statusEl.textContent = "Chirp playback complete";
  } catch (error) {
    logWarn("Chirp auto-run failed:", error);
    // Update status to show the error to user
    statusEl.textContent = `Auto-run failed: ${error.message}`;
    statsEl.textContent = "Check browser console for details";
  }
}

// Wait for DOM to be ready before attempting WebGPU initialization
document.addEventListener('DOMContentLoaded', () => {
  // Initialize canvas size and spectrogram
  syncCanvasSize();
  initSpectrogram();

  // No autoplay. User will click a button to start audio or test rendering.
});

function buildFrequencyArray(rate) {
  const freqs = new Float32Array(NUM_BINS);
  const minNorm = MIN_FREQ / rate;
  const maxNorm = MAX_FREQ / rate;
  const logMin = Math.log(minNorm);
  const logMax = Math.log(maxNorm);
  const delta = logMax - logMin;
  for (let i = 0; i < NUM_BINS; ++i) {
    const ratio = i / (NUM_BINS - 1);
    freqs[i] = Math.exp(logMin + delta * ratio);
  }
  return freqs;
}

function updateFrequencyBuffer(rate) {
  if (!freqBuffer) {
    return;
  }
  const freqs = buildFrequencyArray(rate);
  queue.writeBuffer(freqBuffer, 0, freqs.buffer, freqs.byteOffset, freqs.byteLength);
}
