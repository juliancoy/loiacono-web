struct StreamParams {
  chunk_start : u32,
  samples_valid : u32,
  finalize : u32,
  _pad : u32,
};

@group(0) @binding(0) var<storage, read> chunk : array<f32, 256>;
@group(0) @binding(1) var<storage, read_write> chunk_real : array<f32, 512>;
@group(0) @binding(2) var<storage, read_write> chunk_imag : array<f32, 512>;
@group(0) @binding(3) var<storage, read> freqs : array<f32, 512>;
@group(0) @binding(4) var<uniform> params : StreamParams;

const PI : f32 = 3.141592653589793;
const MULTIPLE : f32 = 15.0;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x >= 512u) {
    return;
  }

  var real_sum = 0.0;
  var imag_sum = 0.0;
  let freq = freqs[gid.x];
  let scale = sqrt(freq / MULTIPLE);
  let start = params.chunk_start;
  let valid = params.samples_valid;

  for (var i: u32 = 0u; i < valid; i = i + 1u) {
    let sample = chunk[i];
    let angle = 2.0 * PI * freq * f32(start + i);
    real_sum = real_sum + sample * cos(angle) * scale;
    imag_sum = imag_sum - sample * sin(angle) * scale;
  }

  chunk_real[gid.x] = real_sum;
  chunk_imag[gid.x] = imag_sum;
}
