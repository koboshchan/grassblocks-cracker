/**
 * WebGPU implementation of the dense patch detector kernel.
 */

export class WebGpuDetector {
  private adapter: GPUAdapter;
  private device: GPUDevice;
  private pipeline: GPUComputePipeline;
  private refinePipeline: GPUComputePipeline | null = null;

  private imageTexture: GPUTexture | null = null;
  private imageView: GPUTextureView | null = null;
  private sampler: GPUSampler | null = null;
  private imageWidth = 0;
  private imageHeight = 0;

  private paramsBuffer: GPUBuffer;
  private refStatsBuffer: GPUBuffer | null = null;
  private refsBuffer: GPUBuffer | null = null;

  private destroyed = false;

  private constructor(
    adapter: GPUAdapter,
    device: GPUDevice,
    pipeline: GPUComputePipeline,
    refinePipeline: GPUComputePipeline | null = null
  ) {
    this.adapter = adapter;
    this.device = device;
    this.pipeline = pipeline;
    this.refinePipeline = refinePipeline;

    this.device.lost.then(info => {
      console.warn(`WebGPU device lost: ${info.message}`);
      this.destroyed = true;
    });

    this.paramsBuffer = this.device.createBuffer({
      size: 16,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });

    this.sampler = this.device.createSampler({
      magFilter: 'linear',
      minFilter: 'linear',
    });
  }

  /** Surface async validation/device errors before returning any GPU result. */
  async checked<T>(work: () => Promise<T>): Promise<T> {
    if (this.destroyed) throw Error('WebGPU device unavailable');
    const device = this.device;
    device.pushErrorScope('internal');
    device.pushErrorScope('out-of-memory');
    device.pushErrorScope('validation');
    try {
      const lost = device.lost.then(info => { throw Error('WebGPU device lost: ' + info.message); });
      const value = await Promise.race([work(), lost]);
      await Promise.race([device.queue.onSubmittedWorkDone(), lost]);
      return value;
    } finally {
      const errors = await Promise.all([device.popErrorScope(), device.popErrorScope(), device.popErrorScope()]);
      const error = errors.find(Boolean);
      if (error) throw Error('WebGPU execution failed: ' + error.message);
    }
  }

  static async isSupported(): Promise<boolean> {
    if (typeof navigator === 'undefined' || !navigator.gpu) {
      return false;
    }
    try {
      const adapter = await navigator.gpu.requestAdapter();
      return adapter !== null;
    } catch {
      return false;
    }
  }

  static async create(shaderCode: string): Promise<WebGpuDetector | null> {
    if (typeof navigator === 'undefined' || !navigator.gpu) {
      return null;
    }
    try {
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) {
        return null;
      }
      const device = await adapter.requestDevice();

      const shaderModule = device.createShaderModule({ code: shaderCode });
      const compilationInfo = await shaderModule.getCompilationInfo();
      const errors = compilationInfo.messages.filter(m => m.type === 'error');
      if (errors.length > 0) {
        device.destroy();
        throw new Error(`Shader compilation failed:\n${errors.map(e => e.message).join('\n')}`);
      }

      const match = shaderCode.match(/@compute[\s\S]*?fn\s+([a-zA-Z0-9_]+)\s*\(/);
      const entryPoint = match ? match[1] : 'classify_main';

      const pipeline = await device.createComputePipelineAsync({
        layout: 'auto',
        compute: {
          module: shaderModule,
          entryPoint,
        },
      });

      let refinePipeline: GPUComputePipeline | null = null;
      if (shaderCode.includes('refine_main')) {
        refinePipeline = await device.createComputePipelineAsync({
          layout: 'auto',
          compute: {
            module: shaderModule,
            entryPoint: 'refine_main',
          },
        });
      }

      return new WebGpuDetector(adapter, device, pipeline, refinePipeline);
    } catch (err) {
      console.warn('Failed to create WebGpuDetector:', err);
      return null;
    }
  }

  setImage(rgba: Uint8Array, width: number, height: number): void {
    if (this.destroyed) throw new Error('WebGpuDetector is destroyed');
    if (!Number.isInteger(width) || !Number.isInteger(height) || width < 2 || height < 2) {
      throw new Error(`Invalid image dimensions: ${width}x${height}`);
    }

    if (this.imageTexture) {
      this.imageTexture.destroy();
      this.imageTexture = null;
      this.imageView = null;
    }

    this.imageWidth = width;
    this.imageHeight = height;

    this.imageTexture = this.device.createTexture({
      size: [width, height, 1],
      format: 'r8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
    });
    this.imageView = this.imageTexture.createView();

    const isRgba = rgba.length >= width * height * 4;
    const alignedBytesPerRow = Math.ceil(width / 256) * 256;
    let textureData: Uint8Array;

    if (alignedBytesPerRow === width && !isRgba) {
      textureData = rgba;
    } else {
      textureData = new Uint8Array(alignedBytesPerRow * height);
      for (let y = 0; y < height; y++) {
        const srcRow = y * width;
        const dstRow = y * alignedBytesPerRow;
        if (isRgba) {
          for (let x = 0; x < width; x++) {
            textureData[dstRow + x] = rgba[(srcRow + x) * 4 + 1];
          }
        } else {
          textureData.set(rgba.subarray(srcRow, srcRow + width), dstRow);
        }
      }
    }

    this.device.queue.writeTexture(
      { texture: this.imageTexture },
      textureData,
      { bytesPerRow: alignedBytesPerRow, rowsPerImage: height },
      { width, height }
    );
  }

  setRefs(refs: number[][]): void {
    if (this.destroyed) throw new Error('WebGpuDetector is destroyed');

    let rotations: number[][];
    if (refs.length === 16 && refs[0].length === 16) {
      // 16x16 reference matrix -> generate 4 clockwise rotations
      rotations = [];
      for (let k = 0; k < 4; k++) {
        const rot: number[] = [];
        for (let y = 0; y < 16; y++) {
          for (let x = 0; x < 16; x++) {
            let xx = x;
            let yy = y;
            for (let j = 0; j < k; j++) {
              const t = xx;
              xx = yy;
              yy = 15 - t;
            }
            rot.push(refs[yy][xx]);
          }
        }
        rotations.push(rot);
      }
    } else if (refs.length === 4 && refs[0].length === 256) {
      rotations = refs;
    } else {
      throw new Error('Expected 4 flat 16x16 rotations or a 16x16 reference matrix');
    }

    // 4-way reference patterns: array<vec4<f32>, 256>
    const refPatternData = new Float32Array(256 * 4);
    for (let idx = 0; idx < 256; idx++) {
      refPatternData[idx * 4 + 0] = rotations[0][idx];
      refPatternData[idx * 4 + 1] = rotations[1][idx];
      refPatternData[idx * 4 + 2] = rotations[2][idx];
      refPatternData[idx * 4 + 3] = rotations[3][idx];
    }

    // Reference statistics: 14x14 inner region (196 pixels) and 4 quadrants (49 pixels each)
    const sb = [0, 0, 0, 0];
    const bb = [0, 0, 0, 0];
    const q_sb = [
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ];
    const q_bb = [
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ];

    for (let j = 1; j < 15; j++) {
      for (let i = 1; i < 15; i++) {
        const q = (i >= 8 ? 1 : 0) + (j >= 8 ? 2 : 0);
        const idx = j * 16 + i;
        for (let k = 0; k < 4; k++) {
          const val = rotations[k][idx];
          sb[k] += val;
          bb[k] += val * val;
          q_sb[q][k] += val;
          q_bb[q][k] += val * val;
        }
      }
    }

    const vb = [0, 0, 0, 0];
    for (let k = 0; k < 4; k++) {
      vb[k] = bb[k] - (sb[k] * sb[k]) / 196.0;
    }

    const q_vb = [
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
      [0, 0, 0, 0],
    ];
    for (let q = 0; q < 4; q++) {
      for (let k = 0; k < 4; k++) {
        q_vb[q][k] = q_bb[q][k] - (q_sb[q][k] * q_sb[q][k]) / 49.0;
      }
    }

    // RefStats struct packing:
    // struct RefStats {
    //   sb: vec4<f32>,             // offset 0 (4 floats)
    //   bb: vec4<f32>,             // offset 4 (4 floats)
    //   vb: vec4<f32>,             // offset 8 (4 floats)
    //   q_sb: array<vec4<f32>, 4>, // offset 12 (16 floats)
    //   q_bb: array<vec4<f32>, 4>, // offset 28 (16 floats)
    //   q_vb: array<vec4<f32>, 4>, // offset 44 (16 floats)
    // };
    const refStatsData = new Float32Array(64);
    for (let k = 0; k < 4; k++) {
      refStatsData[0 + k] = sb[k];
      refStatsData[4 + k] = bb[k];
      refStatsData[8 + k] = vb[k];
    }
    for (let q = 0; q < 4; q++) {
      for (let k = 0; k < 4; k++) {
        refStatsData[12 + q * 4 + k] = q_sb[q][k];
        refStatsData[28 + q * 4 + k] = q_bb[q][k];
        refStatsData[44 + q * 4 + k] = q_vb[q][k];
      }
    }

    if (!this.refStatsBuffer) {
      this.refStatsBuffer = this.device.createBuffer({
        size: 256,
        usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
    this.device.queue.writeBuffer(this.refStatsBuffer, 0, refStatsData);

    if (!this.refsBuffer) {
      this.refsBuffer = this.device.createBuffer({
        size: 256 * 16,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      });
    }
    this.device.queue.writeBuffer(this.refsBuffer, 0, refPatternData);
  }

  async classifyBatch(quads: Float32Array): Promise<Float32Array> {
    if (this.destroyed) throw new Error('WebGpuDetector is destroyed');
    if (!this.imageTexture || !this.imageView || !this.sampler) {
      throw new Error('Image not set. Call setImage() before classifyBatch()');
    }
    if (!this.refStatsBuffer || !this.refsBuffer) {
      throw new Error('References not set. Call setRefs() before classifyBatch()');
    }
    if (quads.length % 8 !== 0) {
      throw new Error('Quads length must be a multiple of 8 (8 floats per quad)');
    }

    const count = quads.length / 8;
    if (count === 0) {
      return new Float32Array(0);
    }

    const paramsData = new Uint32Array([this.imageWidth, this.imageHeight, count, 0]);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, paramsData);

    const quadsByteLength = Math.max(32, quads.byteLength);
    const quadsBuffer = this.device.createBuffer({
      size: quadsByteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.device.queue.writeBuffer(quadsBuffer, 0, quads);

    const resultByteLength = Math.max(20, count * 20);
    const resultsBuffer = this.device.createBuffer({
      size: resultByteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const readBuffer = this.device.createBuffer({
      size: resultByteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    let mapped = false;
    try {
      const bindGroup = this.device.createBindGroup({
        layout: this.pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.paramsBuffer } },
          { binding: 1, resource: { buffer: this.refStatsBuffer } },
          { binding: 2, resource: { buffer: quadsBuffer } },
          { binding: 3, resource: { buffer: resultsBuffer } },
          { binding: 4, resource: this.imageView },
          { binding: 5, resource: this.sampler },
          { binding: 6, resource: { buffer: this.refsBuffer } },
        ],
      });

      const commandEncoder = this.device.createCommandEncoder();
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.pipeline);
      pass.setBindGroup(0, bindGroup);
      const workgroupCount = Math.ceil(count / 64);
      pass.dispatchWorkgroups(workgroupCount, 1, 1);
      pass.end();

      commandEncoder.copyBufferToBuffer(resultsBuffer, 0, readBuffer, 0, count * 20);
      this.device.queue.submit([commandEncoder.finish()]);

      await readBuffer.mapAsync(GPUMapMode.READ);
      mapped = true;

      const out = new Float32Array(count * 5);
      out.set(new Float32Array(readBuffer.getMappedRange(0, count * 20)));
      return out;
    } finally {
      if (mapped) {
        readBuffer.unmap();
      }
      quadsBuffer.destroy();
      resultsBuffer.destroy();
      readBuffer.destroy();
    }
  }

  async refineBatch(quads: Float32Array): Promise<Float32Array> {
    if (this.destroyed) throw new Error('WebGpuDetector is destroyed');
    if (!this.refinePipeline) {
      throw new Error('Refine pipeline not initialized');
    }
    if (!this.imageTexture || !this.imageView || !this.sampler) {
      throw new Error('Image not set. Call setImage() before refineBatch()');
    }
    if (!this.refStatsBuffer || !this.refsBuffer) {
      throw new Error('References not set. Call setRefs() before refineBatch()');
    }
    if (quads.length % 8 !== 0) {
      throw new Error('Quads length must be a multiple of 8 (8 floats per quad)');
    }

    const count = quads.length / 8;
    if (count === 0) {
      return new Float32Array(0);
    }

    const paramsData = new Uint32Array([this.imageWidth, this.imageHeight, count, 0]);
    this.device.queue.writeBuffer(this.paramsBuffer, 0, paramsData);

    const quadsByteLength = Math.max(32, quads.byteLength);
    const quadsBuffer = this.device.createBuffer({
      size: quadsByteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.device.queue.writeBuffer(quadsBuffer, 0, quads);

    const resultByteLength = Math.max(20, count * 20);
    const resultsBuffer = this.device.createBuffer({
      size: resultByteLength,
      usage: GPUBufferUsage.STORAGE,
    });

    const readQuadsBuffer = this.device.createBuffer({
      size: quadsByteLength,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });

    let mapped = false;
    try {
      const bindGroup = this.device.createBindGroup({
        layout: this.refinePipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: this.paramsBuffer } },
          { binding: 1, resource: { buffer: this.refStatsBuffer } },
          { binding: 2, resource: { buffer: quadsBuffer } },
          { binding: 3, resource: { buffer: resultsBuffer } },
          { binding: 4, resource: this.imageView },
          { binding: 5, resource: this.sampler },
          { binding: 6, resource: { buffer: this.refsBuffer } },
        ],
      });

      const commandEncoder = this.device.createCommandEncoder();
      const pass = commandEncoder.beginComputePass();
      pass.setPipeline(this.refinePipeline);
      pass.setBindGroup(0, bindGroup);
      const workgroupCount = Math.ceil(count / 64);
      pass.dispatchWorkgroups(workgroupCount, 1, 1);
      pass.end();

      commandEncoder.copyBufferToBuffer(quadsBuffer, 0, readQuadsBuffer, 0, count * 32);
      this.device.queue.submit([commandEncoder.finish()]);

      await readQuadsBuffer.mapAsync(GPUMapMode.READ);
      mapped = true;

      const out = new Float32Array(count * 8);
      out.set(new Float32Array(readQuadsBuffer.getMappedRange(0, count * 32)));
      return out;
    } finally {
      if (mapped) {
        readQuadsBuffer.unmap();
      }
      quadsBuffer.destroy();
      resultsBuffer.destroy();
      readQuadsBuffer.destroy();
    }
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;

    this.paramsBuffer?.destroy();
    this.refStatsBuffer?.destroy();
    this.refsBuffer?.destroy();
    this.imageTexture?.destroy();
    this.device?.destroy();

    this.refStatsBuffer = null;
    this.refsBuffer = null;
    this.imageTexture = null;
    this.imageView = null;
    this.sampler = null;
  }
}
