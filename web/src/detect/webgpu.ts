import {eccScratchBudget,withEccAllocationRetry,eccThroughputPolicy,runEccWindow} from './ecc-batch';
import {BatchedGpuEcc} from './ecc';
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
  private paramsData = new Uint32Array(4);
  private refStatsData: Float32Array | null = null;
  private refPatternData: Float32Array | null = null;

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
      const requiredLimits:Record<string,number>={};
      for(const key of ['maxStorageBufferBindingSize','maxBufferSize','maxComputeWorkgroupsPerDimension'] as const){const value=adapter.limits?.[key];if(Number.isSafeInteger(value)&&value>0)requiredLimits[key]=value;}
      let device: GPUDevice;
      if (adapter.features?.has('timestamp-query')) {
        try {
          device = await adapter.requestDevice({
            requiredLimits,
            requiredFeatures: ['timestamp-query'],
          });
        } catch {
          device = await adapter.requestDevice({ requiredLimits }).catch(() => adapter.requestDevice());
        }
      } else {
        device = await adapter.requestDevice({ requiredLimits }).catch(() => adapter.requestDevice());
      }

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

    const refPatternData = (this.refPatternData ??= new Float32Array(256 * 4));

    if (refs.length === 16 && refs[0].length === 16) {
      // 16x16 reference matrix -> generate 4 clockwise rotations
      for (let y = 0; y < 16; y++) {
        const row = refs[y];
        const yRot2 = 15 - y;
        for (let x = 0; x < 16; x++) {
          const idx = (y << 4) | x;
          const base = idx << 2;
          refPatternData[base + 0] = row[x];
          refPatternData[base + 1] = refs[15 - x][y];
          refPatternData[base + 2] = refs[yRot2][15 - x];
          refPatternData[base + 3] = refs[x][yRot2];
        }
      }
    } else if (refs.length === 4 && refs[0].length === 256) {
      const r0 = refs[0];
      const r1 = refs[1];
      const r2 = refs[2];
      const r3 = refs[3];
      for (let idx = 0; idx < 256; idx++) {
        const base = idx << 2;
        refPatternData[base + 0] = r0[idx];
        refPatternData[base + 1] = r1[idx];
        refPatternData[base + 2] = r2[idx];
        refPatternData[base + 3] = r3[idx];
      }
    } else {
      throw new Error('Expected 4 flat 16x16 rotations or a 16x16 reference matrix');
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
    const refStatsData = (this.refStatsData ??= new Float32Array(64));
    refStatsData.fill(0);

    // Reference statistics: 14x14 inner region (196 pixels) and 4 quadrants (49 pixels each)
    for (let j = 1; j < 15; j++) {
      const qRow = j >= 8 ? 8 : 0;
      for (let i = 1; i < 15; i++) {
        const qOffset = (i >= 8 ? 4 : 0) + qRow;
        const base = ((j << 4) | i) << 2;
        const qSb = 12 + qOffset;
        const qBb = 28 + qOffset;

        const v0 = refPatternData[base + 0];
        const v1 = refPatternData[base + 1];
        const v2 = refPatternData[base + 2];
        const v3 = refPatternData[base + 3];

        refStatsData[0] += v0;
        refStatsData[1] += v1;
        refStatsData[2] += v2;
        refStatsData[3] += v3;

        refStatsData[4] += v0 * v0;
        refStatsData[5] += v1 * v1;
        refStatsData[6] += v2 * v2;
        refStatsData[7] += v3 * v3;

        refStatsData[qSb + 0] += v0;
        refStatsData[qSb + 1] += v1;
        refStatsData[qSb + 2] += v2;
        refStatsData[qSb + 3] += v3;

        refStatsData[qBb + 0] += v0 * v0;
        refStatsData[qBb + 1] += v1 * v1;
        refStatsData[qBb + 2] += v2 * v2;
        refStatsData[qBb + 3] += v3 * v3;
      }
    }

    for (let k = 0; k < 4; k++) {
      const sbVal = refStatsData[k];
      refStatsData[8 + k] = refStatsData[4 + k] - (sbVal * sbVal) / 196.0;
    }

    for (let q = 0; q < 4; q++) {
      const qOffset = q << 2;
      for (let k = 0; k < 4; k++) {
        const qSbVal = refStatsData[12 + qOffset + k];
        refStatsData[44 + qOffset + k] = refStatsData[28 + qOffset + k] - (qSbVal * qSbVal) / 49.0;
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
    const classifiedAt=performance.now();
    this.diagnostics.classified+=quads.length/8;
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

    this.paramsData[0] = this.imageWidth;
    this.paramsData[1] = this.imageHeight;
    this.paramsData[2] = count;
    this.paramsData[3] = 0;
    this.device.queue.writeBuffer(this.paramsBuffer, 0, this.paramsData);

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

      const out = new Float32Array(readBuffer.getMappedRange(0, count * 20).slice(0));
      return out;
    } finally {
      if (mapped) {
        readBuffer.unmap();
      }
      this.diagnostics.classifyMs+=performance.now()-classifiedAt;
      quadsBuffer.destroy();
      resultsBuffer.destroy();
      readBuffer.destroy();
    }
  }

  private eccComponent: Promise<BatchedGpuEcc> | null = null;
  public diagnostics = {classified:0,eccCandidates:0,classifyMs:0,eccMs:0,eccIterations:0,eccConverged:0,eccIterationLimits:0,eccFailures:0};
  get adapterInfo(): Record<string,unknown> {
    const info=(this.adapter as any).info;
    return info ? {vendor:info.vendor,architecture:info.architecture,device:info.device,description:info.description,isFallbackAdapter:(this.adapter as any).isFallbackAdapter} : {description:'adapter info unavailable'};
  }
  eccBatchPolicy:Record<string,unknown>|null=null;
  eccBatchTimings:import("./ecc").EccHostTiming[]=[];
  async refineBatch(quads: Float32Array, rotations?:number[]): Promise<Float32Array> {
    if (this.destroyed || !this.imageView || !this.refsBuffer) throw Error('ECC image/references unavailable');
    this.eccComponent ??= BatchedGpuEcc.create(this.device);
    const component=await this.eccComponent;
    const count=quads.length/8;
    if (!Number.isInteger(count)) throw Error('Invalid ECC quad buffer');
    if (!rotations) {
      const v = await this.classifyBatch(quads);
      rotations = new Array(count);
      for (let i = 0; i < count; i++) {
        rotations[i] = Math.round(v[i * 5 + 1]);
      }
    }
    const output=new Float32Array(quads.length);const start=performance.now();
    const policy=eccThroughputPolicy(component.capacity);
    const allocationFallbacks:{from:number;to:number}[]=[];
    this.eccBatchPolicy={capacity:component.capacity,chosenBatchSize:Math.min(count,policy.batchSize),inFlight:policy.inFlight,workingSetPolicy:'device-scaled batching up to capacity, two independent batches in flight',scratchBudgetBytes:eccScratchBudget(this.device.limits),bytesPerQuad:1180560,allocationFallbacks,limits:{maxStorageBufferBindingSize:this.device.limits.maxStorageBufferBindingSize,maxBufferSize:this.device.limits.maxBufferSize,maxComputeWorkgroupsPerDimension:this.device.limits.maxComputeWorkgroupsPerDimension}};
    const traces=await runEccWindow(count,component.capacity,async(base,n)=>{
      const batch:import('./ecc').EccTrace[]=[];
      for(let offset=0;offset<n;){
        const result=await withEccAllocationRetry(n-offset,n-offset,async size=>{
          const qs: number[][][] = new Array(size);
          for (let j = 0; j < size; j++) {
            const idx = (base + offset + j) * 8;
            qs[j] = [
              [quads[idx], quads[idx + 1]],
              [quads[idx + 2], quads[idx + 3]],
              [quads[idx + 4], quads[idx + 5]],
              [quads[idx + 6], quads[idx + 7]],
            ];
          }
          return component.run(this.imageView!,this.refsBuffer!,this.imageWidth,this.imageHeight,qs,rotations!.slice(base+offset,base+offset+size),undefined,timing=>this.eccBatchTimings.push({...timing}));
        },(from,to)=>allocationFallbacks.push({from,to}));
        batch.push(...result.value);offset+=result.count;
      }
      return batch;
    });
    const sizes=this.eccBatchTimings.map(t=>t.count);
    this.eccBatchPolicy.batchHistogram=Object.fromEntries([...new Set(sizes)].map(n=>[n,sizes.filter(x=>x===n).length]));
    this.eccBatchPolicy.actualBatchMin=sizes.length?Math.min(...sizes):0;this.eccBatchPolicy.actualBatchMax=sizes.length?Math.max(...sizes):0;
    this.eccBatchPolicy.maxWorkingSetBytes=policy.batchSize*policy.inFlight*1180560+policy.inFlight*16;
    this.eccBatchPolicy.reusedBatches=this.eccBatchTimings.filter(t=>t.reusedBuffers).length;
    for (let j = 0; j < traces.length; j++) {
      const t = traces[j];
      const q = t.quad;
      const dst = j * 8;
      output[dst] = q[0][0];
      output[dst + 1] = q[0][1];
      output[dst + 2] = q[1][0];
      output[dst + 3] = q[1][1];
      output[dst + 4] = q[2][0];
      output[dst + 5] = q[2][1];
      output[dst + 6] = q[3][0];
      output[dst + 7] = q[3][1];
      this.diagnostics.eccIterations += t.iterations;
      this.diagnostics.eccConverged += Number(t.converged);
      this.diagnostics.eccIterationLimits += Number(t.iterationLimitReached);
      this.diagnostics.eccFailures += Number(t.status < 0);
    }
    this.diagnostics.eccCandidates+=count;this.diagnostics.eccMs+=performance.now()-start;
    return output;
  }

  destroy(): void {
    if (this.destroyed) return;
    this.destroyed = true;

    void this.eccComponent?.then(component=>component.destroy()).catch(()=>{});
    this.paramsBuffer?.destroy();
    this.refStatsBuffer?.destroy();
    this.refsBuffer?.destroy();
    this.imageTexture?.destroy();
    this.device?.destroy();

    this.refStatsBuffer = null;
    this.refsBuffer = null;
    this.refStatsData = null;
    this.refPatternData = null;
    this.imageTexture = null;
    this.imageView = null;
    this.sampler = null;
  }
}
