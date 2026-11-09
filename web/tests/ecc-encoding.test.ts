import test from 'node:test';
import assert from 'node:assert/strict';
import {BatchedGpuEcc} from '../src/detect/ecc';

test('ECC compacts to one compute pass without changing the 303 dispatch schedule',async()=>{
 Object.assign(globalThis,{GPUBufferUsage:{STORAGE:1,COPY_DST:2,COPY_SRC:4,UNIFORM:8,MAP_READ:16},GPUMapMode:{READ:1}});
 const dispatches:number[][]=[],cleared:any[]=[];
 let begins=0,ends=0,submits=0,current=-1;
 const pipelines=Array.from({length:8},(_,index)=>({index,getBindGroupLayout:()=>({})}));
 const compute={setBindGroup(){},setPipeline(p:any){current=p.index;},dispatchWorkgroups(x:number,y:number){dispatches.push([current,x,y]);},end(){ends++;}};
 const device:any={limits:{maxStorageBufferBindingSize:128*1024*1024,maxBufferSize:256*1024*1024,maxComputeWorkgroupsPerDimension:65535},pushErrorScope(){},popErrorScope:async()=>null,
 createBuffer:({size}:any)=>({destroy(){},mapAsync:async()=>{},getMappedRange:()=>new ArrayBuffer(size),unmap(){}}),createBindGroup:()=>({}),
 createCommandEncoder:()=>({clearBuffer(b:any){cleared.push(b);},beginComputePass(){begins++;return compute;},copyBufferToBuffer(){},finish:()=>({})}),queue:{writeBuffer(){},submit(){submits++;}}};
 const component=new (BatchedGpuEcc as any)(device,pipelines);
 const quad=[[0,0],[128,0],[128,128],[0,128]];
 await component.run({}, {},256,256,[quad],[0]);
 assert.equal(begins,1);assert.equal(ends,1);assert.equal(submits,1);assert.equal(cleared.length,4);
 const expected=[[0,256,1],[1,256,1],[2,256,1]];
 for(let i=0;i<60;i++)expected.push([3,256,1],[4,1,1],[5,256,1],[6,1,1],[7,1,1]);
 assert.deepEqual(dispatches,expected);assert.equal(component.lastHostTiming.passes,1);assert.equal(component.lastHostTiming.dispatches,303);
 component.destroy();
});
