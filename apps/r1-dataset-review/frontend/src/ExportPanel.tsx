import { useState } from 'react';
import { Alert, Button, Checkbox, InputNumber, Modal, Tag } from 'antd';
import { api, post } from './api';

type Summary = {snapshot_sha256:string;trajectory_count:number;review_status_counts:Record<string,number>;
  total_frames:number;excluded_training_starts:number|null;allowed_training_starts:number|null;warnings:string[];
  output_dir?:string;files?:string[]};

export default function ExportPanel({datasetId,view,converted,dirty}:{datasetId:string;view:string;converted:boolean;dirty:boolean}) {
  const [open,setOpen]=useState(false),[filter,setFilter]=useState(false),[chunk,setChunk]=useState(50);
  const [preview,setPreview]=useState<Summary|null>(null),[result,setResult]=useState<Summary|null>(null);
  const [busy,setBusy]=useState(false),[error,setError]=useState('');
  const inspect=async()=>{setBusy(true);setError('');setResult(null);try{
    setPreview(await api<Summary>(`datasets/${datasetId}/export-preview?view=${view}&include_training_filter=${filter}&action_chunk_size=${chunk}`));
  }catch(e){setPreview(null);setError(e instanceof Error?e.message:String(e));}finally{setBusy(false);}};
  const save=async()=>{if(!preview)return;setBusy(true);setError('');try{
    setResult(await post<Summary>(`datasets/${datasetId}/exports`,{view,include_training_filter:filter,action_chunk_size:chunk,expected_snapshot_hash:preview.snapshot_sha256}));
  }catch(e){setPreview(null);setError(e instanceof Error?e.message:String(e));}finally{setBusy(false);}};
  return <><Button disabled={!datasetId||dirty} title={dirty?'请先保存或取消当前标注':'导出当前数据集、当前视图'} onClick={()=>setOpen(true)}>导出标注</Button>
    <Modal title="导出标注与训练过滤" open={open} width={700} onCancel={()=>{if(!busy)setOpen(false);}} footer={null}>
      <p>范围：当前数据集的{view==='full'?'完整轨迹':'训练分段'}视图。两种视图的标注独立，不自动混合。</p>
      <Checkbox disabled={!converted||busy} checked={filter} onChange={e=>{setFilter(e.target.checked);setPreview(null);setResult(null);}}>同时生成 training_filter.json</Checkbox>
      {!converted&&<p>原始数据可导出标注；转换成 LeRobot 后才能生成训练过滤。</p>}
      {filter&&<p>动作块长度 <InputNumber min={1} max={10000} value={chunk} disabled={busy} onChange={n=>{if(n!==null){setChunk(n);setPreview(null);setResult(null);}}}/> 必须与训练模型的 chunk_size 一致，不是 n_action_steps。</p>}
      <Alert style={{margin:'16px 0'}} type="info" showIcon message="源数据不修改；不会自动启用训练过滤或重算统计量。" description="未审核默认保留；恢复段默认保留；整条废弃或明确勾选的质量区间才排除。子任务标注仅作标签，不会自动启用模型高层规划。"/>
      {error&&<Alert type="error" showIcon message={error}/>}
      {preview&&<section><p><Tag>{preview.trajectory_count} 条轨迹</Tag><Tag>{preview.total_frames} 帧</Tag></p>
        <p>通过 {preview.review_status_counts.approved||0} · 未审核 {preview.review_status_counts.unreviewed||0} · 待复核 {preview.review_status_counts.needs_review||0} · 整条废弃 {preview.review_status_counts.excluded||0}</p>
        {filter&&<p>训练起始帧：保留 {preview.allowed_training_starts}，排除 {preview.excluded_training_starts}。排除范围已按动作块向前扩展，不跨分段。</p>}
        {preview.warnings.map(w=><p key={w} style={{color:'#d6b46d'}}>{w}</p>)}
      </section>}
      {result&&<Alert style={{margin:'16px 0'}} type="success" message="已导出到独立新目录" description={<><p style={{overflowWrap:'anywhere',userSelect:'text'}}>{result.output_dir}</p><p>{result.files?.join(' · ')}</p><p>COMPLETE.json 为完成标志。训练前核对数据集指纹、动作块长度，并按过滤后的样本重新计算统计。</p></>}/>}
      <div style={{display:'flex',gap:12,marginTop:20}}><Button loading={busy} onClick={()=>void inspect()}>预览计数</Button><Button type="primary" disabled={!preview||!!result||busy||dirty} onClick={()=>void save()}>导出到新目录</Button></div>
    </Modal></>;
}
