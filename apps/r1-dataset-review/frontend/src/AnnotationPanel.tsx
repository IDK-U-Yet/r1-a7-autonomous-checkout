import { useCallback, useEffect, useState } from 'react';
import { Button, Input, InputNumber, Select, Switch, Tag } from 'antd';
import type { Annotation, Interval, QualityInterval } from './types';

interface Props {
  value:Annotation; current:number; dirty:boolean;
  onChange:(value:Annotation)=>void; onSave:()=>Promise<boolean>; onNext:()=>void; onSeek:(frame:number)=>void; onPending:(pending:boolean)=>void;
}
const statusOptions=[{value:'unreviewed',label:'未审核'},{value:'approved',label:'通过'},{value:'needs_review',label:'待复核'},{value:'excluded',label:'建议排除'}];
const categories=[{value:'error',label:'错误动作'},{value:'pause',label:'不必要停顿'},{value:'recovery',label:'恢复（默认保留）'},{value:'custom',label:'其他'}];

export default function AnnotationPanel({value,current,dirty,onChange,onSave,onNext,onSeek,onPending}:Props) {
  const [layer,setLayer]=useState<'stages'|'quality_intervals'>('stages');
  const [label,setLabel]=useState('抓取'),[category,setCategory]=useState<QualityInterval['category']>('error');
  const [exclude,setExclude]=useState(true),[start,setStart]=useState<number|null>(null),[end,setEnd]=useState<number|null>(null);
  const [editing,setEditing]=useState<number|null>(null),[error,setError]=useState(''),[saving,setSaving]=useState(false);
  const [success,setSuccess]=useState('');
  const hasDraft=start!==null||end!==null;
  useEffect(()=>onPending(hasDraft),[hasDraft,onPending]);
  const markStart=useCallback(()=>{setStart(current);setEnd(null);setError('');},[current]);
  const markEnd=useCallback(()=>{if(start===null){setError('请先设起点');return;}setEnd(current+1);setError('');},[current,start]);
  const save=useCallback(async(next=false)=>{
    setError('');setSuccess('');
    if(hasDraft){setError('区间尚未加入列表：请先添加区间或取消打点');return;}
    setSaving(true);
    try{if(await onSave()){setSuccess('已保存');if(next)onNext();}else setError('保存失败，标注仍保留；请查看页面错误提示');}
    finally{setSaving(false);}
  },[hasDraft,onSave,onNext]);
  useEffect(()=>{
    const listener=(event:KeyboardEvent)=>{
      if((event.ctrlKey||event.metaKey)&&event.code==='KeyS'){event.preventDefault();void save();return;}
      if((event.target as HTMLElement).closest('input,textarea,select,[contenteditable="true"],.ant-select'))return;
      if(event.code==='KeyI'){event.preventDefault();markStart();}
      if(event.code==='KeyO'){event.preventDefault();markEnd();}
    };
    window.addEventListener('keydown',listener);return()=>window.removeEventListener('keydown',listener);
  },[save,markStart,markEnd]);
  const clear=()=>{setStart(null);setEnd(null);setEditing(null);setError('');};
  const add=()=>{
    if(start===null||end===null||!Number.isInteger(start)||!Number.isInteger(end)||start<0||start>=end||end>value.total_frames||!label.trim()){
      setError(`需要标签和有效区间：0 ≤ 起点 < 终点 ≤ ${value.total_frames}`);return;
    }
    const interval:Interval={label:label.trim(),start_frame:start,end_frame_exclusive:end};
    const candidate=layer==='stages'?interval:{...interval,category,exclude_from_training:exclude};
    const intervals:Interval[]=[...value[layer]];
    if(editing===null)intervals.push(candidate);else intervals[editing]=candidate;
    const ordered=[...intervals].sort((a,b)=>a.start_frame-b.start_frame);
    if(ordered.some((item,i)=>i>0&&ordered[i-1].end_frame_exclusive>item.start_frame)){
      setError('同层区间不能重叠；请修改起点/终点');return;
    }
    onChange({...value,[layer]:intervals});clear();setSuccess('');
  };
  const edit=(name:'stages'|'quality_intervals',index:number)=>{
    const interval=value[name][index];setLayer(name);setEditing(index);setLabel(interval.label);
    setStart(interval.start_frame);setEnd(interval.end_frame_exclusive);setError('');
    if(name==='quality_intervals'){const quality=interval as QualityInterval;setCategory(quality.category);setExclude(quality.exclude_from_training);}
  };
  const list=(name:'stages'|'quality_intervals')=>value[name].map((interval,index)=><div className="interval-item" key={index}>
    <button className="interval-jump" onClick={()=>onSeek(interval.start_frame)}><strong>{interval.label}</strong><small>[{interval.start_frame}, {interval.end_frame_exclusive})</small></button>
    {name==='quality_intervals'&&<Tag color={(interval as QualityInterval).exclude_from_training?'orange':'green'}>{(interval as QualityInterval).exclude_from_training?'排除':'保留'}</Tag>}
    <button className="text-button" onClick={()=>edit(name,index)}>编辑</button>
    <button className="text-button danger" onClick={()=>{onChange({...value,[name]:value[name].filter((_,i)=>i!==index)});clear();}}>移除</button>
  </div>);
  return <section className="annotation-panel">
    <div className="panel-title">人工标注 <Tag color={dirty?'orange':'green'}>{dirty?'未保存':`修订 ${value.revision}`}</Tag></div>
    <label>审核状态</label><Select aria-label="审核状态" value={value.review_status} options={statusOptions} onChange={review_status=>onChange({...value,review_status})}/>
    <div className="annotation-fields"><div><label>轨迹类型</label><Select aria-label="轨迹类型" value={value.episode_type} options={[{value:'',label:'未指定'},{value:'standard_pick',label:'standard_pick'},{value:'tabletop_recovery',label:'tabletop_recovery'}]} onChange={episode_type=>onChange({...value,episode_type})}/></div><div><label>朝向</label><Select aria-label="商品朝向" value={value.orientation} options={[{value:'',label:'未指定'},...Array.from({length:8},(_,i)=>({value:`O${i+1}`,label:`O${i+1}`}))]} onChange={orientation=>onChange({...value,orientation})}/></div></div>
    <h3>帧区间 · I / O 打点</h3><Select aria-label="区间图层" value={layer} options={[{value:'stages',label:'动作阶段'},{value:'quality_intervals',label:'质量区间'}]} onChange={name=>{clear();setLayer(name);setLabel(name==='stages'?'抓取':'错误动作');}}/>
    {layer==='stages'?<div className="stage-presets">{['抓取','重新定向 / 翻转','扫码','放置'].map(text=><button key={text} onClick={()=>setLabel(text)}>{text}</button>)}</div>:<Select aria-label="质量类型" value={category} options={categories} onChange={next=>{setCategory(next);setExclude(next==='error'||next==='pause');setLabel(categories.find(c=>c.value===next)!.label);}}/>}
    <Input aria-label="区间标签" placeholder="可自定义阶段标签" value={label} onChange={e=>setLabel(e.target.value)}/>
    <div className="range-buttons"><Button onClick={markStart}>设起点 I</Button><Button onClick={markEnd}>设终点 O</Button></div>
    <div className="annotation-fields"><div><label>起始帧</label><InputNumber aria-label="起始帧" min={0} max={value.total_frames-1} precision={0} value={start} onChange={setStart}/></div><div><label>结束帧（不含）</label><InputNumber aria-label="结束帧不含" min={1} max={value.total_frames} precision={0} value={end} onChange={setEnd}/></div></div>
    {layer==='quality_intervals'&&<div className="exclude-switch"><Switch checked={exclude} onChange={setExclude}/>排除训练（仅标记，尚未导出）</div>}
    <div className="range-buttons"><Button onClick={add}>{editing===null?'添加区间':'更新区间'}</Button><Button onClick={clear}>取消打点</Button></div>
    <h3>动作阶段 {value.stages.length}</h3>{list('stages')}
    <h3>质量区间 {value.quality_intervals.length}</h3>{list('quality_intervals')}
    <label>备注</label><Input.TextArea aria-label="标注备注" rows={3} value={value.notes} onChange={e=>onChange({...value,notes:e.target.value})}/>
    {error&&<div role="alert" className="error">{error}</div>}{success&&!dirty&&<p className="saved">{success}</p>}
    <div className="annotation-save"><Button type="primary" loading={saving} onClick={()=>void save()}>保存 Ctrl+S</Button><Button loading={saving} onClick={()=>void save(true)}>保存并下一条</Button></div>
    <p className="muted">标注独立保存。动作阶段不改写 task；建议排除不会删除 Episode。完整轨迹和训练分段暂时独立标注，请优先使用完整轨迹。</p>
  </section>;
}
