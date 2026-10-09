import { useCallback, useEffect, useRef, useState } from 'react';
import { Button, Checkbox, Input, Modal, Select, Segmented, Slider, Tag } from 'antd';
import { ChevronLeft, ChevronRight, Expand, Film, Pause, Play, ScanLine, SkipBack, SkipForward } from 'lucide-react';
import { api, put } from './api';
import type { Annotation, Bundle, Camera, Dataset, Detail, Trajectory } from './types';
import AnnotationPanel from './AnnotationPanel';
import ExportPanel from './ExportPanel';

const time = (seconds:number) => `${Math.floor(seconds/60).toString().padStart(2,'0')}:${(seconds%60).toFixed(2).padStart(5,'0')}`;
const phaseLabel = (task:string) => /^place\b/i.test(task) ? '扫码后放置' : /scan/i.test(task) ? '拾取 / 扫码' : '任务阶段';
export default function App() {
  const [datasets,setDatasets] = useState<Dataset[]>([]), [datasetId,setDatasetId] = useState('');
  const [view,setView] = useState('full'), [rows,setRows] = useState<Trajectory[]>([]), [selected,setSelected] = useState('');
  const [detail,setDetail] = useState<Detail|null>(null), [bundle,setBundle] = useState<Bundle|null>(null);
  const [playing,setPlaying] = useState(false), [speed,setSpeed] = useState(1), [loading,setLoading] = useState(false);
  const [error,setError] = useState(''), [search,setSearch] = useState(''), [enlarged,setEnlarged] = useState<Camera|null>(null);
  const [annotation,setAnnotation]=useState<Annotation|null>(null),[pending,setPending]=useState(false),[unreviewedOnly,setUnreviewedOnly]=useState(false);
  const savedAnnotation=useRef(''),dirtyRef=useRef(false);
  const dirty=!!annotation&&(JSON.stringify(annotation)!==savedAnnotation.current||pending);
  dirtyRef.current=dirty;
  const [leaveAction,setLeaveAction]=useState<(()=>void)|null>(null);
  const requestLeave=useCallback((action:()=>void)=>{if(dirtyRef.current)setLeaveAction(()=>action);else action();},[]);
  const selectTrajectory=useCallback((id:string)=>{if(id!==selected)requestLeave(()=>setSelected(id));},[requestLeave,selected]);
  useEffect(()=>{const listener=(event:BeforeUnloadEvent)=>{if(dirtyRef.current){event.preventDefault();event.returnValue='';}};window.addEventListener('beforeunload',listener);return()=>window.removeEventListener('beforeunload',listener);},[]);
  const generation = useRef(0), busy = useRef(false), displayed = useRef(0);
  const frameRequest = useRef<AbortController|null>(null);
  const clock = useRef({wall:performance.now(),media:0});
  const dataset = datasets.find(d=>d.id===datasetId);
  const base = `datasets/${datasetId}/trajectories/${selected}`;

  useEffect(()=> {const controller=new AbortController(); api<Dataset[]>('datasets',controller.signal).then(result=>{
    setDatasets(result); if(result[0]){setDatasetId(result[0].id);setView(result[0].has_full_view?'full':'segments');}
  }).catch(e=>{if(e.name!=='AbortError')setError(String(e.message));});return()=>controller.abort();},[]);
  useEffect(()=>{
    if(!datasetId)return;
    const controller=new AbortController(); setPlaying(false);generation.current++;busy.current=false;setBundle(null);setDetail(null);setRows([]);setSelected('');
    api<Trajectory[]>(`datasets/${datasetId}/trajectories?view=${view}`,controller.signal).then(result=>{setRows(result);setSelected(result[0]?.id||'');}).catch(e=>{if(e.name!=='AbortError')setError(e.message);});
    return()=>{controller.abort();frameRequest.current?.abort();};
  },[datasetId,view]);
  useEffect(()=>{
    if(!selected)return;
    const controller=new AbortController(); generation.current++;busy.current=false;setPlaying(false);setBundle(null);setDetail(null);setError('');
    setAnnotation(null);setPending(false);savedAnnotation.current='';
    api<Detail>(base,controller.signal).then(setDetail).catch(e=>{if(e.name!=='AbortError')setError(e.message);});
    api<Annotation>(base+'/annotation',controller.signal).then(value=>{savedAnnotation.current=JSON.stringify(value);setAnnotation(value);}).catch(e=>{if(e.name!=='AbortError')setError(e.message);});
    return()=>controller.abort();
  },[base,selected]);

  const load = useCallback(async (index:number, resetClock=true)=>{
    if(!detail)return;
    const frame=Math.max(0,Math.min(index,detail.total_frames-1));
    frameRequest.current?.abort();
    const controller = new AbortController(); frameRequest.current=controller;
    const version=++generation.current; busy.current=true;setLoading(true);
    try {
      const next=await api<Bundle>(`${base}/frames/${frame}`,controller.signal);
      // Decode every available view before swapping the frame bundle atomically.
      await Promise.all(next.cameras.map(async camera=>{
        if(camera.image){const image=new Image();image.src=camera.image;try{await image.decode();}catch{camera.image=null;camera.error='浏览器无法解码此帧';}}
      }));
      if(generation.current!==version)return;
      setBundle(next);displayed.current=frame;setError('');
      if(resetClock)clock.current={wall:performance.now(),media:next.timestamp_s};
      return true;
    }catch(e){if(generation.current===version){setError(e instanceof Error?e.message:String(e));setPlaying(false);}}
    finally{if(generation.current===version){busy.current=false;setLoading(false);}}
  },[base,detail]);
  useEffect(()=>{if(detail)void load(0);},[detail,load]);
  useEffect(()=>{
    if(!playing||!detail)return;
    clock.current={wall:performance.now(),media:detail.timestamps_s[displayed.current]};
    const timer=window.setInterval(()=>{
      if(busy.current)return;
      const target=clock.current.media+(performance.now()-clock.current.wall)*speed/1000;
      let low=0,high=detail.total_frames-1;
      while(low<high){const mid=Math.ceil((low+high)/2);if(detail.timestamps_s[mid]<=target)low=mid;else high=mid-1;}
      if(low!==displayed.current)void load(low,false);
      else if(low===detail.total_frames-1)setPlaying(false);
    },20);
    return()=>window.clearInterval(timer);
  },[playing,speed,detail,load]);
  const changeTrajectory=useCallback((delta:number)=>{const i=rows.findIndex(row=>row.id===selected);const next=rows[i+delta];if(next)selectTrajectory(next.id);},[rows,selected,selectTrajectory]);
  const saveAnnotation=useCallback(async()=>{
    if(!annotation)return false;
    try{const saved=await put<Annotation>(base+'/annotation',annotation);savedAnnotation.current=JSON.stringify(saved);dirtyRef.current=false;setAnnotation(saved);setError('');setRows(previous=>previous.map(row=>row.id===saved.trajectory_id?{...row,review_status:saved.review_status}:row));return true;}
    catch(e){setError(e instanceof Error?e.message:String(e));return false;}
  },[annotation,base]);
  const nextUnreviewed=useCallback(()=>{
    const index=rows.findIndex(row=>row.id===selected);
    const next=[...rows.slice(index+1),...rows.slice(0,index)].find(row=>row.review_status==='unreviewed');
    if(next)selectTrajectory(next.id);
  },[rows,selected,selectTrajectory]);
  useEffect(()=>{
    const listener=(event:KeyboardEvent)=>{
      const element=event.target as HTMLElement;
      if(element.closest('input,textarea,select,[contenteditable="true"],.ant-select'))return;
      if(event.code==='Space'){event.preventDefault();setPlaying(p=>!p);}
      else if(event.code==='ArrowRight'){event.preventDefault();setPlaying(false);void load(displayed.current+1);}
      else if(event.code==='ArrowLeft'){event.preventDefault();setPlaying(false);void load(displayed.current-1);}
    };
    window.addEventListener('keydown',listener);return()=>window.removeEventListener('keydown',listener);
  },[load]);
  const current=bundle?.frame_index??0;
  const activeTask=detail?.task_boundaries.filter(boundary=>boundary.frame_index<=current).at(-1)?.task;
  const filtered=rows.filter(row=>(row.name+' '+row.source_batch).toLowerCase().includes(search.toLowerCase())&&(!unreviewedOnly||row.review_status==='unreviewed'));
  const cameraView=(camera:Camera,large=false)=><div className={`camera ${large?'large':''}`} key={camera.key}>
    <div className="picture">{camera.image?<img src={camera.image} alt={camera.title}/>:<div className="missing">画面不可用<br/><small>{camera.error}</small></div>}</div>
    <div className="camera-footer"><strong>{camera.title}</strong><span>{camera.width?`${camera.width} × ${camera.height}`:'缺帧'}{!large&&<button aria-label={`放大${camera.title}`} onClick={()=>setEnlarged(camera)}><Expand size={17}/></button>}</span></div>
  </div>;
  return <div className="app">
    <header><div className="brand"><Film size={26}/><div><h1>数据回放与标注</h1><span>R1-A7 · 离线数据工作台</span></div></div><div className="dataset-select"><label>数据集</label><Select value={datasetId||undefined} options={datasets.map(d=>({value:d.id,label:d.name}))} onChange={id=>requestLeave(()=>{const d=datasets.find(x=>x.id===id)!;setView(d.has_full_view?'full':'segments');setDatasetId(id);})}/></div><ExportPanel key={`${datasetId}:${view}`} datasetId={datasetId} view={view} converted={dataset?.format==='lerobot-v3'} dirty={dirty}/><Tag color="green">源数据只读 · 可标注</Tag></header>
    <div className="workspace"><aside className="navigation"><div className="panel-title">轨迹列表 <span>{rows.length} 条</span></div><Input placeholder="搜索轨迹 / 批次" value={search} onChange={e=>setSearch(e.target.value)}/><Checkbox checked={unreviewedOnly} onChange={e=>setUnreviewedOnly(e.target.checked)}>仅未审核</Checkbox><small className="review-count">已标 {rows.filter(row=>row.review_status!=='unreviewed').length} / {rows.length}</small><div className="episode-list">{filtered.map((row)=><button key={row.id} className={row.id===selected?'episode active':'episode'} onClick={()=>selectTrajectory(row.id)}><span className="episode-number">{String(rows.indexOf(row)+1).padStart(3,'0')}</span><div><strong>{row.name}</strong><small>{row.source_batch}</small><small>{row.total_frames} 帧 · {({unreviewed:'未审核',approved:'通过',needs_review:'待复核',excluded:'建议排除'})[row.review_status]}</small></div></button>)}</div><footer>源文件只读<br/>Space 播放 · ← → 逐帧</footer></aside>
    <main><div className="view-bar"><Segmented value={view} options={[{label:'完整轨迹',value:'full',disabled:!dataset?.has_full_view},{label:'训练分段',value:'segments'}]} onChange={next=>{if(next!==view)requestLeave(()=>setView(next));}}/><span>{dataset?.format} · {detail?.fps??dataset?.fps} FPS{loading?' · 加载中':''}</span></div>
      {error&&<div role="alert" className="error">{error}</div>}
      <div className="camera-grid">{bundle?bundle.cameras.map(c=>cameraView(c)):['全局相机','扫码区相机','左腕相机','右腕相机'].map(title=><div className="camera" key={title}><div className="picture missing">{loading?'正在读取画面':'请选择轨迹'}</div><div className="camera-footer">{title}</div></div>)}</div>
<section className="playback"><div className="transport"><div><Button aria-label="上一条" onClick={()=>changeTrajectory(-1)} icon={<SkipBack size={18}/>}/><Button aria-label="上一帧" disabled={!bundle} onClick={()=>{setPlaying(false);void load(current-1);}} icon={<ChevronLeft size={18}/>}/><Button type="primary" aria-label={playing?'暂停':'播放'} disabled={!bundle} onClick={()=>{if(current===detail!.total_frames-1){void load(0).then(ok=>{if(ok)setPlaying(true);});}else setPlaying(p=>!p);}} icon={playing?<Pause size={19}/>:<Play size={19}/>}>{playing?'暂停':'播放'}</Button><Button aria-label="下一帧" disabled={!bundle} onClick={()=>{setPlaying(false);void load(current+1);}} icon={<ChevronRight size={18}/>}/><Button aria-label="下一条" onClick={()=>changeTrajectory(1)} icon={<SkipForward size={18}/>}/></div><Select aria-label="播放速度" value={speed} options={[.25,.5,1,2].map(x=>({value:x,label:`${x}×`}))} onChange={setSpeed}/><strong>{time(bundle?.timestamp_s??0)} <span>/ {time(detail?.duration_s??0)}</span></strong><span>帧 {current} / {Math.max(0,(detail?.total_frames??1)-1)}</span></div>
      <Slider aria-label="回放时间轴" min={0} max={Math.max(1,(detail?.total_frames??1)-1)} value={current} tooltip={{formatter:i=>`帧 ${i}`}} onChange={i=>{setPlaying(false);void load(i);}}/>
      <div className="phase-track">{detail?.task_boundaries.map((boundary,i)=>{const end=detail.task_boundaries[i+1]?.frame_index??detail.total_frames;return <button key={i} style={{width:`${100*(end-boundary.frame_index)/detail.total_frames}%`}} onClick={()=>{setPlaying(false);void load(boundary.frame_index);}} title={boundary.task}>{phaseLabel(boundary.task)}</button>;})}</div><div className="event-track">{detail?.events.map((event,i)=><button key={i} style={{left:`${event.frame_index/Math.max(1,detail.total_frames-1)*100}%`}} onClick={()=>{setPlaying(false);void load(event.frame_index);}} title={`帧 ${event.frame_index}`}><ScanLine size={14}/>{event.label}</button>)}</div>
      {annotation&&(['stages','quality_intervals'] as const).map(layer=><div className={`annotation-track ${layer}`} key={layer}><span>{layer==='stages'?'阶段':'质量'}</span><div>{annotation[layer].map((interval,i)=><button key={i} style={{left:`${100*interval.start_frame/annotation.total_frames}%`,width:`${100*(interval.end_frame_exclusive-interval.start_frame)/annotation.total_frames}%`}} onClick={()=>{setPlaying(false);void load(interval.start_frame);}} title={`${interval.label} [${interval.start_frame}, ${interval.end_frame_exclusive})`}>{interval.label}</button>)}</div></div>)}
      </section>
    </main>
    <aside className="inspector"><div className="panel-title">当前轨迹 <Tag>{dirty?'未保存':'可标注'}</Tag></div><h2>{detail?.name??'—'}</h2><div className="source">{detail?.source_batch}</div>{annotation&&<AnnotationPanel key={annotation.trajectory_id} value={annotation} current={current} dirty={dirty} onChange={setAnnotation} onSave={saveAnnotation} onNext={nextUnreviewed} onSeek={frame=>{setPlaying(false);void load(frame);}} onPending={setPending}/>}<details className="metadata-details"><summary>商品 / task / 扫码事件（只读）</summary><section><h3>轨迹信息</h3><dl><dt>商品</dt><dd>{detail?.product_name||'未提供'}</dd><dt>条形码</dt><dd>{detail?.barcode||'未提供'}</dd><dt>SKU</dt><dd>{detail?.sku||'未提供'}</dd><dt>总帧数</dt><dd>{detail?.total_frames??'—'}</dd><dt>时长</dt><dd>{time(detail?.duration_s??0)}</dd></dl></section><section><h3>当前任务</h3><p className="task">{activeTask||'未提供任务文本'}</p></section><section><h3>扫码与任务事件</h3>{detail?.events.length?detail.events.map((event,i)=><button className="event-row" key={i} onClick={()=>{setPlaying(false);void load(event.frame_index);}}><ScanLine size={18}/><div><strong>{event.label}</strong><small>帧 {event.frame_index} {event.barcode}</small></div></button>):<p className="muted">没有可核实的扫码事件</p>}</section></details></aside></div>
    <Modal open={!!enlarged} footer={null} width="85vw" onCancel={()=>setEnlarged(null)} title={enlarged?.title}>{enlarged&&cameraView(bundle?.cameras.find(c=>c.key===enlarged.key)??enlarged,true)}</Modal>
    <Modal open={!!leaveAction} title="标注尚未保存" okText="放弃并切换" cancelText="继续标注" onOk={()=>{leaveAction?.();setLeaveAction(null);}} onCancel={()=>setLeaveAction(null)}>当前标注或打点未保存，切换将放弃这些改动。</Modal>
  </div>;
}
