export interface Dataset {id:string;name:string;format:string;fps:number;full_count:number;segment_count:number;has_full_view:boolean}
export interface Trajectory {id:string;name:string;source_batch:string;total_frames:number;tasks:string[];review_status:ReviewStatus}
export interface Detail extends Trajectory {fps:number;duration_s:number;timestamps_s:number[];product_name:string;barcode:string;sku:string;events:{frame_index:number;label:string;barcode?:string}[];task_boundaries:{frame_index:number;task:string}[]}
export interface Camera {key:string;title:string;image:string|null;error:string|null;width?:number;height?:number;media_timestamp_s?:number}
export interface Bundle {trajectory_id:string;frame_index:number;timestamp_s:number;cameras:Camera[]}
export type ReviewStatus = 'unreviewed'|'approved'|'needs_review'|'excluded';
export interface Interval {label:string;start_frame:number;end_frame_exclusive:number}
export interface QualityInterval extends Interval {category:'error'|'pause'|'recovery'|'custom';exclude_from_training:boolean}
export interface Annotation {schema_version:number;dataset_id:string;trajectory_id:string;total_frames:number;metadata_fingerprint:string;revision:number;updated_at:string|null;review_status:ReviewStatus;episode_type:''|'standard_pick'|'tabletop_recovery';orientation:string;stages:Interval[];quality_intervals:QualityInterval[];notes:string}
