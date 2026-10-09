export async function api<T>(path:string, signal?:AbortSignal):Promise<T> {
  const response = await fetch('/api/'+path,{signal});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
  return result;
}

async function write<T>(path:string, value:unknown, method:string):Promise<T> {
  const response=await fetch('/api/'+path,{method,headers:{'Content-Type':'application/json','X-R1-Annotation':'1'},body:JSON.stringify(value)});
  const result=await response.json();
  if(!response.ok)throw new Error(result.error||`HTTP ${response.status}`);
  return result;
}
export const put = <T>(path:string,value:unknown) => write<T>(path,value,'PUT');
export const post = <T>(path:string,value:unknown) => write<T>(path,value,'POST');
