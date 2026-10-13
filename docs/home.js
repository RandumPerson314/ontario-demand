/* Home page: live tracker, IESO forecast, and every model's this-hour / next-hour prediction. */
(function(){
 const H=36e5,key=ms=>new Date(ms).toISOString().slice(0,13)+":00",
 tl=ms=>new Date(ms).toLocaleString("sv-SE",{timeZone:"America/Toronto"}).slice(0,13)+":00";
 function why(){const b=((window.LOG_META||{}).status||[]).find(s=>/IESO/.test(s.name)&&!s.ok);return b?" ("+b.name+": "+b.detail.slice(0,70)+")":""}
 function render(){
  banner();
  const L=window.LOG||[],M=window.LOG_META||{},F=window.FC,t0=Math.floor(Date.now()/H)*H,row=ms=>L.find(r=>r.u==key(ms));
  const fcAt=(arr,ms)=>{const i=F?F.u.indexOf(key(ms)):-1;return i<0||!arr?null:arr[i]};
  /* most recent completed hour */
  const prev=row(t0-H);
  if(prev&&prev.actual!=null){$("hA").textContent=MW(prev.actual);$("hAt").textContent=tl(t0-H).slice(5)+" Toronto time"}
  else{$("hA").textContent="-";$("hAt").textContent=tl(t0-H).slice(5)+" hour · not published yet"}
  /* this hour so far, only if it really is this hour */
  if(M.live_mw!=null&&lab(M.live_t)==tl(t0)){$("hL").textContent=MW(M.live_mw);$("hLt").textContent=tl(t0).slice(5)+" hour · average of "+M.live_n+" readings"}
  else{$("hL").textContent="-";$("hLt").textContent="waiting for this hour's first readings"}
  /* IESO official forecasts */
  const ie=(ms,f)=>{const r=row(ms);return r&&r[f]!=null?r[f]:fcAt(F&&F.ieso,ms)};
  const iN=ie(t0+H,"ieso_n"),iH=ie(t0,"ieso_h");
  $("hI").textContent=iN==null?"-":MW(iN);$("hIt").textContent=tl(t0+H).slice(5)+" Toronto time"+(iN==null?" · not available"+why():"");
  $("hJ").textContent=iH==null?"-":MW(iH);$("hJt").textContent=tl(t0).slice(5)+" Toronto time"+(iH==null?" · not available"+why():"");
  /* tracker chart */
  const D=[];for(let k=24;k>=1;k--){const r=row(t0-k*H);D.push({t:tl(t0-k*H),a:r&&r.actual!=null?r.actual:null,i:r?r.ieso_h:null,h:false})}
  const rc=row(t0);D.push({t:tl(t0),a:M.live_mw!=null&&lab(M.live_t)==tl(t0)?M.live_mw:null,i:rc?rc.ieso_h:null,h:true});
  if(D.some(d=>d.a!=null))plot($("tr"),D.map((_,i)=>i),[{ys:D.map(d=>d.i),color:IESO.c},{ys:D.map(d=>d.a),color:css("--fg"),dots:true,hollow:D.map(d=>d.h)}],
   {nt:7,xf:(x,j)=>D[j].t.slice(11,13)+"h"});
  /* model cards */
  const cell=(id,sfx,ms)=>{const r=row(ms);let v=r&&r[id+sfx]!=null?r[id+sfx]:null,src="";
   if(v==null&&sfx=="_h"&&r&&r[id+"_n"]!=null){v=r[id+"_n"];src="*"}
   if(v==null){v=fcAt(F&&F.m&&F.m[id],ms);if(v!=null)src="*"}return[v,src]};
  const rowsHtml=id=>{const[a,as]=cell(id,"_h",t0),[b,bs]=cell(id,"_n",t0+H);
   return`<tr><td>This hour <small>${tl(t0).slice(11)}</small></td><td>${a==null?"-":MW(a)+as}</td></tr><tr><td>Next hour <small>${tl(t0+H).slice(11)}</small></td><td>${b==null?"-":MW(b)+bs}</td></tr>`};
  $("mg").innerHTML=MIDS.map(id=>`<div class="mc" style="border-left-color:${MODELS[id].c}"><a href="${MODELS[id].page}">${MODELS[id].n}</a>
   <table>${rowsHtml(id)}</table></div>`).join("")+
   `<div class="mc" style="border-left-color:${IESO.c}"><a href="compare.html">${IESO.n}</a><table><tr><td>This hour <small>${tl(t0).slice(11)}</small></td><td>${iH==null?"-":MW(iH)}</td></tr>
   <tr><td>Next hour <small>${tl(t0+H).slice(11)}</small></td><td>${iN==null?"-":MW(iN)}</td></tr></table></div>
   <p style="grid-column:1/-1;margin:0" class="mut">* from the model's 7-day forecast run, used when its hourly run isn't available yet (usually just after the hour changes).</p>`}
 function reload(){let n=2;for(const f of["demand_log.js","forecast.js"]){const s=document.createElement("script");s.src=f+"?"+Date.now();
  s.onload=s.onerror=()=>{s.remove();if(!--n)render()};document.head.appendChild(s)}}
 addEventListener("resize",render);setInterval(reload,300000);render();
})();
