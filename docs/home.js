/* Home page: the leaderboard (every model and the IESO, scored against actual demand), the last 24 hours
   with every model on one chart, and a one-line "right now" summary. */
(function(){
 const H=36e5,key=hkey,tl=htl,ENTRIES=MIDS.map(id=>({id,f:id+"_h",n:MODELS[id].n,c:MODELS[id].c,page:MODELS[id].page,dash:!MODELS[id].demand}))
  .concat([{id:"ieso",f:"ieso_h",n:IESO.n,c:IESO.c,page:"compare.html",ieso:1}]);
 /* leaderboard: each model twice. "hour of" = its this-hour prediction (made in the first minutes of the hour);
    "at IESO's time" = the newest prediction it had made when the IESO first published the forecast that was scored */
 const K=i=>MIDS.indexOf(i),BOARD=ENTRIES.map(e=>e.ieso?{...e,g:r=>r.ieso_h,ga:a=>a.i}:{...e,hof:1,g:r=>r[e.f],ga:a=>a.h[K(e.id)]})
  .concat(MIDS.map(id=>({id:"@"+id,fid:id,n:MODELS[id].n,c:MODELS[id].c,page:MODELS[id].page,at:1,
   g:r=>r.mi&&!r.mil?r.mi[id]:null,ga:a=>a.m&&!a.ml?a.m[K(id)]:null})));
 let hidden=new Set();try{hidden=new Set(JSON.parse(localStorage.getItem("home_hidden")||"[]"))}catch(e){}
 function why(){const b=((window.LOG_META||{}).status||[]).find(s=>/IESO/.test(s.name)&&!s.ok);return b?" ("+b.name+": "+b.detail.slice(0,70)+")":""}

 /* average miss over the last `hours` completed hours. Models count only predictions made before the hour
    began once every model has at least 12 of them in the window ("live"); until then all hours are used and
    the table says so. The IESO's forecast and the "at IESO's time" predictions are saved as they happen
    (or were rebuilt from the site's own commit history), so all their hours count. Works on log rows (g)
    or on the whole-project archive (ga). */
 function scores(L,t0,hours,arc){
  const from=hours?key(t0-hours*H):"",now=key(t0),done=L.filter(r=>(arc?r.a:r.actual)!=null&&r.u>=from&&r.u<now),act=r=>arc?r.a:r.actual;
  const val=(e,r)=>arc?e.ga(r):e.g(r),live=BOARD.filter(e=>e.hof).every(e=>done.filter(r=>!r.bf&&val(e,r)!=null).length>=12),out={live};
  for(const e of BOARD){const rs=done.filter(r=>val(e,r)!=null&&(!e.hof||!live||!r.bf));
   out[e.id]=rs.length?{mae:rs.reduce((s,r)=>s+Math.abs(val(e,r)-act(r)),0)/rs.length,
    pct:100*rs.reduce((s,r)=>s+Math.abs(val(e,r)-act(r))/act(r),0)/rs.length,n:rs.length}:null}
  return out}
 const hm=m=>{m=Math.abs(m);return m<60?m+" min":Math.floor(m/60)+" h"+(m%60?" "+m%60+" min":"")},
  when=(r,mins)=>clockAt(Date.parse(r.u+":00Z")+mins*6e4),
  clockAt=ms=>{const d=htl(ms).slice(0,10),t=new Date(ms).toLocaleTimeString("en-CA",{timeZone:"America/Toronto",hour:"2-digit",minute:"2-digit",hour12:false});
   return(d==htl(Date.now()).slice(0,10)?"":d.slice(5)+" ")+t},
  SRC={"7d":"7-day run",n:"next-hour",h:"this-hour"};

 function render(){
  banner();
  const L=window.LOG||[],F=window.FC,LV=window.LIVE||{},t0=Math.floor(Date.now()/H)*H,row=ms=>L.find(r=>r.u==key(ms));
  const fcAt=(arr,ms)=>{const i=F?F.u.indexOf(key(ms)):-1;return i<0||!arr?null:arr[i]};

  /* ---- right now ---- */
  const ld=lastDone(L);
  $("hA").textContent=ld?MW(ld.v):"-";
  $("hAt").textContent=ld?(ld.fresh?"last completed hour, "+tl(ld.ms).slice(11):"latest on record, "+tl(ld.ms).slice(5)+" · "+ago(ld.ms)+", delayed"):"last completed hour · not published yet";
  const lv=liveNow();
  $("hL").textContent=lv.v==null?"-":MW(lv.v);$("hLt").textContent=lv.fresh?"so far this hour ("+tl(t0).slice(11)+", live)":"latest reading, not this hour";
  const ie=(ms,f)=>{const r=row(ms);return r&&r[f]!=null?r[f]:fcAt(F&&F.ieso,ms)};
  const iN=ie(t0+H,"ieso_n"),iH=ie(t0,"ieso_h");
  $("hI").textContent=iN==null?"-":MW(iN);$("hIt").textContent="IESO forecast for next hour ("+tl(t0+H).slice(11)+")"+(iN==null?" · not available"+why():"");

  /* ---- leaderboard ---- */
  const cell=(e,sfx,ms)=>{const r=row(ms);
   if(e.at){if(sfx=="_n"||!r||!r.mi||r.mi[e.fid]==null)return[null,""];   // known once the IESO's number for the hour is saved
    return[r.mi[e.fid],"",`made ${when(r,r.mim[e.fid])}, ${SRC[r.mis[e.fid]]||""}`+(r.mil?"<br>after the IESO published, so not scored":"")]}
   if(e.ieso&&sfx=="_h"&&r&&r.ieso_h!=null&&r.ip!=null)
    return[r.ieso_h,"",`first published ${when(r,r.ip)}, ${hm(r.ip)} ${r.ip<0?"before":"after"} the hour${r.ipf?" (or earlier)":""}`];
   const f=e.ieso?"ieso"+sfx:e.id+sfx;let v=r&&r[f]!=null?r[f]:null,src="";
   if(e.ieso){if(v==null)v=fcAt(F&&F.ieso,ms);return[v,""]}
   if(v==null&&sfx=="_h"&&r&&r[e.id+"_n"]!=null){v=r[e.id+"_n"];src="*"}
   if(v==null){v=fcAt(F&&F.m&&F.m[e.id],ms);if(v!=null)src="*"}return[v,src]};
  const d1=scores(L,t0,24),d7=scores(L,t0,168),A=window.ARC,dA=A?scores(A.rows,t0,0,1):{},rank=e=>(d7[e.id]||d1[e.id]||{mae:Infinity}).mae;
  const order=BOARD.slice().sort((a,b)=>rank(a)-rank(b)),worst=Math.max(...BOARD.map(e=>d7[e.id]?d7[e.id].mae:0))||1;
  const miss=(s,bar,c)=>s?`<b>±${Math.round(s.mae).toLocaleString()} MW</b> <small>${s.pct.toFixed(1)}%</small>`+
   (bar?`<div class="bar"><i style="width:${(100*s.mae/worst).toFixed(1)}%;background:${c}"></i></div>`:""):`<small>${"not scored yet"}</small>`;
  let medal=0;
  $("lb").innerHTML=`<thead><tr><th>#</th><th>Forecaster</th><th>This hour<br><small>${tl(t0).slice(11)}</small></th><th class="nx">Next hour<br><small>${tl(t0+H).slice(11)}</small></th>
   <th class="d1">Avg miss<br><small>last 24 h</small></th><th>Avg miss<br><small>last 7 days</small></th><th>Avg miss<br><small>whole project</small></th></tr></thead><tbody>`+
   order.map(e=>{const[a,as,an]=cell(e,"_h",t0),[b,bs]=cell(e,"_n",t0+H),scored=rank(e)<Infinity,pos=scored?++medal:"";
    const tag=e.ieso?"benchmark":e.at?"at IESO's time":"hour of";
    return`<tr class="${e.ieso?"ieso":""}${e.at?" at":""}${pos===1?" top":""}"><td>${pos}</td><td><span class="sw${e.at?" ring":""}" style="--c:${e.c}"></span><a href="${e.page}">${e.n}</a><br><small class="tag${e.at?" tat":""}">${tag}</small></td>
     <td>${a==null?(e.at?"<small>once the IESO's<br>number is saved</small>":"-"):MW(a)+as}${an?`<br><small>${an}</small>`:""}</td><td class="nx">${b==null?"-":MW(b)+bs}</td>
     <td class="d1">${miss(d1[e.id])}</td><td>${miss(d7[e.id],1,e.c)}</td><td class="wp">${miss(dA[e.id])}</td></tr>`}).join("")+"</tbody>";
  const ie7=d7.ieso,beat=ie7&&ie7.n>=12?MIDS.filter(id=>d7[id]&&d7[id].mae<ie7.mae).length:null;
  const allI=dA.ieso,beatA=allI?MIDS.filter(id=>dA["@"+id]&&dA["@"+id].mae<allI.mae).length:null;
  $("lbs").innerHTML=`<b>Hour of</b> = the model's prediction made in the first minutes of the hour, using demand up to the hour before. `+
   `<b>At IESO's time</b> = the newest prediction the model had already made when the IESO first published the number it is scored on (from the model's hourly 7-day run, or its next-hour prediction if the IESO published within the last hour), so it is a like-for-like race with the IESO. `+
   (beatA!=null?`Over the whole project, <b>${beatA} of ${MIDS.length}</b> models beat the IESO at its own time (${dA.ieso.n} hours). `:"")+(beat==null?"The IESO's forecast is saved each hour as it is published; it will be ranked once it has 12 scored hours."
    : `Ranked by average miss over the last 7 days (lower is better).`) +(d7.live?"Models are scored only on predictions made before each hour began."
    :"Not every model has 12 hours of predictions made in advance yet, so the scores also include hours filled in afterwards from past data, which flatters the models a little.")+
   (order.some(e=>!e.ieso&&[cell(e,"_h",t0)[1],cell(e,"_n",t0+H)[1]].includes("*"))?" * from the model's 7-day forecast run, used until its hourly run is available.":"");

  /* ---- the last 24 hours, every forecaster on one chart ---- */
  const hrs=[];for(let k=24;k>=0;k--)hrs.push(t0-k*H);
  const act=ms=>ms==t0?(lv.fresh?lv.v:null):(r=>r&&r.actual!=null?r.actual:null)(row(ms));
  const S=ENTRIES.filter(e=>!hidden.has(e.id)).map(e=>({ys:hrs.map(ms=>(r=>r?r[e.f]:null)(row(ms))),color:e.c,w:e.ieso?2:1.5,dash:e.dash?[5,4]:null}));
  S.push({ys:hrs.map(act),color:css("--fg"),w:2.5,dots:true,hollow:hrs.map(ms=>ms==t0)});
  if(hrs.some(ms=>act(ms)!=null))plot($("tr"),hrs.map((_,i)=>i),S,{nt:7,xf:(x,j)=>tl(hrs[j]).slice(11,13)+"h"});
  $("lg").innerHTML=`<span class="chip on" style="--c:${css("--fg")}"><i></i>Actual demand</span>`+ENTRIES.map(e=>
   `<button class="chip${hidden.has(e.id)?"":" on"}${e.dash?" dash":""}" data-id="${e.id}" style="--c:${e.c}"><i></i>${e.n}</button>`).join("")}

 $("lg").addEventListener("click",ev=>{const b=ev.target.closest("button[data-id]");if(!b)return;const id=b.dataset.id;
  hidden.has(id)?hidden.delete(id):hidden.add(id);try{localStorage.setItem("home_hidden",JSON.stringify([...hidden]))}catch(e){}render()});
 function reload(){let n=3;for(const f of["demand_log.js","forecast.js","score_archive.js"]){const s=document.createElement("script");s.src=f+"?"+Date.now();
  s.onload=s.onerror=()=>{s.remove();if(!--n)render()};document.head.appendChild(s)}}
 window.onLive=render;addEventListener("resize",render);setInterval(reload,300000);render();
})();
