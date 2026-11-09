/* Home page: the leaderboard (every model and the IESO, scored against actual demand), the last 24 hours
   with every model on one chart, and a one-line "right now" summary. */
(function(){
 const H=36e5,key=hkey,tl=htl,ENTRIES=MIDS.map(id=>({id,f:id+"_h",n:MODELS[id].n,c:MODELS[id].c,page:MODELS[id].page,dash:!MODELS[id].demand}))
  .concat([{id:"ieso",f:"ieso_h",n:IESO.n,c:IESO.c,page:"compare.html",ieso:1}]);
 let hidden=new Set();try{hidden=new Set(JSON.parse(localStorage.getItem("home_hidden")||"[]"))}catch(e){}
 function why(){const b=((window.LOG_META||{}).status||[]).find(s=>/IESO/.test(s.name)&&!s.ok);return b?" ("+b.name+": "+b.detail.slice(0,70)+")":""}

 /* average miss over the last `hours` completed hours. Models count only predictions made before the hour
    began once every model has at least 12 of them in the window ("live"); until then all hours are used and
    the table says so. The IESO's forecast is only ever saved before the hour, so all its hours count. */
 function scores(L,t0,hours){
  const from=key(t0-hours*H),now=key(t0),done=L.filter(r=>r.actual!=null&&r.u>=from&&r.u<now);
  const live=MIDS.every(id=>done.filter(r=>!r.bf&&r[id+"_h"]!=null).length>=12),out={live};
  for(const e of ENTRIES){const rs=done.filter(r=>r[e.f]!=null&&(e.ieso||!live||!r.bf));
   out[e.id]=rs.length?{mae:rs.reduce((s,r)=>s+Math.abs(r[e.f]-r.actual),0)/rs.length,
    pct:100*rs.reduce((s,r)=>s+Math.abs(r[e.f]-r.actual)/r.actual,0)/rs.length,n:rs.length}:null}
  return out}

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
  const cell=(e,sfx,ms)=>{const r=row(ms),f=e.ieso?"ieso"+sfx:e.id+sfx;let v=r&&r[f]!=null?r[f]:null,src="";
   if(e.ieso){if(v==null)v=fcAt(F&&F.ieso,ms);return[v,""]}
   if(v==null&&sfx=="_h"&&r&&r[e.id+"_n"]!=null){v=r[e.id+"_n"];src="*"}
   if(v==null){v=fcAt(F&&F.m&&F.m[e.id],ms);if(v!=null)src="*"}return[v,src]};
  const d1=scores(L,t0,24),d7=scores(L,t0,168),rank=e=>(d7[e.id]||d1[e.id]||{mae:Infinity}).mae;
  const order=ENTRIES.slice().sort((a,b)=>rank(a)-rank(b)),worst=Math.max(...ENTRIES.map(e=>d7[e.id]?d7[e.id].mae:0))||1;
  const miss=(s,bar,c)=>s?`<b>±${Math.round(s.mae).toLocaleString()} MW</b> <small>${s.pct.toFixed(1)}%</small>`+
   (bar?`<div class="bar"><i style="width:${(100*s.mae/worst).toFixed(1)}%;background:${c}"></i></div>`:""):`<small>${"not scored yet"}</small>`;
  let medal=0;
  $("lb").innerHTML=`<thead><tr><th>#</th><th>Forecaster</th><th>This hour<br><small>${tl(t0).slice(11)}</small></th><th class="nx">Next hour<br><small>${tl(t0+H).slice(11)}</small></th>
   <th class="d1">Avg miss<br><small>last 24 h</small></th><th>Avg miss<br><small>last 7 days</small></th></tr></thead><tbody>`+
   order.map(e=>{const[a,as]=cell(e,"_h",t0),[b,bs]=cell(e,"_n",t0+H),scored=rank(e)<Infinity,pos=scored?++medal:"";
    return`<tr class="${e.ieso?"ieso":""}${pos===1?" top":""}"><td>${pos}</td><td><span class="sw" style="background:${e.c}"></span><a href="${e.page}">${e.n}</a>${e.ieso?' <small class="tag">benchmark</small>':""}</td>
     <td>${a==null?"-":MW(a)+as}</td><td class="nx">${b==null?"-":MW(b)+bs}</td><td class="d1">${miss(d1[e.id])}</td><td>${miss(d7[e.id],1,e.c)}</td></tr>`}).join("")+"</tbody>";
  const ie7=d7.ieso,beat=ie7&&ie7.n>=12?MIDS.filter(id=>d7[id]&&d7[id].mae<ie7.mae).length:null;
  $("lbs").innerHTML=(beat==null?"The IESO's forecast is saved each hour as it is published; it will be ranked once it has 12 scored hours.")
    :` Ranked by average miss over the last 7 days (lower is better). `+(d7.live?"Models are scored only on predictions made before each hour began."
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
 function reload(){let n=2;for(const f of["demand_log.js","forecast.js"]){const s=document.createElement("script");s.src=f+"?"+Date.now();
  s.onload=s.onerror=()=>{s.remove();if(!--n)render()};document.head.appendChild(s)}}
 window.onLive=render;addEventListener("resize",render);setInterval(reload,300000);render();
})();
