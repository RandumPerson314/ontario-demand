/* One page design for every model. The wrapper html sets window.MODEL_ID. */
(function(){
 const id=window.MODEL_ID,Mo=MODELS[id];document.title=Mo.n;
 const info=window.MI&&MI[id],det=info&&info[Mo.demand?"wk":"all"],isPoly=window.MJ&&MJ[id]&&MJ[id].kind=="poly";
 $("app").innerHTML=`<h1>${Mo.n}</h1><div id="banner" class="warn" style="display:none"></div>
 <p class="lead">${Mo.d}</p>
 <p><span class="tag ${Mo.demand?"yes":"no"}">${Mo.demand?"Uses recent demand":"Weather and calendar only"}</span>
 <a href="compare.html">How does it compare with the other models?</a></p>
 <div class="cards">
 <div class="card"><span>This hour: prediction</span><b id="p">-</b><small id="pt"></small></div>
 <div class="card"><span>Next hour: prediction</span><b id="n">-</b><small id="nt"></small></div>
 <div class="card"><span>Actual demand this hour (so far)</span><b id="l">-</b><small id="lt"></small></div>
 <div class="card"><span>Last completed hour (IESO)</span><b id="a">-</b><small id="at"></small></div>
 <div class="card"><span>Accuracy (last 7 days)</span><b id="m">-</b><small id="mt"></small></div></div>
 <div class="panel"><h2>7-day forecast</h2><canvas id="cf"></canvas>
 <p>Coloured line = this model, purple = the IESO's own official forecast (it reaches only the next day or so unless its long-range report is readable). <b>Click the graph to pick an hour</b> (dashed line) to explore below. Times are Toronto time.</p><div id="days"></div></div>
 <div class="panel"><h2>Past hours: prediction vs actual</h2>
 <label style="display:block;margin:0 0 4px">Show <select id="rng"><option value="24">last 24 hours</option><option value="72" selected>last 3 days</option>
 <option value="168">last 7 days</option><option value="336">last 14 days</option></select></label>
 <canvas id="c"></canvas><p id="msg"></p>
 <p>Coloured line = this model's this-hour prediction, dark = actual IESO demand, purple = the IESO's forecast for that hour. Solid dots were predicted live, before the hour started; hollow dots were filled in afterwards.</p></div>
 <div class="panel" id="ex"><h2>What drives this model's forecast?</h2>
 <p id="exTitle">Loading...</p>
 <p>This explores the model's 7-day forecast for the hour you clicked above. It runs in your browser, so move any slider and the prediction updates.</p>
 <div class="ctl" id="ctl"></div><button onclick="resetSc();drawEx()">Reset to the forecast</button>
 <h3>Which factors move demand most?</h3>
 <p>Each bar spans the lowest to highest demand as that one factor sweeps its whole range, everything else as set above.</p><div id="tor"></div>
 <h3>Prediction across each factor</h3>
 <label style="display:block"><input type="checkbox" id="shy" checked> Same y-axis for all graphs</label>
 <div class="grid" id="pd"></div></div>
 ${isPoly?`<div class="panel"><h2>The polynomial, written out</h2>
 <p>Every term of the fitted formula, with its contribution at the hour you picked. <code>z_</code> terms are products of standardized core inputs; other inputs are in their raw units.</p>
 <label style="display:block;margin:6px 0"><input type="checkbox" id="srt"> Sort by size of current contribution</label><pre id="fm"></pre></div>`:""}
 <div class="panel"><h2>About this model</h2><div id="about"></div></div>`;

 /* ---------- cards, forecast, past hours ---------- */
 function forecast(){
  const F=window.FC,arr=F&&F.m&&F.m[id];
  if(!arr){$("days").textContent="Forecast not available yet.";return}
  plot($("cf"),F.u.map((_,i)=>i),[{ys:arr,color:Mo.c},{ys:F.ieso||[],color:IESO.c}],{nt:8,mark:sel,
   xf:(x,j)=>DOW[new Date(F.t[j].slice(0,10)+"T12:00:00").getDay()]+" "+F.t[j].slice(11,13)+"h"});
  $("days").innerHTML=dailyTable(F,arr)}
 $("cf").onclick=e=>{const c=e.target,m=c._m,F=window.FC;if(!m||!F)return;const r=c.getBoundingClientRect();
  sel=Math.max(0,Math.min(F.u.length-1,Math.round(m.x0+(e.clientX-r.left-m.Lm)/(m.w-m.Lm-m.R)*(m.x1-m.x0))));resetSc();forecast();drawEx()};
 function past(){
  const L=window.LOG||[];if(!L.length){$("msg").textContent="No data yet: the first update hasn't run.";return}
  const D=L.slice(-+$("rng").value);
  plot($("c"),D.map((_,i)=>i),[{ys:D.map(r=>r.actual),color:css("--fg")},{ys:D.map(r=>r.ieso_h),color:IESO.c},
   {ys:D.map(r=>r[id+"_h"]),color:Mo.c,dots:D.length<=100,hollow:D.map(r=>r.bf)}],{xf:(x,j)=>D[j].t.slice(5,13).replace("T"," ")+"h"});
  $("msg").textContent="Last update: "+(window.LOG_META&&LOG_META.updated?new Date(LOG_META.updated).toLocaleString():"-")}

 /* ---------- explorer ---------- */
 let sel=0,Sc={};
 const ready=()=>window.FC&&FC.names&&FC.u&&FC.u.length&&window.MJ&&MJ[id]&&window.EX;
 const x0=()=>Object.fromEntries(FC.names.map((n,k)=>[n,FC.X[sel][k]]));
 const range=(a,b,n)=>[...Array(n)].map((_,i)=>a+(b-a)*i/(n-1)),st=k=>[EX.stats[k][0],EX.stats[k][2]];
 const pred=(x,s)=>evalModel(id,mod(x,s));
 const SLD=()=>[["dT","Temperature change",-15,15,.5,v=>(v>0?"+":"")+v.toFixed(1)+" °C"],["humidity","Humidity",...st("humidity"),1,v=>Math.round(v)+"%"],
  ["wind","Wind",...st("wind"),.5,v=>v.toFixed(1)+" km/h"],["cloud","Cloud cover",...st("cloud"),1,v=>Math.round(v)+"%"],
  ["solar","Solar radiation",0,150,5,v=>Math.round(v)+"% of forecast"],["precip","Rain",...st("precip"),.1,v=>v.toFixed(1)+" mm/h"],
  ["snow","Snowfall",...st("snow"),.1,v=>v.toFixed(1)+" cm/h"],["month","Month",1,12,1,v=>MON[Math.round(v)-1]]];
 const FACT=()=>[{k:"dT",n:"Temperature",v:range(-15,15,21),f:(v,x)=>(x.temp+v).toFixed(0)+"°C"},
  {k:"humidity",n:"Humidity",v:range(20,100,9),f:v=>Math.round(v)+"%"},{k:"wind",n:"Wind",v:range(0,40,9),f:v=>Math.round(v)+" km/h"},
  {k:"cloud",n:"Cloud cover",v:range(0,100,6),f:v=>Math.round(v)+"%"},{k:"solar",n:"Solar radiation",v:range(0,150,8),f:v=>Math.round(v)+"%"},
  {k:"precip",n:"Rain",v:range(0,5,6),f:v=>v.toFixed(1)+" mm/h"},{k:"snow",n:"Snowfall",v:range(0,2,6),f:v=>v.toFixed(1)+" cm/h"},
  {k:"month",n:"Season (month)",v:range(1,12,12),f:v=>MON[Math.round(v)-1]},
  {k:"day",n:"Day type",v:["weekday","saturday","sunday","holiday"],f:v=>v}];
 const IN={},LB={};let built=false,FA=null;
 function build(){
  built=true;FA=FACT();const box=$("ctl");
  SLD().forEach(([k,n,lo,hi,step,fmt])=>{const l=document.createElement("label");l.innerHTML=n+"<b></b>";
   const i=document.createElement("input");i.type="range";i.min=lo;i.max=Math.max(hi,lo+step);i.step=step;i.oninput=()=>{Sc[k]=+i.value;drawEx()};
   l.appendChild(i);box.appendChild(l);IN[k]=i;LB[k]=[l.querySelector("b"),fmt]});
  const l=document.createElement("label");l.innerHTML='Day type<select id="dayType"><option value="">as forecast</option><option>weekday</option><option>saturday</option><option>sunday</option><option>holiday</option></select>';
  box.appendChild(l);l.querySelector("select").onchange=e=>{Sc.day=e.target.value||null;drawEx()};
  FA.filter(f=>f.k!="day").forEach(f=>{const q=document.createElement("div");q.innerHTML=`<p style="margin:0"><b>${f.n}</b></p><canvas class="sm"></canvas>`;$("pd").appendChild(q);f.c=q.querySelector("canvas")});
  $("shy").onchange=drawEx;if($("srt"))$("srt").onchange=drawEx}
 window.resetSc=function(){if(!ready())return;const x=x0();Sc={dT:0,humidity:x.humidity,wind:x.wind,cloud:x.cloud,solar:100,precip:x.precip,snow:x.snow,month:x.month,day:null};
  if($("dayType"))$("dayType").value=""};
 window.drawEx=function(){
  if(!ready()){$("exTitle").textContent="Waiting for the forecast and model data (the explorer appears once the first update and training files are published).";return}
  if(!built){build();resetSc()}
  const x=x0(),base=pred(x,Sc);
  SLD().forEach(([k,,,,,fmt])=>{IN[k].value=Sc[k];LB[k][0].textContent=fmt(Sc[k])});
  $("exTitle").innerHTML=`<b>${lab(FC.t[sel])}</b> Toronto time: forecast ${MW(FC.m[id][sel])}; with your settings <b>${MW(base)}</b>`;
  const R=FA.map(f=>({f,ys:f.v.map(v=>pred(x,{...Sc,[f.k]:v}))}));
  const T=R.map(r=>{const lo=Math.min(...r.ys),hi=Math.max(...r.ys);return{f:r.f,lo,hi,vl:r.f.v[r.ys.indexOf(lo)],vh:r.f.v[r.ys.indexOf(hi)]}}).sort((a,b)=>(b.hi-b.lo)-(a.hi-a.lo));
  const sc=Math.max(...T.map(t=>Math.max(Math.abs(t.lo-base),Math.abs(t.hi-base))))||1,sg=d=>(d>0?"+":"")+Math.round(d);
  $("tor").innerHTML=T.map(t=>`<div class=row><span>${t.f.n}</span><div class=bar><u></u><i style="left:${50+(t.lo-base)/sc*50}%;width:${(t.hi-t.lo)/sc*50}%"></i></div>
   <span>${t.f.f(t.vl,x)}: ${sg(t.lo-base)} MW<br>${t.f.f(t.vh,x)}: ${sg(t.hi-base)} MW</span></div>`).join("");
  const pr=R.filter(r=>r.f.k!="day"),all=pr.flatMap(r=>r.ys),yr=$("shy").checked?[Math.min(...all),Math.max(...all)]:null;
  pr.forEach(r=>{const f=r.f,xs=f.k=="dT"?f.v.map(v=>x.temp+v):f.v,cur=f.k=="dT"?x.temp+Sc.dT:Sc[f.k];
   plot(f.c,xs,[{ys:r.ys,color:Mo.c}],{yr,color:Mo.c,dot:[cur,base],nt:4,xf:v=>f.k=="dT"?v.toFixed(0)+"°":f.k=="month"?MON[Math.round(v)-1]:String(f.f(v,x)).replace(/ .*/,"")})});
  if(isPoly)$("fm").textContent=polyText(id,x,$("srt")&&$("srt").checked)};

 /* ---------- about ---------- */
 function about(){
  const feats=window.MJ&&MJ[id]?MJ[id].features||[...MJ[id].core,...MJ[id].linear]:null;
  let h=`<p>${Mo.demand?"It comes in three versions: one for this hour, one for next hour and one for the 7-day forecast, each shown only the demand figures that would genuinely be known at that moment.":"One version serves this hour, next hour and the 7-day forecast, because it never looks at demand."}</p>`;
  if(det){const w=det.val_base,c=det.val_city,better=c<w*0.995;
   h+=`<p><b>Temperature inputs:</b> ${det.city?"each of the seven cities' own temperatures are used as separate inputs, as well as the province-wide average":"the population-weighted province-wide temperature"}. During training both options were tried and the better one kept (validation error ${Math.round(w)} MW with the weighted temperature alone vs ${Math.round(c)} MW with every city's own temperature${better?"":", which was not enough of an improvement to keep"}).</p>`}
  if(feats)h+=`<p><b>Inputs (${feats.length}):</b> ${feats.join(", ")}.</p>`;
  $("about").innerHTML=h}

 function render(){banner();cardsFor(id);forecast();past();drawEx();about()}
 function reload(){let n=2;for(const f of["demand_log.js","forecast.js"]){const s=document.createElement("script");s.src=f+"?"+Date.now();
  s.onload=s.onerror=()=>{s.remove();if(!--n){if(window.FC&&FC.u)sel=Math.min(sel,FC.u.length-1);render()}};document.head.appendChild(s)}}
 $("rng").onchange=past;addEventListener("resize",render);setInterval(reload,300000);
 if(ready())resetSc();render();
})();
