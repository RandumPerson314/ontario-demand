/* Generic page for one model. The wrapper html sets window.MODEL_ID. */
(function(){
 const id=window.MODEL_ID,Mo=MODELS[id];document.title=Mo.n;
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
 <p>Coloured line = this model, purple = the IESO's own official forecast (which only reaches the next day or so unless its long-range report is available). Times are Toronto time.</p><div id="days"></div></div>
 <div class="panel"><h2>Past hours: prediction vs actual</h2>
 <label style="display:block;margin:0 0 4px">Show <select id="rng"><option value="24">last 24 hours</option><option value="72" selected>last 3 days</option>
 <option value="168">last 7 days</option><option value="336">last 14 days</option></select></label>
 <canvas id="c"></canvas><p id="msg"></p>
 <p>Coloured line = this model's this-hour prediction, dark = actual IESO demand, purple = the IESO's forecast for that hour. Solid dots were predicted live, before the hour started; hollow dots were filled in afterwards.</p></div>`;
 function forecast(){
  const F=window.FC,arr=F&&F.m&&F.m[id];
  if(!arr){$("days").textContent="Forecast not available yet.";return}
  plot($("cf"),F.u.map((_,i)=>i),[{ys:arr,color:Mo.c},{ys:F.ieso||[],color:IESO.c}],{nt:8,
   xf:(x,j)=>DOW[new Date(F.t[j].slice(0,10)+"T12:00:00").getDay()]+" "+F.t[j].slice(11,13)+"h"});
  $("days").innerHTML=dailyTable(F,arr)}
 function past(){
  const L=window.LOG||[];if(!L.length){$("msg").textContent="No data yet: the first update hasn't run.";return}
  const D=L.slice(-+$("rng").value);
  plot($("c"),D.map((_,i)=>i),[{ys:D.map(r=>r.actual),color:css("--fg")},{ys:D.map(r=>r.ieso_h),color:IESO.c},
   {ys:D.map(r=>r[id+"_h"]),color:Mo.c,dots:D.length<=100,hollow:D.map(r=>r.bf)}],{xf:(x,j)=>D[j].t.slice(5,13).replace("T"," ")+"h"});
  $("msg").textContent="Last update: "+(window.LOG_META&&LOG_META.updated?new Date(LOG_META.updated).toLocaleString():"-")}
 function render(){banner();cardsFor(id);forecast();past()}
 function reload(){let n=2;for(const f of["demand_log.js","forecast.js"]){const s=document.createElement("script");s.src=f+"?"+Date.now();
  s.onload=s.onerror=()=>{s.remove();if(!--n)render()};document.head.appendChild(s)}}
 $("rng").onchange=past;addEventListener("resize",render);setInterval(reload,300000);render();
})();
