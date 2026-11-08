const $=id=>document.getElementById(id),css=n=>getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const MW=v=>Math.round(v).toLocaleString()+" MW",DOW=["Sun","Mon","Tue","Wed","Thu","Fri","Sat"],
MON=["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"],lab=t=>t.replace("T"," ");
function setup(c){const r=c.getBoundingClientRect(),d=devicePixelRatio||1;c.width=r.width*d;c.height=r.height*d;const g=c.getContext("2d");g.scale(d,d);return[g,r.width,r.height]}
function plot(c,xs,S,o={}){
 const[g,w,h]=setup(c),Lm=54,R=10,T=8,Bt=26,all=S.flatMap(s=>s.ys).filter(v=>v!=null);
 let lo=o.yr?o.yr[0]:Math.min(...all),hi=o.yr?o.yr[1]:Math.max(...all);const p=(hi-lo)*.08||1;lo-=p;hi+=p;
 const x0=xs[0],x1=xs[xs.length-1],X=x=>Lm+(x-x0)/((x1-x0)||1)*(w-Lm-R),Y=y=>T+(hi-y)/(hi-lo)*(h-T-Bt);
 g.font="11px system-ui";g.lineWidth=1;g.fillStyle=css("--mut");g.strokeStyle=css("--bd");
 for(let i=0;i<=4;i++){const y=lo+(hi-lo)*i/4;g.beginPath();g.moveTo(Lm,Y(y));g.lineTo(w-R,Y(y));g.stroke();g.fillText(Math.round(y).toLocaleString(),2,Y(y)+4)}
 const nt=o.nt||5;for(let i=0;i<nt;i++){const j=Math.round((xs.length-1)*i/(nt-1));g.fillText(o.xf(xs[j],j),Math.min(X(xs[j])-12,w-34),h-8)}
 S.forEach(s=>{g.strokeStyle=s.color;g.lineWidth=2;g.beginPath();let pen=false;
  xs.forEach((x,i)=>{const v=s.ys[i];if(v==null){pen=false;return}pen?g.lineTo(X(x),Y(v)):g.moveTo(X(x),Y(v));pen=true});g.stroke();
  if(s.dots)xs.forEach((x,i)=>{const v=s.ys[i];if(v==null)return;g.beginPath();g.arc(X(x),Y(v),3,0,7);
   if(s.hollow&&s.hollow[i]){g.lineWidth=1.5;g.stroke()}else{g.fillStyle=s.color;g.fill()}})});
 if(o.mark!=null){g.strokeStyle=css("--fg");g.lineWidth=1.5;g.setLineDash([4,3]);g.beginPath();g.moveTo(X(o.mark),T);g.lineTo(X(o.mark),h-Bt);g.stroke();g.setLineDash([])}
 if(o.dot){g.fillStyle=css("--fg");g.beginPath();g.arc(X(o.dot[0]),Y(o.dot[1]),5,0,7);g.fill()}
 c._m={x0,x1,Lm,R,w}}

/* ---------- model registry (colours, names, plain-language descriptions) ---------- */
const MODELS={
 gbm:{n:"Gradient boosting + recent demand",c:"#d9480f",page:"dashboard.html",demand:true,
  d:"A team of hundreds of small decision trees, each one correcting the mistakes of the ones before it. Besides the weather and the calendar it is shown recent demand: the last few hours for the this-hour and next-hour predictions, and demand from a week or two earlier for the 7-day forecast."},
 gbm0:{n:"Gradient boosting, no demand",c:"#f08c00",page:"gbm_weather.html",demand:false,
  d:"The same decision-tree technique, but it is never shown any demand numbers at all, only the weather and the calendar. Comparing it with its sibling shows how much knowing recent demand really helps."},
 poly:{n:"Polynomial + recent demand",c:"#1971c2",page:"poly_demand.html",demand:true,
  d:"One fixed formula (a polynomial: sums of products of the inputs) fitted to history. Its terms combine temperature, time of day, sunshine and workday-or-not, and it also has terms for recent demand, so it can follow how demand has been running lately."},
 poly0:{n:"Polynomial, no demand",c:"#4dabf7",page:"polynomial.html",demand:false,
  d:"The same kind of formula using only the weather and the calendar, never recent demand. Its full equation and a set of sliders are on its page."},
 nn:{n:"Neural network + recent demand",c:"#2f9e44",page:"nn_demand.html",demand:true,
  d:"A small neural network (layers of weighted sums with a bend between them), trained with PyTorch and averaged over several training runs. It is shown recent demand as well as the weather and the calendar."},
 nn0:{n:"Neural network, no demand",c:"#74b816",page:"nn_weather.html",demand:false,
  d:"The same kind of neural network, but with no demand numbers as input, only the weather and the calendar."}};
const IESO={n:"IESO official forecast",c:"#7048e8"},MIDS=Object.keys(MODELS);

/* ---------- freshness / problem banner ---------- */
function banner(){
 const M=window.LOG_META||{},el=document.getElementById("banner");if(!el)return;
 const age=M.updated?(Date.now()-Date.parse(M.updated))/6e4:null,bad=(M.status||[]).filter(s=>!s.ok);let msg="";
 if(age==null)msg="No update has run yet, so there is no data to show.";
 else if(age>35)msg="The data was last updated "+(age<120?Math.round(age)+" minutes":Math.round(age/60)+" hours")+" ago, so the automatic update is running late. Numbers on this page may be out of date.";
 if(bad.length)msg+=(msg?" ":"")+"Problem with: "+bad.map(s=>s.name+(s.name.startsWith("Update job")?" ("+s.detail.slice(0,160)+")":"")).join(", ")+" (details on the Comparison page).";
 el.style.display=msg?"block":"none";el.textContent=msg}

/* ---------- helpers for "right now" numbers: fresh browser data first, then what the site last saved ---------- */
const hkey=ms=>new Date(ms).toISOString().slice(0,13)+":00",
 htl=ms=>new Date(ms).toLocaleString("sv-SE",{timeZone:"America/Toronto"}).slice(0,13)+":00",
 ago=ms=>{const h=Math.round((Date.now()-ms)/36e5);return h<1?"under an hour ago":h+(h==1?" hour":" hours")+" ago"};
function liveNow(){   // demand so far this hour; if that isn't available, the newest reading, clearly labelled
 const LV=window.LIVE,M=window.LOG_META||{},H=36e5,t0=Math.floor(Date.now()/H)*H,k0=hkey(t0);
 if(LV&&LV.rt&&LV.rt.key==k0)return{v:LV.rt.avg,fresh:1,sub:htl(t0).slice(5)+" hour · average of "+LV.rt.n+" readings, latest at "+
  new Date(LV.rt.last).toLocaleTimeString("en-CA",{timeZone:"America/Toronto",hour:"2-digit",minute:"2-digit",hour12:false})};
 if(M.live_mw!=null&&lab(M.live_t)==htl(t0))return{v:M.live_mw,fresh:1,sub:htl(t0).slice(5)+" hour · average of "+M.live_n+" readings, latest at "+M.live_last};
 const c=[];if(LV&&LV.rt)c.push({v:LV.rt.avg,k:htl(Date.parse(LV.rt.key+":00Z"))});if(M.live_mw!=null)c.push({v:M.live_mw,k:lab(M.live_t)});
 c.sort((a,b)=>a.k<b.k?1:-1);
 return c.length?{v:c[0].v,fresh:0,sub:"not this hour: the newest reading is for the "+c[0].k.slice(5)+" hour (Toronto time), so the data is delayed"}
  :{v:null,fresh:0,sub:"no reading available yet"}}
function lastDone(L){  // actual demand for the hour that just ended; if missing, the newest completed hour on record
 const LV=window.LIVE,H=36e5,t0=Math.floor(Date.now()/H)*H,k1=hkey(t0-H),kn=hkey(t0);
 if(LV&&LV.actual&&LV.actual[k1]!=null)return{v:LV.actual[k1],ms:t0-H,fresh:1};
 const r=L.find(r=>r.u==k1);if(r&&r.actual!=null)return{v:r.actual,ms:t0-H,fresh:1};
 const best=[];if(LV&&LV.actual)for(const k in LV.actual)if(k<kn)best.push([k,LV.actual[k]]);
 L.forEach(r=>{if(r.actual!=null&&r.u<kn)best.push([r.u,r.actual])});best.sort((a,b)=>a[0]<b[0]?1:-1);
 return best.length?{v:best[0][1],ms:Date.parse(best[0][0]+":00Z"),fresh:0}:null}

/* ---------- the five cards, for any model id ---------- */
function cardsFor(id){
 const L=window.LOG||[],F=window.FC,H=36e5,t0=Math.floor(Date.now()/H)*H,
 arr=F&&(F.m?F.m[id]:id=="gbm"?F.pred:null),fcAt=k=>{const i=F?F.u.indexOf(k):-1;return i<0||!arr?null:arr[i]};
 const pick=(k,fields)=>{const r=L.find(r=>r.u==k);if(r)for(const[f,s]of fields)if(r[f]!=null)return[r[f],s];
  const v=fcAt(k);return v==null?[null,""]:[v,"7-day forecast run"]};
 const show=(b,t,v,s,ms)=>{$(b).textContent=v==null?"-":MW(v);$(t).textContent=htl(ms).slice(5)+" Toronto time"+(v==null?" · not available yet":" · "+s)};
 const[a,as]=pick(hkey(t0),[[id+"_h","made this hour"],[id+"_n","made an hour ahead"]]),[b,bs]=pick(hkey(t0+H),[[id+"_n","made this hour"]]);
 show("p","pt",a,as,t0);show("n","nt",b,bs,t0+H);
 const lv=liveNow();                                   // actual demand so far this hour
 $("l").textContent=lv.v==null?"-":MW(lv.v);
 $("lt").textContent=lv.sub+(lv.fresh&&a!=null?" · prediction is "+((100*(a/lv.v-1)).toFixed(1)).replace(/^(\d)/,"+$1")+"%":"");
 if($("l").previousElementSibling)$("l").previousElementSibling.textContent=lv.fresh?"Actual demand this hour (so far)":"Latest reading (not this hour)";
 const ld=lastDone(L);                                 // last completed hour
 if(ld){const r=L.find(r=>r.u==hkey(ld.ms));$("a").textContent=MW(ld.v);
  $("at").textContent=htl(ld.ms).slice(5)+" Toronto time"+(ld.fresh?"":" · "+ago(ld.ms)+", data is delayed")+(r&&r[id+"_h"]!=null?" · model said "+MW(r[id+"_h"]):"")}
 else{$("a").textContent="-";$("at").textContent=htl(t0-H).slice(5)+" hour · not published yet"}
 if($("a").previousElementSibling)$("a").previousElementSibling.textContent=ld&&!ld.fresh?"Latest completed hour on record":"Last completed hour (IESO)";
 const wk=L.slice(-168).filter(r=>r.actual!=null&&r[id+"_h"]!=null),live=wk.filter(r=>!r.bf),use=live.length>=12?live:wk;
 if(use.length){const e=use.map(r=>r[id+"_h"]-r.actual),mae=e.reduce((s,x)=>s+Math.abs(x),0)/e.length;
  $("m").textContent="±"+Math.round(mae)+" MW";
  $("mt").textContent="this-hour predictions, "+use.length+(use===live?" live":" retrospective")+" hours, "+(100*e.reduce((s,x,i)=>s+Math.abs(x)/use[i].actual,0)/e.length).toFixed(2)+"% error"}
 else{$("m").textContent="-";$("mt").textContent="not enough scored hours yet"}}

/* ---------- daily peak/low table for a 7-day forecast array ---------- */
function dailyTable(F,arr){
 const D={};F.t.forEach((t,i)=>(D[t.slice(0,10)]=D[t.slice(0,10)]||[]).push(i));
 return"<table><tr><th>Day</th><th>Peak</th><th>At</th><th>Lowest</th><th>Avg temp</th></tr>"+Object.entries(D).map(([d,ix])=>{
  const pk=ix.reduce((a,i)=>arr[i]>arr[a]?i:a),mn=Math.min(...ix.map(i=>arr[i])),tp=ix.reduce((s,i)=>s+F.temp[i],0)/ix.length;
  return`<tr><td>${DOW[new Date(d+"T12:00:00").getDay()]} ${d.slice(5)}${ix.length<24?" (partial)":""}</td><td>${MW(arr[pk])}</td><td>${F.t[pk].slice(11,13)}:00</td><td>${MW(mn)}</td><td>${tp.toFixed(1)} °C</td></tr>`}).join("")+"</table>"}

/* ---------- evaluate any exported model in the browser (trees, polynomial, or neural network) ---------- */
function evalModel(id,x){
 const S=MJ[id],lag=S.lag?x[S.lag]:0;
 if(S.kind=="gbm"){const v=S.features.map(n=>x[n]);let s=S.base;
  for(const T of S.trees){let n=0;while(T.f[n]>=0)n=v[T.f[n]]<=T.t[n]?T.l[n]:T.r[n];s+=T.v[n]}return lag+s}
 if(S.kind=="poly"){const z=S.core.map((c,i)=>(x[c]-S.mean[i])/S.scale[i]);let y=S.icpt,k=0;
  for(const p of S.powers){let v=1;for(let i=0;i<p.length;i++)if(p[i])v*=Math.pow(z[i],p[i]);y+=S.coef[k++]*v}
  for(const c of S.linear)y+=S.coef[k++]*x[c];return lag+y}
 if(!S._w)S._w=S.nets.map(L=>L.map(([W,b])=>[W.map(r=>Float64Array.from(r)),Float64Array.from(b)]));
 const z=S.features.map((n,i)=>(x[n]-S.mu[i])/S.sd[i]);let tot=0;
 for(const L of S._w){let a=z;L.forEach(([W,b],li)=>{const o=new Array(W.length);
   for(let j=0;j<W.length;j++){let s=b[j];const w=W[j];for(let i=0;i<a.length;i++)s+=w[i]*a[i];
    o[j]=li<L.length-1?s/(1+Math.exp(-Math.max(-60,Math.min(60,s)))):s}a=o});tot+=a[0]}
 return lag+(tot/S._w.length)*S.ys+S.ym}

/* "what if" changes applied to one hour's inputs; features a model doesn't use are simply ignored */
const TEMPK=["temp","feels","temp_lag1","temp_lag3","temp_lag24","temp_roll24","temp_roll72"];
const dewp=(T,H)=>{const g=Math.log(Math.max(H,1)/100)+17.62*T/(243.12+T);return 243.12*g/(17.62-g)};
function mod(x0,s){
 const x={...x0},has=n=>n in x0;
 for(const n in x0)if(TEMPK.includes(n)||n.startsWith("ct_"))x[n]=x0[n]+s.dT;      // every temperature input moves together
 if(has("hdd"))x.hdd=Math.max(EX.base-x.temp,0);if(has("cdd"))x.cdd=Math.max(x.temp-EX.base,0);
 if(has("dtemp168"))x.dtemp168=x0.dtemp168+s.dT;
 for(const k of["humidity","wind","cloud","precip","snow"])if(has(k))x[k]=s[k];
 if(has("solar"))x.solar=x0.solar*s.solar/100;
 if(has("dew"))x.dew=x0.dew+dewp(x.temp,s.humidity)-dewp(x0.temp,x0.humidity);
 if(s.month!=x0.month){x.month=s.month;x.doy=(s.month-.5)*30.44;const b=2*Math.PI*x.doy/365.25;
  if(has("doy_s")){x.doy_s=Math.sin(b);x.doy_c=Math.cos(b)}for(let m=1;m<=12;m++)if(has("mo_"+m))x["mo_"+m]=m==s.month?1:0}
 if(s.day){const d={weekday:[2,0,0],saturday:[5,1,0],sunday:[6,1,0],holiday:[0,1,1]}[s.day];
  x.dow=d[0];x.nonwork=d[1];x.hol=d[2];for(let k=1;k<=6;k++)if(has("dow"+k))x["dow"+k]=d[0]==k?1:0}
 return x}

/* the written-out polynomial (for polynomial models): every term with its current contribution */
function polyText(id,x,sortBySize){
 const S=MJ[id],z=S.core.map((c,i)=>(x[c]-S.mean[i])/S.scale[i]);
 const names=S.powers.map(p=>p.map((e,i)=>e?"z_"+S.core[i]+(e>1?"^"+e:""):"").filter(Boolean).join("*")||"1").concat(S.linear);
 const vals=S.powers.map(p=>{let v=1;p.forEach((e,i)=>{if(e)v*=Math.pow(z[i],e)});return v}).concat(S.linear.map(c=>x[c]));
 const rows=names.map((n,i)=>({n,c:S.coef[i],m:S.coef[i]*vals[i]}));if(sortBySize)rows.sort((a,b)=>Math.abs(b.m)-Math.abs(a.m));
 const sg=(v,d,w)=>(v<0?"-":"+")+Math.abs(v).toFixed(d).padStart(w),tot=rows.reduce((a,r)=>a+r.m,S.icpt);
 return"where\n"+S.core.map((c,i)=>`  z_${c} = (${c} - ${S.mean[i].toPrecision(6)}) / ${S.scale[i].toPrecision(6)}`).join("\n")+
  "\n\n"+(S.lag?`change_in_MW (added to ${S.lag}) = `:"demand_MW = ")+S.icpt.toFixed(3)+"\n"+
  rows.map(r=>`  ${sg(r.c,4,12)} * ${r.n.padEnd(30)} ${sg(r.m,1,9)} MW`).join("\n")+
  `\n\nTotal at the selected hour = ${Math.round(tot+(S.lag?x[S.lag]:0)).toLocaleString()} MW`}
