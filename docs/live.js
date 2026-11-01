/* Fresh IESO numbers straight into the visitor's browser, so the "right now" cards don't depend on GitHub's update schedule.
   Reads the 5-minute report (this hour so far) and the IESO's hourly chart feed (recent completed hours + its forecast). */
(function(){
 const RT="https://reports-public.ieso.ca/public/RealtimeTotals/PUB_RealtimeTotals.xml",
  MD="https://www.ieso.ca/-/media/Files/IESO/Power-Data/Ontario-Demand-multiday.ashx",H=36e5,
  key=ms=>new Date(ms).toISOString().slice(0,13)+":00";
 async function get(url){
  const p=window.CFG&&CFG.proxy,tries=p?[p+(p.includes("?")?"&":"?")+"u="+encodeURIComponent(url),url]:[url];
  for(const u of tries){try{const ac=new AbortController(),t=setTimeout(()=>ac.abort(),9000);
    const r=await fetch(u,{signal:ac.signal,cache:"no-store"});clearTimeout(t);if(r.ok)return await r.text()}catch(e){}}
  return null}
 const xml=t=>new DOMParser().parseFromString(t,"text/xml"),all=(d,n)=>[...d.getElementsByTagName("*")].filter(e=>(e.localName||e.nodeName.split(":").pop())==n),
  txt=e=>e?e.textContent.trim():"";
 function rt(t){const d=xml(t),date=txt(all(d,"DeliveryDate")[0]),hr=+txt(all(d,"DeliveryHour")[0]);if(!date||!hr)return null;
  const start=Date.parse(date+"T00:00:00Z")+(hr-1)*H+5*H,vals=[];        // the IESO clock is EST, so UTC = EST + 5 h
  all(d,"IntervalEnergy").forEach(ie=>{const iv=+txt(all(ie,"Interval")[0]);
   all(ie,"MQ").forEach(mq=>{const k=all(mq,"MarketQuantity")[0],e=all(mq,"EnergyMW")[0];
    if(k&&txt(k)=="ONTARIO DEMAND"&&e&&isFinite(+txt(e)))vals.push([iv,+txt(e)])})});
  if(!vals.length)return null;vals.sort((a,b)=>a[0]-b[0]);
  return{key:key(start),n:vals.length,avg:vals.reduce((s,v)=>s+v[1],0)/vals.length,last:start+(vals[vals.length-1][0]-1)*3e5}}
 function wallToUtc(naive){   // a Toronto wall-clock time (given as if it were UTC) -> real UTC
  const off=t=>Date.parse(new Date(t).toLocaleString("sv-SE",{timeZone:"America/Toronto"}).replace(" ","T")+"Z")-t;
  let g=naive-off(naive+5*H);g=naive-off(g);return g}
 function md(t,cal){const d=xml(t),sd=txt(all(d,"StartDate")[0]);if(!sd||!cal)return null;
  const s0=Date.parse(sd.slice(0,19)+"Z"),out={};
  all(d,"DataSet").forEach(ds=>{const nm=(ds.getAttribute("Series")||"").toLowerCase(),
   kind=nm.includes("actual")?"actual":(nm.includes("project")||nm.includes("forecast"))?"proj":null;if(!kind)return;
   const m={};all(ds,"Value").forEach((v,i)=>{const s=txt(v);if(s===""||!isFinite(+s))return;const naive=s0+i*H;
    m[key((cal.name=="est"?naive+5*H:wallToUtc(naive))+cal.shift*H)]=+s});out[kind]=m});return out}
 async function refresh(){
  const[a,b]=await Promise.all([get(RT),get(MD)]),L={t:Date.now(),ok:false};
  try{if(a){L.rt=rt(a);if(L.rt)L.ok=true}}catch(e){}
  try{if(b){const m=md(b,(window.LOG_META||{}).feed_cal);if(m){L.actual=m.actual||{};L.proj=m.proj||{};L.ok=true}}}catch(e){}
  window.LIVE=L;if(window.onLive)window.onLive()}
 window.__liveParsers={rt,md};refresh();setInterval(refresh,300000);
})();
