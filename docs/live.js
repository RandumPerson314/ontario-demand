/* Fresh IESO numbers straight into the visitor's browser, so the "right now" cards don't depend on GitHub's update schedule.
   Reads the 5-minute report (this hour so far); completed hours come from the site's own log, updated every few minutes. */
(function(){
 const RT="https://reports-public.ieso.ca/public/RealtimeTotals/PUB_RealtimeTotals.xml",H=36e5,
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
 async function refresh(){
  const a=await get(RT),L={t:Date.now(),ok:false};
  try{if(a){L.rt=rt(a);if(L.rt)L.ok=true}}catch(e){}
  window.LIVE=L;if(window.onLive)window.onLive()}
 window.__liveParsers={rt};refresh();setInterval(refresh,300000);
})();
