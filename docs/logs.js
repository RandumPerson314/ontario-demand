/* Data logs page: every saved hour with links to the official IESO file behind each IESO number. */
(function(){
 const H=36e5,BASE="https://reports-public.ieso.ca/public/",pad=n=>String(n).padStart(2,"0");
 const COLS=MIDS.map(id=>({id,n:MODELS[id].n,c:MODELS[id].c}));
 const est=u=>{const d=new Date(Date.parse(u+":00Z")-5*H);return{date:d.toISOString().slice(0,10),he:d.getUTCHours()+1}};
 const clock=ms=>new Date(ms).toLocaleTimeString("en-CA",{timeZone:"America/Toronto",hour:"2-digit",minute:"2-digit",hour12:false});
 const tor=u=>htl(Date.parse(u+":00Z"));                       // Toronto "YYYY-MM-DDTHH:00"
 const savedAt=(r,k)=>r.at&&r.at[k]!=null?Date.parse(r.u+":00Z")+r.at[k]*6e4:null;
 const rel=m=>m==null?"":(m<0?"−"+(-m):"+"+m)+" min";
 const link=(href,text)=>`<a href="${href}" target="_blank" rel="noopener">${text}</a>`;

 function actualSrc(r){
  const e=est(r.u),d=e.date.replace(/-/g,""),f5=`PUB_RealtimeTotals_${d}${pad(e.he)}.xml`,yr=e.date.slice(0,4);
  const recent=Date.now()-Date.parse(r.u+":00Z")<29*24*H;      // the IESO keeps the per-hour files about a month
  if(r.src=="5min")return link(BASE+"RealtimeTotals/"+f5,"5-min report");
  return link(BASE+`Demand/PUB_Demand_${yr}.csv`,"yearly file")+(recent?" · "+link(BASE+"RealtimeTotals/"+f5,"5-min"):"")}
 function iesoCell(r,k){
  const v=r[k];if(v==null)return"-";const e=est(r.u),f=r.f&&r.f[k],t=savedAt(r,k);
  if(!f&&!t)return`${MW(v)}${miss(v,r.actual)}<small>${link(BASE+`Adequacy3/PUB_Adequacy3_${e.date.replace(/-/g,"")}.xml`,"day's file")} · version not recorded</small>`;
  const file=f?link(BASE+"Adequacy3/"+f,f.replace("PUB_Adequacy3_","").replace(".xml","")):link(BASE+`Adequacy3/PUB_Adequacy3_${e.date.replace(/-/g,"")}.xml`,"day's file");
  return`${MW(v)}${miss(v,r.actual)}<small>${file}${t?" · saved "+clock(t)+" ("+rel(r.at[k])+")":""}</small>`}
 function miss(v,a){if(!$("diff").checked||v==null||a==null)return"";const d=Math.round(v-a);
  return`<small class="${d>0?"pos":d<0?"neg":""}">${d>0?"+":""}${d.toLocaleString()}</small>`}

 function render(){
  banner();
  const L=(window.LOG||[]).slice().reverse(),sfx=$("kind").value,day=$("day").value,k="ieso"+sfx;
  const rows=L.filter(r=>day=="all"||tor(r.u).slice(0,10)==day);
  let html=`<thead><tr><th>Hour (Toronto)<small>IESO date · hour ending (EST)</small></th><th>Actual<small>source</small></th>
   <th>IESO forecast<small>${sfx=="_h"?"saved during the hour":"saved the hour before"}</small></th>`+
   COLS.map(c=>`<th><span class="sw" style="background:${c.c}"></span>${c.n}</th>`).join("")+
   `<th>Saved<small>models, Toronto time</small></th><th>Timing</th></tr></thead><tbody>`,last="";
  for(const r of rows){
   const t=tor(r.u),e=est(r.u);
   if(day=="all"&&t.slice(0,10)!=last){last=t.slice(0,10);html+=`<tr class="day"><td colspan="${COLS.length+5}">${DOW[new Date(last+"T12:00:00").getDay()]} ${last}</td></tr>`}
   const ts=COLS.map(c=>r.at&&r.at[c.id+sfx]).filter(x=>x!=null),lo=Math.min(...ts),hi=Math.max(...ts);
   const saved=ts.length?clock(Date.parse(r.u+":00Z")+lo*6e4)+(hi>lo?"–"+clock(Date.parse(r.u+":00Z")+hi*6e4):"")+`<small>${rel(lo)}${hi>lo?" to "+rel(hi):""}</small>`:"<small>not recorded</small>";
   html+=`<tr><td><b>${t.slice(11)}</b> ${t.slice(5,10)}<small>${e.date} · HE ${e.he}</small></td>
    <td>${r.actual==null?"<small>not in yet</small>":`<b>${MW(r.actual)}</b><small>${actualSrc(r)}</small>`}</td>
    <td>${iesoCell(r,k)}</td>`+
    COLS.map(c=>{const v=r[c.id+sfx];return`<td>${v==null?"-":MW(v)+miss(v,r.actual)}</td>`}).join("")+
    `<td>${saved}</td><td>${r.bf?"after the fact":"on time"}</td></tr>`}
  $("tb").innerHTML=html+"</tbody>";
  const done=rows.filter(r=>r.actual!=null).length;
  $("cnt").textContent=`${rows.length} hours shown (${done} with actual demand). The site keeps the last 14 days; older hours are in the GitHub history.`}

 function fillDays(){
  const days=[...new Set((window.LOG||[]).map(r=>tor(r.u).slice(0,10)))].reverse(),cur=$("day").value||days[0];
  $("day").innerHTML=days.map(d=>`<option value="${d}">${DOW[new Date(d+"T12:00:00").getDay()]} ${d}</option>`).join("")+'<option value="all">All 14 days</option>';
  $("day").value=days.includes(cur)||cur=="all"?cur:days[0]}

 function csv(){
  const keys=MIDS.flatMap(id=>[id+"_h",id+"_n"]).concat(["ieso_h","ieso_n"]);
  const head=["hour_start_utc","hour_start_toronto","ieso_date_est","ieso_hour_ending","actual_mw","actual_source","timing"]
   .concat(keys,keys.map(k=>k+"_saved_min_after_hour_start"),["ieso_h_file","ieso_n_file"]);
  const q=v=>v==null?"":String(v);
  const lines=(window.LOG||[]).map(r=>{const e=est(r.u);return[r.u,tor(r.u),e.date,e.he,r.actual,r.actual==null?"":(r.src||"yearly or 5min (not recorded)"),r.bf?"after the fact":"on time"]
   .concat(keys.map(k=>r[k]),keys.map(k=>r.at?r.at[k]:null),[r.f&&r.f.ieso_h,r.f&&r.f.ieso_n]).map(q).join(",")});
  const a=document.createElement("a");a.href=URL.createObjectURL(new Blob([[head.join(",")].concat(lines).join("\n")],{type:"text/csv"}));
  a.download="ontario-demand-log.csv";document.body.appendChild(a);a.click();a.remove()}

 ["day","kind","diff"].forEach(id=>$(id).addEventListener("change",render));$("dl").addEventListener("click",csv);
 function reload(){const s=document.createElement("script");s.src="demand_log.js?"+Date.now();s.onload=s.onerror=()=>{s.remove();fillDays();render()};document.head.appendChild(s)}
 fillDays();render();setInterval(reload,300000);
})();
