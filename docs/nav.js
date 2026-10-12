(function(){
 const models=[["dashboard.html","Gradient boosting + recent demand"],["gbm_weather.html","Gradient boosting, weather only"],
  ["poly_demand.html","Polynomial + recent demand"],["polynomial.html","Polynomial, weather only"],
  ["nn_demand.html","Neural network + recent demand"],["nn_weather.html","Neural network, weather only"]];
 const here=location.pathname.split("/").pop()||"index.html",onModel=models.some(m=>m[0]==here);
 const st=document.createElement("style");
 st.textContent="nav.site{display:flex;gap:4px;flex-wrap:wrap;align-items:center;padding:8px 16px;border-bottom:1px solid var(--bd);background:var(--card);font:14px system-ui,sans-serif;position:sticky;top:0;z-index:5}"+
  "nav.site a,nav.site summary{color:var(--mut);text-decoration:none;padding:6px 12px;border-radius:8px;cursor:pointer;list-style:none}nav.site summary::-webkit-details-marker{display:none}"+
  "nav.site a:hover,nav.site summary:hover{color:var(--fg)}nav.site .on{background:var(--ac);color:#fff}"+
  "nav.site details{position:relative}nav.site .menu{position:absolute;top:100%;left:0;background:var(--card);border:1px solid var(--bd);border-radius:10px;padding:4px;min-width:250px;box-shadow:0 6px 18px rgba(0,0,0,.18)}"+
  "nav.site .menu a{display:block}";
 document.head.appendChild(st);
 const n=document.createElement("nav");n.className="site";
 n.innerHTML=`<a href="index.html"${here=="index.html"?' class="on"':""}>Home</a>
 <details><summary${onModel?' class="on"':""}>Models &#9662;</summary><div class="menu">${models.map(([h,t])=>`<a href="${h}"${h==here?' class="on"':""}>${t}</a>`).join("")}</div></details>
 <a href="compare.html"${here=="compare.html"?' class="on"':""}>Comparison</a>`;
 document.addEventListener("click",e=>{const d=n.querySelector("details");if(d&&!d.contains(e.target))d.open=false});
 document.addEventListener("DOMContentLoaded",()=>document.body.prepend(n));
})();
