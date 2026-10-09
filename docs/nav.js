(function(){
 const pages=[["index.html","Home"],["dashboard.html","Gradient boosting"],["polynomial.html","Polynomial model"],["compare.html","Comparison"]];
 const here=location.pathname.split("/").pop()||"index.html";
 const st=document.createElement("style");
 st.textContent="nav.site{display:flex;gap:4px;flex-wrap:wrap;padding:8px 16px;border-bottom:1px solid var(--bd);background:var(--card);font:14px system-ui,sans-serif;position:sticky;top:0;z-index:5}"+
  "nav.site a{color:var(--mut);text-decoration:none;padding:6px 12px;border-radius:8px}nav.site a:hover{color:var(--fg)}nav.site a.on{background:var(--ac);color:#fff}";
 document.head.appendChild(st);
 const n=document.createElement("nav");n.className="site";
 n.innerHTML=pages.map(([h,t])=>`<a href="${h}"${h==here?' class="on"':""}>${t}</a>`).join("");
 document.addEventListener("DOMContentLoaded",()=>document.body.prepend(n));
})();
