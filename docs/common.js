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
