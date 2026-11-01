/* Optional. If you set up the free Cloudflare Worker (see cloudflare/worker.js), put its address here,
   e.g.  window.CFG = { proxy: "https://ontario-demand.YOURNAME.workers.dev/" };
   Leave it empty and the pages will try to read the IESO's feeds directly from the browser, then fall
   back to the data saved by the hourly update. */
window.CFG = { proxy: "" };
