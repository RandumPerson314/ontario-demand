// Optional helper that runs free on Cloudflare Workers. It does two things:
//
// 1. SCHEDULER: every 10 minutes it asks GitHub to run the "update-demand-log" workflow. GitHub's own
//    cron can be hours late; a request from here starts the job within seconds.
// 2. RELAY: https://<your-worker>.workers.dev/?u=<IESO url> fetches an IESO feed and adds the CORS
//    header browsers need, so the website can read live IESO numbers directly. Only the two IESO
//    feeds the site uses are allowed, and answers are cached for 60 seconds.
//
// Settings (Worker > Settings > Variables and Secrets):
//   REPO          text    your-github-name/ontario-demand
//   BRANCH        text    main            (the branch the workflow lives on)
//   GITHUB_TOKEN  secret  a fine-grained personal access token for that one repository, with the
//                         permission "Actions: Read and write"
// Trigger (Worker > Settings > Triggers > Cron):   */10 * * * *

const ALLOWED = [
  "https://reports-public.ieso.ca/public/RealtimeTotals/PUB_RealtimeTotals.xml",
  "https://www.ieso.ca/-/media/Files/IESO/Power-Data/Ontario-Demand-multiday.ashx",
];
const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Methods": "GET, OPTIONS",
};

export default {
  async fetch(request) {
    if (request.method === "OPTIONS") return new Response(null, { headers: CORS });
    const target = new URL(request.url).searchParams.get("u");
    if (!target || !ALLOWED.includes(target)) {
      return new Response("Only the IESO demand feeds used by the website can be relayed.", { status: 403, headers: CORS });
    }
    const upstream = await fetch(target, { cf: { cacheTtl: 60, cacheEverything: true } });
    return new Response(upstream.body, {
      status: upstream.status,
      headers: {
        ...CORS,
        "Content-Type": upstream.headers.get("Content-Type") || "text/xml",
        "Cache-Control": "public, max-age=60",
      },
    });
  },

  async scheduled(event, env, ctx) {
    const url = `https://api.github.com/repos/${env.REPO}/actions/workflows/update.yml/dispatches`;
    ctx.waitUntil(
      fetch(url, {
        method: "POST",
        headers: {
          Authorization: `Bearer ${env.GITHUB_TOKEN}`,
          Accept: "application/vnd.github+json",
          "X-GitHub-Api-Version": "2022-11-28",
          "User-Agent": "ontario-demand-trigger",
        },
        body: JSON.stringify({ ref: env.BRANCH || "main" }),
      }).then(async (r) => {
        if (!r.ok) console.log("GitHub refused the trigger:", r.status, await r.text());
      })
    );
  },
};
