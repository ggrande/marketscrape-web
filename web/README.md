# ICO Aggregator (web/)

Next.js 14 + TypeScript front-end and Stagehand-powered scrape workers for the
Instant Cash Offer aggregator.

## Architecture

```
  Browser
     │  (EventSource)
     ▼
  /api/offers/stream   ──► decodeVin (NHTSA vPIC)
     │ fan-out
     ├─► CarMaxScraper   ─► Stagehand ─► Browserbase / local Chromium
     ├─► CarvanaScraper  (todo)
     └─► …               (todo)
```

- `src/scrapers/base.ts` — `BaseScraper` abstract class. Handles session
  lifecycle, deadlines, and error normalisation so provider implementations
  only describe the UI flow.
- `src/scrapers/carmax.ts` — Phase 1 proof of concept. Uses `page.act()` and
  `page.extract()` with natural-language instructions — no hardcoded CSS
  selectors, so UI tweaks on CarMax self-heal.
- `src/app/api/offers/stream/route.ts` — SSE endpoint. Streams a `vehicle`
  event after VIN decode, then per-provider `status` / `result` events, then
  `done`.
- `scripts/test-scrapers.ts` — daily smoke test (`pnpm test:scrapers`).

## Getting started

```bash
cd web
npm install
cp .env.example .env.local   # fill in BROWSERBASE / OPENAI keys
npm run dev
```

For local development without Browserbase, keep `STAGEHAND_ENV=LOCAL` and
install Playwright's Chromium:

```bash
npx playwright install chromium
```

## Adding a provider

1. Create `src/scrapers/<provider>.ts` extending `BaseScraper`.
2. Implement `run(stagehand, ctx)` with `page.act()` / `page.extract()` — no
   CSS selectors.
3. Register the class in `src/scrapers/registry.ts` and add a row to
   `providerMeta` in `src/app/page.tsx`.
4. Run `npm run test:scrapers` against a sample VIN to verify.
