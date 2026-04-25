/**
 * Daily smoke test. Runs every registered provider against a known-good
 * "Sample VIN" and reports which ones still return a valid offer.
 *
 * Intended to run in CI on a schedule (e.g. GitHub Actions cron @daily).
 * Exits 1 if any provider errors so the job fails loudly.
 *
 *   pnpm test:scrapers           # default VIN/zip
 *   SAMPLE_VIN=... SAMPLE_ZIP=... pnpm test:scrapers
 */

import { decodeVin } from "../src/lib/vin";
import { providers } from "../src/scrapers/registry";

// 2019 Toyota Camry LE — a common, well-supported profile.
const DEFAULT_VIN = "4T1B11HK5KU812345";
const DEFAULT_ZIP = "10001";
const PROVIDER_BUDGET_MS = 120_000;

async function main() {
  const vin = process.env.SAMPLE_VIN ?? DEFAULT_VIN;
  const zip = process.env.SAMPLE_ZIP ?? DEFAULT_ZIP;

  console.log(`[test-scrapers] VIN=${vin} ZIP=${zip}`);

  const vehicle = await decodeVin(vin);
  console.log(
    `[test-scrapers] decoded: ${[vehicle.year, vehicle.make, vehicle.model].filter(Boolean).join(" ")}`,
  );

  const deadline = Date.now() + PROVIDER_BUDGET_MS;
  const abort = new AbortController();

  const results = await Promise.all(
    providers.map(async (provider) => {
      const started = Date.now();
      const result = await provider.scrape({
        request: { vin, zip },
        vehicle,
        deadlineMs: deadline,
        signal: abort.signal,
      });
      const elapsed = Date.now() - started;
      return { provider: provider.meta, result, elapsed };
    }),
  );

  let failed = 0;
  for (const { provider, result, elapsed } of results) {
    const tag = result.status === "success" ? "OK " : result.status.toUpperCase();
    const amount = result.amountUsd != null ? `$${result.amountUsd.toLocaleString()}` : "—";
    console.log(
      `  [${tag}] ${provider.name.padEnd(12)} ${amount.padStart(10)}  (${elapsed}ms)` +
        (result.error ? `  ${result.error}` : ""),
    );
    if (result.status === "error") failed += 1;
  }

  if (failed > 0) {
    console.error(`[test-scrapers] ${failed} provider(s) errored`);
    process.exit(1);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
