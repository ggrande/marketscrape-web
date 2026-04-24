import type { Stagehand } from "@browserbasehq/stagehand";
import { createStagehand } from "@/lib/stagehand";
import type {
  DecodedVehicle,
  OfferRequest,
  OfferResult,
  ProviderId,
  ProviderMeta,
} from "@/lib/types";

export interface ScrapeContext {
  request: OfferRequest;
  vehicle: DecodedVehicle;
  /** Hard deadline — providers should bail out rather than exceed this. */
  deadlineMs: number;
  signal: AbortSignal;
}

/**
 * Base class for every ICO provider. Concrete providers implement {@link run}
 * using Stagehand's `act()` / `extract()` methods with natural-language
 * instructions so that minor UI changes on the target site self-heal.
 */
export abstract class BaseScraper {
  abstract readonly meta: ProviderMeta;

  /** Default budget a provider gets before we surface a timeout. */
  readonly defaultTimeoutMs = 90_000;

  protected get id(): ProviderId {
    return this.meta.id;
  }

  /**
   * Public entry point. Spins up its own Stagehand session, runs the provider
   * implementation, normalises timeouts / errors into an OfferResult, and
   * guarantees the browser is closed.
   */
  async scrape(ctx: ScrapeContext): Promise<OfferResult> {
    const started = Date.now();
    let stagehand: Stagehand | null = null;
    try {
      stagehand = await createStagehand();
      const result = await this.withDeadline(
        this.run(stagehand, ctx),
        ctx.deadlineMs - started,
        ctx.signal,
      );
      return {
        ...result,
        providerId: this.id,
        durationMs: Date.now() - started,
      };
    } catch (err) {
      return {
        providerId: this.id,
        status: "error",
        error: err instanceof Error ? err.message : String(err),
        durationMs: Date.now() - started,
      };
    } finally {
      if (stagehand) {
        await stagehand.close().catch(() => {});
      }
    }
  }

  /** Provider-specific scrape. Return an OfferResult missing `providerId`/`durationMs`. */
  protected abstract run(
    stagehand: Stagehand,
    ctx: ScrapeContext,
  ): Promise<Omit<OfferResult, "providerId" | "durationMs">>;

  private async withDeadline<T>(
    p: Promise<T>,
    ms: number,
    signal: AbortSignal,
  ): Promise<T> {
    if (ms <= 0) throw new Error("Provider deadline already elapsed");
    return await new Promise<T>((resolve, reject) => {
      const to = setTimeout(
        () => reject(new Error(`Provider ${this.id} timed out after ${ms}ms`)),
        ms,
      );
      const onAbort = () => reject(new Error("Aborted"));
      signal.addEventListener("abort", onAbort, { once: true });
      p.then(
        (v) => {
          clearTimeout(to);
          signal.removeEventListener("abort", onAbort);
          resolve(v);
        },
        (e) => {
          clearTimeout(to);
          signal.removeEventListener("abort", onAbort);
          reject(e);
        },
      );
    });
  }
}

/** Parse "$12,345" / "12345" / "12,345.00" etc. into a number, or null. */
export function parseCurrency(raw: string | null | undefined): number | null {
  if (!raw) return null;
  const digits = raw.replace(/[^0-9.]/g, "");
  if (!digits) return null;
  const n = Number.parseFloat(digits);
  return Number.isFinite(n) ? Math.round(n) : null;
}
