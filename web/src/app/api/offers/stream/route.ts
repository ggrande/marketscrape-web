import { NextRequest } from "next/server";
import { offerRequestSchema, type OfferStreamEvent } from "@/lib/types";
import { decodeVin } from "@/lib/vin";
import { providers } from "@/scrapers/registry";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const PROVIDER_BUDGET_MS = 90_000;

/**
 * SSE fan-out. On GET the client streams:
 *   - `vehicle` once, after VIN decode
 *   - `status` transitions per provider (pending -> running -> success/error)
 *   - `result` when each provider finishes
 *   - `done` when every provider has settled
 */
export async function GET(req: NextRequest) {
  const parsed = offerRequestSchema.safeParse({
    vin: req.nextUrl.searchParams.get("vin") ?? "",
    zip: req.nextUrl.searchParams.get("zip") ?? "",
  });

  if (!parsed.success) {
    return new Response(parsed.error.issues.map((i) => i.message).join("; "), {
      status: 400,
    });
  }

  const request = parsed.data;
  const abort = new AbortController();
  req.signal.addEventListener("abort", () => abort.abort(), { once: true });

  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      const encoder = new TextEncoder();
      const send = (event: OfferStreamEvent) => {
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(event)}\n\n`));
      };

      try {
        const vehicle = await decodeVin(request.vin, abort.signal);
        send({ type: "vehicle", vehicle });

        for (const p of providers) {
          send({ type: "status", providerId: p.meta.id, status: "pending" });
        }

        const deadline = Date.now() + PROVIDER_BUDGET_MS;
        await Promise.all(
          providers.map(async (provider) => {
            send({ type: "status", providerId: provider.meta.id, status: "running" });
            const result = await provider.scrape({
              request,
              vehicle,
              deadlineMs: deadline,
              signal: abort.signal,
            });
            send({ type: "result", result });
          }),
        );

        send({ type: "done" });
      } catch (err) {
        send({
          type: "error",
          message: err instanceof Error ? err.message : String(err),
        });
      } finally {
        controller.close();
      }
    },
    cancel() {
      abort.abort();
    },
  });

  return new Response(stream, {
    headers: {
      "Content-Type": "text/event-stream",
      "Cache-Control": "no-cache, no-transform",
      Connection: "keep-alive",
    },
  });
}
