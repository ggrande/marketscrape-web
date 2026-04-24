import { z } from "zod";
import type { Stagehand } from "@browserbasehq/stagehand";
import { BaseScraper, parseCurrency, type ScrapeContext } from "./base";
import type { OfferResult, ProviderMeta } from "@/lib/types";

/**
 * CarMax "Sell My Car" Instant Cash Offer.
 *
 * Uses Stagehand natural-language actions instead of CSS selectors so that
 * CarMax's frequent UI tweaks don't silently break the scraper.
 */
export class CarMaxScraper extends BaseScraper {
  readonly meta: ProviderMeta = {
    id: "carmax",
    name: "CarMax",
    logoUrl: "https://www.carmax.com/favicon.ico",
    siteUrl: "https://www.carmax.com/sell-my-car",
  };

  protected async run(
    stagehand: Stagehand,
    ctx: ScrapeContext,
  ): Promise<Omit<OfferResult, "providerId" | "durationMs">> {
    const { vin, zip } = ctx.request;
    const page = stagehand.page;

    await page.goto(this.meta.siteUrl, { waitUntil: "domcontentloaded" });

    await page.act("Dismiss any cookie banner or promotional modal if one is visible");

    await page.act(`Enter the VIN "${vin}" in the VIN input field`);
    await page.act(`Enter the ZIP code "${zip}" in the ZIP code input field`);
    await page.act("Click the button to continue or get my offer");

    // CarMax flows through a few screens that confirm mileage / condition /
    // contact details. The POC declines anything optional and accepts the
    // default condition. A production build would ingest mileage + condition.
    await page.act(
      "If asked for mileage, enter 60000. If asked about condition, choose 'Good'. " +
        "If asked about accident or title history, answer 'No'. " +
        "Skip or decline any optional upsell, account-creation, or phone-verification step. " +
        "Keep clicking the continue / next / get-my-offer button until an offer amount is displayed.",
    );

    const offer = await page.extract({
      instruction:
        "Extract the instant cash offer amount, the expiration date of the offer, " +
        "and the URL the user should visit to redeem the offer (if shown).",
      schema: z.object({
        offerAmount: z
          .string()
          .nullable()
          .describe('The cash offer, e.g. "$18,450". Null if no offer is displayed.'),
        expiresAt: z
          .string()
          .nullable()
          .describe("Offer expiration as an ISO-8601 date, or null if not shown."),
        claimUrl: z
          .string()
          .nullable()
          .describe("URL to claim/redeem the offer, or null if not shown."),
      }),
    });

    const amount = parseCurrency(offer.offerAmount);
    if (amount == null) {
      return { status: "no_offer", note: "CarMax did not return an offer amount" };
    }

    return {
      status: "success",
      amountUsd: amount,
      expiresAt: offer.expiresAt ?? undefined,
      claimUrl: offer.claimUrl ?? page.url(),
    };
  }
}
