import { z } from "zod";

export const vinSchema = z
  .string()
  .trim()
  .toUpperCase()
  .length(17, "VIN must be exactly 17 characters")
  .regex(/^[A-HJ-NPR-Z0-9]{17}$/, "VIN contains invalid characters (no I, O, or Q)");

export const zipSchema = z
  .string()
  .trim()
  .regex(/^\d{5}$/, "Zip must be 5 digits");

export const offerRequestSchema = z.object({
  vin: vinSchema,
  zip: zipSchema,
});

export type OfferRequest = z.infer<typeof offerRequestSchema>;

export interface DecodedVehicle {
  vin: string;
  year: number | null;
  make: string | null;
  model: string | null;
  trim: string | null;
  bodyClass: string | null;
}

export type ProviderId =
  | "carmax"
  | "carvana"
  | "vroom"
  | "driveway"
  | "autonation"
  | "echopark"
  | "peddle"
  | "wheelzy"
  | "kbb"
  | "edmunds"
  | "truecar"
  | "cargurus"
  | "cars"
  | "caredge";

export interface ProviderMeta {
  id: ProviderId;
  name: string;
  logoUrl: string;
  siteUrl: string;
}

export type OfferStatus = "pending" | "running" | "success" | "no_offer" | "error";

export interface OfferResult {
  providerId: ProviderId;
  status: OfferStatus;
  amountUsd?: number;
  expiresAt?: string; // ISO-8601
  claimUrl?: string;
  note?: string; // e.g. "Requires inspection"
  error?: string;
  durationMs?: number;
}

/** Event emitted over SSE as a provider progresses. */
export type OfferStreamEvent =
  | { type: "vehicle"; vehicle: DecodedVehicle }
  | { type: "status"; providerId: ProviderId; status: OfferStatus }
  | { type: "result"; result: OfferResult }
  | { type: "done" }
  | { type: "error"; message: string };
