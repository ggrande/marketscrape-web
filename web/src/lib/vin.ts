import type { DecodedVehicle } from "./types";

interface NhtsaRow {
  Variable: string;
  Value: string | null;
}

interface NhtsaResponse {
  Results: NhtsaRow[];
}

const NHTSA_URL = "https://vpic.nhtsa.dot.gov/api/vehicles/decodevin";

/**
 * Decode a VIN via NHTSA's free vPIC endpoint. No key required. Returns a
 * best-effort DecodedVehicle even when some fields are missing — downstream
 * providers frequently only need year/make/model.
 */
export async function decodeVin(vin: string, signal?: AbortSignal): Promise<DecodedVehicle> {
  const url = `${NHTSA_URL}/${encodeURIComponent(vin)}?format=json`;
  const res = await fetch(url, { signal, cache: "no-store" });
  if (!res.ok) throw new Error(`NHTSA vPIC request failed: ${res.status}`);
  const body = (await res.json()) as NhtsaResponse;
  const rows = new Map(body.Results.map((r) => [r.Variable, r.Value]));

  const yearRaw = rows.get("Model Year");
  const year = yearRaw ? Number.parseInt(yearRaw, 10) : NaN;

  return {
    vin,
    year: Number.isFinite(year) ? year : null,
    make: rows.get("Make") || null,
    model: rows.get("Model") || null,
    trim: rows.get("Trim") || null,
    bodyClass: rows.get("Body Class") || null,
  };
}
