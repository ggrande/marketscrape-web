"use client";

import { useCallback, useMemo, useRef, useState } from "react";
import { OfferForm } from "@/components/OfferForm";
import { OffersTable } from "@/components/OffersTable";
import type {
  DecodedVehicle,
  OfferResult,
  OfferStatus,
  OfferStreamEvent,
  ProviderId,
} from "@/lib/types";

// Mirror of the server-side registry meta. Kept tiny for now (one provider)
// and designed so additional providers can be appended.
const providerMeta = [
  {
    id: "carmax" as ProviderId,
    name: "CarMax",
    logoUrl: "https://www.carmax.com/favicon.ico",
    siteUrl: "https://www.carmax.com/sell-my-car",
  },
];

export default function Home() {
  const [vehicle, setVehicle] = useState<DecodedVehicle | null>(null);
  const [statuses, setStatuses] = useState<Record<ProviderId, OfferStatus>>(
    () => Object.fromEntries(providerMeta.map((p) => [p.id, "pending"])) as Record<
      ProviderId,
      OfferStatus
    >,
  );
  const [results, setResults] = useState<Record<ProviderId, OfferResult>>(
    {} as Record<ProviderId, OfferResult>,
  );
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);

  const rows = useMemo(
    () =>
      providerMeta.map((provider) => ({
        provider,
        status: statuses[provider.id] ?? "pending",
        result: results[provider.id],
      })),
    [statuses, results],
  );

  const start = useCallback((vin: string, zip: string) => {
    esRef.current?.close();
    setVehicle(null);
    setStatuses(
      Object.fromEntries(providerMeta.map((p) => [p.id, "pending"])) as Record<
        ProviderId,
        OfferStatus
      >,
    );
    setResults({} as Record<ProviderId, OfferResult>);
    setError(null);
    setRunning(true);

    const url = `/api/offers/stream?vin=${encodeURIComponent(vin)}&zip=${encodeURIComponent(zip)}`;
    const es = new EventSource(url);
    esRef.current = es;

    es.onmessage = (ev) => {
      const event = JSON.parse(ev.data) as OfferStreamEvent;
      if (event.type === "vehicle") {
        setVehicle(event.vehicle);
      } else if (event.type === "status") {
        setStatuses((prev) => ({ ...prev, [event.providerId]: event.status }));
      } else if (event.type === "result") {
        setResults((prev) => ({ ...prev, [event.result.providerId]: event.result }));
        setStatuses((prev) => ({ ...prev, [event.result.providerId]: event.result.status }));
      } else if (event.type === "error") {
        setError(event.message);
      } else if (event.type === "done") {
        setRunning(false);
        es.close();
      }
    };

    es.onerror = () => {
      setError("Connection lost");
      setRunning(false);
      es.close();
    };
  }, []);

  return (
    <main className="mx-auto flex min-h-screen max-w-3xl flex-col items-center gap-8 px-6 py-16">
      <header className="text-center">
        <h1 className="text-4xl font-semibold tracking-tight">Instant Cash Offer Aggregator</h1>
        <p className="mt-2 text-neutral-600">
          Enter your VIN once. Get every major buyer&apos;s offer in parallel.
        </p>
      </header>

      <OfferForm disabled={running} onSubmit={start} />

      {vehicle && (
        <div className="w-full max-w-3xl rounded-xl bg-white px-5 py-3 text-sm text-neutral-700 shadow-sm">
          <span className="text-neutral-500">Decoded:</span>{" "}
          <span className="font-medium">
            {[vehicle.year, vehicle.make, vehicle.model, vehicle.trim].filter(Boolean).join(" ")}
          </span>
        </div>
      )}

      {error && (
        <div className="w-full max-w-3xl rounded-xl border border-red-200 bg-red-50 px-5 py-3 text-sm text-red-800">
          {error}
        </div>
      )}

      <OffersTable rows={rows} />
    </main>
  );
}
