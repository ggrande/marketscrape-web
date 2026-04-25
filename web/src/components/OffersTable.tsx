"use client";

import type { OfferResult, OfferStatus, ProviderMeta } from "@/lib/types";

type Row = {
  provider: ProviderMeta;
  status: OfferStatus;
  result?: OfferResult;
};

const statusLabel: Record<OfferStatus, string> = {
  pending: "Queued",
  running: "Checking…",
  success: "Offer",
  no_offer: "No offer",
  error: "Error",
};

const currency = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

export function OffersTable({ rows }: { rows: Row[] }) {
  const sorted = [...rows].sort((a, b) => {
    const av = a.result?.amountUsd ?? -1;
    const bv = b.result?.amountUsd ?? -1;
    return bv - av;
  });

  return (
    <div className="w-full max-w-3xl overflow-hidden rounded-2xl bg-white shadow-sm">
      <table className="w-full text-left text-sm">
        <thead className="bg-neutral-100 text-xs uppercase tracking-wide text-neutral-500">
          <tr>
            <th className="px-4 py-3">Buyer</th>
            <th className="px-4 py-3">Offer</th>
            <th className="px-4 py-3">Expires</th>
            <th className="px-4 py-3">Status</th>
            <th className="px-4 py-3" />
          </tr>
        </thead>
        <tbody>
          {sorted.map((row) => (
            <tr key={row.provider.id} className="border-t border-neutral-100">
              <td className="px-4 py-3">
                <div className="flex items-center gap-2">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={row.provider.logoUrl}
                    alt=""
                    className="h-5 w-5 rounded"
                    loading="lazy"
                  />
                  <span className="font-medium">{row.provider.name}</span>
                </div>
              </td>
              <td className="px-4 py-3 font-mono">
                {row.result?.amountUsd != null
                  ? currency.format(row.result.amountUsd)
                  : "—"}
              </td>
              <td className="px-4 py-3 text-neutral-500">
                {row.result?.expiresAt
                  ? new Date(row.result.expiresAt).toLocaleDateString()
                  : "—"}
              </td>
              <td className="px-4 py-3">
                <StatusPill status={row.status} />
                {row.status === "error" && row.result?.error && (
                  <div className="mt-1 text-xs text-red-600">{row.result.error}</div>
                )}
              </td>
              <td className="px-4 py-3 text-right">
                {row.result?.claimUrl && (
                  <a
                    href={row.result.claimUrl}
                    target="_blank"
                    rel="noreferrer"
                    className="rounded-md bg-neutral-900 px-3 py-1 text-xs font-semibold text-white hover:bg-neutral-700"
                  >
                    Claim Offer
                  </a>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StatusPill({ status }: { status: OfferStatus }) {
  const className = {
    pending: "bg-neutral-100 text-neutral-600",
    running: "bg-amber-100 text-amber-800 animate-pulse",
    success: "bg-emerald-100 text-emerald-800",
    no_offer: "bg-neutral-100 text-neutral-600",
    error: "bg-red-100 text-red-800",
  }[status];
  return (
    <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${className}`}>
      {statusLabel[status]}
    </span>
  );
}
