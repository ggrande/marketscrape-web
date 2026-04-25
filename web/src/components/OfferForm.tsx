"use client";

import { useState } from "react";

interface Props {
  disabled?: boolean;
  onSubmit: (vin: string, zip: string) => void;
}

export function OfferForm({ disabled, onSubmit }: Props) {
  const [vin, setVin] = useState("");
  const [zip, setZip] = useState("");

  return (
    <form
      className="flex w-full max-w-3xl flex-col gap-3 rounded-2xl bg-white p-6 shadow-sm sm:flex-row sm:items-end"
      onSubmit={(e) => {
        e.preventDefault();
        onSubmit(vin.trim().toUpperCase(), zip.trim());
      }}
    >
      <label className="flex-1">
        <span className="mb-1 block text-xs font-medium uppercase tracking-wide text-neutral-500">
          VIN
        </span>
        <input
          value={vin}
          onChange={(e) => setVin(e.target.value)}
          placeholder="17-character VIN"
          maxLength={17}
          className="w-full rounded-lg border border-neutral-300 px-3 py-2 font-mono text-base uppercase tracking-wide outline-none focus:border-neutral-900"
          required
        />
      </label>
      <label className="sm:w-40">
        <span className="mb-1 block text-xs font-medium uppercase tracking-wide text-neutral-500">
          Zip
        </span>
        <input
          value={zip}
          onChange={(e) => setZip(e.target.value)}
          placeholder="e.g. 10001"
          inputMode="numeric"
          pattern="\d{5}"
          maxLength={5}
          className="w-full rounded-lg border border-neutral-300 px-3 py-2 font-mono text-base outline-none focus:border-neutral-900"
          required
        />
      </label>
      <button
        type="submit"
        disabled={disabled}
        className="rounded-lg bg-neutral-900 px-5 py-2 text-sm font-semibold text-white transition hover:bg-neutral-700 disabled:cursor-not-allowed disabled:opacity-50"
      >
        {disabled ? "Checking…" : "Get Offers"}
      </button>
    </form>
  );
}
