import { Suspense } from "react";

import CountryTable from "@/components/CountryTable";

export default function Home() {
  return (
    <main className="mx-auto w-full max-w-4xl p-8">
      <h1 className="text-3xl font-semibold">SentinX</h1>
      <p className="mt-1 text-zinc-500">
        {/* current it isn't global, our data is only Asia-Pacific */}
        Supply-chain disruption risk forecast for global countries, highest risk first.
      </p>
      <Suspense fallback={<p className="mt-6 text-zinc-500">Loading countries…</p>}>
        <CountryTable />
      </Suspense>
    </main>
  );
}