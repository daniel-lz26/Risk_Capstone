import { getCountries } from "@/lib/api";

// grabs phone from file 1 (calls api and waits for country list)
export default async function CountryTable() {
  const countries = await getCountries();
  // sort highest score first (makes a copy first tho so we dont mess with og)
  const ranked = [...countries].sort((a, b) => b.fragility_score - a.fragility_score);

  // thead = header row (column titles)
  // tbody = actual country rows
  return (
    <table className="mt-6 w-full text-left text-sm">
      <thead className="border-b text-zinc-500">
        <tr>
          <th className="py-2">#</th>
          <th className="py-2">Country</th>
          <th className="py-2 text-right">Fragility score</th>
          <th className="py-2 text-right">Disruption chance</th>
          <th className="py-2 text-right">Alert</th>
        </tr>
      </thead>
      <tbody>
        {/* .map = for each country c at position 1 make 1 row */}
        {/* rank number, starts at 1 */}
        {ranked.map((c, i) => (
          <tr key={c.iso3} className="border-b border-zinc-100 dark:border-zinc-800">
            <td className="py-2 text-zinc-400">{i + 1}</td>
            <td className="py-2">{c.country}</td>
            <td className="py-2 text-right font-mono">{c.fragility_score.toFixed(1)}</td>
            {/* turns 0.153 into 15% */}
            <td className="py-2 text-right font-mono">{Math.round(c.alert_prob * 100)}%</td>
            <td className="py-2 text-right">{c.alert ? "Alert" : "—"}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
