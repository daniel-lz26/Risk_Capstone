// Where the FastAPI backend lives. Override with NEXT_PUBLIC_API_URL in .env.local.
export const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

// Matches CountrySummary in api/schemas.py
export type CountrySummary = {
  iso3: string;
  country: string;
  latest_year: number;
  fragility_score: number;
  alert_prob: number;
  alert: boolean;
};

// Function that calls API and returns list of countries
export async function getCountries(): Promise<CountrySummary[]> {
  const res = await fetch(`${API_URL}/countries`);
  if (!res.ok) throw new Error(`GET /countries failed: ${res.status}`);
  return res.json();
}
