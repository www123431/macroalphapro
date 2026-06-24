// Server wrapper: declares pre-rendered slugs for static export, then
// delegates to the client view. Required by Next.js 16 + `output: export`
// — dynamic routes need generateStaticParams() at build time.

import { getAllPaperIds } from "@/lib/static-params";
import PaperDetailView from "./_view";

export const dynamicParams = false;

export function generateStaticParams() {
  return getAllPaperIds().map((paper_id) => ({ paper_id }));
}

export default function Page({ params }: { params: Promise<{ paper_id: string }> }) {
  return <PaperDetailView params={params} />;
}
