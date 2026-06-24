// Server wrapper for static export: declares pre-rendered paper_id
// slugs at build time, then renders the client reader view.

import { getAllPaperIds } from "@/lib/static-params";
import PaperReaderView from "./_view";

export const dynamicParams = false;

export function generateStaticParams() {
  return getAllPaperIds().map((paper_id) => ({ paper_id }));
}

export default function Page({ params }: { params: Promise<{ paper_id: string }> }) {
  return <PaperReaderView params={params} />;
}
