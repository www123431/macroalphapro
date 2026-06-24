// Server wrapper for static export: declares pre-rendered lesson_id
// slugs at build time, then renders the client lesson-detail view.

import { getAllLessonIds } from "@/lib/static-params";
import LessonDetailView from "./_view";

export const dynamicParams = false;

export function generateStaticParams() {
  return getAllLessonIds().map((lesson_id) => ({ lesson_id }));
}

export default function Page({ params }: { params: Promise<{ lesson_id: string }> }) {
  return <LessonDetailView params={params} />;
}
