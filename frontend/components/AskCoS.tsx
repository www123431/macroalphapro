"use client";

import Link from "next/link";
import { MessagesSquare } from "lucide-react";
import { useI18n } from "@/lib/i18n";
import { cn } from "@/components/ui";

// Deep-link from a data surface into the Chief-of-Staff chat with a context-tagged question
// pre-filled (/chat?q=…). Connects the data layer to the agentic layer: you ask about WHAT YOU'RE
// LOOKING AT, in context, instead of typing from scratch.
export function AskCoS({ q, label, className = "" }: { q: string; label?: string; className?: string }) {
  const { t } = useI18n();
  return (
    <Link href={`/chat?q=${encodeURIComponent(q)}`} onClick={(e) => e.stopPropagation()}
      className={cn("inline-flex items-center gap-1 text-xs text-muted transition-colors hover:text-accent", className)}>
      <MessagesSquare className="h-3 w-3" /> {label ?? t("chat.ask_cos")}
    </Link>
  );
}
