"use client";

import { motion } from "framer-motion";
import { Languages } from "lucide-react";
import { useI18n, Lang } from "@/lib/i18n";
import { fadeUp, stagger } from "@/lib/motion";
import { Card, SectionTitle, cn } from "@/components/ui";

const LANGS: { value: Lang; label: string }[] = [
  { value: "en", label: "English" },
  { value: "zh", label: "中文" },
];

export default function SettingsPage() {
  const { lang, setLang, t } = useI18n();

  return (
    <div className="flex min-h-[68vh] flex-col items-center justify-center">
      <motion.div variants={stagger(0.08)} initial="hidden" animate="show" className="w-full max-w-xl space-y-6">
        <motion.div variants={fadeUp} className="text-center">
          <h1 className="text-2xl font-semibold tracking-tight">{t("settings.title")}</h1>
          <p className="mt-1 text-sm text-muted">{t("settings.subtitle")}</p>
        </motion.div>

        <motion.div variants={fadeUp}>
          <SectionTitle>
            <span className="inline-flex items-center gap-1.5"><Languages className="h-3.5 w-3.5" /> {t("settings.language")}</span>
          </SectionTitle>
          <Card className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
            <p className="max-w-md text-sm leading-relaxed text-muted">{t("settings.language_desc")}</p>
            <div className="inline-flex shrink-0 rounded-lg border border-border bg-panel2 p-1">
              {LANGS.map((l) => (
                <button key={l.value} onClick={() => setLang(l.value)}
                  className={cn("rounded-md px-4 py-1.5 text-sm transition-colors",
                    lang === l.value ? "bg-accent/15 text-accent" : "text-muted hover:text-foreground")}>
                  {l.label}
                </button>
              ))}
            </div>
          </Card>
        </motion.div>

        <motion.p variants={fadeUp} className="text-center text-xs text-muted/70">{t("settings.more_soon")}</motion.p>
      </motion.div>
    </div>
  );
}
