"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { motion } from "framer-motion";
import { fadeUp, stagger } from "@/lib/motion";
import { Background } from "@/components/Background";
import { Logo } from "@/components/Logo";
import { useI18n } from "@/lib/i18n";
import { api } from "@/lib/api";

export default function Landing() {
  const { t } = useI18n();
  const [online, setOnline] = useState<boolean | null>(null);

  useEffect(() => {
    let alive = true;
    api.health().then(() => alive && setOnline(true)).catch(() => alive && setOnline(false));
    return () => { alive = false; };
  }, []);

  return (
    <>
      <Background />

      {/* Brand bar — logo-only. The ONLINE status pill was previously
          here next to the wordmark and read as visual noise on the
          landing surface. Moved to a fixed bottom-right "status pill"
          (Vercel / Stripe / Linear convention — system status belongs
          at the page edge, not in the header). */}
      <header className="mx-auto flex max-w-6xl items-center justify-between px-6 py-5">
        <Logo />
      </header>

      <span
        title={online == null ? "" : online ? "Backend reachable" : "Backend unreachable"}
        className={`fixed bottom-4 right-4 z-30 inline-flex items-center gap-1.5 rounded-full border bg-background/70 px-2.5 py-1 text-[10px] uppercase tracking-[0.12em] backdrop-blur-sm shadow-sm ${
          online == null ? "border-muted/30 text-muted/70"
            : online      ? "border-ok/30 text-ok/90"
            : "border-alert/35 text-alert/90"
        }`}>
        <span className={`h-1.5 w-1.5 rounded-full ${
          online == null ? "bg-muted/60"
            : online      ? "bg-ok live-dot"
            : "bg-alert"
        }`} />
        {online == null ? t("land.connecting") : online ? t("land.online") : t("land.offline")}
      </span>

      {/* hero — the project name is the statement; generous whitespace, few words */}
      <section className="mx-auto flex min-h-[78vh] max-w-3xl flex-col items-center justify-center px-6 text-center">
        <motion.div variants={stagger(0.12)} initial="hidden" animate="show" className="flex flex-col items-center">
          <motion.span variants={fadeUp} className="mb-7 text-xs uppercase tracking-[0.3em] text-muted">
            {t("land.kicker")}
          </motion.span>

          <motion.h1 variants={fadeUp}
            className="text-6xl font-bold tracking-tight sm:text-8xl">
            <span className="bg-gradient-to-b from-foreground to-foreground/55 bg-clip-text text-transparent">Macro</span>
            <span className="bg-gradient-to-b from-accent to-accent/60 bg-clip-text text-transparent">Alpha</span>
            <span className="bg-gradient-to-b from-foreground to-foreground/55 bg-clip-text text-transparent">Pro</span>
          </motion.h1>

          <motion.p variants={fadeUp} className="mt-6 text-sm tracking-wide text-muted">
            {t("land.tagline")}
          </motion.p>

          <motion.div variants={fadeUp} className="mt-10">
            <Link href="/dashboard"
              className="btn-glow group inline-flex items-center gap-2 rounded-lg px-7 py-3 text-sm font-medium text-foreground">
              {t("land.enter")}
              <span className="transition-transform duration-200 group-hover:translate-x-1">→</span>
            </Link>
          </motion.div>

          <motion.p variants={fadeUp} className="mt-14 flex flex-wrap items-center justify-center gap-x-3 gap-y-1 text-xs tracking-wide text-muted/70">
            <span>{t("land.f1")}</span><span className="text-muted/25">·</span>
            <span>{t("land.f2")}</span><span className="text-muted/25">·</span>
            <span>{t("land.f3")}</span><span className="text-muted/25">·</span>
            <span>{t("land.f4")}</span>
          </motion.p>
        </motion.div>
      </section>

      <footer className="py-6 text-center text-xs text-muted/60">
        deterministic-decision quant terminal · {new Date().getFullYear()}
      </footer>
    </>
  );
}
