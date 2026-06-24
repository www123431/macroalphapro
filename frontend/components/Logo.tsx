import Link from "next/link";

// Brand lockup: flat-bordered icon mark + MacroAlphaPro wordmark.
// Institutional terminals (Bloomberg / Citadel) use flat icons, not
// glowing badges — glow reads as "marketing", flat reads as "tool".
export function Logo({ href = "/", terminal = false }: { href?: string; terminal?: boolean }) {
  return (
    <Link href={href} className="group flex shrink-0 items-center gap-2.5">
      <span className="relative inline-flex h-8 w-8 items-center justify-center rounded-md border border-accent/35 bg-accent/[0.06] text-accent transition-colors group-hover:border-accent/60 group-hover:bg-accent/10">
        <svg viewBox="0 0 24 24" className="h-[17px] w-[17px]" fill="none" stroke="currentColor"
          strokeWidth={2.2} strokeLinecap="round" strokeLinejoin="round" aria-hidden>
          <path d="M3 17 L9 9 L13 13.5 L20.5 5" />
          <path d="M20.5 5 L15.5 5 M20.5 5 L20.5 10" />
        </svg>
      </span>
      <span className="flex items-baseline gap-1.5">
        <span className="text-[16px] font-semibold tracking-[-0.01em]">
          Macro<span className="text-accent">Alpha</span>Pro
        </span>
        {terminal && (
          <span className="hidden text-[10px] font-normal uppercase tracking-[0.15em] text-muted/60 sm:inline">
            Terminal
          </span>
        )}
      </span>
    </Link>
  );
}
