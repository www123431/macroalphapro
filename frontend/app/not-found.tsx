import Link from "next/link";
import { Background } from "@/components/Background";

export default function NotFound() {
  return (
    <>
      <Background />
      <div className="flex min-h-screen flex-col items-center justify-center px-6 text-center">
        <div className="text-7xl font-bold tracking-tight text-muted/30">404</div>
        <p className="mt-3 text-muted">This route doesn&apos;t exist in the terminal.</p>
        <Link href="/" className="btn-glow mt-6 inline-flex items-center gap-2 rounded-lg px-5 py-2.5 text-sm font-medium text-foreground">
          Back to start <span aria-hidden>→</span>
        </Link>
      </div>
    </>
  );
}
