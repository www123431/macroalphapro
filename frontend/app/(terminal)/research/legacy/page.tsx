"use client";

// /research/legacy — REDIRECT to consolidated /research/lessons view.
//
// 2026-06-04 (PR-A+B consolidation): the standalone "legacy" surface
// was a duplicate of /research/lessons with a hardcoded filter. URL
// preserved for outside links; behavior is now a one-tick redirect
// into the canonical lessons page with the right query params.

import { useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";


export default function LegacyRedirect() {
  const router = useRouter();
  useEffect(() => {
    router.replace("/research/lessons?grounding_method=pretrain_grounded&include_legacy=true");
  }, [router]);

  return (
    <div className="p-6 text-sm text-muted">
      Redirecting to{" "}
      <Link href="/research/lessons?grounding_method=pretrain_grounded&include_legacy=true"
            className="underline text-accent">
        /research/lessons (legacy filter)
      </Link>
      …
    </div>
  );
}
