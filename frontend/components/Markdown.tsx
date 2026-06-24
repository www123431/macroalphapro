"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

// Dark-theme markdown renderer for chat answers (GFM tables/lists/code). Styled via an explicit
// component map (no typography plugin needed). Compact spacing to sit well inside chat bubbles.
export function Markdown({ children }: { children: string }) {
  return (
    <div className="space-y-2 text-sm leading-relaxed">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => <p className="leading-relaxed">{children}</p>,
          strong: ({ children }) => <strong className="font-semibold text-foreground">{children}</strong>,
          em: ({ children }) => <em className="italic">{children}</em>,
          h1: ({ children }) => <h3 className="mt-2 text-base font-semibold text-foreground">{children}</h3>,
          h2: ({ children }) => <h3 className="mt-2 text-base font-semibold text-foreground">{children}</h3>,
          h3: ({ children }) => <h4 className="mt-2 text-sm font-semibold text-foreground">{children}</h4>,
          ul: ({ children }) => <ul className="list-disc space-y-1 pl-5">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal space-y-1 pl-5">{children}</ol>,
          li: ({ children }) => <li className="leading-relaxed">{children}</li>,
          a: ({ children, href }) => <a href={href} target="_blank" rel="noreferrer" className="text-accent underline underline-offset-2">{children}</a>,
          hr: () => <hr className="my-3 border-border" />,
          blockquote: ({ children }) => <blockquote className="border-l-2 border-border pl-3 text-muted">{children}</blockquote>,
          code: ({ className, children }) => {
            const block = (className ?? "").includes("language-");
            return block
              ? <code className="block overflow-x-auto rounded-lg bg-panel2 p-3 text-xs">{children}</code>
              : <code className="tnum rounded bg-panel2 px-1 py-0.5 text-[0.85em] text-accent">{children}</code>;
          },
          pre: ({ children }) => <pre className="overflow-x-auto">{children}</pre>,
          table: ({ children }) => (
            <div className="my-2 overflow-x-auto">
              <table className="w-full border-collapse text-xs">{children}</table>
            </div>
          ),
          thead: ({ children }) => <thead className="text-muted">{children}</thead>,
          th: ({ children }) => <th className="border border-border px-2 py-1 text-left font-medium">{children}</th>,
          td: ({ children }) => <td className="tnum border border-border px-2 py-1">{children}</td>,
        }}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}
