// @ts-check
import { defineConfig } from 'astro/config';
import mdx from '@astrojs/mdx';
import tailwindcss from '@tailwindcss/vite';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

/** Restart the dev server when new content files appear so the
 *  content collection re-syncs and picks them up. */
function restartOnNewContent() {
  return {
    name: 'restart-on-new-content',
    configureServer(server) {
      const onAddOrUnlink = (file) => {
        if (/\/src\/content\/.*\.(md|mdx)$/.test(file)) {
          server.restart();
        }
      };
      server.watcher.on('add', onAddOrUnlink);
      server.watcher.on('unlink', onAddOrUnlink);
    },
  };
}

export default defineConfig({
  site: 'https://eoinmurray.github.io',
  base: '/demolab2',
  integrations: [mdx()],
  markdown: {
    remarkPlugins: [remarkMath],
    rehypePlugins: [rehypeKatex],
  },
  vite: {
    plugins: [tailwindcss(), restartOnNewContent()],
    server: {
      watch: { usePolling: true, interval: 300 },
    },
  },
});
