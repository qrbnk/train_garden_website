<!--
Repository-specific Copilot instructions for the Train Garden static website.
Designed to give an AI coding agent immediate, actionable context for edits.
Keep this short and concrete — reference real filenames and code examples below.
-->

# Copilot instructions — Train Garden (static site)

Summary
- This repository is a small static website (HTML/CSS/vanilla JS) served from the project root. Primary pages: `index.html`, `map.html`, `subscribe.html`. Styles live in `style.css` and small JS is inline in `index.html`.
- The site uses Google embed integrations: a Google Form (newsletter) in `subscribe.html` and an embedded Google Map in `map.html`.
- Deployment is expected via GitHub Pages (this repo contains a `CNAME` file). Treat file paths as root-relative.

What to know before editing
- No build system / no package.json: changes are direct edits to files in the repository root.
- Preserve filenames and relative paths for images (e.g., `logo.png`, `nature.png`, `hero-blue.png`), because HTML references them directly from the root.
- External embeds:
  - Newsletter form: open `subscribe.html` and update the iframe `src` if the Google Form changes. Example:

```html
<iframe src="https://docs.google.com/forms/d/e/…/viewform?embedded=true">…</iframe>
```

  - Map embed: open `map.html` and update the iframe `src` to change the map. Example:

```html
<iframe src="https://www.google.com/maps/d/u/4/embed?mid=1r0HJw53KoNuucSnhGcNFn5v5IHZ_U6w&ehbc=2E312F"></iframe>
```
 - Contact and CTA links: the header links in `index.html` are root-relative. Keep them consistent when renaming or moving pages. Example nav snippet:

```html
<ul class="nav-links">
  <li><a href="#home">Home</a></li>
  <li><a href="map.html">Skills Hubs Around You</a></li>
  <li><a href="https://forms.gle/..." target="_blank">For Jobseekers</a></li>
  <li><a href="mailto:traingarden25@gmail.com">Contact Us</a></li>
</ul>
```

Local run / testing
- Quick local server (recommended) — start a simple static server from the project root and open `http://localhost:8000` to test layouts and embeds. (Example commands are in the README.)
- Use a mobile viewport in the browser dev tools to check responsive breakpoints — CSS uses breakpoints at 768px and 1024px.

Patterns and conventions to follow
- Single-page content per HTML file: add new pages at the repo root and link with root-relative paths.
- CSS is global (`style.css`) — prefer adding scoped class names and updating `style.css` rather than introducing many new inline styles.
- Small interactive behaviors are inline in `index.html` (e.g., `resizeElements()`); keep these changes minimal and consider moving larger scripts to a new `main.js` in the root if functionality grows.
- Image sizing uses `clamp()` and utility classes (`small`, `medium`, `large`) — follow that pattern when adding responsive images.

Integration & security notes
- Embedded Google content (forms/maps) loads from external domains. When updating these, verify the embed URL and confirm it allows `iframe` embedding.
- Links that open a new tab use `target="_blank"` and `rel="noopener noreferrer"` in-nav (follow that pattern for external links).

PR guidance
- Preview the site locally (server) and verify:
  - All image assets load (no broken links to `*.png` files in root).
  - Embedded iframes render and are not blocked by CSP or browser blocking.
  - Navigation links work and header remains sticky on scroll.
- Keep diffs small and describe the intent in PR titles (e.g., "Update newsletter embed URL" or "Add volunteer page and link in header").

If you need more
- If you plan to add build tooling (webpack/Vite/etc.), note that it changes deployment flow — open an issue before converting this repo to a built site.
- If you can't find an asset referenced by the HTML (e.g., `logo.png`), list missing assets in your PR so the maintainer can confirm where to source them.

Files referenced (primary)
- `index.html` — home page (hero, services, newsletter CTA)
- `style.css` — global styles + responsive breakpoints
- `subscribe.html` — Google Form iframe (newsletter)
- `map.html` — Google Maps embed
- `CNAME` — indicates GitHub Pages host name

-- End of file
