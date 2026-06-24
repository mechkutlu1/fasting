// Fasting Console — offline cache (cache-first for app shell)
const CACHE = 'fasting-console-v1';
const SHELL = ['./', 'index.html', 'manifest.webmanifest', 'icon-192.png', 'icon-512.png'];

self.addEventListener('install', e => {
  e.waitUntil(caches.open(CACHE).then(c => c.addAll(SHELL)).then(() => self.skipWaiting()));
});
self.addEventListener('activate', e => {
  e.waitUntil(
    caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
      .then(() => self.clients.claim())
  );
});
self.addEventListener('fetch', e => {
  const req = e.request;
  if (req.method !== 'GET') return;                 // never cache API writes
  if (new URL(req.url).pathname.endsWith('/state')) return; // always hit network for sync
  e.respondWith(
    caches.match(req).then(hit => hit || fetch(req).then(res => {
      // runtime-cache same-origin GETs (e.g. fonts handled by browser cache otherwise)
      return res;
    }).catch(() => hit))
  );
});
