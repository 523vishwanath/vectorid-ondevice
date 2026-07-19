// VectorID service worker — caches everything for true offline use.
// Bump CACHE_VERSION whenever you change index.html or the model to force a refresh.
const CACHE_VERSION = 'vectorid-v1';
const ASSETS = [
  './',
  './index.html',
  './vectorcam.onnx',
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js',
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort-wasm-simd-threaded.wasm',
  'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort-wasm-simd-threaded.jsep.wasm',
  'https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700&family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@500&display=swap'
];

// Install: cache everything up front
self.addEventListener('install', (e) => {
  e.waitUntil(
    caches.open(CACHE_VERSION).then((cache) =>
      // addAll fails if any one asset 404s; use individual puts so one miss
      // (e.g. a wasm filename that varies) doesn't break the whole install
      Promise.allSettled(ASSETS.map((url) =>
        fetch(url, { mode: 'no-cors' }).then((r) => cache.put(url, r)).catch(() => {})
      ))
    ).then(() => self.skipWaiting())
  );
});

// Activate: clean up old caches
self.addEventListener('activate', (e) => {
  e.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE_VERSION).map((k) => caches.delete(k)))
    ).then(() => self.clients.claim())
  );
});

// Fetch: serve from cache first, fall back to network, and cache new network hits
self.addEventListener('fetch', (e) => {
  e.respondWith(
    caches.match(e.request).then((cached) => {
      if (cached) return cached;
      return fetch(e.request).then((resp) => {
        // cache successful GETs for next time (esp. the wasm files ort loads at runtime)
        if (e.request.method === 'GET' && resp && (resp.status === 200 || resp.type === 'opaque')) {
          const copy = resp.clone();
          caches.open(CACHE_VERSION).then((cache) => cache.put(e.request, copy));
        }
        return resp;
      }).catch(() => cached);
    })
  );
});
