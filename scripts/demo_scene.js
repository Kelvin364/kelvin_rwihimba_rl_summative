/* AgriScout — my WebGL field view (three.js r160).
 *
 * This renders the SAME recorded trace as the 2D grid, so the two views can never
 * disagree: both read `episodes[ep].frames[idx]`. I added the 3D view to make the
 * demo legible at a glance -- crop height and colour show field health, the rover is
 * a recognisable machine rather than a cube, and treatments fire a visible effect at
 * the treated cell, so you can see WHAT the agent did and WHERE without reading
 * anything.
 *
 * Everything is instanced or pooled and only re-posed each frame; I create and
 * destroy nothing during playback.
 */
const Scene = (() => {
  let renderer, scene, camera, rover, beacon, sun;
  let cropStalks, cropCanopy, pestMesh, ringPool = [], ringNext = 0;
  let rows = 0, cols = 0, mounted = false, raf = null;
  let orbit = { az: -0.72, el: 0.92, dist: 15, auto: true, drag: false, px: 0, py: 0 };
  // Rover pose is interpolated toward its target so playback glides between cells
  // instead of teleporting -- the single biggest "is this a demo or a debug view"
  // difference at 110ms/step.
  let pose = { x: 0, z: 0, tx: 0, tz: 0, yaw: 0, tyaw: 0 };
  const V = new THREE.Vector3(), M = new THREE.Matrix4(), Q = new THREE.Quaternion();
  const C = new THREE.Color(), SCALE = new THREE.Vector3(1, 1, 1);

  const CELL = 1.0;
  const wx = c => (c - (cols - 1) / 2) * CELL;
  const wz = r => (r - (rows - 1) / 2) * CELL;

  /* Palette mirrors the 2D view's roles, read from CSS so both follow the theme. */
  function themeColors() {
    const cs = getComputedStyle(document.documentElement);
    const get = n => cs.getPropertyValue(n).trim();
    return {
      soil: new THREE.Color(get('--soil')),
      crop: new THREE.Color(get('--crop')),
      irrigate: new THREE.Color(get('--irrigate')),
      spray: new THREE.Color(get('--spray')),
      depot: new THREE.Color(get('--depot')),
      bad: new THREE.Color(get('--bad')),
      dark: document.documentElement.getAttribute('data-theme') === 'dark'
        || (!document.documentElement.getAttribute('data-theme')
            && matchMedia('(prefers-color-scheme: dark)').matches),
    };
  }

  function buildRover(t) {
    const g = new THREE.Group();
    const body = new THREE.Mesh(
      new THREE.BoxGeometry(0.54, 0.2, 0.38),
      new THREE.MeshStandardMaterial({ color: 0xf2f2ef, metalness: 0.25, roughness: 0.45 }));
    body.position.y = 0.20; body.castShadow = true; g.add(body);

    const deck = new THREE.Mesh(
      new THREE.BoxGeometry(0.34, 0.08, 0.3),
      new THREE.MeshStandardMaterial({ color: 0x2f3440, metalness: 0.5, roughness: 0.35 }));
    deck.position.set(-0.04, 0.34, 0); deck.castShadow = true; g.add(deck);

    // Solar panel — reads instantly as "field robot".
    const panel = new THREE.Mesh(
      new THREE.BoxGeometry(0.46, 0.02, 0.32),
      new THREE.MeshStandardMaterial({ color: 0x14213d, metalness: 0.75, roughness: 0.18 }));
    panel.position.set(-0.02, 0.40, 0); panel.castShadow = true; g.add(panel);

    // Wheels.
    const wheelGeo = new THREE.CylinderGeometry(0.11, 0.11, 0.07, 18);
    const wheelMat = new THREE.MeshStandardMaterial({ color: 0x1b1b1d, roughness: 0.85 });
    for (const [dx, dz] of [[0.18, 0.21], [0.18, -0.21], [-0.18, 0.21], [-0.18, -0.21]]) {
      const w = new THREE.Mesh(wheelGeo, wheelMat);
      w.rotation.x = Math.PI / 2; w.position.set(dx, 0.11, dz); w.castShadow = true; g.add(w);
    }

    // Forward sensor boom — doubles as an unmistakable heading indicator.
    const boom = new THREE.Mesh(
      new THREE.CylinderGeometry(0.022, 0.022, 0.3, 10),
      new THREE.MeshStandardMaterial({ color: 0x9aa0aa, metalness: 0.6, roughness: 0.4 }));
    boom.rotation.z = Math.PI / 2; boom.position.set(0.34, 0.26, 0); g.add(boom);
    const head = new THREE.Mesh(
      new THREE.ConeGeometry(0.075, 0.16, 14),
      new THREE.MeshStandardMaterial({ color: 0xffc83d, metalness: 0.3, roughness: 0.35 }));
    head.rotation.z = -Math.PI / 2; head.position.set(0.5, 0.26, 0);
    head.castShadow = true; g.add(head);

    // Beacon: recoloured per action, so the machine itself announces what it did.
    beacon = new THREE.Mesh(
      new THREE.SphereGeometry(0.07, 16, 12),
      new THREE.MeshStandardMaterial({ color: 0xffffff, emissive: 0x000000,
        emissiveIntensity: 1.6, roughness: 0.3 }));
    beacon.position.set(-0.12, 0.5, 0); g.add(beacon);

    // Locator ring: at true scale the rover is a speck among 54 crops. This keeps it
    // findable from any camera angle without my having to inflate the machine.
    const loc = new THREE.Mesh(
      new THREE.TorusGeometry(0.46, 0.028, 8, 32),
      new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.9 }));
    loc.rotation.x = -Math.PI / 2; loc.position.y = 0.035; g.add(loc);
    g.userData.locator = loc;

    // Scaled up: legibility beats literal proportion in a demo this size.
    g.scale.setScalar(1.45);
    return g;
  }

  function mount(container, nRows, nCols) {
    rows = nRows; cols = nCols;
    const t = themeColors();

    if (!renderer) {
      // preserveDrawingBuffer keeps the frame readable by canvas.toDataURL after the
      // draw call returns, which is what scripts/record_video_web.py captures. Costs
      // a little memory bandwidth; worth it to make the good-looking view recordable.
      renderer = new THREE.WebGLRenderer({
        antialias: true, alpha: true, preserveDrawingBuffer: true });
      renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
      renderer.shadowMap.enabled = true;
      renderer.shadowMap.type = THREE.PCFSoftShadowMap;
      renderer.outputColorSpace = THREE.SRGBColorSpace;
      renderer.toneMapping = THREE.ACESFilmicToneMapping;
      // Below 1.0: at 1.05 the tan soil bed washed out to near-white under the
      // light-mode key light, flattening the whole plot.
      renderer.toneMappingExposure = 0.92;
      container.appendChild(renderer.domElement);
      attachControls(renderer.domElement);
    }
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(42, 16 / 9, 0.1, 200);

    // Keep fill well under the key light -- a strong hemisphere fill erases the
    // shadows that make crop height readable.
    scene.add(new THREE.HemisphereLight(t.dark ? 0x33405c : 0xbfd9ff,
                                        t.dark ? 0x0b0b0a : 0x6b5b40, t.dark ? 0.9 : 0.85));
    sun = new THREE.DirectionalLight(0xfff4e0, t.dark ? 1.6 : 2.1);
    sun.position.set(6, 11, 5);
    sun.castShadow = true;
    const span = Math.max(rows, cols) * 0.85 + 2;
    Object.assign(sun.shadow.camera, { left: -span, right: span, top: span, bottom: -span,
                                       near: 1, far: 40 });
    sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.bias = -0.0012;
    sun.shadow.camera.updateProjectionMatrix();
    scene.add(sun);

    // Ground: a soil bed slightly larger than the planted area, plus a rim so the
    // field reads as a plot rather than a floating grid.
    const bedW = cols * CELL + 1.6, bedD = rows * CELL + 1.6;
    const bed = new THREE.Mesh(
      new THREE.BoxGeometry(bedW, 0.35, bedD),
      new THREE.MeshStandardMaterial({ color: t.soil.clone().multiplyScalar(0.78),
                                       roughness: 0.97 }));
    bed.position.y = -0.175; bed.receiveShadow = true; scene.add(bed);
    const rim = new THREE.Mesh(
      new THREE.BoxGeometry(bedW + 0.3, 0.1, bedD + 0.3),
      new THREE.MeshStandardMaterial({ color: t.soil.clone().multiplyScalar(0.7),
                                       roughness: 1.0 }));
    rim.position.y = -0.3; rim.receiveShadow = true; scene.add(rim);

    // Furrow lines: cheap, and they make the plot read as cultivated ground.
    const furrowMat = new THREE.MeshStandardMaterial({
      color: t.soil.clone().multiplyScalar(0.82), roughness: 1.0 });
    for (let r = 0; r < rows; r++) {
      const f = new THREE.Mesh(new THREE.BoxGeometry(cols * CELL + 0.8, 0.02, 0.09), furrowMat);
      f.position.set(0, 0.005, wz(r)); f.receiveShadow = true; scene.add(f);
    }

    // Depot pad at grid (0,0) — where RETURN_TO_DEPOT actually refills.
    const pad = new THREE.Mesh(
      new THREE.BoxGeometry(0.92, 0.04, 0.92),
      new THREE.MeshStandardMaterial({ color: t.depot, roughness: 0.5, metalness: 0.1 }));
    pad.position.set(wx(0), 0.02, wz(0)); pad.receiveShadow = true; scene.add(pad);
    const padRing = new THREE.Mesh(
      new THREE.TorusGeometry(0.52, 0.03, 8, 28),
      new THREE.MeshStandardMaterial({ color: 0xffc83d, emissive: 0x3a2c00 }));
    padRing.rotation.x = -Math.PI / 2; padRing.position.set(wx(0), 0.045, wz(0));
    scene.add(padRing);

    const n = rows * cols;
    cropStalks = new THREE.InstancedMesh(
      new THREE.CylinderGeometry(0.035, 0.05, 1, 7),
      new THREE.MeshStandardMaterial({ color: 0x6b5433, roughness: 0.9 }), n);
    cropCanopy = new THREE.InstancedMesh(
      new THREE.IcosahedronGeometry(0.34, 1),
      new THREE.MeshStandardMaterial({ color: 0xffffff, roughness: 0.72, flatShading: true }), n);
    pestMesh = new THREE.InstancedMesh(
      new THREE.IcosahedronGeometry(0.13, 0),
      new THREE.MeshStandardMaterial({ color: t.bad, emissive: t.bad,
        emissiveIntensity: 0.45, roughness: 0.6, flatShading: true }), n);
    for (const m of [cropStalks, cropCanopy, pestMesh]) {
      m.castShadow = true; m.receiveShadow = true;
      m.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      scene.add(m);
    }

    // Pooled effect rings — reused, never allocated mid-playback.
    ringPool = [];
    for (let i = 0; i < 6; i++) {
      const ring = new THREE.Mesh(
        new THREE.TorusGeometry(0.3, 0.045, 8, 24),
        new THREE.MeshBasicMaterial({ transparent: true, opacity: 0 }));
      ring.rotation.x = -Math.PI / 2; ring.visible = false;
      ring.userData.t = 1; scene.add(ring); ringPool.push(ring);
    }

    rover = buildRover(t); scene.add(rover);
    // Frame the plot rather than assuming a fixed distance, so the view is filled
    // for any field size.
    orbit.dist = Math.max(rows, cols) * 1.28 + 2.2;
    mounted = true;
    resize();
    if (!raf) loop();
  }

  /* Minimal orbit: drag to rotate, wheel to zoom. three.js dropped OrbitControls from
     examples/js in r150+, and I would rather write one small handler than take on a
     second dependency for this. */
  function attachControls(el) {
    el.style.touchAction = 'none';
    el.addEventListener('pointerdown', e => {
      orbit.drag = true; orbit.auto = false; orbit.px = e.clientX; orbit.py = e.clientY;
      el.setPointerCapture(e.pointerId);
    });
    el.addEventListener('pointermove', e => {
      if (!orbit.drag) return;
      orbit.az -= (e.clientX - orbit.px) * 0.008;
      orbit.el = Math.max(0.18, Math.min(1.45, orbit.el - (e.clientY - orbit.py) * 0.006));
      orbit.px = e.clientX; orbit.py = e.clientY;
    });
    el.addEventListener('pointerup', e => {
      orbit.drag = false; try { el.releasePointerCapture(e.pointerId); } catch (_) {}
    });
    el.addEventListener('wheel', e => {
      e.preventDefault();
      orbit.dist = Math.max(6, Math.min(30, orbit.dist + Math.sign(e.deltaY) * 0.9));
    }, { passive: false });
  }

  function resize() {
    if (!renderer || !renderer.domElement.parentElement) return;
    const box = renderer.domElement.parentElement.getBoundingClientRect();
    const w = Math.max(200, box.width), h = Math.max(200, box.width * 0.62);
    renderer.setSize(w, h, false);
    renderer.domElement.style.width = w + 'px';
    renderer.domElement.style.height = h + 'px';
    if (camera) { camera.aspect = w / h; camera.updateProjectionMatrix(); }
  }

  /* Push one recorded frame into the scene. `snap` skips interpolation (scrubbing). */
  function update(f, successHealth, snap) {
    if (!mounted) return;
    const soil = themeColors().soil, crop = themeColors().crop;
    let i = 0;
    for (let r = 0; r < rows; r++) {
      for (let c = 0; c < cols; c++, i++) {
        const h = Math.max(0, Math.min(1, f.h[r][c]));
        const stalkH = 0.12 + h * 0.5;
        M.compose(V.set(wx(c), stalkH / 2, wz(r)), Q.identity(),
                  SCALE.set(1, stalkH, 1));
        cropStalks.setMatrixAt(i, M);
        // Canopy: height AND size track health, so the field's state is legible in
        // silhouette alone -- not only by colour.
        const s = 0.42 + h * 0.72;
        M.compose(V.set(wx(c), stalkH + 0.16 * s, wz(r)), Q.identity(),
                  SCALE.set(s, s * 0.82, s));
        cropCanopy.setMatrixAt(i, M);
        cropCanopy.setColorAt(i, C.copy(soil).lerp(crop, h));

        const p = f.p[r][c];
        if (p > 0.02) {
          const ps = 0.5 + Math.min(1, p / 0.6) * 1.5;
          M.compose(V.set(wx(c) + 0.26, stalkH + 0.34, wz(r) + 0.24), Q.identity(),
                    SCALE.set(ps, ps, ps));
        } else {
          M.compose(V.set(0, -50, 0), Q.identity(), SCALE.set(0.001, 0.001, 0.001));
        }
        pestMesh.setMatrixAt(i, M);
      }
    }
    cropStalks.instanceMatrix.needsUpdate = true;
    cropCanopy.instanceMatrix.needsUpdate = true;
    if (cropCanopy.instanceColor) cropCanopy.instanceColor.needsUpdate = true;
    pestMesh.instanceMatrix.needsUpdate = true;

    pose.tx = wx(f.cell[1]); pose.tz = wz(f.cell[0]);
    const YAW = { MOVE_E: 0, MOVE_W: Math.PI, MOVE_N: -Math.PI / 2, MOVE_S: Math.PI / 2 };
    if (f.a in YAW) {
      let d = YAW[f.a] - pose.tyaw;
      while (d > Math.PI) d -= 2 * Math.PI;
      while (d < -Math.PI) d += 2 * Math.PI;
      pose.tyaw += d;
    }
    if (snap) { pose.x = pose.tx; pose.z = pose.tz; pose.yaw = pose.tyaw; }

    const t = themeColors();
    const bc = f.kind === 'irrigate' ? t.irrigate : f.kind === 'spray' ? t.spray
             : f.kind === 'depot' ? t.depot : new THREE.Color(0x9aa0aa);
    beacon.material.color.copy(bc);
    beacon.material.emissive.copy(f.kind === 'move' || f.kind === 'idle'
      ? new THREE.Color(0x000000) : bc);

    if (!snap && (f.kind === 'irrigate' || f.kind === 'spray' || f.kind === 'depot')) {
      const ring = ringPool[ringNext++ % ringPool.length];
      ring.material.color.copy(bc);
      ring.position.set(pose.tx, 0.1, pose.tz);
      ring.userData.t = 0; ring.visible = true;
    }
  }

  function loop() {
    raf = requestAnimationFrame(loop);
    step();
  }

  /* One frame of animation + draw, callable synchronously.
     Headless Chrome under --virtual-time-budget never fires requestAnimationFrame,
     so the offline recorder (scripts/record_video_web.py) drives this directly
     instead of waiting on the rAF loop that never ticks. */
  function step() {
    if (!mounted) return;
    if (orbit.auto) orbit.az += 0.0016;
    const k = 0.22;
    pose.x += (pose.tx - pose.x) * k;
    pose.z += (pose.tz - pose.z) * k;
    pose.yaw += (pose.tyaw - pose.yaw) * k;
    rover.position.set(pose.x, 0, pose.z);
    rover.rotation.y = -pose.yaw;
    const pulse = 1.2 + Math.sin(performance.now() * 0.006) * 0.8;
    beacon.material.emissiveIntensity = pulse;
    const loc = rover.userData.locator;
    if (loc) {
      loc.material.color.copy(beacon.material.color);
      loc.material.opacity = 0.45 + 0.3 * (pulse / 2);
    }

    for (const ring of ringPool) {
      if (!ring.visible) continue;
      ring.userData.t += 0.045;
      const t = ring.userData.t;
      if (t >= 1) { ring.visible = false; continue; }
      const s = 0.5 + t * 2.4;
      ring.scale.set(s, s, 1);
      ring.material.opacity = 0.85 * (1 - t);
    }

    const cy = Math.sin(orbit.el) * orbit.dist, cr = Math.cos(orbit.el) * orbit.dist;
    camera.position.set(Math.cos(orbit.az) * cr, cy, Math.sin(orbit.az) * cr);
    camera.lookAt(0, 0.4, 0);
    renderer.render(scene, camera);
  }

  return {
    mount, update, resize,
    renderNow: step,
    // Offline recording forces this to 1: headless Chrome on a Retina display
    // reports devicePixelRatio 2, which quadruples the drawing buffer and made
    // software-GL capture time out.
    setPixelRatio: r => { if (renderer) { renderer.setPixelRatio(r); resize(); } },
    setAuto: v => { orbit.auto = v; },
    getAuto: () => orbit.auto,
    isMounted: () => mounted,
    reset: () => { mounted = false; if (renderer) renderer.domElement.remove(); renderer = null; },
  };
})();
