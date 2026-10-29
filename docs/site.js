/* How the sky dates a tablet.
 *
 * Every value comes from data/*.json, which scripts/site/build_site_data.py writes from
 * the committed results. The page draws; it computes no verdict. The verdict of each
 * clause in each year is the "grid" of the record. The dials draw where the Sun and the
 * planets stood, from data/sky.json, so the reader can see why a year agrees.
 *
 * The page follows one record in three scrolling chapters, each with a sticky figure.
 * When a step reaches the reading line, its figure moves to the state of that step. A
 * forward step plays its animation; a jump sets the end state at once. Chapter 2 has one
 * step per decisive clause of the record, so its steps are written again per record.
 */
(function () {
  'use strict';

  const N = 701;
  const MAX_DIALS = 7;            // decisive clauses that get a dial of their own
  const motion = window.matchMedia('(prefers-reduced-motion: reduce)');
  const still = () => motion.matches;
  let META = null;
  let SKY = null;
  let REC = null;
  let CLOCKS = [];                // the active decisive clauses, in the order of Fig. 3
  let C = {};
  const tip = document.querySelector('.tooltip');

  // ------------------------------------------------------------------ helpers
  const yearLabel = (y) => (y <= 0 ? `${1 - y} BCE` : `AD ${y}`);
  const spanLabel = (y) => (y <= 0 ? `${1 - y}/${-y} BCE` : `AD ${y}/${y + 1}`);
  const fmt = d3.format(',');
  const plural = (n, one, many) => `${fmt(n)} ${n === 1 ? one : many}`;
  const esc = (s) => String(s ?? '').replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const narrow = (n) => (n > 0 && n < N ? Math.log2(N / n) / Math.log2(N) : 0);
  const capital = (s) => s.charAt(0).toUpperCase() + s.slice(1);
  const showMonth = (m) => (m ? m.replace('2', '₂') : m);
  const BODY = { sun: 'the Sun', moon: 'the Moon', mercury: 'Mercury', venus: 'Venus', mars: 'Mars',
    jupiter: 'Jupiter', saturn: 'Saturn', lunar: 'the Moon', solar: 'the Sun' };
  const PLANETS = new Set(['mercury', 'venus', 'mars', 'jupiter', 'saturn']);
  const POSITIONS = new Set(['in_sign', 'entry', 'station', 'first_app', 'last_app']);
  const VERDICT = { a: 'agrees', c: 'contradicts', '.': 'untestable' };
  const CLOCK_NAME = { mars: 'Mars', jupiter: 'Jupiter', saturn: 'Saturn', lunar_eclipse: 'lunar eclipses',
    seasonal_sirius: 'the solar year' };
  const medal = (name) => `img/sky/${name}.webp`;
  // The drawings of the signs and the bodies, loaded once for the canvas.
  const ART = {};
  function art(name) {
    if (!ART[name]) { ART[name] = new Image(); ART[name].src = medal(name); }
    return ART[name];
  }

  function readColors() {
    const cs = getComputedStyle(document.documentElement);
    ['bg', 'bg-2', 'card', 'ink', 'ink-2', 'muted', 'faint', 'rule', 'gold', 'gold-ink', 'gold-leaf', 'cross', 'cross-wash', 'seal',
      'c-mars', 'c-jupiter', 'c-saturn', 'c-moon', 'c-sun', 'c-venus', 'c-mercury']
      .forEach((k) => { C[k] = cs.getPropertyValue(`--${k}`).trim(); });
  }
  // The clock of a clause decides its ink and its medallion.
  function clockOf(c) {
    if (c.clock === 'lunar_eclipse' || c.kind === 'eclipse') return { color: C['c-moon'], medal: 'moon', name: 'lunar eclipses' };
    if (c.clock === 'seasonal_sirius' || c.kind === 'season' || c.kind === 'sirius') return { color: C['c-sun'], medal: 'sun', name: 'the solar year' };
    const body = PLANETS.has(c.clock) ? c.clock : (PLANETS.has(c.target) ? c.target : c.body);
    return { color: C[`c-${body}`] || C.gold, medal: ['sun', 'moon', ...PLANETS].includes(body) ? body : 'moon', name: BODY[body] || body };
  }
  function tipShow(ev, html) {
    tip.innerHTML = html;
    tip.style.opacity = 1;
    const r = tip.getBoundingClientRect();
    let x = ev.clientX + 14;
    let y = ev.clientY + 14;
    if (x + r.width > innerWidth - 8) x = ev.clientX - r.width - 14;
    if (y + r.height > innerHeight - 8) y = ev.clientY - r.height - 14;
    tip.style.left = `${x}px`;
    tip.style.top = `${y}px`;
  }
  const tipHide = () => { tip.style.opacity = 0; };
  function passage(c, max = 60) {
    const t = c.passage.replace(/\s+/g, ' ').replace(/\s([,.;:])/g, '$1');
    return t.length > max ? `${t.slice(0, max - 1)}…` : t;
  }
  function fields(c) {
    const parts = [BODY[c.body] ? capital(BODY[c.body]) : c.body];
    if (c.target) parts.push(`to ${BODY[c.target] || c.target}`);
    if (c.star) parts.push(`at ${c.star}`);
    if (c.sign) parts.push(`in ${c.sign}`);
    parts.push(c.month ? `month ${String(c.month).split('|').map(showMonth).join(' or ')}${c.day ? `, day ${c.day}` : ''}` : 'no month');
    return `${c.type} · ${parts.filter(Boolean).join(' · ')}`;
  }
  function canvasFor(fig, w, h, cls) {
    const dpr = window.devicePixelRatio || 1;
    const cv = fig.append('canvas').attr('class', cls).attr('width', Math.round(w * dpr)).attr('height', Math.round(h * dpr))
      .style('width', `${w}px`).style('height', `${h}px`);
    const ctx = cv.node().getContext('2d');
    ctx.scale(dpr, dpr);
    return { cv, ctx };
  }

  // ------------------------------------------------------------------ the clocks of a record
  // The active decisive clauses, sorted by how many years each agrees with, as in Fig. 3a,
  // and for each year whether every clause up to it agrees.
  function findClocks(rec) {
    const list = rec.clauses.map((c, j) => ({ c, j, row: rec.grid[j] }))
      .filter((d) => d.c.decisive && d.c.bits > 0)
      .map((d) => ({ ...d, agree: d.row.split('').reduce((n, ch) => n + (ch === 'a'), 0) }))
      .sort((a, b) => d3.descending(a.agree, b.agree) || d3.ascending(a.j, b.j));
    let running = new Uint8Array(N).fill(1);
    list.forEach((d) => {
      running = running.map((v, i) => (v && d.row[i] === 'a' ? 1 : 0));
      d.cross = running;
      d.crossCount = running.reduce((n, v) => n + v, 0);
      d.clock = clockOf(d.c);
    });
    return list;
  }

  // Where a clause's body stood in each candidate year, as arcs of longitude in degrees.
  // A clause with a month gets the track of that month; one without a month gets every
  // month of its year, as its rule tests the whole year. A clause after a year change
  // looks at the year after the candidate. An eclipse has no body to follow, so it sits
  // at its phase in the 18-year cycle of eclipses.
  function dialOf(c) {
    const off = c.year_offset || 0;
    // A clause can allow two months, such as VI or VI2 in a year with an extra month.
    const months = c.month ? String(c.month).split('|') : [];
    const ks = months.map((x) => SKY.months.indexOf(x)).filter((k) => k >= 0);
    const bodyIndex = (b) => SKY.bodies.indexOf(b);
    const nb = SKY.bodies.length;
    const cells = (i) => {
      const row = SKY.lon[i + off];
      if (!row) return [];
      if (months.length) return ks.map((k) => row[k]).filter(Boolean);
      return row.filter(Boolean);
    };
    const segs = new Array(N);
    const out = { segs, wedge: null, rim: 'signs', what: '', medal: clockOf(c).medal };
    const track = (fn) => {
      for (let i = 0; i < N; i += 1) {
        const list = cells(i).map(fn).filter(Boolean);
        segs[i] = list.length ? list : null;
      }
    };
    const when = months.length ? `in month ${months.map(showMonth).join(' or ')} of that year` : 'over that whole year';
    if (PLANETS.has(c.body) && POSITIONS.has(c.kind)) {
      const b = bodyIndex(c.body);
      track((cell) => [cell[b] / 10, cell[b + nb] / 10]);
      const s = SKY.signs.indexOf(c.sign);
      if (s >= 0) out.wedge = [30 * s, 30 * s + 30, c.sign];
      out.what = `where ${BODY[c.body]} moved ${when}`;
    } else if (c.kind === 'acronychal' && PLANETS.has(c.body)) {
      const b = bodyIndex(c.body);
      const sun = bodyIndex('sun');
      track((cell) => [(cell[b] - cell[sun] + 3600) % 3600 / 10, (cell[b + nb] - cell[sun + nb] + 3600) % 3600 / 10]);
      out.wedge = [165, 195, 'opposite the Sun'];
      out.rim = 'elongation';
      out.what = `the angle of ${BODY[c.body]} from the Sun ${when}`;
    } else if (PLANETS.has(c.target) || PLANETS.has(c.body)) {
      const body = PLANETS.has(c.target) ? c.target : c.body;
      const b = bodyIndex(body);
      track((cell) => [cell[b] / 10, cell[b + nb] / 10]);
      out.what = `where ${BODY[body]} moved ${when}`;
      out.medal = body;
    } else if (c.kind === 'season' || c.kind === 'sirius') {
      const sun = bodyIndex('sun');
      track((cell) => [cell[sun] / 10, cell[sun + nb] / 10]);
      out.what = `where the Sun moved ${when}`;
    } else {
      for (let i = 0; i < N; i += 1) {
        const t = 360 * ((i / 18.03) % 1);
        segs[i] = [[t - 4, t + 4]];
      }
      out.rim = 'cycle';
      out.what = 'the place of that year in the 18-year cycle of eclipses. This dial is a diagram, not a computed position';
    }
    return out;
  }

  // ------------------------------------------------------------------ hero ring
  function setupRing() {
    const wheel = document.querySelector('.zodiac-ring .wheel');
    if (!wheel || !SKY) return;
    wheel.innerHTML = SKY.signs.map((s, i) => {
      const t = ((i * 30 - 90) * Math.PI) / 180;
      const x = 50 + 40.5 * Math.cos(t) - 9.5;
      const y = 50 + 40.5 * Math.sin(t) - 9.5;
      art(s.toLowerCase());
      return `<img src="${medal(s.toLowerCase())}" alt="" style="left:${x}%;top:${y}%">`;
    }).join('');
  }

  // ------------------------------------------------------------------ the chooser
  // The swatch darkens as the ranking narrows: one top year is darkest.
  function swatchOpacity(r) { return r.n_decisive ? 0.14 + 0.86 * narrow(r.first) : 0.06; }
  function setupChooser() {
    const input = document.getElementById('chooser-input');
    const list = document.getElementById('chooser-list');
    const sw = document.querySelector('.combo-swatch');
    const rows = META.records.slice().sort((a, b) => d3.descending(a.n_decisive > 0, b.n_decisive > 0)
      || d3.ascending(a.strict, b.strict) || d3.ascending(a.id, b.id));
    let shown = [];
    let active = -1;
    let query = '';
    const rowHTML = (r, i) => `<li role="option" id="opt-${i}" data-i="${i}" aria-selected="${i === active}">
        <span class="sw" style="opacity:${swatchOpacity(r).toFixed(2)}"></span>
        <span class="id">${r.id}</span><span class="what">${esc(museumLabel(r.museum_no))} · ${esc(r.volume)}</span>
        <span class="yrs">${r.n_decisive ? `${plural(r.first, 'top year', 'top years')} · ${fmt(r.strict)} stay` : 'no decisive clause'}</span></li>`;
    function render(q) {
      const query = q.trim().toLowerCase();
      const match = rows.filter((r) => !query || r.id.toLowerCase().includes(query)
        || String(r.museum_no || '').toLowerCase().includes(query) || museumLabel(r.museum_no).toLowerCase().includes(query));
      const featured = query ? [] : META.featured.map((id) => rows.find((r) => r.id === id)).filter(Boolean);
      shown = featured.concat(match);
      list.innerHTML = (featured.length ? '<li class="head">Suggested</li>' : '')
        + featured.map(rowHTML).join('')
        + (featured.length ? '<li class="head">All tablets · the darker the dot, the fewer the top years</li>' : '')
        + match.slice(0, 420).map((r, i) => rowHTML(r, i + featured.length)).join('');
    }
    const open = () => { query = ''; render(''); list.hidden = false; input.setAttribute('aria-expanded', 'true'); };
    const close = () => {
      list.hidden = true; input.setAttribute('aria-expanded', 'false'); input.removeAttribute('aria-activedescendant');
      active = -1; if (REC) input.value = REC.id;
    };
    input.addEventListener('focus', () => { input.select(); open(); });
    input.addEventListener('input', () => { active = -1; query = input.value; render(query); list.hidden = false; });
    input.addEventListener('keydown', (ev) => {
      if (ev.key === 'ArrowDown') { active = Math.min(active + 1, shown.length - 1); render(query); ev.preventDefault(); }
      else if (ev.key === 'ArrowUp') { active = Math.max(active - 1, 0); render(query); ev.preventDefault(); }
      else if (ev.key === 'Enter') { const r = shown[Math.max(active, 0)]; if (r) { choose(r.id); input.blur(); } }
      else if (ev.key === 'Escape') input.blur();
      if (active >= 0) input.setAttribute('aria-activedescendant', `opt-${active}`);
      const li = list.querySelector('[aria-selected="true"]');
      if (li) li.scrollIntoView({ block: 'nearest' });
    });
    list.addEventListener('mousedown', (ev) => {
      const li = ev.target.closest('li');
      if (!li || li.classList.contains('head')) return;
      ev.preventDefault();
      choose(shown[+li.dataset.i].id);
      input.blur();
    });
    input.addEventListener('blur', () => setTimeout(close, 100));
    document.querySelectorAll('.open-chooser').forEach((b) => b.addEventListener('click', () => {
      input.focus();
    }));
    document.querySelector('.random').addEventListener('click', () => {
      const pool = META.records.filter((r) => r.n_decisive > 0 && r.id !== REC?.id);
      choose(pool[Math.floor(Math.random() * pool.length)].id);
    });
    return (rec) => {
      const row = META.records.find((r) => r.id === rec.id);
      input.value = rec.id;
      sw.style.opacity = row ? swatchOpacity(row) : 0.1;
    };
  }
  let showChoice = () => {};

  // Each record is downloaded once; a failed download is tried again the next time.
  const RECORDS = new Map();
  function record(id) {
    if (!RECORDS.has(id)) RECORDS.set(id, d3.json(`data/records/${id}.json`).catch((err) => { RECORDS.delete(id); throw err; }));
    return RECORDS.get(id);
  }
  function choose(id, stay) {
    if (REC && REC.id === id) return;
    record(id).then((rec) => {
      const switching = !!REC;
      const sky = document.getElementById('ch-sky');
      const skyTop = sky.getBoundingClientRect().top;
      REC = rec;
      CLOCKS = findClocks(rec);
      history.replaceState(null, '', `#${id}`);
      tipHide();
      REVEALED = false;
      document.body.classList.remove('revealed');
      d3.select('.provenance-panel').attr('hidden', true);
      showChoice(rec);
      writeText();
      writeClockSteps();
      sections.forEach((s) => { if (s.impl.stop) s.impl.stop(); s.current = -1; s.rendered = false; s.waiting = false; });
      // A reader at chapter IV stays there, and the game follows the new tablet; chapter IV keeps its place on the
      // screen when the text above it changes length. Elsewhere, a new tablet starts its story at the beginning, so
      // its year does not show at once.
      const start = document.getElementById('ch-text');
      if (switching && skyTop < innerHeight * 0.5) scrollBy(0, sky.getBoundingClientRect().top - skyTop);
      else if (switching && !stay && start.getBoundingClientRect().top < 0) start.scrollIntoView();
      gameSync(id);
      onScroll();
    });
  }

  // ------------------------------------------------------------------ the words of the steps
  const PLACE = ['first place', 'second place', 'third place'];
  // The reveal stays out of the page, and out of reach of a screen reader, until its step.
  // Once the reader reaches the reveal, it stays for this tablet, so the year shows when they scroll back.
  let REVEAL_HTML = '';
  let REVEALED = false;
  const REVEAL_WAIT = 'The last step brings in the date that the edition gives.';
  function setRevealed(on) {
    REVEALED = REVEALED || on;
    document.body.classList.toggle('revealed', REVEALED);
    d3.select('.u-reveal').html(REVEALED ? REVEAL_HTML : REVEAL_WAIT);
    document.querySelectorAll('.chron').forEach((el) => el.setAttribute('aria-hidden', REVEALED ? 'false' : 'true'));
  }
  function writeText() {
    const r = REC;
    const dec = r.clauses.filter((c) => c.decisive);
    d3.select('.rec-line').text(`${r.id}${r.museum_no ? `, ${museumLabel(r.museum_no)}` : ''}, from ${r.volume}`);
    writeMuseum(r);
    const idle = r.clauses.filter((c) => c.idle).length;
    d3.select('.t-intro').html(!(r.text || '').trim()
      ? `${esc(r.id)} has no English translation, so the system cannot read it.`
      : r.tablet
        ? `This is ${esc(r.id)}, as the edition reads its signs. The cuneiform comes from that reading, sign by sign,
          with the ORACC Sign List. The dark hollows show where the clay is broken. The faint signs in them are the signs
          that Sachs and Hunger restored. Move the pointer over a line to read it in Latin letters.`
        : `This is the English translation of ${esc(r.id)}. The edition gives no reading of its signs line by line.`);
    d3.select('.t-clauses').html(`The system reads only the English translation. Words on a <span class="restored">hatched
      ground</span> are in square brackets. Sachs and Hunger restored them where the clay is broken. The shaded words are year numbers and
      names of kings. Code hides them from the language model. The dating uses year numbers only to count the years
      between two year headings. The record number also holds the number of the edition, and thus its year, but no step
      of the dating reads it. This page hides these words only for the story. It cannot keep them secret, because the
      record number and the data files hold them. It shows them at the end.`);
    d3.select('.t-decisive').html(r.clauses.length
      ? `The system finds ${plural(r.clauses.length, 'clause', 'clauses')} in the text. Each clause is one report about the
        sky. The inked clauses are <span class="k k-gold">decisive</span>. Only some types of clause can be decisive. The
        authors chose them by how well they agree with the published dates. They are most clauses about Mars, Jupiter or
        Saturn, lunar eclipses, solstices, equinoxes and dates of Sirius. A decisive clause also needs text that is not
        restored. It needs a secure date, or a test over its whole year. The scribe must not mark the event as
        unobserved. Fixed rules apply these conditions, not a model. ${dec.length
        ? `Here ${plural(dec.length, 'clause is', 'clauses are')} decisive.` : 'Here no clause is decisive.'}${idle
        ? ` ${plural(idle, 'other clause has', 'other clauses have')} a decisive type, but ${idle === 1 ? 'it agrees' : 'they agree'}
          with each year that the tools can test, so ${idle === 1 ? 'it does' : 'they do'} not narrow the years.` : ''}
        The other clauses only put the years in order.`
      : `The system finds no clause about the sky in this text, so the sky cannot date it.`);
    writeModel(r);
    const maxK = r.K.length ? d3.max(r.K) : 0;
    const nMax = r.K.filter((k) => k === maxK).length;
    d3.select('.u-bars').html(CLOCKS.length
      ? `Each decisive clause adds weight to the years that it agrees with. The weight is in bits. It is the base-2
        logarithm of the years that the tools can test, divided by the years that agree. Thus a clause that agrees with
        1 year in 4 adds 2 bits. Clauses with the same verdicts share their bits. In the strip, the ink of each year gets
        darker as the clauses add their weight.`
      : 'This record has no decisive clause. Thus no year gets decisive weight, and all 701 years stay.');
    d3.select('.u-clocks').html(CLOCKS.length
      ? `Two decisive clauses of one clock count as one clock. Thus the rules also count the clocks that agree with each
        year. The gold dots show the years with the most clocks: ${plural(maxK, 'clock', 'clocks')}, in
        ${plural(nMax, 'year', 'years')}.`
      : 'With no decisive clause, no clock agrees with any year.');
    const strict = r.strict.length;
    d3.select('.u-strict').html(CLOCKS.length
      ? `One year beats another year if it has at least as many clocks and at least as much decisive weight, and more of one.
        The years that no other year beats <span class="k k-gold">stay</span>. Here ${plural(strict, 'year stays', 'years stay')}${strict <= 4
        ? `: ${r.strict.map(yearLabel).join(', ')}` : ''}. Gold marks them in the strip. The paper calls them the strict set.`
      : 'No year beats another year, so all 701 years stay.');
    const g1 = r.groups[0] || [];
    d3.select('.u-ladder').html(CLOCKS.length
      ? `The rules then rank the years. The years that stay come first. Next come the years that stay when the rules
        remove one decisive statement, the safety set${!r.safety ? '' : r.safety.length === N
        ? ' (here all 701 years: when the rules remove one decisive statement, no year is removed)'
        : ` (here ${plural(r.safety.length, 'year', 'years')})`}.
        All other years come last. In each part, more clocks come first, then more decisive weight. Years with equal scores
        share a place. The line under each year shows its weight, in the colors of the clocks that agree with it. The
        years in first place are the top years. ${g1.length === 1 ? 'Here one year is in first place, alone.'
        : `Here ${g1.length} years share first place.`}`
      : (r.clauses.length ? `The supporting clauses can still put the years in order. The rules do not date a record
        from them, so this page does not draw that order.` : 'With no clause, the rules cannot put the years in order.'));
    const sup = r.clauses.filter((c) => !c.decisive);
    const named = sup.slice().sort((a, b) => d3.descending(a.bits, b.bits)).slice(0, 3).map((c) => `“${esc(passage(c, 46))}”`);
    d3.select('.u-support').html(!CLOCKS.length ? (sup.length
      ? `The record has ${plural(sup.length, 'supporting clause', 'supporting clauses')}. They cannot remove a year.` : 'The record has no clause that the tools can test.') : sup.length
      ? `The record also has ${plural(sup.length, 'supporting clause', 'supporting clauses')}, such as ${named.length > 1
        ? `${named.slice(0, -1).join(', ')} and ${named[named.length - 1]}` : named[0]}. They cannot remove a year. They
        add the gray part of each weight line. When years tie on their clocks and on their decisive weight, this part puts
        them in order. If years still tie, the year that fewer statements contradict comes first.`
      : 'This record has no supporting clause, so the decisive clauses alone set the order.');
    const pub = r.published_astro;
    const safe = r.safety || [];
    let reveal = 'The edition gives no single year for this record.';
    if (pub != null) {
      const rank = r.groups.findIndex((g) => g.includes(pub));
      const inStrict = r.strict.includes(pub);
      const prefer = !g1.length ? 'no year' : (g1.length <= 4 ? g1.map(yearLabel).join(', ') : `the ${g1.length} top years`);
      const desig = String(r.designation || '').replace(/^AD\s*/, '').replace(/^-/, '−');
      reveal = `Now the date of the edition enters. Sachs and Hunger list this record as no. ${esc(desig)} of
        ${esc(r.volume)}, from <span class="k k-seal">${spanLabel(pub)}</span>. `;
      const i = pub - META.years[0];
      const j = g1.length ? g1[0] - META.years[0] : -1;
      if (pub < META.years[0] || pub > META.years[1]) reveal += 'This year is outside the 701 candidate years, so the system cannot find it.';
      else if (strict === N) reveal += 'The sky alone cannot narrow this record.';
      else if (rank === 0) reveal += g1.length === 1 ? 'This is the one top year.' : `This year is one of the ${g1.length} top years.`;
      else {
        const where = rank > 0 ? `in ${PLACE[rank] || `place ${rank + 1}`}` : 'below third place';
        const stays = inStrict ? 'This year stays' : (safe.includes(pub)
          ? 'This year does not stay, but it is in the safety set' : 'This year does not stay');
        reveal += `${stays}, and it is ${where}. `;
        if (j >= 0 && r.K[i] === r.K[j] && Math.abs(r.ID[i] - r.ID[j]) < 1e-6) {
          reveal += `The decisive clauses do not separate it from the top years. Only the supporting clauses put ${prefer} first.`;
        } else {
          reveal += `The rules put ${prefer} first. A specialist can check if a clause is misread, or if the date of the
            edition needs a check.`;
        }
      }
    }
    let claim = 'The rules do not date this record.';
    if (r.claim === 'committed') {
      claim = r.basis === 'secure_lunar_eclipse'
        ? 'The rules commit to this year. Only this year stays, and an observed lunar eclipse agrees with it.'
        : `The rules commit to this year. Only this year stays, and two or more clocks agree with it. It also stays alone
          when the rules remove any one decisive statement.`;
    } else if (r.claim === 'ranked') {
      claim = strict > 1 ? 'More than one year stays, so the rules rank the years and do not commit to one.'
        : (safe.length > 1 ? `Only one year stays, but the safety set holds ${safe.length} years. Thus the rules rank the years and do not commit to one.`
          : 'Only one year stays, but only one clock agrees with it. Thus the rules rank the years and do not commit to one.');
    } else if (r.claim === 'decline' && CLOCKS.length && strict === N) {
      claim = 'The decisive clauses do not narrow the years, so the rules do not date this record.';
    } else if (r.claim === 'decline' && !CLOCKS.length && r.clauses.length) {
      claim = 'The record has no decisive clause, so the rules do not date it.';
    } else if (r.claim === 'decline' && !r.clauses.length) {
      claim = 'The record has no clause that the tools can test, so the rules do not date it.';
    }
    REVEAL_HTML = `${reveal} <span class="small">${claim}</span>`;
    d3.select('.u-reveal').html(REVEAL_WAIT);
    d3.select('.data-links').html(`This record as data: <a href="data/records/${r.id}.json">${r.id}.json</a>.
      The code and the results of all records are at
      <a href="https://github.com/tingyuansen/Babylonian_Dating">github.com/tingyuansen/Babylonian_Dating</a>.`);
  }

  // Where the tablet is now. Only the museum number tells this, so the card gives no other fact about the object.
  // Only the search of the British Museum is linked: it finds a tablet by its number.
  const MUSEUMS = {
    BM: { name: 'The British Museum', city: 'London', img: 'bm',
      search: (no) => `https://www.britishmuseum.org/collection/search?keyword=${encodeURIComponent(no.startsWith('Rm')
        ? no.replace(/^Rm\.?\s*/, 'Rm.') : no.replace(/^BM\s*/, ''))}` },
    VAT: { name: 'The Vorderasiatisches Museum', city: 'Berlin', img: 'berlin' },
    MNB: { name: 'The Louvre', city: 'Paris', img: 'louvre' },
    CBS: { name: 'The Penn Museum', city: 'Philadelphia', img: 'penn' },
  };
  // The edition spaces museum numbers unevenly ("Rm.845", "BM32332"); the page prints each series one way.
  // Rm and BM stay apart: they are two series of numbers of the British Museum.
  function museumLabel(no) {
    return String(no || '').trim().replace(/\b(Rm|W)\.\s*/g, '$1 ')
      .replace(/\b(BM|VAT|MNB|CBS)\s*(?=\d)/g, '$1 ').replace(/\s*\+\s*/g, ' + ');
  }
  function writeMuseum(r) {
    const box = d3.select('.museum').html('');
    const no = (r.museum_no || '').trim();
    const m = /^(BM|Rm|VAT|MNB|CBS|W)\b\.?/.exec(no);
    if (!m) return;
    const pieces = no.split('+').length;
    const join = pieces > 1 ? ` The tablet is a join of ${pieces} pieces.` : '';
    const label = museumLabel(no);
    if (m[1] === 'W') {
      box.html(`<p class="where">Uruk (Warka), Iraq</p><p class="facts">${esc(label)} is an excavation number from
        Uruk.${join}</p>`);
      return;
    }
    const mus = MUSEUMS[m[1] === 'Rm' ? 'BM' : m[1]];
    const first = no.split('+')[0].trim();
    const inBM = META.records.filter((x) => /^(BM|Rm)\b/.test(x.museum_no || '')).length;
    box.html(`<img src="img/museums/${mus.img}.webp" alt="" loading="lazy">
      <p class="where">${mus.name}, ${mus.city}</p>
      <p class="facts">Museum number ${esc(label)}.${/\bRm\b/.test(label) ? ' Rm numbers and BM numbers are two series of numbers of the British Museum.' : ''}${join}${mus.img === 'bm' ? ` Of the ${META.records.length} records on this
        page, ${inBM} are in the British Museum.` : ''}${mus.search ? ` <a href="${mus.search(first)}" target="_blank"
        rel="noopener">Find it in the collection</a>` : ''}</p>`);
  }

  function clockSentence(d) {
    const c = d.c;
    const dial = dialOf(c);
    const img = `<img class="medal" src="${medal(dial.medal)}" alt="">`;
    const where = dial.wedge ? (dial.rim === 'elongation'
      ? ' The gold wedge shows the angle opposite the Sun.'
      : ` The gold wedge shows ${esc(dial.wedge[2])}.`) : '';
    const offset = c.year_offset ? ' The clause comes after a change of year, so the tools read it in the next year.' : '';
    const joint = c.joint && c.joint.length
      ? ` The text also puts ${c.joint.map((b) => BODY[b] || b).join(' and ')} in that sign at the same time. Thus some
        years with ${BODY[c.body] || c.body} in ${esc(c.sign)} still contradict.` : '';
    const arc = c.month && dial.rim !== 'cycle' ? ' Each arc joins the position at the start of the month with the position at its end, so it does not show a turn back inside the month.' : '';
    return `${img}“${esc(passage(c, 80))}”. Each arc is one candidate year. The arc shows ${dial.what}.${arc}${where}${offset}${joint}
      The clause agrees with ${plural(d.agree, 'year', 'years')}, in the color of ${esc(d.clock.name)}.`;
  }
  function writeClockSteps() {
    const box = d3.select('#s-clock .steps').html('');
    if (!CLOCKS.length) {
      box.append('div').attr('class', 'step').html('<p>This record has no decisive clause, so it has no clock. Its other clauses can only put the years in order.</p>');
    } else {
      const first = CLOCKS[0];
      box.append('div').attr('class', 'step').html(`<p>${clockSentence(first)}</p>
        <p class="small">The years go outward, from 626 BCE near the center to AD 75 at the edge. Move the pointer over the
        dial to read a year.</p>`);
      const yrs = d3.range(N).filter((i) => first.row[i] === 'a');
      const gaps = d3.rollups(yrs.slice(1).map((y, k) => y - yrs[k]), (v) => v.length, (g) => g).sort((a, b) => b[1] - a[1]);
      const gap = yrs.length >= 3 && gaps.length ? gaps[0][0] : null;
      box.append('div').attr('class', 'step').html(`<p>Now the dial unwinds. Each year moves to its position on a line of
        years. The years that agree become a row of marks.${gap ? ` Most often, two years that agree are ${plural(gap, 'year', 'years')} apart.` : ''}</p>`);
      CLOCKS.slice(1, MAX_DIALS).forEach((d) => {
        box.append('div').attr('class', 'step').html(`<p>${clockSentence(d)}</p>
          <p class="small">Only <span class="k k-ink">${plural(d.crossCount, 'year agrees', 'years agree')}</span> with all the clauses so far.</p>`);
      });
      if (CLOCKS.length > MAX_DIALS) {
        const last = CLOCKS[CLOCKS.length - 1];
        box.append('div').attr('class', 'step').html(`<p>The other ${CLOCKS.length - MAX_DIALS} decisive clauses unwind in
          the same way. With all of them, <span class="k k-ink">${plural(last.crossCount, 'year agrees', 'years agree')}</span>
          with all the clauses.</p>`);
      }
      if (CLOCKS[CLOCKS.length - 1].crossCount === 0) {
        box.append('div').attr('class', 'step').html(`<p>No year agrees with all the clauses. A broken or misread clause can
          cause this. Thus the rules do not require that all clauses agree. They add the weights of the clauses.</p>`);
      }
    }
    const s = sections.find((x) => x.id === 's-clock');
    if (s) s.steps = Array.from(s.el.querySelectorAll('.step'));
  }

  // The part of the language model: it checks the parser clauses and searches for clauses that the parser misses.
  // Code checks each of its citations. The date uses the parser clauses as the model leaves them, and the clauses that
  // it finds.
  function writeModel(r) {
    const rv = r.review;
    const found = r.found || [];
    const changes = (rv && rv.changes) || [];
    const acc = changes.filter((x) => x.accepted);
    const rej = changes.filter((x) => !x.accepted);
    const agent = r.clauses.filter((c) => c.agent_rule).length;
    const n = r.clauses.length;
    const np = r.clauses.filter((c) => !c.found).length;
    let t = n ? `A rule-based parser reads ${np === n ? (n === 1 ? 'this clause' : 'these clauses') : `${np} of these clauses`}. ` : '';
    t += `A language model then reads the same text. It checks each clause of the parser, and it searches for clauses
      that the parser misses. It must cite the words that support each change and each new clause, and code checks each
      citation.`;
    if (rv) {
      const n = rv.kept + rv.modified + rv.dropped;
      t += rv.modified === 0 && rv.dropped === 0
        ? ` Here it keeps ${n === 1 ? 'the clause' : `all ${n} clauses`} as the parser reads ${n === 1 ? 'it' : 'them'}.`
        : ` Here it keeps ${plural(rv.kept, 'clause', 'clauses')} and changes ${plural(rv.modified, 'clause', 'clauses')}.`;
      const order = ['day', 'month', 'restored'];
      const pick = acc.slice().sort((a, b) => (order.indexOf(a.field) + 1 || 9) - (order.indexOf(b.field) + 1 || 9))[0];
      if (pick) t += ` For example, it ${changeNote(pick).replace(/^The model /, '').replace(/\s*The checks accept the change\.\s*$/, '')}`;
      if (rej.length) t += ` The checks reject ${plural(rej.length, 'other proposed change', 'other proposed changes')}, because the sentence of the clause does not state ${rej.length === 1 ? 'it' : 'them'}.`;
    }
    t += found.length
      ? ` It finds ${plural(found.length, 'clause', 'clauses')} that the parser misses. A pencil line marks ${found.length === 1 ? 'it' : 'them'}.`
      : ' It finds no clause that the parser misses.';
    if (changes.length) t += ' A dashed pencil line marks a changed clause.';
    d3.select('.t-model').html(t);
    d3.select('.t-model-2').html(`The date on this page uses all these clauses.${agent
      ? ` A coding agent also proposed some rules of the parser, and the authors wrote and tested them. Here these rules
        read ${plural(agent, 'clause', 'clauses')}.` : ''}`);
  }

  // A change that the language model proposed for a parser clause, and what the checks did with it.
  function changeNote(x) {
    const v = (y) => (y == null || y === '' ? 'none' : esc(String(y)));
    const what = x.field === 'restored' ? 'marks the clause as resting on words that Sachs and Hunger restored'
      : x.field === 'day' ? `sets the day to ${v(x.to)}`
      : x.field === 'month' ? `sets the month to ${v(x.to)}`
      : x.field === 'event_subtype' ? `sets the kind of event to “${v(x.to)}”`
      : `sets ${esc(String(x.field).replace(/_/g, ' '))} from ${v(x.from)} to ${v(x.to)}`;
    const field = x.field === 'event_subtype' ? 'kind of event' : esc(String(x.field).replace(/_/g, ' '));
    if (!x.accepted && x.to === undefined) {
      return `The model proposes a new ${field}${x.quote ? `, from “${esc(String(x.quote).replace(/\s+/g, ' ').slice(0, 90))}”` : ''}.
        The checks reject the change.`;
    }
    const quote = String(x.quote || '').replace(/\s+/g, ' ');
    return `The model ${what}, from “${esc(quote.slice(0, 90))}${quote.length > 90 ? '…' : ''}”.
      ${x.accepted ? 'The checks accept the change.' : 'The checks reject the change.'}`;
  }

  // ------------------------------------------------------------------ chapter 1: tablet to parchment
  // One line of the tablet in cuneiform. Whole signs are pressed into the clay; damaged signs are faint; where
  // the clay is lost, a dark hollow holds the signs that Sachs and Hunger restored. A reading with no sign in the
  // ORACC Sign List stays in Latin letters.
  function glyphHTML(tokens) {
    let html = '';
    let gap = '';
    let space = false;
    const flush = () => { if (gap) html += `<span class="gap">${gap}</span>`; gap = ''; };
    (tokens || []).forEach((t) => {
      const k = t[0];
      const v = esc(t.slice(1));
      if (k === 's') { space = true; return; }
      const lost = k === 'b' || k === 'g';
      if (lost) {
        if (space && gap) gap += ' ';
        else if (space) html += ' ';
        gap += k === 'b' ? `<span class="rs">${v}</span>` : '<span class="lost"></span>';
      } else {
        flush();
        if (space) html += ' ';
        if (k === 'o') html += v;
        else if (k === 'd') html += `<span class="dm">${v}</span>`;
        else if (k === 'u') html += '<span class="ux"></span>';
        else if (k === 'w') html += '<span class="lost"></span>';
        else if (k === 'l') html += `<span class="lt">${v}</span>`;
      }
      space = false;
    });
    flush();
    return html;
  }
  function translitHTML(t) {
    let s = esc(t).replace(/\s*-\s*/g, '-').replace(/\s+\.\s+/g, '.').replace(/\s+/g, ' ');
    s = s.replace(/\[[^\]]*\]?|\[|\]/g, (m) => `<span class="gap">${m}</span>`);
    s = s.replace(/⸢([^⸣]*)⸣/g, (m, inner) => `<span class="dmg">${inner}</span>`);
    return s;
  }
  function secText(fig) {
    let slab = null;
    let lines = null;
    function render() {
      fig.selectAll('*').remove();
      const hasTablet = !!REC.tablet;
      slab = fig.append('div').attr('class', `slab${hasTablet ? '' : ' parch no-tablet'}`);
      slab.append('div').attr('class', 'slab-head')
        .html(`<span><b>${esc(REC.id)}</b>${REC.museum_no ? ` · ${esc(museumLabel(REC.museum_no))}` : ''}</span><span class="mode">${hasTablet ? 'the signs, as the edition reads them' : 'the English translation'}</span><button type="button" class="read-all">Full text</button>`);
      slab.select('.read-all').on('click', textPanel);
      lines = slab.append('div').attr('class', 'slab-body').append('div').attr('class', 'slab-lines');
      const text = REC.text || '';
      const owner = new Int16Array(text.length).fill(-1);
      const hid = new Uint8Array(text.length);
      (REC.hidden || []).forEach(([a, b]) => hid.fill(1, a, b));
      // The words that Sachs and Hunger restored, inside square brackets, over the whole text.
      const rst = new Uint8Array(text.length);
      for (let k = 0, open = false; k < text.length; k += 1) {
        if (text[k] === '[') open = true;
        if (open) rst[k] = 1;
        if (text[k] === ']') open = false;
      }
      const fnd = new Int16Array(text.length).fill(-1);
      (REC.found || []).forEach((f, q) => { for (let k = f.start; k < f.end && k < text.length; k += 1) if (fnd[k] < 0) fnd[k] = q; });
      const changed = new Set(((REC.review || {}).changes || []).map((x) => x.j));
      REC.clauses.map((c, j) => [c, j]).sort((a, b) => b[0].decisive - a[0].decisive)
        .forEach(([c, j]) => { for (let k = c.start; k < c.end && k < text.length; k += 1) if (owner[k] < 0) owner[k] = j; });
      const textLines = text.split('\n');
      const starts = [];
      let pos = 0;
      textLines.forEach((line) => { starts.push(pos); pos += line.length + 1; });
      const lineHTML = (li) => {
        const line = textLines[li];
        const start = starts[li];
        let html = '';
        let dec = false;
        let model = false;
        let k = 0;
        while (k < line.length) {
          const o = owner[start + k];
          const hd = hid[start + k];
          const f = fnd[start + k];
          const rs = rst[start + k];
          let e = k;
          while (e < line.length && owner[start + e] === o && hid[start + e] === hd && fnd[start + e] === f
            && rst[start + e] === rs) e += 1;
          let seg = esc(line.slice(k, e));
          if (rs) seg = `<span class="restored">${seg}</span>`;
          if (hd) seg = `<span class="chron" aria-hidden="true" title="Shown at the last step">${seg}</span>`;
          if (o >= 0) {
            const c = REC.clauses[o];
            dec = dec || c.decisive;
            seg = `<span class="clause ${c.decisive ? 'dec' : 'sup'}${changed.has(o) ? ' rev' : ''}" data-j="${o}" style="--kc:${clockOf(c).color}">${seg}</span>`;
            model = model || changed.has(o);
          }
          if (f >= 0) { seg = `<span class="found" data-q="${f}">${seg}</span>`; model = true; }
          html += seg;
          k = e;
        }
        return { html: html || '&nbsp;', dec, model };
      };
      const rows = hasTablet ? REC.tablet : textLines.map((_, li) => ({ label: String(li + 1), translit: '', line: li }));
      // A line that names the year or the king keeps its signs and its Latin letters hidden until the reveal.
      const dated = (row) => (row.line != null && hid.subarray(starts[row.line], starts[row.line] + textLines[row.line].length).some((v) => v))
        || /\bMU\b|LUGAL|KAM₂?\b/.test(row.translit || '');
      rows.forEach((row, n) => {
        const tr = row.line != null ? lineHTML(row.line) : { html: '', dec: false };
        lines.append('div').attr('class', `tl-row${tr.dec ? ' has-dec' : ''}${tr.model ? ' has-model' : ''}`)
          .html(`<span class="n">${esc(row.label)}</span><span class="tl-cell" style="--d:${Math.min(n * 30, 1400)}ms">
            <span class="tl${dated(row) ? ' dated' : ''}" aria-hidden="true" data-n="${n}">${row.signs ? glyphHTML(row.signs) : translitHTML(row.translit)}</span><span class="tr">${tr.html}</span></span>`);
      });
      lines.selectAll('.tl').on('mousemove', (ev) => {
        const row = rows[+ev.currentTarget.dataset.n];
        tipShow(ev, `<b>Line ${esc(row.label)}</b><br>${dated(row) && !document.body.classList.contains('revealed')
          ? '<span class="small">This line names the year or the king. The page shows it at the end.</span>'
          : `<span class="translit">${translitHTML(row.translit)}</span>`}`);
      }).on('mouseleave', tipHide);
      lines.selectAll('.clause').on('mousemove', (ev) => {
        if (ev.target.closest('.found') && ev.target.closest('.found') !== ev.currentTarget) return;
        const j = +ev.currentTarget.dataset.j;
        const c = REC.clauses[j];
        const notes = ((REC.review || {}).changes || []).filter((x) => x.j === j).map(changeNote);
        tipShow(ev, `<b>${c.decisive ? 'Decisive' : (c.idle ? 'A decisive type that narrows no year' : 'Supporting')}</b><br>${esc(fields(c))}${c.agent_rule
          ? '<br><span class="small">A coding agent proposed the parser rule that reads this clause. The authors wrote and tested it.</span>' : ''}${c.restored
          ? '<br><span class="small">It counts as restored, so it can only support.</span>' : ''}${notes.length
          ? `<br><span class="pencil">${notes.join('<br>')}</span>` : ''}`);
      }).on('mouseleave', tipHide);
      lines.selectAll('.restored').on('mousemove', (ev) => {
        if (ev.target.closest('.clause, .found')) return;
        tipShow(ev, '<b>Restored</b><br><span class="small">Sachs and Hunger restored these words where the tablet is broken.</span>');
      }).on('mouseleave', tipHide);
      lines.selectAll('.found').on('mousemove', (ev) => {
        ev.stopPropagation();
        const f = REC.found[+ev.currentTarget.dataset.q];
        const c = REC.clauses[f.j] || {};
        tipShow(ev, `<b class="pencil">Found by the language model</b><br>${esc(f.what)}<br>“${esc(f.quote)}”<br><span class="small">${f.restored
          ? 'It rests on words that Sachs and Hunger restored or query, so it can only support.' : (c.decisive ? 'Decisive.' : (c.idle ? 'A decisive type that narrows no year.' : 'Supporting.'))} The date uses it like the other clauses.</span>`);
      }).on('mouseleave', tipHide);
    }
    function update(i) {
      if (!slab) return;
      const hasTablet = !!REC.tablet;
      slab.classed('parch', !hasTablet || i >= 1).classed('inked', i >= 2).classed('focus', i === 2)
        .classed('model', i >= 3).classed('focus-model', i >= 3);
      slab.select('.mode').text(!hasTablet || i >= 1 ? 'the English translation, which the system reads' : 'the signs, as the edition reads them');
      const first = i >= 3 ? (lines.select('.tl-row.has-model').node() || lines.select('.clause.dec').node())
        : (i >= 2 ? lines.select('.clause.dec').node() : null);
      const box = slab.select('.slab-body').node().clientHeight;
      const row = first ? first.closest('.tl-row') : null;
      const shift = row ? Math.max(0, row.offsetTop - box * 0.3) : 0;
      lines.style('transform', `translateY(${-shift}px)`);
    }
    return { render, update };
  }

  // Shorten an SVG label, with an ellipsis, until it fits its width in pixels.
  function fitText(node, max) {
    let t = node.textContent;
    while (node.getComputedTextLength() > max && t.length > 8) {
      t = t.slice(0, -2).trimEnd();
      node.textContent = `${t}…`;
    }
  }

  // ------------------------------------------------------------------ chapter 2: the clocks
  function secClock(fig) {
    let st = null;
    function geometry(w, h) {
      const narrowScreen = w < 600;
      const rows = Math.min(CLOCKS.length, narrowScreen ? 5 : 10);
      const rh = narrowScreen ? 26 : Math.max(20, Math.min(30, (h * 0.32) / Math.max(rows, 1)));
      const rowsH = rh * rows + 40;
      const med = narrowScreen ? 34 : Math.max(44, Math.min(76, w * 0.085));
      const room = h - rowsH - 36;
      const R = Math.max(70, Math.min(room / 2 - med - 6, w / 2 - med - 14));
      const cx = w / 2;
      const cy = 8 + med + 6 + R;
      return { narrowScreen, rows, rh, med, R, cx, cy, r0: R * 0.2,
        rowsTop: cy + R + med + (narrowScreen ? 30 : 26), left: narrowScreen ? 26 : Math.min(210, w * 0.25),
        right: narrowScreen ? 64 : 104 };
    }
    const toCanvas = (deg) => ((deg - 90) * Math.PI) / 180;
    function radius(i) { return st.g.r0 + (st.g.R - st.g.r0) * (i / (N - 1)); }
    function polar(deg, r) { const t = toCanvas(deg); return [st.g.cx + r * Math.cos(t), st.g.cy + r * Math.sin(t)]; }
    function render() {
      fig.selectAll('*').remove();
      const w = fig.node().clientWidth;
      const h = fig.node().clientHeight;
      if (st) stopTimers();                 // a dial of the last tablet may still sweep
      st = { w, h, dialFor: -1, shownRows: 0, timers: [] };
      if (!CLOCKS.length) {
        fig.append('svg').attr('width', w).attr('height', h).append('text').attr('class', 'note')
          .attr('x', w / 2).attr('y', h / 2).attr('text-anchor', 'middle').text('No decisive clause, so no clock.');
        return;
      }
      const g = geometry(w, h);
      st.g = g;
      st.x = d3.scaleLinear().domain([0, N - 1]).range([g.left, w - g.right]);
      st.base = canvasFor(fig, w, h, 'base');
      st.fly = canvasFor(fig, w, h, 'fly');
      const svg = fig.append('svg').attr('width', w).attr('height', h);
      st.svg = svg;
      st.under = svg.append('g');            // ink columns behind the rows
      st.dial = svg.append('g');
      st.rows = svg.append('g');
      st.hover = svg.append('g').style('pointer-events', 'none');
      // The ledger: axis and count headings.
      const axisY = g.rowsTop + g.rh * g.rows - g.rh / 2 + 10;
      st.rows.append('g').attr('class', 'axis').attr('transform', `translate(0,${axisY})`)
        .call(d3.axisBottom(d3.scaleLinear().domain([-625, 75]).range([g.left, w - g.right]))
          .tickValues(g.narrowScreen ? [-600, -400, -200, 0] : [-600, -500, -400, -300, -200, -100, 0]).tickFormat(yearLabel).tickSizeOuter(0));
      const colA = w - g.right + (g.narrowScreen ? 28 : 46);
      const headY = g.rowsTop - g.rh / 2 - (g.narrowScreen ? 10 : 6);
      st.rows.append('text').attr('class', 'note').attr('x', colA).attr('y', headY).attr('text-anchor', 'end').text('agree');
      st.rows.append('text').attr('class', 'note').attr('x', w - 4).attr('y', headY).attr('text-anchor', 'end').text('all');
      st.colA = colA;
      st.rowG = CLOCKS.map(() => st.rows.append('g').attr('opacity', 0));
      CLOCKS.forEach((d, k) => { if (k < g.rows) drawRow(k); });
      if (CLOCKS.length > g.rows) {
        st.more = st.rows.append('text').attr('class', 'note').attr('x', g.left).attr('y', axisY + 34).attr('opacity', 0)
          .text(`+${plural(CLOCKS.length - g.rows, 'more decisive clause', 'more decisive clauses')}, in the counts only`);
      }
      st.ctl = fig.append('div').attr('class', 'fig-controls');
      st.ctl.append('button').attr('type', 'button').text('↻ Replay').on('click', () => {
        if (st.dialFor < 0) return;
        const s = stateOf(st.cur || 0);
        sweep(s.dial, () => showRows(s.rows, true));
      });
      // Hover anywhere on the dial or the rows.
      svg.on('mousemove', (ev) => hover(d3.pointer(ev))).on('mouseleave', () => { st.hover.selectAll('*').remove(); tipHide(); });
    }
    function rowY(k) { return st.g.rowsTop + k * st.g.rh; }
    function drawRow(k) {
      const d = CLOCKS[k];
      const g = st.g;
      const row = st.rowG[k];
      const y = rowY(k);
      const tw = Math.max(1.8, st.x(1) - st.x(0));
      const th = g.rh * (g.narrowScreen ? 0.45 : 0.6);   // on a phone the label sits above the marks
      row.append('line').attr('x1', g.left).attr('x2', st.w - g.right).attr('y1', y + g.rh / 2).attr('y2', y + g.rh / 2)
        .attr('stroke', C.rule).attr('stroke-width', 0.8);
      const agree = d3.range(N).filter((i) => d.row[i] === 'a');
      row.selectAll('rect.a').data(agree).join('rect').attr('class', 'a').attr('rx', 0.8)
        .attr('x', (i) => st.x(i) - tw / 2).attr('y', y - th / 2).attr('width', tw).attr('height', th).attr('fill', d.clock.color);
      const iconSize = Math.min(18, g.rh - 4);
      if (g.narrowScreen) {
        row.append('image').attr('href', medal(d.clock.medal)).attr('x', g.left).attr('y', y - th / 2 - iconSize - 1)
          .attr('width', iconSize * 0.8).attr('height', iconSize * 0.8);
        const label = row.append('text').attr('class', 'label').attr('x', g.left + iconSize).attr('y', y - th / 2 - 4)
          .attr('font-size', 11).text(passage(d.c, 60));
        fitText(label.node(), st.colA - 40 - g.left - iconSize);
      } else {
        row.append('image').attr('href', medal(d.clock.medal)).attr('x', g.left - iconSize - 6).attr('y', y - iconSize / 2)
          .attr('width', iconSize).attr('height', iconSize);
        row.append('text').attr('class', 'label').attr('x', g.left - iconSize - 12).attr('y', y + 5).attr('text-anchor', 'end')
          .text(passage(d.c, Math.round((g.left - 30) / 7.4)));
      }
      row.append('text').attr('class', 'count').attr('x', st.colA).attr('y', y + 5).attr('text-anchor', 'end')
        .attr('fill', C['ink-2']).text(d.agree);
      row.append('text').attr('class', 'count').attr('x', st.w - 4).attr('y', y + 5).attr('text-anchor', 'end')
        .attr('fill', C.cross).text(d.crossCount);
    }
    // The ink columns: the years that every row shown so far agrees with.
    function drawColumns(rows) {
      st.under.selectAll('*').remove();
      if (rows < 2) return;
      const d = CLOCKS[rows - 1];
      const g = st.g;
      const tw = Math.max(2.4, st.x(1) - st.x(0) + 1);
      st.under.selectAll('rect').data(d3.range(N).filter((i) => d.cross[i])).join('rect')
        .attr('x', (i) => st.x(i) - tw / 2).attr('width', tw)
        .attr('y', g.rowsTop - g.rh / 2).attr('height', g.rh * Math.min(rows, g.rows))
        .attr('fill', C.cross).attr('opacity', 0.22);
    }
    // The dial frame of clause k: rim, wedge, signs and the medallion at the centre.
    function drawFrame(k) {
      const g = st.g;
      const d = CLOCKS[k];
      const dial = dialOf(d.c);
      st.dialFor = k;
      st.dialSpec = dial;
      const D = st.dial.html('');
      D.append('circle').attr('cx', g.cx).attr('cy', g.cy).attr('r', g.R + 4).attr('fill', 'none').attr('stroke', C.rule).attr('stroke-width', 1.2);
      D.append('circle').attr('cx', g.cx).attr('cy', g.cy).attr('r', g.R + 8).attr('fill', 'none').attr('stroke', C.rule).attr('stroke-width', 0.6);
      if (dial.wedge) {
        const [a0, a1] = dial.wedge;
        D.append('path').attr('transform', `translate(${g.cx},${g.cy})`)
          .attr('d', d3.arc()({ innerRadius: g.r0 * 0.85, outerRadius: g.R + 4, startAngle: (a0 * Math.PI) / 180, endAngle: (a1 * Math.PI) / 180 }))
          .attr('fill', C['gold-leaf']).attr('stroke', C.gold).attr('stroke-width', 1.2);
      }
      if (dial.rim === 'signs') {
        SKY.signs.forEach((s, i) => {
          const [x1, y1] = polar(i * 30, g.r0 * 0.85);
          const [x2, y2] = polar(i * 30, g.R + 4);
          D.append('line').attr('x1', x1).attr('y1', y1).attr('x2', x2).attr('y2', y2).attr('stroke', C.rule).attr('stroke-width', 0.7);
          const lit = dial.wedge && dial.wedge[2] === s;
          const size = lit ? g.med * 1.3 : g.med;
          const [mx, my] = polar(i * 30 + 15, g.R + 8 + g.med / 2);
          if (lit) D.append('circle').attr('cx', mx).attr('cy', my).attr('r', size * 0.56).attr('fill', C['gold-leaf']);
          D.append('image').attr('href', medal(s.toLowerCase())).attr('x', mx - size / 2).attr('y', my - size / 2)
            .attr('width', size).attr('height', size).attr('opacity', lit || !dial.wedge ? 1 : 0.75)
            .append('title').text(s);
        });
      } else if (dial.rim === 'elongation') {
        const [sx, sy] = polar(0, g.R + 10 + g.med / 2);
        D.append('image').attr('href', medal('sun')).attr('x', sx - g.med / 2).attr('y', sy - g.med / 2).attr('width', g.med).attr('height', g.med);
        const [ox, oy] = polar(180, g.R + 22);
        D.append('text').attr('class', 'label').attr('x', ox).attr('y', oy + 5).attr('text-anchor', 'middle').attr('fill', C['gold-ink']).text('opposite the Sun');
      } else {
        const [tx, ty] = polar(0, g.R + 20);
        D.append('text').attr('class', 'note').attr('x', tx).attr('y', ty).attr('text-anchor', 'middle').text('one turn = 18 years, the cycle of eclipses');
      }
      st.center = D.append('g');
      const cm = g.r0 * 1.6;
      st.center.append('image').attr('href', medal(dial.medal)).attr('x', g.cx - cm / 2).attr('y', g.cy - cm / 2).attr('width', cm).attr('height', cm);
      st.hand = D.append('g').attr('opacity', 0);
      st.hand.append('line').attr('stroke', C.gold).attr('stroke-width', 1.6);
      st.hand.append('image').attr('href', medal(dial.medal)).attr('width', g.med * 0.8).attr('height', g.med * 0.8);
      // The count sits a full line below the year, so the two never touch on a small dial.
      const fs = Math.max(18, g.R * 0.13);
      st.yearText = D.append('text').attr('class', 'year-big halo').attr('x', g.cx).attr('y', g.cy - 2)
        .attr('text-anchor', 'middle').attr('font-size', fs).attr('opacity', 0);
      st.countText = D.append('text').attr('class', 'note halo').attr('x', g.cx).attr('y', g.cy + fs * 0.95)
        .attr('text-anchor', 'middle').attr('opacity', 0);
    }
    // Draw the arcs of years [from, to) on the base canvas.
    function drawYears(from, to) {
      const ctx = st.base.ctx;
      const d = CLOCKS[st.dialFor];
      const spec = st.dialSpec;
      const order = (i) => d.row[i] === 'a';
      for (let pass = 0; pass < 2; pass += 1) {
        for (let i = from; i < to; i += 1) {
          if (order(i) !== (pass === 1)) continue;
          const segs = spec.segs[i];
          if (!segs) continue;
          const v = d.row[i];
          ctx.strokeStyle = v === 'a' ? d.clock.color : C.ink;
          ctx.globalAlpha = v === 'a' ? 0.95 : v === 'c' ? 0.16 : 0.05;
          ctx.lineWidth = v === 'a' ? 3 : 1;
          const r = radius(i);
          segs.forEach(([a0, a1]) => {
            const dd = ((a1 - a0 + 540) % 360) - 180;
            const s = Math.min(a0, a0 + dd);
            const e = Math.max(a0, a0 + dd) + 0.6;
            ctx.beginPath();
            ctx.arc(st.g.cx, st.g.cy, r, toCanvas(s), toCanvas(e));
            ctx.stroke();
          });
        }
      }
      ctx.globalAlpha = 1;
    }
    // Clear the dial, then draw the named sign large inside its wedge, faint, under the arcs.
    function clearBase() {
      const ctx = st.base.ctx;
      ctx.clearRect(0, 0, st.w, st.h);
      const spec = st.dialSpec;
      if (!spec || !spec.wedge || spec.rim !== 'signs') return;
      const img = art(spec.wedge[2].toLowerCase());
      const draw = () => {
        const g = st.g;
        const size = (g.R - g.r0) * 0.86;
        const [x, y] = polar(spec.wedge[0] + 15, g.r0 + (g.R - g.r0) * 0.56);
        ctx.save();
        ctx.globalAlpha = 0.34;
        ctx.drawImage(img, x - size / 2, y - size / 2, size, size);
        ctx.restore();
      };
      if (img.complete && img.naturalWidth) draw(); else img.onload = draw;
    }
    function stopTimers() { st.timers.forEach((t) => t.stop()); st.timers = []; }
    // The hand sweeps the years outward and leaves each year's arc behind it.
    function sweep(k, then) {
      stopTimers();
      drawFrame(k);
      clearBase();
      st.fly.ctx.clearRect(0, 0, st.w, st.h);
      if (still()) {
        drawYears(0, N);
        if (then) then();
        return;
      }
      const d = CLOCKS[k];
      const T = 2800;
      let drawn = 0;
      let agreed = 0;
      st.hand.attr('opacity', 1);
      st.yearText.attr('opacity', 1);
      st.countText.attr('opacity', 1);
      st.center.attr('opacity', 0.25);
      const timer = d3.timer((elapsed) => {
        const p = Math.min(1, elapsed / T);
        const upto = Math.round(d3.easeCubicInOut(p) * (N - 1)) + 1;
        if (upto > drawn) {
          drawYears(drawn, upto);
          for (let i = drawn; i < upto; i += 1) agreed += d.row[i] === 'a' ? 1 : 0;
          drawn = upto;
        }
        const i = drawn - 1;
        const segs = st.dialSpec.segs[i];
        const end = segs ? segs[segs.length - 1][1] : 0;
        const [hx, hy] = polar(end, radius(i));
        st.hand.select('line').attr('x1', st.g.cx).attr('y1', st.g.cy).attr('x2', hx).attr('y2', hy);
        const ms = st.g.med * 0.8;
        st.hand.select('image').attr('x', hx - ms / 2).attr('y', hy - ms / 2);
        st.yearText.text(yearLabel(META.years[0] + i));
        st.countText.text(`${fmt(agreed)} agreeing years`).attr('fill', C['ink-2']);
        if (p >= 1) {
          timer.stop();
          st.hand.transition().duration(500).attr('opacity', 0);
          st.yearText.transition().duration(500).attr('opacity', 0);
          st.countText.transition().duration(500).attr('opacity', 0);
          st.center.transition().duration(500).attr('opacity', 1);
          if (then) st.timers.push(d3.timeout(then, 450));
        }
      });
      st.timers.push(timer);
    }
    // Unwind: each year flies from its arc to its place on the line of row k.
    function unwind(k, then) {
      const d = CLOCKS[k];
      if (still() || st.dialFor !== k || k >= st.g.rows) { showRows(k + 1); if (then) then(); return; }
      const ctx = st.fly.ctx;
      const spec = st.dialSpec;
      const pts = d3.range(N).filter((i) => spec.segs[i]).map((i) => {
        const segs = spec.segs[i];
        const mid = (segs[0][0] + segs[segs.length - 1][1]) / 2;
        const [x0, y0] = polar(mid, radius(i));
        return { i, x0, y0, x1: st.x(i), y1: rowY(k), a: d.row[i] === 'a' };
      });
      const T = 1500;
      const timer = d3.timer((elapsed) => {
        ctx.clearRect(0, 0, st.w, st.h);
        let done = true;
        pts.forEach((p) => {
          const t = Math.max(0, Math.min(1, (elapsed - p.i * 1.1) / 700));
          if (t < 1) done = false;
          const e = d3.easeCubicInOut(t);
          if (!p.a && t > 0.5) return;
          ctx.globalAlpha = p.a ? 1 : 0.25 * (1 - t * 2);
          ctx.fillStyle = p.a ? d.clock.color : C.ink;
          ctx.beginPath();
          ctx.arc(p.x0 + (p.x1 - p.x0) * e, p.y0 + (p.y1 - p.y0) * e, p.a ? 2.4 : 1.4, 0, 6.283);
          ctx.fill();
        });
        ctx.globalAlpha = 1;
        if (done || elapsed > T + N * 1.1) {
          timer.stop();
          showRows(k + 1, true);
          st.timers.push(d3.timeout(() => ctx.clearRect(0, 0, st.w, st.h), 300));
          if (then) then();
        }
      });
      st.timers.push(timer);
    }
    function showRows(n, fade) {
      st.rowG.forEach((row, k) => {
        row.interrupt();
        if (k < n && k < st.g.rows) row.transition().duration(fade ? 350 : 0).attr('opacity', 1);
        else row.attr('opacity', 0);
      });
      if (st.more) st.more.attr('opacity', n > st.g.rows ? 1 : 0);
      drawColumns(n);
      st.shownRows = n;
    }
    // Step i: the dial shown, and how many rows are unwound.
    function stateOf(i) {
      const n = CLOCKS.length;
      const shown = Math.min(n, MAX_DIALS);
      if (i <= 0) return { dial: 0, rows: 0 };
      if (i === 1) return { dial: 0, rows: 1 };
      if (i <= shown) return { dial: i - 1, rows: i };
      return { dial: shown - 1, rows: n };
    }
    function update(i, dur) {
      if (!st || !CLOCKS.length) return;
      st.cur = i;
      const s = stateOf(i);
      stopTimers();
      st.fly.ctx.clearRect(0, 0, st.w, st.h);
      const animate = dur > 0;
      if (!animate) {
        if (s.dial !== st.dialFor || true) { drawFrame(s.dial); clearBase(); drawYears(0, N); }
        showRows(s.rows);
        return;
      }
      const newDial = s.dial !== st.dialFor;
      const oneMore = s.rows === st.shownRows + 1;
      showRows(oneMore ? st.shownRows : s.rows);
      if (newDial) {
        sweep(s.dial, oneMore ? () => unwind(s.rows - 1) : null);
      } else if (oneMore) {
        unwind(s.rows - 1);
      }
    }
    function hover([mx, my]) {
      if (!st || !st.g || st.dialFor < 0) return;
      const g = st.g;
      st.hover.selectAll('*').remove();
      let i = null;
      const r = Math.hypot(mx - g.cx, my - g.cy);
      if (r >= g.r0 - 2 && r <= g.R + 2) i = Math.round(((r - g.r0) / (g.R - g.r0)) * (N - 1));
      else if (my > g.rowsTop - g.rh && mx >= g.left && mx <= st.w - g.right) i = Math.round(st.x.invert(mx));
      if (i == null || i < 0 || i >= N) { tipHide(); return; }
      st.hover.append('circle').attr('cx', g.cx).attr('cy', g.cy).attr('r', radius(i)).attr('fill', 'none')
        .attr('stroke', C.ink).attr('stroke-width', 1).attr('stroke-dasharray', '3,3').attr('opacity', 0.6);
      if (st.shownRows) {
        st.hover.append('line').attr('x1', st.x(i)).attr('x2', st.x(i)).attr('y1', g.rowsTop - g.rh / 2)
          .attr('y2', rowY(st.shownRows - 1) + g.rh / 2).attr('stroke', C.ink).attr('stroke-width', 1).attr('opacity', 0.6);
      }
      const d = CLOCKS[st.dialFor];
      const lines = CLOCKS.slice(0, Math.max(st.shownRows, 1)).map((x) =>
        `<span style="color:${x.clock.color}">●</span> ${esc(passage(x.c, 36))}: ${VERDICT[x.row[i]]}`).join('<br>');
      tipShow({ clientX: st.svg.node().getBoundingClientRect().left + mx, clientY: st.svg.node().getBoundingClientRect().top + my },
        `<b>${yearLabel(META.years[0] + i)}</b><br>${st.shownRows ? lines : `${esc(passage(d.c, 40))}: ${VERDICT[d.row[i]]}`}`);
    }
    return { render, update, stop: () => { if (st) stopTimers(); } };
  }

  // The rulers of Babylon, after the tables of Parker and Dubberstein. From 311 BCE the years count in the Seleucid
  // Era, so the band shows the era there and the tooltip its year.
  function reignName(y) {
    const r = (META.reigns || []).find((x) => y >= x.first && y <= x.last);
    if (y >= -310) {
      const se = y + 311;
      return `Seleucid Era year ${se}${se > 64 ? `, Arsacid Era year ${se - 64}` : ''}`;
    }
    return r ? r.name : '';
  }
  function drawReigns(svg, x, y, h) {
    const P = META.periods || [];
    if (!P.length) return;
    const g = svg.append('g').attr('class', 'reigns');
    P.forEach((p, k) => {
      const x0 = x(p.first - 0.5);
      const x1 = x(p.last + 0.5);
      const wd = x1 - x0;
      g.append('rect').attr('x', x0).attr('y', y).attr('width', wd).attr('height', h)
        .attr('fill', k % 2 ? 'rgba(122, 86, 52, 0.16)' : 'rgba(122, 86, 52, 0.07)');
      (META.reigns || []).filter((r) => r.first > p.first && r.first <= p.last).forEach((r) => {
        g.append('line').attr('x1', x(r.first - 0.5)).attr('x2', x(r.first - 0.5)).attr('y1', y + 3).attr('y2', y + h - 3)
          .attr('stroke', C.faint).attr('stroke-width', 0.8);
      });
      const icon = h + 6;
      if (wd > icon * 0.8) {
        g.append('image').attr('href', `img/periods/${p.key}.webp`).attr('x', x0 + Math.max(1, Math.min(4, (wd - icon) / 2)))
          .attr('y', y - 3).attr('width', icon).attr('height', icon);
      }
      const label = wd > 190 ? p.name : p.short;
      if (wd > icon + label.length * 6.4 + 8) {
        g.append('text').attr('class', 'note').attr('x', x0 + icon + 8).attr('y', y + h / 2 + 4.5).attr('font-size', 13)
          .attr('fill', C['ink-2']).text(label);
      }
    });
    g.append('rect').attr('x', x(META.years[0] - 0.5)).attr('y', y).attr('width', x(META.years[1] + 0.5) - x(META.years[0] - 0.5))
      .attr('height', h).attr('fill', 'transparent')
      .on('mousemove', (ev) => {
        const yr = Math.round(x.invert(d3.pointer(ev)[0]));
        const p = P.find((q) => yr >= q.first && yr <= q.last);
        const r = (META.reigns || []).find((q) => yr >= q.first && yr <= q.last);
        tipShow(ev, `<b>${yearLabel(yr)}</b><br>${p ? esc(p.name) : ''}${r && yr < -310 ? `<br>${esc(r.name)}, ${yearLabel(r.first)} to
          ${yearLabel(r.last)}` : `<br>${esc(reignName(yr))}`}`);
      }).on('mouseleave', tipHide);
  }

  // ------------------------------------------------------------------ chapter 3: the clocks add up
  function secSum(fig) {
    let st = null;
    const SEAL = '#9e1f22';
    function render() {
      fig.selectAll('*').remove();
      const w = fig.node().clientWidth;
      const h = fig.node().clientHeight;
      const svg = fig.append('svg').attr('width', w).attr('height', h);
      st = { svg, w, h, step: -1, soaked: -1 };
      if (!CLOCKS.length) {
        svg.append('text').attr('class', 'note').attr('x', w / 2).attr('y', h / 2).attr('text-anchor', 'middle').text('All 701 years stay.');
        return;
      }
      const narrowScreen = w < 600;
      const left = narrowScreen ? 30 : 40;
      const right = 12;
      const rows = Math.min(CLOCKS.length, 10);
      const rh = Math.max(7, Math.min(11, (h * 0.13) / rows));
      const rowsTop = 18;
      const stripTop = rowsTop + rows * rh + 46;
      const stripH = narrowScreen ? 26 : 34;
      const ladderTop = stripTop + stripH + (narrowScreen ? 100 : 108);
      const y0 = META.years[0];
      st.L = { narrowScreen, left, right, rows, rh, rowsTop, stripTop, stripH, ladderTop };
      st.x = d3.scaleLinear().domain([y0 - 0.5, META.years[1] + 0.5]).range([left, w - right]);
      st.maxID = d3.max(REC.ID) || 1;
      st.maxTotal = d3.max(REC.ID.map((v, i) => v + (REC.IS[i] || 0))) || 1;
      st.maxK = d3.max(REC.K) || 0;
      const cw = st.x(y0 + 1) - st.x(y0);
      // The unwound rows of chapter 2.
      CLOCKS.slice(0, rows).forEach((d, k) => {
        const y = rowsTop + k * rh;
        svg.append('g').selectAll('rect').data(d3.range(N).filter((i) => d.row[i] === 'a')).join('rect')
          .attr('x', (i) => st.x(y0 + i) - Math.max(1.4, cw) / 2).attr('width', Math.max(1.4, cw))
          .attr('y', y - rh * 0.34).attr('height', rh * 0.68).attr('fill', d.clock.color);
        const icon = Math.min(14, rh + 4);
        svg.append('image').attr('href', medal(d.clock.medal)).attr('x', left - icon - 6).attr('y', y - icon / 2)
          .attr('width', icon).attr('height', icon).append('title').text(passage(d.c, 80));
      });
      // The strength strip: one cell per year, soaked in ink by its weight.
      st.leaf = svg.append('g');
      st.cells = svg.append('g').selectAll('rect').data(d3.range(N)).join('rect')
        .attr('x', (i) => st.x(y0 + i) - cw / 2).attr('width', cw + 0.4)
        .attr('y', stripTop).attr('height', stripH).attr('fill', C['bg-2'] || C.bg);
      svg.append('rect').attr('x', left).attr('y', stripTop).attr('width', w - left - right).attr('height', stripH)
        .attr('fill', 'none').attr('stroke', C.rule);
      svg.append('g').attr('class', 'axis').attr('transform', `translate(0,${stripTop + stripH})`)
        .call(d3.axisBottom(st.x).tickValues(narrowScreen ? [-600, -400, -200, 0] : [-600, -500, -400, -300, -200, -100, 0])
          .tickFormat(yearLabel).tickSizeOuter(0));
      const bandY = stripTop + stripH + 28;
      const bandH = narrowScreen ? 18 : 22;
      drawReigns(svg, st.x, bandY, bandH);
      svg.append('text').attr('class', 'note').attr('x', w - right).attr('y', bandY + bandH + 20).attr('text-anchor', 'end')
        .text('weight of each year: darker ink, more bits');
      st.dots = svg.append('g').selectAll('circle').data(d3.range(N).filter((i) => st.maxK && REC.K[i] === st.maxK)).join('circle')
        .attr('cx', (i) => st.x(y0 + i)).attr('cy', stripTop - 5).attr('r', 3.4).attr('fill', C.gold).attr('opacity', 0);
      st.pubG = svg.append('g').attr('opacity', 0).attr('display', 'none').attr('aria-hidden', 'true');
      st.hover = svg.append('g').style('pointer-events', 'none');
      st.ladder = svg.append('g').attr('opacity', 0);
      fig.append('div').attr('class', 'fig-controls').append('button').attr('type', 'button').text('Full ranking')
        .on('click', () => rankingPanel());
      st.pubRow = svg.append('g').attr('display', 'none').attr('aria-hidden', 'true');
      buildLadder();
      drawSeal();
      svg.append('rect').attr('x', left).attr('y', rowsTop - 10).attr('width', w - left - right)
        .attr('height', stripTop + stripH - rowsTop + 10).attr('fill', 'transparent').style('cursor', 'pointer')
        .on('mousemove', (ev) => {
          const [mx] = d3.pointer(ev);
          const yr = Math.round(st.x.invert(mx));
          const i = yr - y0;
          if (i < 0 || i >= N) return;
          st.hover.html('').append('line').attr('x1', st.x(yr)).attr('x2', st.x(yr)).attr('y1', rowsTop - 10)
            .attr('y2', stripTop + stripH).attr('stroke', C.ink).attr('stroke-width', 1).attr('opacity', 0.55);
          const agreeing = CLOCKS.filter((d) => d.row[i] === 'a').map((d) => `<span style="color:${d.clock.color}">●</span> ${esc(passage(d.c, 34))}`);
          tipShow(ev, `<b>${yearLabel(yr)}</b> · ${esc(reignName(yr))} · ${plural(REC.K[i], 'clock', 'clocks')} · ${REC.ID[i].toFixed(1)} bits${REC.strict.includes(yr) ? ' · <b>stays</b>' : ''}<br>${agreeing.join('<br>') || 'no decisive clause agrees'}<br><span style="opacity:.7">Click to see every clause</span>`);
        })
        .on('mouseleave', () => { st.hover.html(''); tipHide(); })
        .on('click', (ev) => provenance(Math.round(st.x.invert(d3.pointer(ev)[0]))));
    }
    // The ink of a year after the first k+1 clauses have added their weight.
    function soak(i, k) {
      let bits = 0;
      for (let q = 0; q <= k && q < CLOCKS.length; q += 1) if (CLOCKS[q].row[i] === 'a') bits += CLOCKS[q].c.bits;
      const t = Math.pow(bits / st.maxID, 0.75);
      return bits ? d3.interpolateRgb(C['bg-2'] || C.bg, C.ink)(0.12 + 0.88 * t) : (C['bg-2'] || C.bg);
    }
    // The ladder: the first three places, each a numeral, a line that says what sets it apart,
    // and a chip per year. Under each year runs a thin line of its weight, split by clock.
    function buildLadder() {
      const { L } = st;
      const g = st.ladder;
      const y0 = META.years[0];
      const chipW = L.narrowScreen ? 74 : 92;
      const chipH = L.narrowScreen ? 30 : 34;
      const maxLines = L.narrowScreen ? 1 : 2;     // a phone shows one line of each place, and the panel holds the rest
      const gap = 8;
      const indent = L.narrowScreen ? 0 : 54;
      const span = st.w - L.left - L.right - indent;
      const perLine = Math.max(2, Math.floor((span + gap) / (chipW + gap)));
      const pub = REC.published_astro;
      const groups = REC.groups.slice(0, 3).filter((x) => x && x.length);
      const rowsData = groups.map((years, gi) => ({ years: years.slice().sort(d3.ascending), gi }));
      const lines = (n) => Math.min(maxLines, Math.ceil(n / perLine));
      const shown = (r) => (r.years.length > maxLines * perLine ? r.years.slice(0, maxLines * perLine - 1) : r.years);
      const pubShown = pub != null && rowsData.some((r) => shown(r).includes(pub));
      if (pub != null && !pubShown && REC.strict.length < N) rowsData.push({ years: [pub], gi: -1 });
      const NUM = ['I', 'II', 'III'];
      // Where the published year is when its row is not shown: in a long place, below third place, or out.
      const pubWhere = () => {
        const gi = REC.groups.findIndex((g) => g.includes(pub));
        if (gi >= 0) return `in ${PLACE[gi]}`;
        if (REC.strict.includes(pub)) return 'stays, below third place';
        return (REC.safety || []).includes(pub) ? 'in the safety set, below third place' : 'does not stay';
      };
      const facts = (i) => {
        const parts = [plural(REC.K[i], 'clock', 'clocks'), `${REC.ID[i].toFixed(1)} decisive bits`];
        if (REC.IS[i] > 0) parts.push(`${REC.IS[i].toFixed(1)} supporting bits`);
        if (REC.contradict && REC.contradict[i]) parts.push(plural(REC.contradict[i], 'contradiction', 'contradictions'));
        return parts.join(' · ');
      };
      // On a narrow screen the label of a place keeps only the clocks and the decisive bits.
      const label = (i) => (L.narrowScreen ? `${plural(REC.K[i], 'clock', 'clocks')} · ${REC.ID[i].toFixed(1)} bits` : facts(i));
      st.tiles = [];
      let top = L.ladderTop;
      let cut = false;
      rowsData.forEach((r) => {
        const list = r.gi >= 0 ? shown(r) : r.years;
        const n = list.length + (r.gi >= 0 && r.years.length > list.length ? 1 : 0);
        const height = 30 + lines(n) * (chipH + gap);
        if (cut || top + height > st.h - 4) { if (r.gi >= 0) cut = true; return; }
        const row = (r.gi === -1 ? st.pubRow : g).append('g');
        if (r.gi >= 0 && !L.narrowScreen) {
          row.append('text').attr('class', 'year-big').attr('x', L.left).attr('y', top + 40).attr('font-size', 40)
            .attr('fill', r.gi === 0 ? C.gold : C.faint).text(NUM[r.gi]);
        }
        const i0 = r.years[0] - y0;
        const title = r.gi >= 0
          ? `${['First', 'Second', 'Third'][r.gi]} place, ${plural(r.years.length, 'year', 'years')} · ${label(i0)}`
          : `The published year · ${pubWhere()} · ${label(i0)}`;
        row.append('text').attr('class', 'label').attr('x', L.left + indent).attr('y', top + 6)
          .attr('font-size', L.narrowScreen ? 13 : 15).attr('font-weight', r.gi === 0 ? 600 : 400)
          .attr('fill', r.gi === -1 ? SEAL : (r.gi === 0 ? C.ink : C['ink-2'])).text(title);
        list.forEach((yr, t) => {
          const i = yr - y0;
          const x = L.left + indent + (t % perLine) * (chipW + gap);
          const y = top + 16 + Math.floor(t / perLine) * (chipH + gap);
          const chip = row.append('g').attr('class', 'token').attr('transform', `translate(${x},${y})`).style('cursor', 'pointer');
          chip.append('rect').attr('class', 'frame').attr('width', chipW).attr('height', chipH).attr('rx', chipH / 2)
            .attr('fill', C.card).attr('stroke', r.gi === 0 ? C.gold : C.rule).attr('stroke-width', r.gi === 0 ? 1.4 : 1);
          chip.append('text').attr('x', chipW / 2).attr('y', 17).attr('text-anchor', 'middle')
            .attr('font-size', L.narrowScreen ? 12 : 14).attr('font-weight', r.gi === 0 ? 600 : 400).text(yearLabel(yr));
          // The weight line: one segment per agreeing clock, then the supporting weight in gray.
          const full = chipW - 24;
          let x0 = 12;
          CLOCKS.forEach((d) => {
            if (d.row[i] !== 'a') return;
            const len = (d.c.bits / st.maxTotal) * full;
            chip.append('rect').attr('x', x0).attr('y', 24).attr('width', Math.max(1, len - 0.6)).attr('height', 3.5).attr('rx', 1.5)
              .attr('fill', d.clock.color);
            x0 += len;
          });
          const cap = ((REC.IS[i] || 0) / st.maxTotal) * full;
          if (cap > 0.5) {
            chip.append('rect').attr('class', 'cap').attr('x', x0).attr('y', 24).attr('width', Math.max(1, cap - 0.6))
              .attr('height', 3.5).attr('rx', 1.5).attr('fill', C.faint).attr('opacity', 0);
          }
          const clocks = [];
          CLOCKS.forEach((d) => { if (d.row[i] === 'a' && !clocks.some((ck) => ck.medal === d.clock.medal)) clocks.push(d.clock); });
          chip.on('mousemove', (ev) => tipShow(ev, `<b>${yearLabel(yr)}</b><br>${esc(facts(i))}<br>${clocks.map((ck) => `<span style="color:${ck.color}">●</span> ${esc(ck.name)}`).join('<br>')}<br><span style="opacity:.7">Click to see every clause</span>`))
            .on('mouseleave', tipHide);
          pressable(chip, `${yearLabel(yr)}: ${facts(i)}`, () => provenance(yr));
          st.tiles.push({ yr, tile: chip });
        });
        if (r.gi >= 0 && r.years.length > list.length) {
          const t = list.length;
          const x = L.left + indent + (t % perLine) * (chipW + gap);
          const y = top + 16 + Math.floor(t / perLine) * (chipH + gap);
          const more = row.append('g').attr('class', 'token').attr('transform', `translate(${x},${y})`).style('cursor', 'pointer');
          more.append('rect').attr('width', chipW).attr('height', chipH).attr('rx', chipH / 2).attr('fill', C.card)
            .attr('stroke', C.muted).attr('stroke-dasharray', '3,3');
          more.append('text').attr('x', chipW / 2).attr('y', chipH / 2 + 5).attr('text-anchor', 'middle')
            .attr('font-size', L.narrowScreen ? 12 : 14).attr('fill', C['ink-2']).text(`+${r.years.length - t} more`);
          const place = r.gi;
          pressable(more, `${r.years.length - t} more years in ${PLACE[place]}: show the full ranking`, () => rankingPanel(place));
        }
        top += height + 14;
      });
      if (cut) {
        const link = g.append('text').attr('class', 'label').attr('x', L.left + indent).attr('y', Math.min(top + 6, st.h - 8))
          .attr('font-size', L.narrowScreen ? 13 : 14).attr('fill', C['ink-2']).style('cursor', 'pointer')
          .style('text-decoration', 'underline').text('More places: show the full ranking');
        pressable(link, 'Show the full ranking', () => rankingPanel());
      }
    }
    // The published year: a red line through the strip and a small red tag above it. The tag opens
    // away from the nearer edge, so it never covers a label or a chip.
    function drawSeal() {
      const { x, L } = st;
      const pub = REC.published_astro;
      if (pub == null || pub < META.years[0] || pub > META.years[1]) return;
      const px = x(pub);
      st.pubG.append('line').attr('x1', px).attr('x2', px).attr('y1', L.stripTop - 22).attr('y2', L.stripTop + L.stripH + 2)
        .attr('stroke', SEAL).attr('stroke-width', 2.2);
      st.pubG.append('circle').attr('cx', px).attr('cy', L.stripTop - 24).attr('r', 5).attr('fill', SEAL);
      const right = px < (L.left + st.w - L.right) / 2;
      st.pubG.append('text').attr('class', 'label halo').attr('x', px + (right ? 10 : -10)).attr('y', L.stripTop - 20)
        .attr('text-anchor', right ? 'start' : 'end').attr('fill', SEAL).attr('font-weight', 600)
        .text(`published: ${spanLabel(pub)}`);
    }
    function update(i, dur) {
      setRevealed(i >= 5);
      if (!st || !CLOCKS.length) return;
      st.step = i;
      const { L } = st;
      const y0 = META.years[0];
      const cw = st.x(y0 + 1) - st.x(y0);
      // The ink soaks in clause by clause the first time; later steps keep it.
      if (st.soaked < CLOCKS.length - 1) {
        const steps = dur ? CLOCKS.length : 1;
        for (let k = (dur ? 0 : CLOCKS.length - 1); k < CLOCKS.length; k += 1) {
          st.cells.transition(`soak${k}`).delay(dur ? k * 380 : 0).duration(dur ? 420 : 0).attr('fill', (yi) => soak(yi, k));
        }
        st.soaked = CLOCKS.length - 1;
        void steps;
      }
      st.dots.transition().duration(dur).attr('opacity', i >= 1 ? 1 : 0);
      st.leaf.selectAll('rect').data(i >= 2 && REC.strict.length < N ? REC.strict : []).join(
        (enter) => enter.append('rect').attr('opacity', 0),
      ).attr('x', (yr) => st.x(yr) - Math.max(cw, 3) / 2 - 1.5).attr('width', Math.max(cw, 3) + 3)
        .attr('y', L.stripTop - 12).attr('height', L.stripH + 18).attr('rx', 2).attr('fill', C.gold)
        .transition().duration(dur).attr('opacity', 0.55);
      st.cells.transition('dim').duration(dur).attr('opacity', (yi) => (i >= 2 && REC.strict.length < N && !REC.strict.includes(y0 + yi) ? 0.45 : 1));
      st.ladder.transition().duration(dur).attr('opacity', i >= 3 ? 1 : 0);
      st.ladder.selectAll('rect.cap').transition().duration(dur).attr('opacity', i >= 4 ? 1 : 0);
      st.pubG.attr('display', i >= 5 ? null : 'none').attr('aria-hidden', i >= 5 ? null : 'true')
        .transition().duration(dur).attr('opacity', i >= 5 ? 1 : 0);
      st.pubRow.attr('display', i >= 5 ? null : 'none').attr('aria-hidden', i >= 5 ? null : 'true');
      const pub = REC.published_astro;
      (st.tiles || []).forEach(({ yr, tile }) => {
        tile.select('rect.frame').transition().duration(dur)
          .attr('stroke', i >= 5 && yr === pub ? SEAL : null).attr('stroke-width', i >= 5 && yr === pub ? 3 : null);
      });
    }
    return { render, update };
  }

  // ------------------------------------------------------------------ the side panel
  // One panel holds what the charts cannot: the verdicts of one year, the full ranking, the full text.
  // It takes the keyboard focus, Escape closes it, and the focus goes back to where it was.
  let panelReturn = null;
  function openPanel(html, label) {
    const panel = d3.select('.provenance-panel').attr('hidden', null).attr('role', 'dialog').attr('aria-label', label);
    if (!panelReturn) panelReturn = document.activeElement;
    panel.html(`<button class="close" type="button" aria-label="Close">×</button>${html}`);
    panel.select('.close').on('click', closePanel);
    panel.attr('tabindex', -1).node().focus();
    return panel;
  }
  function closePanel() {
    d3.select('.provenance-panel').attr('hidden', true);
    if (panelReturn && panelReturn.focus) panelReturn.focus();
    panelReturn = null;
  }
  document.addEventListener('keydown', (ev) => {
    if (ev.key === 'Escape' && d3.select('.provenance-panel').attr('hidden') === null) closePanel();
  });
  // A button-like element in an SVG: Enter and Space work as a click.
  function pressable(sel, label, fn) {
    sel.attr('tabindex', 0).attr('role', 'button').attr('aria-label', label).on('click', fn)
      .on('keydown', (ev) => { if (ev.key === 'Enter' || ev.key === ' ') { ev.preventDefault(); fn(ev); } });
  }

  // The first three places in full. The published year gets its mark only after the reveal.
  function rankingPanel(place) {
    const revealed = document.body.classList.contains('revealed');
    const pub = REC.published_astro;
    const y0 = META.years[0];
    const places = REC.groups.slice(0, 3).filter((g) => g && g.length).map((years, gi) => {
      const sorted = years.slice().sort(d3.ascending);
      const i0 = sorted[0] - y0;
      return `<section class="place" id="place-${gi}"><h4>${['First', 'Second', 'Third'][gi]} place · ${plural(years.length, 'year', 'years')}</h4>
        <p class="why">${plural(REC.K[i0], 'clock', 'clocks')} · ${REC.ID[i0].toFixed(1)} decisive bits</p>
        <div class="yrs">${sorted.map((yr) => `<button type="button" class="yr${revealed && yr === pub ? ' pub' : ''}" data-yr="${yr}">${yearLabel(yr)}</button>`).join('')}</div></section>`;
    }).join('');
    const panel = openPanel(`<h3>The ranking</h3><p class="why">Years with equal scores share a place. Select a year to
      see the verdict of each clause in that year.</p>${places}`, 'The ranking');
    panel.selectAll('button.yr').on('click', (ev) => provenance(+ev.currentTarget.dataset.yr));
    const target = place != null ? panel.select(`#place-${place}`).node() : null;
    if (target) target.scrollIntoView({ block: 'nearest' });
  }

  // The whole translation, with the hidden words kept hidden until the reveal, and the work of the model.
  function textPanel() {
    const revealed = document.body.classList.contains('revealed');
    const text = REC.text || '';
    let html = '';
    let last = 0;
    (REC.hidden || []).slice().sort((a, b) => a[0] - b[0]).forEach(([a, b]) => {
      html += esc(text.slice(last, a)) + (revealed ? `<span class="k k-seal">${esc(text.slice(a, b))}</span>`
        : '<span class="mask">hidden</span>');
      last = b;
    });
    html += esc(text.slice(last));
    const changes = ((REC.review || {}).changes || []).map((x) => `<li>“${esc(passage(REC.clauses[x.j], 60))}”. ${changeNote(x)}</li>`).join('');
    const found = (REC.found || []).map((f) => `<li>${esc(f.what)}: “${esc(f.quote)}”</li>`).join('');
    openPanel(`<h3>The translation</h3><div class="fulltext">${html.replace(/\n/g, '<br>') || 'This record has no English translation.'}</div>
      ${changes ? `<h4>Changes that the language model proposed</h4><ul>${changes}</ul>` : ''}
      ${found ? `<h4>Clauses that the language model found</h4><ul>${found}</ul>` : ''}`, 'The translation');
  }

  // ------------------------------------------------------------------ provenance of one year
  function provenance(yr) {
    const i = yr - META.years[0];
    if (!REC || i < 0 || i >= N) return;
    const why = REC.details[String(yr)];
    const color = { a: C.gold, c: C.seal, '.': C.faint };
    const rows = REC.clauses.map((c, j) => ({ c, j, v: REC.grid[j][i] }))
      .sort((a, b) => d3.descending(a.c.decisive, b.c.decisive) || d3.ascending('ac.'.indexOf(a.v), 'ac.'.indexOf(b.v)));
    openPanel(`<h3>${yearLabel(yr)}</h3>
      <p>${plural(REC.K[i], 'clock', 'clocks')} · ${REC.ID[i].toFixed(2)} decisive bits · ${REC.strict.includes(yr) ? 'stays' : 'does not stay'}</p>
      ${rows.map(({ c, j, v }) => `<div class="row"><span class="dot" style="background:${c.decisive && v === 'a' ? clockOf(c).color : color[v]}"></span><div>
        “${esc(passage(c, 90))}” <span style="color:${c.decisive ? C['gold-ink'] : C.muted}">${c.decisive ? 'decisive' : 'supporting'}</span>
        <div class="why">${capital(VERDICT[v])}${why ? `: ${esc(why[j])}` : '.'}</div></div></div>`).join('')}
      ${why ? '' : '<p class="why">The written reason of each verdict is kept for the first 12 years of the ranking, for the years that stay when they are 40 or fewer, and for the published year.</p>'}`, yearLabel(yr));
  }

  // ------------------------------------------------------------------ scroll engine
  const BUILDERS = { 's-text': secText, 's-clock': secClock, 's-sum': secSum };
  const sections = [];
  const isNarrow = () => innerWidth < 900;
  function setup() {
    Object.entries(BUILDERS).forEach(([id, build]) => {
      const el = document.getElementById(id);
      const fig = d3.select(el).select('figure.graphic');
      sections.push({ id, el, fig, impl: build(fig), steps: Array.from(el.querySelectorAll('.step')), rendered: false, current: -1 });
    });
  }
  function onScroll() {
    const line = innerHeight * (isNarrow() ? 0.8 : 0.55);
    sections.forEach((s) => {
      const r = s.el.getBoundingClientRect();
      if (!s.rendered && REC && r.top < innerHeight * 2 && r.bottom > -innerHeight) {
        s.impl.render();
        s.rendered = true;
        s.current = -1;
      }
      let i = 0;
      s.steps.forEach((step, j) => { if (step.getBoundingClientRect().top < line) i = j; });
      // A chapter waits until the reader can see its graphic, then animates into its step. So the first dial
      // sweeps when the reader arrives, and not while the chapter is still below the screen.
      const visible = r.top < innerHeight * 0.6 && r.bottom > innerHeight * 0.3;
      if (!s.rendered) return;
      if (i !== s.current) s.waiting = true;
      if (s.waiting && visible) {
        const jump = s.current >= 0 && (Math.abs(i - s.current) > 1 || i < s.current);
        s.waiting = false;
        s.current = i;
        s.steps.forEach((step, j) => step.classList.toggle('is-active', j === i));
        s.impl.update(i, jump || still() ? 0 : 700);
      }
    });
    const doc = document.documentElement;
    document.querySelector('.progress-bar').style.width = `${(100 * doc.scrollTop / Math.max(1, doc.scrollHeight - innerHeight)).toFixed(2)}%`;
  }
  let ticking = false;
  const requestScroll = () => {
    // A hidden page runs no animation frames, so it handles the scroll at once.
    if (document.hidden) { onScroll(); return; }
    if (ticking) return;
    ticking = true;
    // Some frames run no animation frame although they count as visible; a timer then does the work.
    let done = false;
    const run = () => { if (done) return; done = true; ticking = false; onScroll(); };
    requestAnimationFrame(run);
    setTimeout(run, 120);
  };
  function measureBar() {
    const bar = document.querySelector('.chooser-bar');
    // The bar floats 10 px below the top, with a small gap under it.
    if (bar) document.documentElement.style.setProperty('--bar-h', `${Math.ceil(bar.getBoundingClientRect().height) + 18}px`);
  }
  function rerender() {
    measureBar();
    readColors();
    if (REC) CLOCKS = findClocks(REC);
    sections.forEach((s) => { s.rendered = false; s.current = -1; });
    if (REC) showChoice(REC);
    onScroll();
  }
  let resizeTimer = null;
  let lastWidth = innerWidth;
  addEventListener('resize', () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (Math.abs(innerWidth - lastWidth) < 2 && isNarrow()) return;   // a phone's address bar
      lastWidth = innerWidth;
      rerender();
    }, 180);
  });
  // ------------------------------------------------------------------ start
  readColors();
  document.querySelectorAll('figure.graphic').forEach((f) => {
    const p = document.createElement('div');
    p.className = 'loading';
    p.textContent = 'Loading the data…';
    f.append(p);
  });
  const metaLoad = d3.json('data/meta.json');
  // The first record downloads while the star list still loads.
  metaLoad.then((meta) => {
    const wanted = (location.hash || '').slice(1);
    record(meta.records.some((r) => r.id === wanted) ? wanted : meta.featured[0]).catch(() => {});
  }, () => {});
  Promise.all([metaLoad, d3.json('data/sky.json')]).then(([meta, sky]) => {
    META = meta;
    SKY = sky;
    setupRing();
    measureBar();
    showChoice = setupChooser();
    setup();
    const wanted = (location.hash || '').slice(1);
    choose(META.records.some((r) => r.id === wanted) ? wanted : META.featured[0]);
    addEventListener('scroll', requestScroll, { passive: true });
    if (document.fonts && document.fonts.ready) document.fonts.ready.then(rerender);
  }, (err) => {
    console.error(err);
    document.querySelectorAll('.loading').forEach((el) => {
      el.textContent = 'The data did not load. Serve this folder with a web server, such as python -m http.server.';
    });
  });
  // A link to a chapter scrolls there and keeps the record in the address, so a reload keeps the record.
  document.querySelectorAll('a[href^="#ch-"]').forEach((a) => a.addEventListener('click', (ev) => {
    const target = document.querySelector(a.getAttribute('href'));
    if (!target) return;
    ev.preventDefault();
    target.scrollIntoView({ behavior: still() ? 'auto' : 'smooth' });
  }));
  // Chapter IV holds the game A Little Babylon. It loads only when the reader asks, it follows the tablet chosen
  // here, and a tablet chosen in the game is chosen here too, without moving the page.
  const GAME = /^(localhost|127\.0\.0\.1)$/.test(location.hostname) ? 'http://localhost:5187/'
    : 'https://tingyuansen.github.io/Babylonian_Observatory/';
  const gameOrigin = new URL(GAME).origin;
  let gameFrame = null;
  let gameVeil = null;
  let gameVisible = true;
  const tellGameVisible = () => gameFrame.contentWindow.postMessage({ type: 'babylon-visible', visible: gameVisible }, gameOrigin);
  // The game shows the tablets that the research dates: those whose first group of years is not all of them.
  const gameDated = (id) => { const r = META && META.records.find((x) => x.id === id); return !!r && r.first < META.years[1] - META.years[0] + 1; };
  function gameSync(id) {
    const dated = gameDated(id);
    document.querySelector('.game-full').href = dated ? `${GAME}?tablet=${id}` : GAME;
    document.querySelector('.game-note').textContent = dated ? ''
      : 'The research gives no year for this tablet, so the game cannot show it.';
    if (gameVeil) gameVeil.hidden = dated;
    if (gameFrame && dated) gameFrame.contentWindow.postMessage({ type: 'babylon-tablet', id }, gameOrigin);
  }
  // A phone opens the game on a page of its own: a game in the page would take the finger that scrolls the page.
  const phone = matchMedia('(pointer: coarse)').matches && Math.min(screen.width, screen.height) < 820;
  if (phone) document.querySelector('.game-start small').textContent = 'The game opens full screen. Drag to look, and hold the circle to walk.';
  document.querySelector('.game-start').addEventListener('click', () => {
    const dated = REC && gameDated(REC.id);
    if (phone) { window.open(`${GAME}?tablet=${dated ? REC.id : 'X301492'}`, '_blank', 'noopener'); return; }
    gameFrame = document.createElement('iframe');
    gameFrame.src = `${GAME}?tablet=${dated ? REC.id : 'X301492'}`;
    gameFrame.title = 'A Little Babylon: the tablet in the sky over Babylon';
    gameFrame.allow = 'fullscreen; autoplay';
    // Over the game, a line tells the reader when the game cannot show the tablet chosen here.
    gameVeil = document.createElement('p');
    gameVeil.className = 'game-veil';
    gameVeil.textContent = 'The research gives no year for this tablet, so the game shows another tablet.';
    gameVeil.hidden = !!dated;
    document.querySelector('.game-frame').replaceChildren(gameFrame, gameVeil);
    // The game shares the page's thread, so it sleeps, music and all, while chapter IV is out of sight.
    new IntersectionObserver((entries) => {
      gameVisible = entries[entries.length - 1].isIntersecting;
      tellGameVisible();
    }).observe(document.querySelector('.game-frame'));
  });
  addEventListener('message', (e) => {
    if (e.origin !== gameOrigin || !gameFrame || e.source !== gameFrame.contentWindow || !e.data || !META) return;
    // The game is ready to listen: send the tablet again, in case the game missed it while it loaded.
    // Then the game takes the keys, so W A S D walk at once, unless the reader is typing elsewhere on the page.
    if (e.data.type === 'babylon-ready') {
      if (REC) gameSync(REC.id);
      tellGameVisible();
      if (document.activeElement === document.body) gameFrame.focus({ preventScroll: true });
      return;
    }
    if (e.data.type !== 'babylon-tablet-changed') return;
    const id = String(e.data.id);
    if (META.records.some((r) => r.id === id) && (!REC || REC.id !== id)) choose(id, true);
  });
  addEventListener('hashchange', () => {
    const id = location.hash.slice(1);
    if (META && META.records.some((r) => r.id === id)) choose(id);
  });
}());
