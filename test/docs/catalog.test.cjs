// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const { test } = require('node:test');
const vm = require('node:vm');

const source = readFileSync(join(__dirname, '../../docs/assets/javascripts/e2s-catalog.js'), 'utf8');
const model = (title, memory, filters = {}) => ({
  title, recommended_vram_gb: memory, kind: 'model', group: 'Prognostic',
  summary: 'Weather', chips: [`Recommended VRAM: ${memory ?? 'unknown'}`],
  filters, url: `../../../models/${title}/`, tone: 'default',
});
const records = [
  model('Small', 24, { provider: ['NVIDIA'], region: ['Global'] }),
  model('Boundary', 80, { provider: ['NVIDIA'], region: ['Regional'] }),
  model('Large', 96), model('Unknown', null), model('Zero', 0),
  model('Decimal', 12.5), model('Missing', undefined), model('Malformed', '40'),
  model('Negative', -1),
  { ...model('Data', null), kind: 'data', chips: [], filters: { provider: ['NOAA'] } },
];

// Minimal DOM/event adapter: run the unmodified script, asserting rendered markup,
// URL updates and user events. Real browser layout is checked separately.
async function catalog(query = '', items = records, inline = true) {
  const controls = new Map();
  let html = '';
  let url = new URL(`https://example.test/catalog/${query}#catalog`);
  const element = (selector, value = '', dataset = {}) => {
    if (!controls.has(selector)) controls.set(selector, {
      value, dataset, checked: false, handlers: {},
      addEventListener(name, fn) { this.handlers[name] = fn; },
      scrollIntoView() {},
    });
    return controls.get(selector);
  };
  const resultPattern = /<section class="e2s-catalog-results"[\s\S]*?<\/section>/;
  const root = {
    get innerHTML() { return html; },
    set innerHTML(value) { html = value; controls.clear(); },
    querySelector(selector) {
      if (selector === '[data-e2s-catalog-data]') return inline ? { textContent: JSON.stringify(items) } : null;
      if (selector === '.e2s-catalog-results') return {
        set outerHTML(value) { html = html.replace(resultPattern, value); },
        scrollIntoView() {},
      };
      if (!html.includes(selector.slice(1, -1))) return null;
      return element(selector);
    },
    querySelectorAll(selector) {
      const attr = selector.slice(1, -1);
      return [...html.matchAll(new RegExp(`<[^>]+${attr}="([^"]+)"[^>]*>`, 'g'))].map(([tag, value]) => {
        const key = attr.slice(5).replace(/-([a-z])/g, (_, c) => c.toUpperCase());
        const inputValue = tag.match(/ value="([^"]+)"/)?.[1] || '';
        const el = element(`${selector}:${value}:${inputValue}`, inputValue, { [key]: value });
        el.checked = tag.includes('checked');
        return el;
      });
    },
  };
  const window = {
    get location() { return url; },
    history: { replaceState(_state, _title, next) { url = new URL(next, url); } },
  };
  vm.runInNewContext(source, {
    document: { querySelector: () => root, currentScript: { src: 'https://example.test/assets/javascripts/e2s-catalog.js' } },
    window, URL, URLSearchParams, fetch: async () => ({ ok: true, json: async () => items }),
  });
  await new Promise(setImmediate);
  return {
    html: () => html,
    titles: () => [...html.matchAll(/<h3><a[^>]+>([^<]+)<\/a><\/h3>/g)].map((match) => match[1]),
    url: () => url,
    change(value) {
      const select = root.querySelector('[data-catalog-vram]');
      assert.ok(select, 'Available GPU VRAM selector is rendered');
      select.value = value;
      select.handlers.change({ target: select });
    },
    search(value) {
      const input = root.querySelector('[data-catalog-search]');
      input.value = value;
      input.handlers.input({ target: input });
    },
    click(selector, value) {
      const el = value ? root.querySelectorAll(selector).find((item) => Object.values(item.dataset).includes(value)) : root.querySelector(selector);
      assert.ok(el, selector);
      el.handlers.click();
    },
    filter(group, value) {
      const el = root.querySelectorAll('[data-filter-group]').find((item) => item.dataset.filterGroup === group && item.value === value);
      assert.ok(el);
      el.checked = true;
      el.handlers.change();
    },
  };
}

test('capacity options come from model metadata, sorted numerically and labelled accessibly', async () => {
  const ui = await catalog();
  assert.match(ui.html(), /<label[^>]+for="e2s-catalog-vram"[^>]*>Available GPU VRAM<\/label>/);
  assert.match(ui.html(), /aria-describedby="e2s-catalog-vram-help"/);
  const options = [...ui.html().matchAll(/<option value="([^"]*)"/g)].map((match) => match[1]);
  assert.deepEqual(options, ['', '0', '12.5', '24', '80', '96']);
});

test('80 GB includes smaller recommendations and the exact boundary, excluding unknowns and invalid metadata', async () => {
  const ui = await catalog();
  ui.change('80');
  assert.deepEqual(ui.titles(), ['Small', 'Boundary', 'Zero', 'Decimal']);
  assert.match(ui.html(), /Recommended VRAM: 80/);
  assert.equal(ui.url().searchParams.get('vram'), '80');
  assert.equal(ui.url().hash, '#catalog');
});

test('zero capacity is an explicit cutoff, not all or unknown', async () => {
  const ui = await catalog('?vram=0');
  assert.deepEqual(ui.titles(), ['Zero']);
});

test('capacity combines with categorical filters and search', async () => {
  const ui = await catalog();
  ui.change('80');
  ui.filter('provider', 'NVIDIA');
  ui.filter('region', 'Global');
  assert.deepEqual(ui.titles(), ['Small']);
  ui.search('boundary');
  assert.deepEqual(ui.titles(), []);
  ui.search('small');
  assert.deepEqual(ui.titles(), ['Small']);
});

test('clear capacity keeps other filters; clear filters resets everything', async () => {
  const ui = await catalog('?vram=80&q=weather&provider=NVIDIA');
  ui.change('');
  assert.deepEqual(ui.titles(), ['Small', 'Boundary']);
  assert.equal(ui.url().searchParams.has('vram'), false);
  ui.click('[data-catalog-clear]');
  assert.equal(ui.titles().length, 9);
  assert.equal(ui.url().search, '?tab=models');
  assert.match(ui.html(), /Recommended VRAM: unknown/);
});

test('URL round trip preserves capacity, search, categories and links with fetched JSON', async () => {
  const ui = await catalog('?tab=models&vram=80&q=weather&provider=NVIDIA', records, false);
  const restored = await catalog(ui.url().search);
  assert.deepEqual(restored.titles(), ['Small', 'Boundary']);
  assert.match(restored.html(), /<option value="80" selected>80 GB<\/option>/);
  assert.match(restored.html(), /href="\.\.\/\.\.\/\.\.\/models\/Boundary\/"/);
});

for (const value of ['garbage', '-1', 'NaN', 'Infinity', '80gb', '80junk', '1e2', '0x50', '81', '80&vram=24']) {
  test(`invalid or unsupported URL capacity is cleared: ${value}`, async () => {
    const ui = await catalog(`?vram=${value}`);
    assert.equal(ui.url().searchParams.has('vram'), false);
    assert.equal(ui.titles().length, 9);
    assert.match(ui.html(), /<option value="" selected>/);
  });
}

test('filter changes reset pagination, and filtered pagination survives reload', async () => {
  const many = Array.from({ length: 25 }, (_, i) => model(`Model${i}`, i < 21 ? 80 : 96));
  const ui = await catalog('?page=3', many);
  ui.change('80');
  assert.match(ui.html(), /Showing 1-10 of 21/);
  assert.equal(ui.url().searchParams.has('page'), false);
  ui.click('[data-catalog-page]', 'next');
  assert.match(ui.html(), /Showing 11-20 of 21/);
  const restored = await catalog(ui.url().search, many);
  assert.deepEqual(restored.titles(), ui.titles());
  ui.change('');
  assert.match(ui.html(), /Showing 1-10 of 25/);
});

test('data-source tab ignores capacity and switching tabs clears it', async () => {
  const data = await catalog('?tab=data&vram=80&provider=NOAA');
  assert.deepEqual(data.titles(), ['Data']);
  assert.doesNotMatch(data.html(), /data-catalog-vram/);
  assert.equal(data.url().searchParams.has('vram'), false);
  const ui = await catalog('?vram=80');
  ui.click('[data-catalog-kind]', 'data');
  assert.deepEqual(ui.titles(), ['Data']);
  ui.click('[data-catalog-kind]', 'model');
  assert.equal(ui.url().searchParams.has('vram'), false);
  assert.equal(ui.titles().length, 9);
});
