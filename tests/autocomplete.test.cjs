const { test } = require('node:test');
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const vm = require('node:vm');

function setup() {
    let listener;
    let timer;
    const pending = [];
    const suggestions = { values: [], replaceChildren(...options) { this.values = options.map(o => o.value); } };
    const input = { value: '', dataset: { autocompleteUrl: '/autocomplete' },
        addEventListener(event, callback) { listener = callback; } };
    vm.runInNewContext(readFileSync(join(__dirname, '../static/autocomplete.js'), 'utf8'), {
        document: { getElementById: id => id === 'autocomplete' ? input : suggestions,
            createElement: () => ({}) },
        AbortController, encodeURIComponent,
        setTimeout(callback) { timer = callback; },
        clearTimeout() { timer = undefined; },
        fetch(url, options) { return new Promise((resolve, reject) => pending.push({ url, options, resolve, reject })); },
    });
    return { input, suggestions, pending,
        type(value) { input.value = value; listener(); },
        runTimer() { return timer(); } };
}

test('an old response cannot populate suggestions during the next debounce', async () => {
    const ui = setup();
    ui.type('old');
    const first = ui.runTimer();
    ui.type('new');
    assert.equal(ui.pending[0].options.signal.aborted, true);
    ui.pending[0].resolve({ ok: true, json: async () => ['Old title'] });
    await first;
    assert.deepEqual(ui.suggestions.values, []);
    const second = ui.runTimer();
    ui.pending[1].resolve({ ok: true, json: async () => ['New title'] });
    await second;
    assert.deepEqual(ui.suggestions.values, ['New title']);
});

test('network failure leaves manual search usable and clears suggestions', async () => {
    const ui = setup();
    ui.type('book');
    const request = ui.runTimer();
    ui.pending[0].reject(new Error('Offline'));
    await request;
    assert.deepEqual(ui.suggestions.values, []);
    assert.equal(ui.input.value, 'book');
});

test('clearing the input cancels an outstanding request and clears suggestions', async () => {
    const ui = setup();
    ui.type('book');
    const request = ui.runTimer();
    ui.type('');
    ui.pending[0].resolve({ ok: true, json: async () => ['Book'] });
    await request;
    assert.deepEqual(ui.suggestions.values, []);
    assert.equal(ui.pending.length, 1);
});

test('detailed suggestions keep the book title as the submitted value', async () => {
    const ui = setup();
    ui.type('book');
    const request = ui.runTimer();
    assert.match(ui.pending[0].url, /details=1/);
    ui.pending[0].resolve({ ok: true, json: async () => [{ title: 'A book', author: 'An author' }] });
    await request;
    assert.deepEqual(ui.suggestions.values, ['A book']);
});
