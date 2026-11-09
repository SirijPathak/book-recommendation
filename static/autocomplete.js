const input = document.getElementById('autocomplete');
const suggestions = document.getElementById('book-suggestions');
let timer;
let latestRequest = 0;
let controller;

input.addEventListener('input', () => {
    clearTimeout(timer);
    const request = ++latestRequest;
    controller?.abort();
    suggestions.replaceChildren();
    const query = input.value.trim();
    if (!query) return;
    timer = setTimeout(async () => {
        controller = new AbortController();
        try {
            const response = await fetch(input.dataset.autocompleteUrl + '?details=1&q=' + encodeURIComponent(query), {
                signal: controller.signal,
            });
            if (!response.ok) throw new Error('Suggestions unavailable');
            const titles = await response.json();
            // The input may have changed while this response was loading.
            if (request !== latestRequest) return;
            suggestions.replaceChildren(...titles.map((title) => {
                const option = document.createElement('option');
                option.value = typeof title === 'string' ? title : title.title;
                if (typeof title !== 'string') option.label = 'by ' + title.author;
                return option;
            }));
        } catch (error) {
            // Manual search remains available when suggestions cannot be loaded.
            if (request === latestRequest) suggestions.replaceChildren();
        }
    }, 150);
});
