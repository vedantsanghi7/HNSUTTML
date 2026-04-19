/* HN Intel frontend - digest pipeline + grounded chat. */

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const el = {
    hero: $('#hero'),
    form: $('#search-form'),
    input: $('#topic-input'),
    analyzeBtn: $('#analyze-btn'),
    progress: $('#progress-panel'),
    progressText: $('#progress-text'),
    progressSteps: $('#progress-steps'),
    workspace: $('#workspace'),
    workspaceTopic: $('#workspace-topic'),
    workspaceMeta: $('#workspace-meta'),
    digestOut: $('#digest-markdown'),
    tabs: $$('.tab'),
    tabPanels: $$('.tab-panel'),
    chatLog: $('#chat-log'),
    chatForm: $('#chat-form'),
    chatInput: $('#chat-input'),
    chatSend: $('#chat-send'),
    errorBanner: $('#error-banner'),
    errorContent: $('#error-banner .error-content'),
    errorDismiss: $('#error-dismiss'),
    newBtn: $('#new-btn'),
    digestList: $('#digest-list'),
    digestEmpty: $('#digest-empty'),
    expectedTime: $('#expected-time'),
};

const state = {
    sessionId: null,
    queryId: null,
    topic: null,
    evSource: null,
    chatBusy: false,
    pipelineStartTime: 0,
    pipelineTimer: null,
};

const EXPECTED_MS = 4 * 60 * 1000 + 30 * 1000; // 4m30s


const citeExtension = {
    name: 'citation',
    level: 'inline',
    start(src) { return src.indexOf('[#'); },
    tokenizer(src) {
        const m = /^\[#([0-9][0-9,\s#]*)\]/.exec(src);
        if (!m) return undefined;
        const ids = m[1].replace(/#/g, '').split(/[,\s]+/).filter(Boolean);
        return { type: 'citation', raw: m[0], ids };
    },
    renderer(token) {
        return token.ids
            .map((id) => `<a class="cite" href="https://news.ycombinator.com/item?id=${id}" target="_blank" rel="noopener" data-cid="${id}">#${id}</a>`)
            .join(' ');
    },
};
marked.use({ extensions: [citeExtension] });

function renderMarkdown(text) {
    return marked.parse(text || '');
}


const STEP_KEYWORDS = [
    { step: 'fetch',       re: /fetching threads/i },
    { step: 'chunk',       re: /context prefix/i },
    { step: 'embed',       re: /embedding/i },
    { step: 'extract',     re: /extracting/i },
    { step: 'cluster',     re: /clustering/i },
    { step: 'synthesize',  re: /synthesizing/i },
];

function updateProgress(msg) {
    el.progressText.textContent = msg;
    const match = STEP_KEYWORDS.find(({ re }) => re.test(msg));
    if (!match) return;
    let passed = true;
    $$('#progress-steps li').forEach((li) => {
        const step = li.dataset.step;
        if (step === match.step) {
            li.classList.add('active');
            li.classList.remove('done');
            passed = false;
        } else if (passed) {
            li.classList.add('done');
            li.classList.remove('active');
        } else {
            li.classList.remove('active', 'done');
        }
    });
}

function markAllDone() {
    $$('#progress-steps li').forEach((li) => {
        li.classList.add('done');
        li.classList.remove('active');
    });
}


function showError(msg) {
    el.errorContent.textContent = msg;
    el.errorBanner.classList.remove('hidden');
}
el.errorDismiss.addEventListener('click', () => el.errorBanner.classList.add('hidden'));


function showHero() {
    el.hero.classList.remove('hidden');
    el.progress.classList.add('hidden');
    el.workspace.classList.add('hidden');
}

function updateTimer() {
    if (!state.pipelineStartTime) return;
    const elapsed = Date.now() - state.pipelineStartTime;
    const remaining = Math.max(0, EXPECTED_MS - elapsed);
    if (remaining > 0) {
        const m = Math.floor(remaining / 60000);
        const s = Math.floor((remaining % 60000) / 1000);
        el.expectedTime.textContent = `Expected time remaining: ~${m}:${s < 10 ? '0'+s : s}`;
    } else {
        el.expectedTime.textContent = `Wrapping up...`;
    }
}

function showProgress() {
    el.hero.classList.add('hidden');
    el.progress.classList.remove('hidden');
    el.workspace.classList.add('hidden');
    $$('#progress-steps li').forEach((li) => li.classList.remove('active', 'done'));
    state.pipelineStartTime = Date.now();
    updateTimer();
    if (state.pipelineTimer) clearInterval(state.pipelineTimer);
    state.pipelineTimer = setInterval(updateTimer, 1000);
}
function showWorkspace() {
    el.hero.classList.add('hidden');
    el.progress.classList.add('hidden');
    el.workspace.classList.remove('hidden');
}

function switchTab(name) {
    el.tabs.forEach((t) => t.classList.toggle('active', t.dataset.tab === name));
    el.tabPanels.forEach((p) => p.classList.toggle('active', p.id === `tab-${name}`));
    if (name === 'chat') setTimeout(() => el.chatInput.focus(), 80);
}
el.tabs.forEach((t) => t.addEventListener('click', () => switchTab(t.dataset.tab)));


el.form.addEventListener('submit', (e) => {
    e.preventDefault();
    const topic = el.input.value.trim();
    if (topic) runPipeline(topic);
});

$$('.chip').forEach((c) => {
    c.addEventListener('click', () => {
        el.input.value = c.dataset.topic;
        runPipeline(c.dataset.topic);
    });
});

el.newBtn.addEventListener('click', () => {
    if (state.evSource) { state.evSource.close(); state.evSource = null; }
    resetChat();
    state.sessionId = null;
    state.queryId = null;
    state.topic = null;
    el.input.value = '';
    showHero();
    el.input.focus();
    refreshDigestList();
});

function runPipeline(topic) {
    if (state.evSource) state.evSource.close();
    el.analyzeBtn.disabled = true;
    showProgress();
    updateProgress('Connecting...');

    const url = `/api/generate?topic=${encodeURIComponent(topic)}`;
    const es = new EventSource(url);
    state.evSource = es;

    es.onmessage = (event) => {
        let data;
        try { data = JSON.parse(event.data); } catch (_) { return; }
        switch (data.type) {
            case 'status':
                updateProgress(data.message);
                break;
            case 'complete':
                if (state.pipelineTimer) clearInterval(state.pipelineTimer);
                markAllDone();
                onDigestReady({
                    content: data.content,
                    query_id: data.query_id,
                    topic: data.topic || topic,
                });
                es.close();
                state.evSource = null;
                el.analyzeBtn.disabled = false;
                break;
            case 'error':
                if (state.pipelineTimer) clearInterval(state.pipelineTimer);
                showError(`Pipeline failed: ${data.message}`);
                es.close();
                state.evSource = null;
                el.analyzeBtn.disabled = false;
                showHero();
                break;
        }
    };
    es.onerror = () => {
        if (state.pipelineTimer) clearInterval(state.pipelineTimer);
        showError('Connection lost while generating the digest.');
        es.close();
        state.evSource = null;
        el.analyzeBtn.disabled = false;
        showHero();
    };
}


async function onDigestReady({ content, query_id, topic }) {
    state.queryId = query_id;
    state.topic = topic;
    el.workspaceTopic.textContent = topic;
    el.workspaceMeta.textContent = `query #${query_id}`;
    el.digestOut.innerHTML = renderMarkdown(content);
    resetChat();
    showWorkspace();
    switchTab('digest');
    refreshDigestList();
    try {
        const res = await fetch('/api/chat/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query_id }),
        });
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        state.sessionId = data.session_id;
    } catch (e) {
        showError(`Chat unavailable: ${e.message || e}`);
    }
}


function resetChat() {
    el.chatLog.innerHTML = '';
    el.chatLog.innerHTML = `
        <div class="chat-hint">
            Ask follow-ups grounded in the fetched HN threads. Try:
            <ul>
                <li>"What did they say about write performance?"</li>
                <li>"How does this compare to Postgres?"</li>
                <li>"Any caveats with WAL mode?"</li>
            </ul>
        </div>
    `;
}

function addUserMsg(text) {
    const hint = el.chatLog.querySelector('.chat-hint');
    if (hint) hint.remove();
    const div = document.createElement('div');
    div.className = 'msg user';
    div.innerHTML = `<div class="bubble"></div>`;
    div.querySelector('.bubble').textContent = text;
    el.chatLog.appendChild(div);
    div.scrollIntoView({ behavior: 'smooth', block: 'end' });
    return div;
}

function addAssistantPlaceholder() {
    const div = document.createElement('div');
    div.className = 'msg assistant';
    div.innerHTML = `<div class="bubble"><div class="typing"><span></span><span></span><span></span></div></div>`;
    el.chatLog.appendChild(div);
    div.scrollIntoView({ behavior: 'smooth', block: 'end' });
    return div;
}

function renderEvidence(rows) {
    if (!rows || !rows.length) return '';
    const items = rows.map((r) => `
        <li class="evidence-item">
            <div class="ev-head">
                <span class="ev-cid"><a class="cite" href="https://news.ycombinator.com/item?id=${r.cid}" target="_blank" rel="noopener">#${r.cid}</a></span>
                <span>${(r.thread_title || '').slice(0, 80)}${(r.thread_title || '').length > 80 ? '…' : ''}</span>
            </div>
            <div class="ev-snippet">${escapeHtml(r.snippet || '')}</div>
        </li>
    `).join('');
    return `<ul class="evidence-list">${items}</ul>`;
}

function escapeHtml(s) {
    return (s || '').replace(/[&<>"']/g, (m) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[m]));
}

function fillAssistantMsg(div, result) {
    const body = renderMarkdown(result.answer || '');
    const metaBits = [];
    if (result.intent) metaBits.push(`intent: ${result.intent}`);
    if (result.rewritten_query && result.rewritten_query.trim()) metaBits.push(`rewritten: “${result.rewritten_query.trim()}”`);
    if (result.used_retrieval) metaBits.push(`evidence: ${result.evidence.length}`);
    const meta = metaBits.length ? `<div class="msg-meta">${metaBits.join(' · ')}</div>` : '';
    div.innerHTML = `<div class="bubble">${body}${renderEvidence(result.evidence)}</div>${meta}`;
    div.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

el.chatForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (state.chatBusy) return;
    const msg = el.chatInput.value.trim();
    if (!msg) return;
    if (!state.sessionId) {
        showError('No active chat session. Run a topic first.');
        return;
    }
    state.chatBusy = true;
    el.chatSend.disabled = true;
    addUserMsg(msg);
    const placeholder = addAssistantPlaceholder();
    el.chatInput.value = '';
    autoresize(el.chatInput);
    try {
        const res = await fetch('/api/chat/message', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: state.sessionId, message: msg }),
        });
        if (!res.ok) {
            const text = await res.text();
            throw new Error(text || res.statusText);
        }
        const data = await res.json();
        fillAssistantMsg(placeholder, data);
    } catch (err) {
        placeholder.innerHTML = `<div class="bubble" style="border-color: rgba(239,68,68,0.5);">⚠ ${escapeHtml(err.message || String(err))}</div>`;
    } finally {
        state.chatBusy = false;
        el.chatSend.disabled = false;
        el.chatInput.focus();
    }
});

function autoresize(ta) {
    ta.style.height = 'auto';
    ta.style.height = Math.min(ta.scrollHeight, 180) + 'px';
}
el.chatInput.addEventListener('input', () => autoresize(el.chatInput));
el.chatInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        el.chatForm.requestSubmit();
    }
});


async function refreshDigestList() {
    try {
        const res = await fetch('/api/digests');
        if (!res.ok) return;
        const { items } = await res.json();
        el.digestList.innerHTML = '';
        if (!items || !items.length) {
            el.digestEmpty.classList.remove('hidden');
            return;
        }
        el.digestEmpty.classList.add('hidden');
        items.forEach((it) => {
            const li = document.createElement('li');
            const btn = document.createElement('button');
            btn.innerHTML = `<div class="topic">${escapeHtml(it.topic)}</div><div class="meta">query #${it.query_id}</div>`;
            if (it.query_id === state.queryId) btn.classList.add('active');
            btn.addEventListener('click', () => loadDigest(it.query_id));
            li.appendChild(btn);
            el.digestList.appendChild(li);
        });
    } catch (_) { /* ignore */ }
}

async function loadDigest(queryId) {
    try {
        const res = await fetch(`/api/digest/${queryId}`);
        if (!res.ok) throw new Error(await res.text());
        const data = await res.json();
        onDigestReady({ content: data.content, query_id: data.query_id, topic: data.topic });
    } catch (e) {
        showError(`Could not load digest: ${e.message || e}`);
    }
}

refreshDigestList();
