/**
 * RAG-RPG 记忆引擎 - SillyTavern 扩展 v2.0
 * 功能：
 *   1. 自动捕获每轮对话并发送至 Python 后端处理
 *   2. AI 生成前自动查询向量库获取剧情约束
 *   3. 将约束文本注入系统提示词，实现记忆驱动的剧情引导
 *   4. 在聊天界面中浮动展示当前生效的剧情约束
 */

import { eventSource, event_types } from '../../../../script.js';
import { extension_settings, getContext } from '../../../extensions.js';
import { saveSettingsDebounced } from '../../../../script.js';

const EXTENSION_NAME = 'RAG-RPG';
const API_TIMEOUT = 15000;

const defaultSettings = {
    api_url: 'http://127.0.0.1:8765',
    api_key: 'rag-rpg-local',
    game_profile: 'default',
    auto_profile: true,
    auto_ingest: true,
    auto_query: true,
    inject_constraints: true,
    show_constraints_panel: true,
    query_turns: 6,
    max_results: 3,
    debug_mode: false,
};

// ─── HTML 模板 ─────────────────────────────────────────────

function getPanelHtml() {
    return `
    <div id="rag-rpg-panel" style="
        position: fixed; bottom: 60px; right: 10px; width: 320px;
        max-height: 280px; overflow-y: auto; z-index: 9999;
        background: rgba(20, 20, 30, 0.92); border: 1px solid #4a4a6a;
        border-radius: 8px; padding: 10px; font-size: 12px;
        color: #d0d0e0; display: none; box-shadow: 0 4px 12px rgba(0,0,0,0.5);
        font-family: 'Segoe UI', sans-serif;">
        <div style="display:flex;justify-content:space-between;margin-bottom:6px;
            border-bottom:1px solid #3a3a5a;padding-bottom:6px;">
            <span style="font-weight:bold;color:#8ab4f8;">RAG-RPG 约束</span>
            <span id="rag-rpg-close-btn" style="cursor:pointer;color:#888;"
                title="关闭">✕</span>
        </div>
        <div id="rag-rpg-profile-badge" style="font-size:10px;color:#6a8;margin-bottom:4px;"></div>
        <div id="rag-rpg-constraints-list">
            <div style="color:#666;text-align:center;padding:10px 0;">
                暂无约束（进行对话后自动更新）
            </div>
        </div>
        <div style="margin-top:6px;border-top:1px solid #3a3a5a;padding-top:4px;
            display:flex;gap:6px;">
            <button id="rag-rpg-refresh-btn" style="flex:1;background:#2a2a4a;
                border:1px solid #4a4a6a;color:#aac;border-radius:4px;padding:3px 8px;
                cursor:pointer;font-size:11px;">刷新</button>
            <span style="color:#555;font-size:10px;align-self:center;">
                ⚡约束注入中</span>
        </div>
    </div>`;
}

// ─── 加载/保存设置 ─────────────────────────────────────────

function loadSettings() {
    if (!extension_settings[EXTENSION_NAME]) {
        extension_settings[EXTENSION_NAME] = {};
    }
    const s = extension_settings[EXTENSION_NAME];
    for (const [key, value] of Object.entries(defaultSettings)) {
        if (s[key] === undefined) {
            s[key] = value;
        }
    }
    if (s.api_url.endsWith('/')) {
        s.api_url = s.api_url.slice(0, -1);
    }
}

function getSettings() {
    return extension_settings[EXTENSION_NAME];
}

function log(...args) {
    if (getSettings().debug_mode) {
        console.log(`[${EXTENSION_NAME}]`, ...args);
    }
}

function alwaysLog(...args) {
    console.log(`[${EXTENSION_NAME}]`, ...args);
}

// ─── 设置面板 HTML（ST 通过全局对象访问此函数）───────

globalThis.getSettingsHtml = function() {
    const s = getSettings();
    return `
    <div class="rag-rpg-settings">
        <div class="inline-drawer">
            <div class="inline-drawer-toggle inline-drawer-header">
                <b>RAG-RPG 记忆引擎</b>
                <div class="inline-drawer-icon fa-solid fa-circle-chevron-down down"></div>
            </div>
            <div class="inline-drawer-content" style="padding:10px;">
                <label class="checkbox_label">
                    <input type="checkbox" id="rag-rpg-auto-ingest" ${s.auto_ingest ? 'checked' : ''}>
                    <span>自动记录对话</span>
                </label>
                <label class="checkbox_label">
                    <input type="checkbox" id="rag-rpg-auto-query" ${s.auto_query ? 'checked' : ''}>
                    <span>自动查询约束</span>
                </label>
                <label class="checkbox_label">
                    <input type="checkbox" id="rag-rpg-inject-constraints" ${s.inject_constraints ? 'checked' : ''}>
                    <span>注入剧情约束</span>
                </label>
                <label class="checkbox_label">
                    <input type="checkbox" id="rag-rpg-show-panel" ${s.show_constraints_panel ? 'checked' : ''}>
                    <span>显示约束浮动面板</span>
                </label>
                <hr>
                <label>API URL:
                    <input type="text" id="rag-rpg-api-url" value="${s.api_url}" style="width:100%;">
                </label>
                <label>API Key:
                    <input type="text" id="rag-rpg-api-key" value="${s.api_key}" style="width:100%;">
                </label>
                <label>查询轮次:
                    <input type="number" id="rag-rpg-query-turns" value="${s.query_turns}" min="1" max="20" style="width:60px;">
                </label>
                <hr>
                <div style="color:#888;font-size:11px;">
                    <div>状态: <span id="rag-rpg-settings-status">未知</span></div>
                    <button id="rag-rpg-test-btn" class="menu_button" style="margin-top:4px;">测试连接</button>
                </div>
            </div>
        </div>
    </div>`;
};

console.log(`[${EXTENSION_NAME}] 扩展已加载, 等待初始化...`);

function onSettingsChange() {
    const s = getSettings();
    s.auto_ingest = $('#rag-rpg-auto-ingest').prop('checked');
    s.auto_query = $('#rag-rpg-auto-query').prop('checked');
    s.inject_constraints = $('#rag-rpg-inject-constraints').prop('checked');
    s.show_constraints_panel = $('#rag-rpg-show-panel').prop('checked');
    s.debug_mode = $('#rag-rpg-debug-mode').prop('checked');
    s.api_url = String($('#rag-rpg-api-url').val() || defaultSettings.api_url);
    s.api_key = String($('#rag-rpg-api-key').val() || defaultSettings.api_key);
    s.query_turns = parseInt($('#rag-rpg-query-turns').val()) || defaultSettings.query_turns;
    if (s.api_url.endsWith('/')) s.api_url = s.api_url.slice(0, -1);
    saveSettingsDebounced();
    alwaysLog('设置已更新');
}

function registerSettingsHandlers() {
    $(document).on('change', '#rag-rpg-auto-ingest', onSettingsChange);
    $(document).on('change', '#rag-rpg-auto-query', onSettingsChange);
    $(document).on('change', '#rag-rpg-inject-constraints', onSettingsChange);
    $(document).on('change', '#rag-rpg-show-panel', onSettingsChange);
    $(document).on('change', '#rag-rpg-debug-mode', onSettingsChange);
    $(document).on('input', '#rag-rpg-api-url', onSettingsChange);
    $(document).on('input', '#rag-rpg-api-key', onSettingsChange);
    $(document).on('input', '#rag-rpg-query-turns', onSettingsChange);
    $(document).on('click', '#rag-rpg-test-btn', async () => {
        const statusEl = document.getElementById('rag-rpg-settings-status');
        statusEl.textContent = '测试中...';
        try {
            const resp = await fetch(`${getSettings().api_url}/api/health`, {
                headers: { 'X-API-Key': getSettings().api_key }
            });
            statusEl.textContent = resp.ok ? '✅ 连接成功' : `❌ HTTP ${resp.status}`;
        } catch (e) {
            statusEl.textContent = `❌ ${e.message}`;
        }
    });
}

// ─── API 调用 ──────────────────────────────────────────────

async function callApi(endpoint, data) {
    const settings = getSettings();
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), API_TIMEOUT);

    try {
        const response = await fetch(`${settings.api_url}${endpoint}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-API-Key': settings.api_key,
                'X-Game-Profile': settings.game_profile,
            },
            body: JSON.stringify(data),
            signal: controller.signal,
        });
        clearTimeout(timeoutId);
        if (!response.ok) {
            const errText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errText}`);
        }
        return await response.json();
    } catch (error) {
        if (error.name === 'AbortError') {
            console.warn(`[${EXTENSION_NAME}] API 请求超时: ${endpoint}`);
        } else {
            console.warn(`[${EXTENSION_NAME}] API 请求失败: ${endpoint}`, error.message);
        }
        return null;
    }
}

async function callApiGet(endpoint) {
    const settings = getSettings();
    try {
        const response = await fetch(`${settings.api_url}${endpoint}`, {
            method: 'GET',
            headers: {
                'X-API-Key': settings.api_key,
                'X-Game-Profile': settings.game_profile,
            },
        });
        if (!response.ok) return null;
        return await response.json();
    } catch (error) {
        return null;
    }
}

// ─── 约束面板展示 ─────────────────────────────────────────

function updateConstraintsPanel(result) {
    const panel = document.getElementById('rag-rpg-panel');
    const list = document.getElementById('rag-rpg-constraints-list');
    if (!panel || !list) return;

    const constraints = result?.active_constraints || [];
    const totalHits = result?.total_hits || 0;

    if (constraints.length === 0) {
        list.innerHTML = `<div style="color:#666;text-align:center;padding:8px 0;">
            ${totalHits > 0 ? `查找到 ${totalHits} 条记忆` : '暂无约束'}
        </div>`;
        return;
    }

    const typeLabels = {
        skill: '⚔️ 技能', mechanic: '🔧 机制',
        setting: '🌍 设定', plot: '📜 剧情',
        dialogue: '💬 记忆',
    };

    list.innerHTML = constraints.map(c => {
        const label = typeLabels[c.type] || `📌 ${c.type}`;
        return `<div style="padding:4px 0;border-bottom:1px solid #2a2a3a;">
            <div style="font-size:11px;color:#8ab4f8;">${label}
                <span style="color:#666;margin-left:6px;">
                    相关度:${c.score.toFixed(2)}</span>
            </div>
            <div style="color:#bbb;font-size:11px;padding:2px 0 0 10px;">
                ${c.content.length > 80 ? c.content.substring(0, 80) + '...' : c.content}
            </div>
        </div>`;
    }).join('');
}

async function refreshConstraintsPanel() {
    const result = await callApiGet('/api/constraints/current');
    if (result) {
        updateConstraintsPanel(result);
    }
}

function updateProfileBadge() {
    const badge = document.getElementById('rag-rpg-profile-badge');
    if (!badge) return;
    const settings = getSettings();
    const context = getContext();
    const charName = context.name2 || context.name || '';
    const label = charName ? `${charName} (${settings.game_profile})` : settings.game_profile;
    badge.textContent = `📂 ${label}`;
}

function togglePanel(show) {
    const panel = document.getElementById('rag-rpg-panel');
    if (panel) {
        panel.style.display = show ? 'block' : 'none';
        if (show) updateProfileBadge();
    }
}

// ─── 事件处理 ──────────────────────────────────────────────

let turnCounter = 0;
let lastProcessedMessage = '';
let currentSessionId = '';
let pendingConstraint = '';

function generateSessionId() {
    const context = getContext();
    const charName = context.name2 || context.name || 'unknown';
    const now = new Date();
    const ts = now.getFullYear()
        + String(now.getMonth() + 1).padStart(2, '0')
        + String(now.getDate()).padStart(2, '0') + '_'
        + String(now.getHours()).padStart(2, '0')
        + String(now.getMinutes()).padStart(2, '0');
    const slug = charName.replace(/[^a-zA-Z0-9\u4e00-\u9fff_-]/g, '_').substring(0, 30);
    return `${slug}_${ts}`;
}

async function handleMessageReceived() {
    const settings = getSettings();
    if (!settings.auto_ingest) return;

    const context = getContext();
    const chat = context.chat;
    if (!chat || chat.length === 0) return;

    const message = chat[chat.length - 1];
    const mes = String(message.mes || '').trim();
    if (!mes || mes.length < 3) return;

    if (mes === lastProcessedMessage) return;
    lastProcessedMessage = mes;

    turnCounter++;
    const speaker = message.is_user ? 'user' : 'ai';
    const name = String(message.name || (speaker === 'user' ? '用户' : 'AI'));

    log(`捕获对话 Turn#${turnCounter} | ${speaker} | ${name}`);

    if (!currentSessionId) {
        currentSessionId = generateSessionId();
        log(`兜底生成 session_id: ${currentSessionId}`);
    }

    // ingest 和 query 并发执行，互不阻塞
    const ingestPromise = callApi('/api/dialogue/ingest', {
        speaker: speaker,
        name: name,
        content: mes,
        turn: turnCounter,
        timestamp: new Date().toISOString(),
        session_id: currentSessionId,
    });

    let queryPromise = Promise.resolve(null);
    if (speaker === 'user') {
        alwaysLog('用户消息已发送，并发预查约束...');
        const context = getContext();
        const chat = context.chat;
        const recentTurns = (chat || []).slice(-getSettings().query_turns);
        const dialogueContext = recentTurns.map(m => ({
            speaker: m.is_user ? 'user' : 'ai',
            content: String(m.mes || ''),
        }));
        queryPromise = callApi('/api/dialogue/query', {
            context: dialogueContext,
            max_results: getSettings().max_results,
            generate_constraint: getSettings().inject_constraints,
        });
    }

    const [ingestResult, queryResult] = await Promise.all([ingestPromise, queryPromise]);

    if (!ingestResult) {
        alwaysLog('ingest 返回空（可能超时）');
    }

    if (speaker === 'user' && queryResult && queryResult.constraint_text) {
        pendingConstraint = queryResult.constraint_text;
        alwaysLog(`约束已预取 (${queryResult.total_hits} 条命中)`);
        const context = getContext();
        context.setExtensionPrompt(EXTENSION_NAME, queryResult.constraint_text);
        alwaysLog('约束已直接注入 setExtensionPrompt');
        if (getSettings().show_constraints_panel && queryResult.active_constraints) {
            updateConstraintsPanel(queryResult);
            togglePanel(true);
        }
    } else if (speaker === 'user') {
        pendingConstraint = '';
        alwaysLog('预查约束返回空');
    }
}

async function handleGenerationBefore() {
    const settings = getSettings();
    const context = getContext();

    // 优先使用预取的约束
    if (pendingConstraint) {
        context.setExtensionPrompt(EXTENSION_NAME, pendingConstraint);
        alwaysLog('使用预取约束注入 setExtensionPrompt');
        pendingConstraint = '';
        return;
    }

    // 兜底：尝试实时查询
    alwaysLog('无预取约束，实时查询...');
    const chat = context.chat;
    if (!chat || chat.length < 2) {
        alwaysLog('对话历史不足2轮, 跳过查询');
        return;
    }

    const recentTurns = chat.slice(-settings.query_turns);
    const dialogueContext = recentTurns.map(m => ({
        speaker: m.is_user ? 'user' : 'ai',
        content: String(m.mes || ''),
    }));

    const result = await callApi('/api/dialogue/query', {
        context: dialogueContext,
        max_results: settings.max_results,
        generate_constraint: settings.inject_constraints,
    });

    if (result && result.constraint_text) {
        context.setExtensionPrompt(EXTENSION_NAME, result.constraint_text);
        alwaysLog(`实时查询约束已注入 (${result.total_hits} 条命中)`);
        if (settings.show_constraints_panel && result.active_constraints) {
            updateConstraintsPanel(result);
            togglePanel(true);
        }
    } else {
        alwaysLog('实时查询约束返回空');
    }
}

async function onChatChanged() {
    turnCounter = 0;
    lastProcessedMessage = '';
    currentSessionId = generateSessionId();
    log(`会话已切换，新 session_id: ${currentSessionId}`);

    const settings = getSettings();
    if (settings.auto_profile) {
        const context = getContext();
        const charId = context.characterId;
        const charName = context.name2 || context.name || '';

        let newProfile = 'default';
        if (charId && charId !== 'undefined' && charId !== '') {
            newProfile = `char_${String(charId).replace(/[^a-zA-Z0-9_-]/g, '_')}`;
        }

        if (settings.game_profile !== newProfile) {
            settings.game_profile = newProfile;
            saveSettingsDebounced();
            log(`自动切换数据存储: ${newProfile}${charName ? ` (${charName})` : ''}`);
        }
    }

    togglePanel(false);
    updateProfileBadge();
}

// ─── 初始化 ────────────────────────────────────────────────

jQuery(async () => {
    loadSettings();

    // 强制开启核心功能（不依赖设置面板）
    const s = getSettings();
    s.auto_query = true;
    s.inject_constraints = true;
    s.auto_ingest = true;
    saveSettingsDebounced();
    alwaysLog('核心设置已强制开启: auto_query=true, inject_constraints=true');

    // 注入面板 HTML
    const panelHtml = getPanelHtml();
    $('body').append(panelHtml);

    // 显示初始 profile
    updateProfileBadge();

    // 初始化时生成 session_id（覆盖已打开对话未触发 CHAT_CHANGED 的情况）
    currentSessionId = generateSessionId();
    log(`初始化 session_id: ${currentSessionId}`);

    // 绑定面板事件
    $(document).on('click', '#rag-rpg-close-btn', () => togglePanel(false));
    $(document).on('click', '#rag-rpg-refresh-btn', refreshConstraintsPanel);

    // 绑定 SillyTavern 事件
    eventSource.on(event_types.MESSAGE_RECEIVED, handleMessageReceived);
    eventSource.on(event_types.MESSAGE_SENT, handleMessageReceived);
    eventSource.on(event_types.GENERATION_BEFORE_COMMANDS, handleGenerationBefore);
    eventSource.on(event_types.CHAT_CHANGED, onChatChanged);

    alwaysLog(`v2.0 已加载 → ${s.api_url}`);
});
