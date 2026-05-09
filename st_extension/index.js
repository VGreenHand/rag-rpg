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
const API_TIMEOUT = 5000;

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

    await callApi('/api/dialogue/ingest', {
        speaker: speaker,
        name: name,
        content: mes,
        turn: turnCounter,
        timestamp: new Date().toISOString(),
        session_id: currentSessionId,
    });
}

async function handleGenerationBefore() {
    const settings = getSettings();
    if (!settings.auto_query) return;

    const context = getContext();
    const chat = context.chat;
    if (!chat || chat.length < 2) return;

    const recentTurns = chat.slice(-settings.query_turns);
    const dialogueContext = recentTurns.map(m => ({
        speaker: m.is_user ? 'user' : 'ai',
        content: String(m.mes || ''),
    }));

    log(`查询向量库 (最近${dialogueContext.length}轮对话)...`);

    const result = await callApi('/api/dialogue/query', {
        context: dialogueContext,
        max_results: settings.max_results,
        generate_constraint: settings.inject_constraints,
    });

    if (result) {
        log(`查询完成: ${result.total_hits} 条命中`);

        if (settings.show_constraints_panel && result.active_constraints) {
            updateConstraintsPanel(result);
            togglePanel(true);
        }

        if (result.constraint_text) {
            context.setExtensionPrompt(EXTENSION_NAME, result.constraint_text);
        } else if (result.formatted) {
            context.setExtensionPrompt(EXTENSION_NAME, result.formatted);
        }
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

    console.log(`[${EXTENSION_NAME}] v2.0 已加载 → ${getSettings().api_url}`);
});
