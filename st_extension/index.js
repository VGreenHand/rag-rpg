/**
 * RAG-RPG 记忆引擎 - SillyTavern 扩展
 * 功能：
 *   1. 自动捕获每轮对话并发送至 Python 后端处理
 *   2. AI 生成前自动查询向量库获取剧情约束
 *   3. 将约束文本注入系统提示词，实现记忆驱动的剧情引导
 *
 * 安装方式：将 st_extension 文件夹复制到 SillyTavern 的
 *   data/default-user/extensions/ 目录下，重命名为 RAG-RPG
 * 然后重启 SillyTavern 并在扩展面板中启用。
 */

import { eventSource, event_types } from '../../../../script.js';
import { extension_settings, getContext } from '../../../extensions.js';
import { saveSettingsDebounced } from '../../../../script.js';

const EXTENSION_NAME = 'RAG-RPG';
const API_TIMEOUT = 5000;

const defaultSettings = {
    api_url: 'http://127.0.0.1:8765',
    api_key: 'rag-rpg-local',
    auto_ingest: true,
    auto_query: true,
    inject_constraints: true,
    query_turns: 6,
    max_results: 3,
    debug_mode: false,
};

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

let turnCounter = 0;
let lastProcessedMessage = '';

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

    log(`捕获对话 Turn#${turnCounter} | ${speaker} | ${name} | ${mes.substring(0, 40)}...`);

    await callApi('/api/dialogue/ingest', {
        speaker: speaker,
        name: name,
        content: mes,
        turn: turnCounter,
        timestamp: new Date().toISOString(),
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

    if (result && result.constraint_text) {
        log(`已注入剧情约束 (${result.total_hits} 条命中)`);
        context.setExtensionPrompt(EXTENSION_NAME, result.constraint_text);
    } else if (result && result.formatted) {
        log(`已注入检索记忆 (${result.total_hits} 条命中)`);
        context.setExtensionPrompt(EXTENSION_NAME, result.formatted);
    }
}

async function onChatChanged() {
    turnCounter = 0;
    lastProcessedMessage = '';
    log('对话已切换，计数器重置');
}

jQuery(async () => {
    loadSettings();

    eventSource.on(event_types.MESSAGE_RECEIVED, handleMessageReceived);
    eventSource.on(event_types.MESSAGE_SENT, handleMessageReceived);
    eventSource.on(event_types.GENERATION_BEFORE_COMMANDS, handleGenerationBefore);
    eventSource.on(event_types.CHAT_CHANGED, onChatChanged);

    console.log(`[${EXTENSION_NAME}] 扩展已加载 → ${getSettings().api_url}`);
});
