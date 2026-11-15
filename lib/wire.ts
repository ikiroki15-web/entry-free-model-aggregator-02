// The client wires, the one upstream wire, and the model field.
//
// The trial's client calls the entry as an OpenAI service (POST /v1/chat/completions) or an
// Anthropic one (POST /v1/messages), with a JSON body or with server-sent events. The entry dials
// the grader's gateway, an OpenAI-compatible endpoint, so every prompt goes upstream as a Chat
// Completions request and the Anthropic path is translated both ways:
//
//   - the OpenAI path passes the client's body (with the dialled model's id) to the gateway and
//     relays the gateway's JSON or SSE answer, with every `model` field set to the id this entry
//     dialled;
//   - the Anthropic path rewrites the request to Chat Completions (system prompt, image blocks,
//     tool definitions, tool results and tool choice included) and rewrites the gateway's answer
//     into a Messages object, or into the message_start / content_block_start / content_block_delta
//     / content_block_stop / message_delta / message_stop event order, with the stop reason and the
//     usage translated. Tool calls become tool_use blocks and input_json_delta events.
//   - a stream that ends without usage or without [DONE] is completed on this side: the OpenAI
//     stream always ends with exactly one [DONE] and carries a final usage chunk when the upstream
//     sent none, and the Anthropic stream's message_start input count is written once the usage
//     chunk has arrived, so neither wire ever reports no usage. Usage the upstream did not report
//     is estimated from the prompt and answer text, about one token per four characters.
//
// THE MODEL FIELD
//   The house records every gateway call and scores the model that really answered, so a response's
//   `model` must be the model this entry dialled. OpenRouter's own answer names a free model without
//   the gateway's `openrouter/` prefix; these functions put the dialled id back on every answer and
//   every streamed chunk. They never make a model up: the id came from the brief and went to the
//   gateway in the request's `model`.
//
// Everything here is a pure function of its arguments, except the stream relays: no clock and no
// randomness, so the same body and the same gateway answer give the same bytes on a replay. The two
// model listings use fixed timestamps for the same reason: a `created` that moved between runs would
// be a nondeterministic document.

import { isRecord, type Json } from './providers.ts';

/** One model the entry serves, as GET /v1/models lists it. */
export interface ServedModel {
  id: string;
  provider: string;
  display_name: string;
}

/** A fixed epoch second for every listed model's `created`: nothing in the reply reads the clock. */
const FIXED_OPENAI_CREATED = 1767225600;
/** A fixed instant for every listed model's `created_at`: nothing in the reply reads the clock. */
const FIXED_ANTHROPIC_CREATED_AT = '2026-01-01T00:00:00Z';

/** One OpenAI list entry for one served model. */
function openAIListEntry(model: ServedModel): Json {
  return { id: model.id, object: 'model', created: FIXED_OPENAI_CREATED, owned_by: model.provider };
}

/** GET /v1/models on an OpenAI-shaped request: the models this entry serves, in OpenAI's list shape. */
export function openAIModelList(models: readonly ServedModel[]): Json {
  const data: Json[] = [];
  for (const model of models) data.push(openAIListEntry(model));
  return { object: 'list', data };
}

/** One Anthropic list entry for one served model. */
function anthropicListEntry(model: ServedModel): Json {
  return { type: 'model', id: model.id, display_name: model.display_name, created_at: FIXED_ANTHROPIC_CREATED_AT };
}

/** GET /v1/models on an Anthropic-shaped request (an x-api-key header): Anthropic's model list shape. */
export function anthropicModelList(models: readonly ServedModel[]): Json {
  const data: Json[] = [];
  for (const model of models) data.push(anthropicListEntry(model));
  return { data, has_more: false };
}

/** An error in Anthropic's shape (the Messages API's error envelope). */
export function anthropicError(type: string, message: string): Json {
  return { type: 'error', error: { type, message } };
}

/** An error in OpenAI's shape (Chat Completions). */
export function openAIError(message: string, type: string, code: string | null): Json {
  return { error: { message, type, code } };
}

/** A number field, or 0 when the value is not a finite number. */
function numberOrZero(value: unknown): number {
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
}

/** A string field, or '' when the value is not a string. */
function stringOrEmpty(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

/**
 * The text of a content value: the string itself, or the text blocks of a block array joined with
 * newlines. Anything else holds no text.
 */
function textOf(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const pieces: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') pieces.push(block.text);
  }
  return pieces.join('\n');
}

/** One token estimate for a text: about one token per four characters, at least one. Deterministic: the same text always estimates the same count, so a replay reproduces it. */
export function estimateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

/** The prompt text of an OpenAI request: its messages' text, joined. Used only to estimate usage when the upstream reports none. */
export function openAIPromptText(body: Json): string {
  const parts: string[] = [];
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const message of messages) {
    if (!isRecord(message)) continue;
    const text = textOf(message.content);
    if (text !== '') parts.push(text);
  }
  return parts.join('\n');
}

/** The prompt text of an Anthropic request: its system prompt and its messages' text, joined. Used only to estimate usage when the upstream reports none. */
export function anthropicPromptText(body: Json): string {
  const parts: string[] = [];
  const system = textOf(body.system);
  if (system !== '') parts.push(system);
  const messages = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of messages) {
    if (!isRecord(raw)) continue;
    const text = textOf(raw.content);
    if (text !== '') parts.push(text);
  }
  return parts.join('\n');
}

/**
 * The token counts to report: the upstream counts when they are whole (an input above zero and
 * an output above zero), else estimates from the prompt text and the answer text. An upstream
 * that reports no usage would otherwise leave every prompt failing the usage check, which needs
 * a whole input count of at least 0 and an output count of at least 1.
 */
function usageOrEstimate(usage: unknown, promptText: string, answerText: string): { promptTokens: number; completionTokens: number } {
  const record = isRecord(usage) ? usage : {};
  const prompt = numberOrZero(record.prompt_tokens ?? record.input_tokens);
  const completion = numberOrZero(record.completion_tokens ?? record.output_tokens);
  return {
    promptTokens: prompt > 0 ? prompt : estimateTokens(promptText),
    completionTokens: completion > 0 ? completion : estimateTokens(answerText),
  };
}

/** An image block's source as an OpenAI image_url part, or null when the source is not usable. */
function imagePartFromSource(source: Json): Json | null {
  if (source.type === 'base64' && typeof source.data === 'string') {
    const media = typeof source.media_type === 'string' ? source.media_type : 'image/png';
    return { type: 'image_url', image_url: { url: `data:${media};base64,${source.data}` } };
  }
  if (source.type === 'url' && typeof source.url === 'string') {
    return { type: 'image_url', image_url: { url: source.url } };
  }
  return null;
}

/** One Anthropic content block as an OpenAI content part: text or an inline image; null for the rest. */
function partOf(block: Json): Json | null {
  if (block.type === 'text' && typeof block.text === 'string') return { type: 'text', text: block.text };
  if (block.type === 'image' && isRecord(block.source)) return imagePartFromSource(block.source);
  return null;
}

/** A tool_result block's content as text: its text blocks, or its JSON when it holds something else. */
function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = textOf(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** A tool call's arguments as an object: the gateway's JSON string, or {} when it is not one. */
function argumentsOf(value: unknown): Json {
  if (isRecord(value)) return value;
  if (typeof value === 'string' && value.trim() !== '') {
    try {
      const parsed: unknown = JSON.parse(value);
      if (isRecord(parsed)) return parsed;
    } catch {
      // Not JSON: an empty input.
    }
  }
  return {};
}

/** OpenAI's finish_reason as Anthropic's stop_reason. */
export function stopReasonOf(finish: unknown): string {
  switch (finish) {
    case 'length':
      return 'max_tokens';
    case 'tool_calls':
    case 'function_call':
      return 'tool_use';
    case 'content_filter':
      return 'refusal';
    default:
      return 'end_turn';
  }
}

/** The system message for a system prompt, or null when the request carries no system text. */
function systemMessage(body: Json): Json | null {
  const system = textOf(body.system);
  return system === '' ? null : { role: 'system', content: system };
}

/** One Anthropic tool_use block as an OpenAI tool_call. */
function toolCallFromBlock(block: Json): Json {
  return {
    id: stringOrEmpty(block.id),
    type: 'function',
    function: {
      name: stringOrEmpty(block.name),
      arguments: JSON.stringify(isRecord(block.input) ? block.input : {}),
    },
  };
}

/** An assistant message from its blocks: its text, and its tool calls when it made any. */
function assistantMessageFromBlocks(blocks: Json[]): Json {
  const toolCalls: Json[] = [];
  for (const block of blocks) {
    if (block.type === 'tool_use') toolCalls.push(toolCallFromBlock(block));
  }
  const text = textOf(blocks);
  const message: Json = { role: 'assistant', content: text !== '' ? text : null };
  if (toolCalls.length > 0) message.tool_calls = toolCalls;
  return message;
}

/**
 * One user-side message's blocks as OpenAI parts, pushing any tool_result block straight out as its
 * own tool message (a tool result is a message of its own, never a part of the user message).
 */
function userPartsAndToolMessages(blocks: Json[]): { parts: Json[]; toolMessages: Json[] } {
  const parts: Json[] = [];
  const toolMessages: Json[] = [];
  for (const block of blocks) {
    if (block.type === 'tool_result') {
      toolMessages.push({
        role: 'tool',
        tool_call_id: stringOrEmpty(block.tool_use_id),
        content: toolResultText(block.content),
      });
      continue;
    }
    const part = partOf(block);
    if (part !== null) parts.push(part);
  }
  return { parts, toolMessages };
}

/** One Anthropic message as its OpenAI messages: a user message and any tool messages it carries. */
function messagesFromAnthropicMessage(raw: unknown): Json[] {
  if (!isRecord(raw)) return [];
  const content = raw.content;
  const role = raw.role === 'assistant' ? 'assistant' : 'user';
  if (typeof content === 'string') return [{ role, content }];
  if (!Array.isArray(content)) return [];
  const blocks: Json[] = [];
  for (const block of content) {
    if (isRecord(block)) blocks.push(block);
  }
  if (role === 'assistant') return [assistantMessageFromBlocks(blocks)];
  const { parts, toolMessages } = userPartsAndToolMessages(blocks);
  const out: Json[] = [];
  // Tool results are emitted as their own messages before the user message of the same turn:
  // the original translation pushes each tool_result the moment it is met and the user message
  // only once all of its parts are known.
  out.push(...toolMessages);
  if (parts.length > 0) out.push({ role: 'user', content: parts });
  return out;
}

/** One Anthropic tool definition as an OpenAI function tool, or null when it names no tool. */
function toolDefinitionFromBlock(raw: unknown): Json | null {
  if (!isRecord(raw) || typeof raw.name !== 'string') return null;
  return {
    type: 'function',
    function: {
      name: raw.name,
      description: stringOrEmpty(raw.description),
      parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
    },
  };
}

/** The request's tool definitions as OpenAI tools, or null when it declares none usable. */
function toolsFromBody(body: Json): Json[] | null {
  if (!Array.isArray(body.tools)) return null;
  const tools: Json[] = [];
  for (const raw of body.tools) {
    const tool = toolDefinitionFromBlock(raw);
    if (tool !== null) tools.push(tool);
  }
  return tools.length > 0 ? tools : null;
}

/** The request's tool_choice as OpenAI spells it, or undefined when it states none. */
function toolChoiceFromBody(body: Json): Json | string | undefined {
  const choice = body.tool_choice;
  if (!isRecord(choice)) return undefined;
  if (choice.type === 'auto') return 'auto';
  if (choice.type === 'any') return 'required';
  if (choice.type === 'tool' && typeof choice.name === 'string') {
    return { type: 'function', function: { name: choice.name } };
  }
  return undefined;
}

/** The scalar generation options that pass through unchanged: cap, temperature, top_p, stop. */
function copyScalarOptions(body: Json, out: Json): void {
  if (typeof body.max_tokens === 'number') out.max_tokens = body.max_tokens;
  if (typeof body.temperature === 'number') out.temperature = body.temperature;
  if (typeof body.top_p === 'number') out.top_p = body.top_p;
  if (Array.isArray(body.stop_sequences) && body.stop_sequences.length > 0) out.stop = body.stop_sequences;
}

/**
 * An Anthropic Messages request as an OpenAI Chat Completions request that names `model`: the system
 * prompt becomes a system message, image blocks become image_url data URLs, tool_use blocks become
 * tool_calls, tool_result blocks become tool messages, tools and tool_choice are rewritten, and the
 * output cap, temperature, top_p, stop sequences and streaming pass through.
 */
export function anthropicToOpenAI(body: Json, model: string): Json {
  const messages: Json[] = [];
  const system = systemMessage(body);
  if (system !== null) messages.push(system);
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) messages.push(...messagesFromAnthropicMessage(raw));

  const out: Json = { model, messages };
  copyScalarOptions(body, out);
  const tools = toolsFromBody(body);
  if (tools !== null) out.tools = tools;
  const toolChoice = toolChoiceFromBody(body);
  if (toolChoice !== undefined) out.tool_choice = toolChoice;
  if (body.stream === true) {
    out.stream = true;
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** The text content of a Chat Completions message: its string, or its text parts. */
function messageText(message: Json): string {
  if (typeof message.content === 'string') return message.content;
  return textOf(message.content);
}

/** One OpenAI tool_call as an Anthropic tool_use block. */
function toolUseFromCall(call: unknown): Json | null {
  if (!isRecord(call) || !isRecord(call.function)) return null;
  return {
    type: 'tool_use',
    id: typeof call.id === 'string' ? call.id : 'toolu_unknown',
    name: stringOrEmpty(call.function.name),
    input: argumentsOf(call.function.arguments),
  };
}

/** A Chat Completions message's content blocks: its text, then its tool calls. */
function contentBlocksFromMessage(message: Json): Json[] {
  const content: Json[] = [];
  const text = messageText(message);
  if (text !== '') content.push({ type: 'text', text });
  const calls = Array.isArray(message.tool_calls) ? message.tool_calls : [];
  for (const call of calls) {
    const block = toolUseFromCall(call);
    if (block !== null) content.push(block);
  }
  if (content.length === 0) content.push({ type: 'text', text: '' });
  return content;
}

/** A Chat Completions answer as an Anthropic Messages object. `model` is the id the entry dialled. `promptText` feeds the usage fallback when the upstream reports no usage. */
export function openAIToAnthropic(body: Json, model: string, promptText = ''): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const usage = usageOrEstimate(body.usage, promptText, messageText(message));
  return {
    id: `msg_${typeof body.id === 'string' ? body.id : 'aggregate'}`,
    type: 'message',
    role: 'assistant',
    model,
    content: contentBlocksFromMessage(message),
    stop_reason: stopReasonOf(choice.finish_reason),
    stop_sequence: null,
    usage: { input_tokens: usage.promptTokens, output_tokens: usage.completionTokens },
  };
}

/**
 * A Chat Completions answer with `model` set to the id the entry dialled: the one truthful claim.
 * Its usage is the upstream counts when they are whole, else estimates from the prompt and the
 * answer, so a prompt is never left with no usage at all. `promptText` feeds that fallback.
 */
export function openAIJsonWithModel(body: Json, model: string, promptText = ''): Json {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const usage = usageOrEstimate(body.usage, promptText, messageText(message));
  const total = numberOrZero(isRecord(body.usage) ? body.usage.total_tokens : undefined);
  return {
    ...body,
    model,
    usage: {
      prompt_tokens: usage.promptTokens,
      completion_tokens: usage.completionTokens,
      total_tokens: total > 0 ? total : usage.promptTokens + usage.completionTokens,
    },
  };
}

/** One server-sent event: `event: <name>` when named, then the JSON data line, then the blank line. */
export function sseEvent(event: string | null, data: unknown): string {
  return `${event === null ? '' : `event: ${event}\n`}data: ${JSON.stringify(data)}\n\n`;
}

/**
 * The SSE grammar, one read at a time: frames end on a blank line, and a frame cut across reads is
 * held until its end arrives. `feed` takes freshly decoded text and returns the frames that ended
 * in it; `flush` returns whatever is left when the body ends, if it holds anything at all.
 * Carriage returns are normalised to line feeds on the way in, as the SSE grammar does.
 */
class SseFrameParser {
  private buffer = '';

  feed(text: string): string[] {
    this.buffer += text.replace(/\r\n/g, '\n');
    const frames: string[] = [];
    let at = this.buffer.indexOf('\n\n');
    while (at !== -1) {
      frames.push(this.buffer.slice(0, at));
      this.buffer = this.buffer.slice(at + 2);
      at = this.buffer.indexOf('\n\n');
    }
    return frames;
  }

  flush(): string | null {
    if (this.buffer.trim() === '') return null;
    const rest = this.buffer;
    this.buffer = '';
    return rest;
  }
}

/** The data payloads of one frame's lines, in order: the text after each `data:` prefix. */
function dataPayloads(frame: string): string[] {
  const payloads: string[] = [];
  for (const line of frame.split('\n')) {
    const trimmed = line.trim();
    if (trimmed.startsWith('data:')) payloads.push(trimmed.slice(5).trim());
  }
  return payloads;
}

/** One SSE line with its data payload's `model` set to `model`; non-data and non-JSON lines pass through. */
function lineWithModel(line: string, model: string): string {
  const trimmed = line.trim();
  if (!trimmed.startsWith('data:')) return line;
  const payload = trimmed.slice(5).trim();
  if (payload === '' || payload === '[DONE]') return line;
  try {
    const chunk: unknown = JSON.parse(payload);
    if (isRecord(chunk)) return `data: ${JSON.stringify({ ...chunk, model })}`;
  } catch {
    // Not JSON: relay the line as it came.
  }
  return line;
}

/** One SSE frame with every data payload's `model` set to `model`, and its blank line kept. */
function frameWithModel(frame: string, model: string): string {
  const lines: string[] = [];
  for (const line of frame.split('\n')) lines.push(lineWithModel(line, model));
  return `${lines.join('\n')}\n\n`;
}

/** Read an upstream body frame by frame, yielding each frame with `rewrite` applied to it. */
async function* readFrames(
  upstream: ReadableStream<Uint8Array>,
  rewrite: (frame: string) => string[],
): AsyncGenerator<string> {
  const reader = upstream.getReader();
  const decoder = new TextDecoder();
  const parser = new SseFrameParser();
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      for (const frame of parser.feed(decoder.decode(value, { stream: true }))) {
        yield* rewrite(frame);
      }
    }
    // Flush the decoder itself: a multi-byte character split across the last reads is only
    // released by a final decode with no stream flag.
    for (const frame of parser.feed(decoder.decode())) {
      yield* rewrite(frame);
    }
    const rest = parser.flush();
    if (rest !== null) yield* rewrite(rest);
  } finally {
    reader.releaseLock();
  }
}

/** One SSE frame with its `data: [DONE]` lines taken out, or null when nothing is left. A held [DONE] is re-emitted once at the end, after any synthetic usage chunk, so the client never sees two and never sees an event after one. */
function stripDoneLines(frame: string): string | null {
  const kept = frame.split('\n').filter((line) => line.trim() !== 'data: [DONE]');
  if (kept.join('').trim() === '') return null;
  return kept.join('\n');
}

/**
 * A Chat Completions SSE stream as it came, frame by frame, every data payload's `model` the id the
 * entry dialled. Chat Completions names the model on its chunks; the client sees the id the entry
 * dialled, never the provider's spelling. The stream always ends with exactly one `data: [DONE]`:
 * an upstream [DONE] is held and re-emitted at the end, and one is added when the upstream closed
 * without one, so a stream the provider ended early is still a well-formed OpenAI stream. When no
 * chunk carried usage, one final chunk carries estimates from the prompt and the answer text, before
 * the [DONE], so the response is never left with no usage at all.
 */
export async function* openAIStreamWithModel(upstream: ReadableStream<Uint8Array>, model: string, promptText = ''): AsyncGenerator<string> {
  let sawDone = false;
  let sawUsage = false;
  let answerText = '';
  const rewrite = (frame: string): string[] => {
    for (const payload of dataPayloads(frame)) {
      if (payload === '[DONE]') {
        sawDone = true;
        continue;
      }
      let chunk: unknown;
      try {
        chunk = JSON.parse(payload);
      } catch {
        continue;
      }
      if (!isRecord(chunk)) continue;
      if (isRecord(chunk.usage)) sawUsage = true;
      const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
      const choice = isRecord(choices[0]) ? choices[0] : null;
      if (choice !== null && isRecord(choice.delta) && typeof choice.delta.content === 'string') {
        answerText += choice.delta.content;
      }
    }
    const stripped = stripDoneLines(frame);
    return stripped === null ? [] : [frameWithModel(stripped, model)];
  };
  yield* readFrames(upstream, rewrite);
  if (!sawUsage) {
    const usage = usageOrEstimate(null, promptText, answerText);
    yield `data: ${JSON.stringify({
      id: 'chatcmpl-aggregate',
      object: 'chat.completion.chunk',
      created: FIXED_OPENAI_CREATED,
      model,
      choices: [],
      usage: {
        prompt_tokens: usage.promptTokens,
        completion_tokens: usage.completionTokens,
        total_tokens: usage.promptTokens + usage.completionTokens,
      },
    })}\n\n`;
  }
  yield 'data: [DONE]\n\n';
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence, held as state: message_start on
 * the first chunk, content_block_start/delta/stop per text block or tool call, then message_delta
 * with the stop reason and usage, then message_stop. `model` is the id the entry dialled, named
 * back to the client. The stop reason is held until the stream's usage chunk and [DONE] have
 * arrived: OpenAI sends the usage after the finish_reason, and Anthropic's message_delta carries
 * the output tokens.
 */
class AnthropicStreamTranslator {
  private readonly model: string;
  private started = false;
  private finished = false;
  private pendingStop: string | null = null;
  private textIndex = -1;
  private nextIndex = 0;
  private readonly toolIndexes = new Map<number, number>();
  private inputTokens = 0;
  private outputTokens = 0;
  private sawUsage = false;
  private answerText = '';

  constructor(model: string) {
    this.model = model;
  }

  /** The message_start event, once, on the first chunk that carries anything. */
  ensureStarted(): string[] {
    if (this.started) return [];
    this.started = true;
    return [
      sseEvent('message_start', {
        type: 'message_start',
        message: {
          id: 'msg_stream',
          type: 'message',
          role: 'assistant',
          model: this.model,
          content: [],
          stop_reason: null,
          stop_sequence: null,
          usage: { input_tokens: this.inputTokens, output_tokens: 0 },
        },
      }),
    ];
  }

  /** The content_block_stop events for every block still open, text first, then tool calls in order. */
  closeOpenBlocks(): string[] {
    const out: string[] = [];
    if (this.textIndex !== -1) {
      out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: this.textIndex }));
      this.textIndex = -1;
    }
    const indexes = [...this.toolIndexes.values()].sort((a, b) => a - b);
    for (const index of indexes) out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index }));
    this.toolIndexes.clear();
    return out;
  }

  /** The closing events for one stop reason, once: blocks stopped, then message_delta and message_stop. */
  finish(stopReason: string): string[] {
    if (this.finished) return [];
    this.finished = true;
    return [
      ...this.closeOpenBlocks(),
      sseEvent('message_delta', { type: 'message_delta', delta: { stop_reason: stopReason, stop_sequence: null }, usage: { output_tokens: this.outputTokens } }),
      sseEvent('message_stop', { type: 'message_stop' }),
    ];
  }

  get isFinished(): boolean {
    return this.finished;
  }

  get heldStopReason(): string {
    return this.pendingStop ?? 'end_turn';
  }

  /**
   * The usage the closing events must carry: the upstream counts when a usage chunk arrived,
   * else estimates from the prompt and the answer text. The message_start event is emitted on
   * the first chunk, before any usage chunk can arrive, so its input count is only known at the
   * end; applyClosingUsage writes the true counts into the buffered events once the stream has
   * closed.
   */
  usageForClose(promptText: string): { inputTokens: number; outputTokens: number } {
    return {
      inputTokens: this.sawUsage ? this.inputTokens : estimateTokens(promptText),
      outputTokens: this.sawUsage ? Math.max(1, this.outputTokens) : estimateTokens(this.answerText),
    };
  }

  /** Rewrite the buffered message_start and message_delta events with the closing usage. */
  applyClosingUsage(events: string[], promptText: string): void {
    const usage = this.usageForClose(promptText);
    patchEventData(events, 'message_start', (data) => {
      if (isRecord(data.message) && isRecord(data.message.usage)) data.message.usage.input_tokens = usage.inputTokens;
    });
    patchEventData(events, 'message_delta', (data) => {
      if (isRecord(data.usage)) data.usage.output_tokens = usage.outputTokens;
    });
  }

  /** Open the text block, if it is not open already, and return its start event. */
  private openTextBlock(): string[] {
    if (this.textIndex !== -1) return [];
    this.textIndex = this.nextIndex;
    this.nextIndex += 1;
    return [
      sseEvent('content_block_start', { type: 'content_block_start', index: this.textIndex, content_block: { type: 'text', text: '' } }),
    ];
  }

  /** The events for one text delta: open the text block on its first piece, then the delta. */
  private textDeltaEvents(text: string): string[] {
    if (text === '') return [];
    this.answerText += text;
    return [
      ...this.openTextBlock(),
      sseEvent('content_block_delta', { type: 'content_block_delta', index: this.textIndex, delta: { type: 'text_delta', text } }),
    ];
  }

  /** Open a tool block for one upstream tool call and return the Anthropic index it was given. */
  private openToolBlock(upstreamIndex: number): number {
    const index = this.nextIndex;
    this.nextIndex += 1;
    this.toolIndexes.set(upstreamIndex, index);
    return index;
  }

  /** The events for one tool call delta: open its block on its first piece, then its argument delta. */
  private toolCallEvents(call: Json): string[] {
    if (!isRecord(call)) return [];
    const out: string[] = [];
    const upstreamIndex = typeof call.index === 'number' ? call.index : this.toolIndexes.size;
    let index = this.toolIndexes.get(upstreamIndex);
    if (index === undefined) {
      if (this.textIndex !== -1) {
        out.push(sseEvent('content_block_stop', { type: 'content_block_stop', index: this.textIndex }));
        this.textIndex = -1;
      }
      index = this.openToolBlock(upstreamIndex);
      const fn = isRecord(call.function) ? call.function : {};
      out.push(
        sseEvent('content_block_start', {
          type: 'content_block_start',
          index,
          content_block: {
            type: 'tool_use',
            id: typeof call.id === 'string' && call.id !== '' ? call.id : `toolu_${index}`,
            name: stringOrEmpty(fn.name),
            input: {},
          },
        }),
      );
    }
    const fn = isRecord(call.function) ? call.function : {};
    if (typeof fn.arguments === 'string' && fn.arguments !== '') {
      out.push(sseEvent('content_block_delta', { type: 'content_block_delta', index, delta: { type: 'input_json_delta', partial_json: fn.arguments } }));
    }
    return out;
  }

  /** One chunk's events: text deltas, tool call deltas, and the stop reason it may carry. */
  private chunkEvents(chunk: Json): string[] {
    const out: string[] = [];
    if (isRecord(chunk.usage)) {
      this.sawUsage = true;
      this.inputTokens = numberOrZero(chunk.usage.prompt_tokens);
      this.outputTokens = numberOrZero(chunk.usage.completion_tokens);
    }
    const choices = Array.isArray(chunk.choices) ? chunk.choices : [];
    const choice = isRecord(choices[0]) ? choices[0] : null;
    if (choice === null) return out;
    const delta = isRecord(choice.delta) ? choice.delta : {};
    out.push(...this.ensureStarted());
    if (typeof delta.content === 'string') out.push(...this.textDeltaEvents(delta.content));
    const calls = Array.isArray(delta.tool_calls) ? delta.tool_calls : [];
    for (const call of calls) out.push(...this.toolCallEvents(call));
    if (choice.finish_reason !== null && choice.finish_reason !== undefined) {
      this.pendingStop = stopReasonOf(choice.finish_reason);
    }
    return out;
  }

  /** One data payload's events: [DONE] closes the message, anything else is one chunk. */
  handlePayload(payload: string): string[] {
    if (payload === '[DONE]') return [...this.ensureStarted(), ...this.finish(this.heldStopReason)];
    let chunk: unknown;
    try {
      chunk = JSON.parse(payload);
    } catch {
      return [];
    }
    if (!isRecord(chunk)) return [];
    return this.chunkEvents(chunk);
  }

  /** One frame's events, in the order its data lines arrived. */
  handleFrame(frame: string): string[] {
    const out: string[] = [];
    for (const payload of dataPayloads(frame)) out.push(...this.handlePayload(payload));
    return out;
  }
}

/** Rewrite one named SSE event's JSON data in place, in a buffered event list. */
function patchEventData(events: string[], eventName: string, patch: (data: Json) => void): void {
  for (let i = 0; i < events.length; i++) {
    const text = events[i] as string;
    if (!text.startsWith(`event: ${eventName}\n`)) continue;
    const dataLine = text.split('\n').find((line) => line.startsWith('data: '));
    if (dataLine === undefined) continue;
    let data: unknown;
    try {
      data = JSON.parse(dataLine.slice(6));
    } catch {
      continue;
    }
    if (!isRecord(data)) continue;
    patch(data);
    events[i] = sseEvent(eventName, data);
  }
}

/**
 * A Chat Completions SSE stream as the Anthropic event sequence: message_start on the first chunk,
 * content_block_start/delta/stop per text block or tool call, then message_delta with the stop reason
 * and usage, then message_stop. `model` is the id the entry dialled, named back to the client. A
 * stream that ends without [DONE] is still closed with the held stop reason, so the client always
 * sees a finished message. The events are buffered until the upstream closes: the message_start
 * event's input count is only known once the usage chunk arrives, at the end of the stream, so the
 * buffered message_start and message_delta are rewritten with the true counts before anything is
 * sent. When no usage chunk arrived, estimates from the prompt and the answer text are used instead.
 */
export async function* anthropicEventsFromOpenAI(upstream: ReadableStream<Uint8Array>, model: string, promptText = ''): AsyncGenerator<string> {
  const translator = new AnthropicStreamTranslator(model);
  const events: string[] = [];
  for await (const event of readFrames(upstream, (frame) => translator.handleFrame(frame))) events.push(event);
  if (!translator.isFinished) {
    events.push(...translator.ensureStarted());
    events.push(...translator.finish(translator.heldStopReason));
  }
  translator.applyClosingUsage(events, promptText);
  yield* events;
}
