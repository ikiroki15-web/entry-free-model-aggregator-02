// free-model-aggregator-02's starter kit: one server, two client wires, and a real free model behind
// every answer.
//
// This is a working entry to start from, not a competitive answer. It is the house baseline for the
// bounty, deliberately weak, and worth a fraction of the score: use it to see the shape of a trial, then
// replace its routing, its failover and its catalog with your own. The gaps are marked TODO below.
//
// WHAT A TRIAL GIVES YOU
//   A trial runs this server as `node /bot/main.ts` from your repository's root, with Node 22's standard
//   library alone (nothing is installed; TypeScript runs directly, and a relative import needs its .ts
//   extension). The env is at $MAKEFUN_ENV_URL; your model calls go through the grader's gateway at
//   $OPENAI_BASE_URL, an OpenAI-compatible endpoint that reaches OpenRouter on the key you pushed to the
//   competition. The brief at $MAKEFUN_PROMPT_FILE carries two documents:
//     - the free model list: each free OpenRouter model's gateway id
//       (`openrouter/<vendor>/<model>:free`), its published rating in milli-points, what it can do
//       (tools, vision, JSON mode, context length) and its free flag;
//     - the six prompts: each one's id, its api (openai or anthropic), whether it streams, its weight
//       (1 or 2) and the requirements a model must meet to serve it. Each prompt carries a unique token
//       the answer must contain.
//   The server:
//     1. listens on 0.0.0.0 at a free port (never 8080: the env itself is there on one layout);
//     2. POSTs $MAKEFUN_ENV_URL/register {"protocol_version": 1, "port": <port>, "catalog": <catalog>};
//     3. serves GET /v1/models (answering in the shape of the auth header it got: x-api-key means
//        Anthropic, Authorization means OpenAI), POST /v1/chat/completions and POST /v1/messages, each
//        with a JSON body or with server-sent events;
//     4. answers each prompt by dialling ONE free model through the gateway. OPENAI_API_KEY in the
//        sandbox is a dummy the gateway swaps for the OpenRouter key you pushed; the real key never
//        enters the sandbox;
//     5. polls GET $MAKEFUN_ENV_URL/work every 100 ms and exits 0 when the env says {"done": true}. A
//        process still running when the wall ends scores zero, so never wait for anything else.
//   `node main.ts --list` prints the same catalog JSON and exits 0 before any server starts; the
//   register body's catalog is that same document, byte for byte.
//
// THE MODEL RULE
//   Whatever this server reports in a response's `model` field must be the model it actually dialled.
//   The house records every gateway call and scores the model that really answered. This starter sends
//   the brief's model id in the gateway request and puts that same id back on every answer and every
//   streamed chunk (the provider's own answer spells it without the gateway's `openrouter/` prefix), so
//   its claim is the call it made, never another. A prompt whose reported model has no matching
//   successful call in the house's record earns nothing.
//
// WHAT THIS ENTRY DOES DIFFERENTLY (the starter's TODOs, implemented)
//   - quality (60%): rankedCandidates() ranks every free candidate that meets the prompt's
//     requirements by rating_milli descending (deterministic id tie-break) and dials the best one:
//     Inkling 2 Small (1475) for plain/tools prompts, Nemotron 3.5 Super VL (1303) for JSON-mode
//     prompts, Gemma 4 31B (1210) next. The catalog still lists all fourteen ids in brief order.
//   - failover (10%): dialWithFailover() retries a refused prompt on the next ranked candidate
//     (max 4 attempts per prompt, so at most 24 calls in a trial, under the 60-call cap) and
//     reports every recovered refusal in the x-makefun-upstream-failures header, so the failover
//     column can see refusals that were retried and then served.
//   - rate limits: failureCounts demotes a model by 400 milli per refusal and skips models with
//     two refusals while any other candidate remains, so one rate-limited model cannot take every
//     later prompt down with it. Ordering is a pure function of the brief and these counts — no
//     wall clock is read — so a replay reproduces the same choices from the same recorded refusals.
//   - auth: a 401 (no OpenRouter key pushed) fails fast after one call instead of burning the
//     trial cap on fourteen doomed attempts; a prompt with no output cap gets max_tokens 1024 so
//     the gateway does not refuse it for max_tokens_required.
//   - the catalog (5%): it lists every free model the brief names, which is exactly the fourteen ids
//     this column scores. TODO(catalog): keep it exact — register that same set and answer GET /v1/models
//     with it on both auth headers; the column checks the register body and both answers against exactly
//     those fourteen ids, so listing fewer loses it as surely as listing more.
//   The trial's rules are in the kit's RULES.txt; read them for the score's columns and the tie-break.
//
// FILES
//   main.ts            this server: the brief, the catalog, the routes, the register and the poll.
//   lib/providers.ts   the brief's types and parser, the free model list, the catalog.
//   lib/wire.ts        Anthropic <-> Chat Completions translation, the model field, the SSE writers.
//   LICENSE            MIT: keep it or replace it with another OSI-approved license.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIJsonWithModel,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this baseline reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's own key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';
/** Max gateway attempts per prompt: enough to fail over, small enough to stay under the 60-call trial cap. */
const MAX_ATTEMPTS_PER_PROMPT = 4;
/** Max rows in the x-makefun-upstream-failures trail header (the spec caps it at 8). */
const MAX_FAILURE_ROWS = 8;
/** Default output cap when a client request names none: the gateway refuses uncapped calls. */
const DEFAULT_MAX_TOKENS = 1024;
/** One recorded gateway refusal, in the trail-header shape. */
interface UpstreamFailure {
  status: number;
  code: string;
  model: string | null;
}
/**
 * Per-model refusal counts in this process. No wall clock is read anywhere: ordering is a pure
 * function of the brief, the request and these counts, so a replay that serves the same recorded
 * refusals makes the same choices. A model that refused twice is skipped while any other candidate
 * remains, so one rate-limited model cannot take every later prompt down with it.
 */
const failureCounts = new Map<string, number>();
function failureCount(model: string): number {
  return failureCounts.get(model) ?? 0;
}
function recordFailure(model: string): void {
  failureCounts.set(model, failureCount(model) + 1);
}

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return freeModels(brief).map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * Free candidates that meet a prompt's requirements, ranked best first.
 *
 * Quality is 60% of the score and is the rating of the model that really served the prompt, over
 * the best rating for that prompt — so the order is rating_milli descending, with a deterministic
 * id tie-break. A model's own refusals demote it (400 milli per refusal): after one refusal the
 * best model (1475) falls below the second-best (1303), so the next prompt tries a fresh model
 * instead of hammering the one that just refused. Models with two or more refusals are skipped
 * while any other candidate remains.
 */
function rankedCandidates(requires: Requirement): ModelRow[] {
  const eligible = freeModels(brief).filter((m) => meetsRequirements(requires, capabilitiesOf(m)));
  const fresh = eligible.filter((m) => failureCount(m.id) < 2);
  const pool = fresh.length > 0 ? fresh : eligible;
  return [...pool].sort((a, b) => {
    const scoreA = a.rating_milli - failureCount(a.id) * 400;
    const scoreB = b.rating_milli - failureCount(b.id) * 400;
    if (scoreB !== scoreA) return scoreB - scoreA;
    if (b.rating_milli !== a.rating_milli) return b.rating_milli - a.rating_milli;
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
  });
}

/** The one free model that serves a request: the best-ranked candidate, or null when none meets it. */
function routeFor(requires: Requirement): ModelRow | null {
  return rankedCandidates(requires)[0] ?? null;
}

/** A gateway refusal's code, from its error body when it parses, else an http_<status> code. */
function failureCode(status: number, text: string): string {
  try {
    const parsed: unknown = JSON.parse(text);
    if (isRecord(parsed) && isRecord(parsed.error)) {
      const code = parsed.error.code ?? parsed.error.type;
      if (typeof code === 'string' && code !== '') return code;
    }
  } catch {
    // Not JSON: fall through to the status code.
  }
  return `http_${status}`;
}

/** The Chat Completions body this entry dials: the client's body, with an output cap the gateway requires. */
function gatewayBody(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  if (typeof out.max_tokens !== 'number' && typeof out.max_completion_tokens !== 'number') {
    out.max_tokens = DEFAULT_MAX_TOKENS;
  }
  if (out.stream === true && !isRecord(out.stream_options)) {
    out.stream_options = { include_usage: true };
  }
  return out;
}

/** The requirements of the prompt now arriving: the brief's row by arrival order, else the body's own. */
function requirementsFor(body: Json): Requirement {
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (entry !== undefined) return entry.requires;
  // TODO(workload): the brief's prompt list is the declared source of each request's requirements; this
  // fallback is a guess for a brief that holds none.
  return requirementsFromBody(body);
}

/** Requirements read off a request body, when the brief has no row for it. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

/** The failover trail header: the refusals this prompt recovered from, at most 8 rows. */
function failureHeaders(failures: UpstreamFailure[]): Record<string, string> {
  if (failures.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(failures.slice(0, MAX_FAILURE_ROWS)) };
}

function sendJson(res: ServerResponse, status: number, body: unknown, extraHeaders: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...extraHeaders });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string): void {
  if (wire === 'anthropic') sendJson(res, status, anthropicError(type, message));
  else sendJson(res, status, openAIError(message, type, null));
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), { method: 'POST', headers, body: JSON.stringify(gatewayBody(body, model)) });
}

/**
 * Dial a prompt through the gateway, failing over across the ranked candidates.
 *
 * Each attempt names a different model, in rating order, and a refusal is recorded in the trail
 * and demotes that model for later prompts. A 401 means no OpenRouter key was pushed: no other
 * model can help, so it fails fast after one call instead of burning the trial's 60-call cap on
 * fourteen doomed attempts. A network error is treated like a refusal of that model and the next
 * candidate is tried. Returns the first successful upstream response with the model that served
 * it and the refusals it recovered from, or null with the full trail when every attempt failed.
 */
async function dialWithFailover(
  requires: Requirement,
  makeBody: (model: string) => Json,
): Promise<{ upstream: Response; model: string; failures: UpstreamFailure[] } | { upstream: null; model: null; failures: UpstreamFailure[] }> {
  const failures: UpstreamFailure[] = [];
  const candidates = rankedCandidates(requires);
  const attempts = Math.min(MAX_ATTEMPTS_PER_PROMPT, candidates.length);
  for (let i = 0; i < attempts; i++) {
    const model = candidates[i].id;
    let upstream: Response;
    try {
      upstream = await dial(model, makeBody(model));
    } catch (err) {
      recordFailure(model);
      failures.push({ status: 0, code: 'fetch_error', model });
      log(`gateway fetch failed for ${model}: ${(err as Error).message}`);
      continue;
    }
    if (upstream.ok && upstream.body !== null) return { upstream, model, failures };
    const text = await upstream.text();
    recordFailure(model);
    failures.push({ status: upstream.status, code: failureCode(upstream.status, text), model });
    log(`gateway refused ${model} with HTTP ${upstream.status}`);
    if (upstream.status === 401) break;
  }
  return { upstream: null, model: null, failures };
}

/** A refused or broken gateway call as a 502 in the client's wire, carrying the refusal trail. */
function refused(res: ServerResponse, wire: 'openai' | 'anthropic', failures: UpstreamFailure[]): void {
  const first = failures[0];
  const status = first?.status ?? 502;
  const text = first !== undefined ? `HTTP ${status} (${first.code})` : 'no gateway response';
  const headers = failureHeaders(failures);
  if (wire === 'anthropic') {
    sendJson(res, 502, anthropicError('api_error', `the gateway answered ${text}`), headers);
  } else {
    sendJson(res, 502, openAIError(`the gateway answered ${text}`, 'upstream_error', null), headers);
  }
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it, with failover. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  if (routeFor(requires) === null) return sendError(res, 'openai', 503, 'no_free_model', 'no free model in the brief meets this request');
  const { upstream, model, failures } = await dialWithFailover(requires, (id) => ({ ...body, model: id }));
  if (upstream === null || model === null) return refused(res, 'openai', failures);
  if (body.stream === true) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      ...failureHeaders(failures),
    });
    try {
      for await (const frame of openAIStreamWithModel(upstream.body, model)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  let answer: unknown;
  try {
    answer = await upstream.json();
  } catch {
    return refused(res, 'openai', [...failures, { status: 502, code: 'bad_upstream_json', model }]);
  }
  sendJson(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, model), failureHeaders(failures));
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back, with failover. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object');
  const requires = requirementsFor(body);
  if (routeFor(requires) === null) return sendError(res, 'anthropic', 503, 'overloaded_error', 'no free model in the brief meets this request');
  const { upstream, model, failures } = await dialWithFailover(requires, (id) => anthropicToOpenAI(body, id));
  if (upstream === null || model === null) return refused(res, 'anthropic', failures);
  if (body.stream === true) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      ...failureHeaders(failures),
    });
    try {
      for await (const event of anthropicEventsFromOpenAI(upstream.body, model)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  let answer: unknown;
  try {
    answer = await upstream.json();
  } catch {
    return refused(res, 'anthropic', [...failures, { status: 502, code: 'bad_upstream_json', model }]);
  }
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, model), failureHeaders(failures));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer');
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
