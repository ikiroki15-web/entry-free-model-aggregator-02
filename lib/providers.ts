// The brief's two documents, the free model list, and the catalog.
//
// A trial's brief (MAKEFUN_PROMPT_FILE) is prose with two JSON documents embedded in it. The prose
// never matters: this module pulls the documents out by their `schema` member and reads only them.
//   - models (`free-model-aggregator-02/models/v1`): one row per free model, keyed by the id the
//     gateway routes (`openrouter/<vendor>/<model>:free`), with its public rating in milli-points,
//     its capabilities (tools, vision, JSON mode, context length) and its free flag. Only a free
//     model may ever serve a prompt; a paid row is never routed.
//   - prompts (`free-model-aggregator-02/prompts/v1`): the six prompts the house client will send,
//     in order, each with its id, its api (`openai` or `anthropic`), whether it streams, its weight
//     (1 or 2) and the requirements a model must meet to serve it.
// A brief may spell the lists `models`/`ratings` and `prompts`/`manifest` (the sibling -01 contest's
// names), and one document may carry both lists; a document is found by the arrays it carries, never
// by a name alone. Everything here is deterministic: no clock, no randomness.

import { readFileSync } from 'node:fs';

export const MODELS_SCHEMA = 'free-model-aggregator-02/models/v1';
export const PROMPTS_SCHEMA = 'free-model-aggregator-02/prompts/v1';

/** Every schema this contest's brief writes names this prefix. */
const SCHEMA_PREFIX = 'free-model-aggregator-02/';

/** The public base URL of the provider the free models belong to: the endpoint the gateway reaches. */
export const OPENROUTER_PUBLIC_BASE = 'https://openrouter.ai/api/v1';
/** The environment variable an entrant's OpenRouter key is pushed under (an entry secret, kept out of the sandbox). */
export const OPENROUTER_KEY_ENV = 'OPENROUTER_API_KEY';

export type Json = Record<string, unknown>;
export const isRecord = (v: unknown): v is Json => v !== null && typeof v === 'object' && !Array.isArray(v);

/** One free model as the brief lists it. `id` is the id the gateway routes. */
export interface ModelRow {
  id: string;
  /** The provider the model belongs to, as the brief names it (`openrouter`). */
  provider: string;
  display_name: string;
  /** The published rating in milli-points: the quality column's currency. */
  rating_milli: number;
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
  free: boolean;
}

/** What one prompt needs: a model without it cannot serve the prompt. */
export interface Requirement {
  tools?: boolean;
  vision?: boolean;
  json?: boolean;
  min_context?: number;
}

/** One prompt the client will send, as the brief lists it. */
export interface PromptRow {
  id: string;
  api: 'openai' | 'anthropic';
  stream: boolean;
  weight: 1 | 2;
  requires: Requirement;
}

/** The brief as this module reads it. `error` is set when a document could not be read. */
export interface Brief {
  models: ModelRow[];
  prompts: PromptRow[];
  error: string | null;
}

/** One model in the catalog document. */
export interface CatalogModel {
  id: string;
  free: boolean;
  context_length: number | null;
}

/** One provider in the catalog document. */
export interface CatalogProvider {
  id: string;
  base_url: string;
  key_env: string;
  wire: 'openai';
  models: CatalogModel[];
}

/** What `--list` prints and what the entry POSTs to /register: the same document, byte for byte. */
export interface CatalogReport {
  providers: CatalogProvider[];
}

/** What a model can do, from the brief's row (null: the brief did not say). */
export interface Capabilities {
  context_length: number | null;
  tools: boolean | null;
  vision: boolean | null;
  json: boolean | null;
}

/**
 * A single left-to-right pass over the brief text that collects every JSON object carrying a string
 * `schema`, in the order the objects close. The pass keeps a stack of the open braces it has seen:
 * a `{` outside a string pushes its position, a `}` outside a string pops the matching opener and
 * yields that span as one candidate object. Braces inside strings never touch the stack, and an
 * escaped quote never ends a string, so prose, code samples and string contents cannot confuse it.
 * A span that is not valid JSON on its own, or that parses to something without a string `schema`,
 * is simply not a document. Nested objects are candidates too, exactly as a scan that re-tried every
 * `{` would find them, but this pass reads each character once.
 */
class JsonDocumentScanner {
  private readonly text: string;
  private readonly openers: number[] = [];
  private inString = false;
  private escaped = false;
  readonly documents: Json[] = [];

  constructor(text: string) {
    this.text = text;
  }

  scan(): Json[] {
    for (let i = 0; i < this.text.length; i++) {
      this.step(this.text[i] as string, i);
    }
    return this.documents;
  }

  private step(ch: string, index: number): void {
    if (this.inString) {
      if (this.escaped) {
        this.escaped = false;
      } else if (ch === '\\') {
        this.escaped = true;
      } else if (ch === '"') {
        this.inString = false;
      }
      return;
    }
    if (ch === '"') {
      this.inString = true;
    } else if (ch === '{') {
      this.openers.push(index);
    } else if (ch === '}') {
      this.closeObject(index);
    }
  }

  private closeObject(end: number): void {
    const start = this.openers.pop();
    if (start === undefined) return;
    const span = this.text.slice(start, end + 1);
    let value: unknown;
    try {
      value = JSON.parse(span);
    } catch {
      return;
    }
    if (isRecord(value) && typeof value.schema === 'string') this.documents.push(value);
  }
}

/** The documents in `text`, found by the single-pass scanner. */
function documentsIn(text: string): Json[] {
  return new JsonDocumentScanner(text).scan();
}

/** Whether a schema string belongs to this contest's brief. */
function schemaBelongsToContest(schema: unknown): boolean {
  return typeof schema === 'string' && schema.includes(SCHEMA_PREFIX);
}

/** A string field, or '' when the value is not a string. */
function asString(value: unknown): string {
  return typeof value === 'string' ? value : '';
}

/** A finite number field, or null when the value is not one. */
function asFiniteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

/** A boolean field, or null when the value is not one. */
function asBoolean(value: unknown): boolean | null {
  return typeof value === 'boolean' ? value : null;
}

/** A capability flag: true only when the brief says true, false only when it says false, else null. */
function validateCapability(value: unknown): boolean | null {
  return asBoolean(value);
}

/** A context length: a finite number, or null when the brief does not give one. */
function validateContextLength(value: unknown): number | null {
  return asFiniteNumber(value);
}

/** A rating in milli-points: the brief's number, or 0 when it gives none. */
function validateRating(value: unknown): number {
  return asFiniteNumber(value) ?? 0;
}

/** A free flag: a row is free unless the brief explicitly says `free: false`. */
function validateFreeFlag(value: unknown): boolean {
  return value !== false;
}

/**
 * A model id as the gateway routes it. The brief may give the bare OpenRouter id
 * (`vendor/model:free`) together with a provider naming `openrouter`; then the gateway spelling
 * (`openrouter/vendor/model:free`) is built. An id that already carries the prefix, or a row under
 * another provider, is left exactly as written. The comparison is case-insensitive, the result keeps
 * the brief's own casing.
 */
function normalizeModelId(raw: string, providerName: string): string {
  if (providerName !== 'openrouter') return raw;
  if (raw.toLowerCase().startsWith('openrouter/')) return raw;
  return `openrouter/${raw}`;
}

/** The provider a row names: its `provider`, else its `wire`, else `openrouter`. */
function validateProviderName(row: Json): string {
  const named = asString(row.provider) !== '' ? asString(row.provider) : asString(row.wire);
  return named === '' ? 'openrouter' : named;
}

/** The display name a row gives, or the raw id when it gives none. */
function validateDisplayName(row: Json, rawId: string): string {
  const name = asString(row.display_name);
  return name !== '' ? name : rawId;
}

/**
 * One model row, validated field by field. The id comes from `id`, or from `model` when the row
 * gives no `id`; a row with neither has no routable id and is dropped. The brief lists free models,
 * so a row is free unless it says `free: false`; a paid row is never routed.
 */
function modelOf(value: unknown): ModelRow | null {
  if (!isRecord(value)) return null;
  const raw = asString(value.id) !== '' ? asString(value.id) : asString(value.model);
  if (raw === '') return null;
  const provider = validateProviderName(value);
  return {
    id: normalizeModelId(raw, provider),
    provider,
    display_name: validateDisplayName(value, raw),
    rating_milli: validateRating(value.rating_milli),
    context_length: validateContextLength(value.context_length),
    tools: validateCapability(value.tools),
    vision: validateCapability(value.vision),
    json: validateCapability(value.json),
    free: validateFreeFlag(value.free),
  };
}

/** A prompt's api: `anthropic` only when the brief says so, otherwise `openai`. */
function validatePromptApi(value: unknown): 'openai' | 'anthropic' {
  return value === 'anthropic' ? 'anthropic' : 'openai';
}

/** A prompt's weight: 2 only when the brief says 2, otherwise 1. */
function validatePromptWeight(value: unknown): 1 | 2 {
  return value === 2 ? 2 : 1;
}

/** A prompt's stream flag: true only when the brief says true. */
function validatePromptStream(value: unknown): boolean {
  return value === true;
}

/**
 * One prompt's requirements, checked field by field. Only a requirement the brief states as true
 * (or, for `min_context`, as a finite number) is kept; anything else is absent, which means the
 * prompt does not ask for it.
 */
function validateRequirement(raw: unknown): Requirement {
  const out: Requirement = {};
  if (!isRecord(raw)) return out;
  if (raw.tools === true) out.tools = true;
  if (raw.vision === true) out.vision = true;
  if (raw.json === true) out.json = true;
  const minContext = asFiniteNumber(raw.min_context);
  if (minContext !== null) out.min_context = minContext;
  return out;
}

/** One prompt row. A row with no id cannot be matched to a client request and is dropped. */
function promptOf(value: unknown): PromptRow | null {
  if (!isRecord(value)) return null;
  const id = asString(value.id);
  if (id === '') return null;
  return {
    id,
    api: validatePromptApi(value.api),
    stream: validatePromptStream(value.stream),
    weight: validatePromptWeight(value.weight),
    requires: validateRequirement(value.requires),
  };
}

/**
 * The first document under this contest's prefix that carries one of `keys` as an array: the document
 * and the key it was found under. A brief that spells its lists in the sibling's names still reads.
 */
function findDocumentWithArray(docs: readonly Json[], keys: readonly string[]): { doc: Json; key: string } | null {
  for (const doc of docs) {
    if (!schemaBelongsToContest(doc.schema)) continue;
    for (const key of keys) {
      if (Array.isArray(doc[key])) return { doc, key };
    }
  }
  return null;
}

/** The rows of one document's array, validated one by one; rows that do not validate are dropped. */
function validatedRows<T>(doc: Json, key: string, validate: (value: unknown) => T | null): T[] {
  const raw = doc[key];
  if (!Array.isArray(raw)) return [];
  const out: T[] = [];
  for (const item of raw) {
    const row = validate(item);
    if (row !== null) out.push(row);
  }
  return out;
}

/** The brief at `path`, its two documents found under this contest's schema prefix. */
export function readBrief(path: string): Brief {
  if (path === '') return { models: [], prompts: [], error: 'MAKEFUN_PROMPT_FILE is not set' };
  let text: string;
  try {
    text = readFileSync(path, 'utf8');
  } catch (err) {
    return { models: [], prompts: [], error: `cannot read ${path}: ${(err as Error).message}` };
  }
  const docs = documentsIn(text);
  const modelsDoc = findDocumentWithArray(docs, ['models', 'ratings']);
  if (modelsDoc === null) {
    return { models: [], prompts: [], error: `the brief at ${path} holds no ${MODELS_SCHEMA} document` };
  }
  const models = validatedRows(modelsDoc.doc, modelsDoc.key, modelOf);
  const promptsDoc = findDocumentWithArray(docs, ['prompts', 'manifest']);
  if (promptsDoc === null) {
    return { models, prompts: [], error: `the brief at ${path} holds no ${PROMPTS_SCHEMA} document` };
  }
  const prompts = validatedRows(promptsDoc.doc, promptsDoc.key, promptOf);
  return { models, prompts, error: null };
}

/** The models this entry may serve: the brief's free ones, in the brief's order. */
export function freeModels(brief: Brief): ModelRow[] {
  return brief.models.filter((model) => model.free);
}

/** What a model can do, as the brief's row says it. */
export function capabilitiesOf(model: ModelRow): Capabilities {
  return {
    context_length: model.context_length,
    tools: model.tools,
    vision: model.vision,
    json: model.json,
  };
}

/** One requirement against one capability: true when the capability meets it, false otherwise. */
function requirementMet(check: { needed: boolean | undefined; have: boolean | null }): boolean {
  if (check.needed !== true) return true;
  return check.have === true;
}

/** Whether a model's capabilities meet one prompt's requirements. An unknown capability never meets one. */
export function meetsRequirements(requires: Requirement, caps: Capabilities): boolean {
  const checks = [
    requirementMet({ needed: requires.tools, have: caps.tools }),
    requirementMet({ needed: requires.vision, have: caps.vision }),
    requirementMet({ needed: requires.json, have: caps.json }),
  ];
  if (!checks.every(Boolean)) return false;
  if (requires.min_context === undefined) return true;
  if (caps.context_length === null) return false;
  return caps.context_length >= requires.min_context;
}

/**
 * The catalog's model list: every free model in the brief, each once, in the brief's order. A
 * repeated id is listed once; a paid model is never listed.
 */
function collectCatalogModels(models: readonly ModelRow[]): CatalogModel[] {
  const seen = new Set<string>();
  const out: CatalogModel[] = [];
  for (const model of models) {
    if (!model.free) continue;
    if (seen.has(model.id)) continue;
    seen.add(model.id);
    out.push({ id: model.id, free: true, context_length: model.context_length });
  }
  return out;
}

/** The single provider the catalog carries: `openrouter`, on its public endpoint, OpenAI wire. */
function makeCatalogProvider(models: CatalogModel[]): CatalogProvider {
  return {
    id: 'openrouter',
    base_url: OPENROUTER_PUBLIC_BASE,
    key_env: OPENROUTER_KEY_ENV,
    wire: 'openai',
    models,
  };
}

/**
 * The catalog the brief's free model list makes: one provider, the models it serves.
 *
 * The base_url is the provider's public endpoint, never the gateway's trial-local address: the
 * gateway's origin differs between a live pass and its replay, and the catalog is part of the state
 * export a replay must reproduce byte for byte.
 *
 * This entry lists every free model in the brief — exactly the fourteen dialable ids the catalog
 * column scores — so keep it that way: register this set and keep GET /v1/models (both auth headers)
 * exactly that set, since listing fewer loses the column as surely as listing more.
 */
export function buildCatalog(brief: Brief): CatalogReport {
  const models = collectCatalogModels(brief.models);
  if (models.length === 0) return { providers: [] };
  return { providers: [makeCatalogProvider(models)] };
}
