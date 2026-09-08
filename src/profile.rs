//! Everything a checkpoint says about how to talk to it, in one validated
//! place: architecture, tokenizer, special tokens, context window and chat
//! template.
//!
//! The engine works on token ids and knows nothing about text; the HTTP server
//! and CLI used to fill the gap with TinyLlama's constants — BOS 1, EOS 2, a
//! hand-written `role: content` chat format — whatever checkpoint was loaded.
//! Those are model properties, and every HF checkpoint ships them:
//!
//! | what              | where                                                  |
//! |-------------------|--------------------------------------------------------|
//! | BOS / EOS ids     | `config.json`, then `generation_config.json`, then the |
//! |                   | tokenizer's own `bos_token` / `eos_token` strings      |
//! | context window    | `config.json` `max_position_embeddings`                |
//! | whether text gets | `tokenizer_config.json` `add_bos_token`                |
//! | a BOS             |                                                        |
//! | chat template     | `tokenizer_config.json` `chat_template`                |
//!
//! # Who owns special tokens
//!
//! Exactly one layer does, and it is this one. Token-id requests are complete
//! model inputs and pass through untouched. Text prompts get exactly one BOS
//! when the model wants one, whether the tokenizer's post-processor adds it or
//! this module has to. Chat requests are rendered through the checkpoint's own
//! template and then tokenized *without* added special tokens, exactly as
//! `apply_chat_template` does in `transformers` — the template decides whether
//! a BOS appears, because that is what the model was trained on.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde_json::Value;
use tokenizers::Tokenizer;

use crate::engine::EngineConfig;
use crate::model::LlamaConfig;

/// One turn of a conversation, as the chat endpoint receives it.
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// Roles the templates this engine serves know how to render. Anything else
/// (`tool`, `function`, a typo) is refused rather than rendered as a user turn.
pub const SUPPORTED_ROLES: [&str; 3] = ["system", "user", "assistant"];

#[derive(Debug)]
pub struct ModelProfile {
    pub config: LlamaConfig,
    pub tokenizer: Option<Tokenizer>,
    /// Token that starts a sequence, if the model uses one.
    pub bos_token: Option<u32>,
    /// Every token that ends generation. Never empty: a model with no
    /// declared EOS anywhere is refused at load, since it could never stop.
    pub eos_tokens: Vec<u32>,
    pub max_context: Option<usize>,
    /// Whether text prompts should start with BOS.
    pub add_bos_token: bool,
    /// The Jinja2 chat template, verbatim from `tokenizer_config.json`.
    pub chat_template: Option<String>,
    /// The special tokens as text, for the template's `bos_token` /
    /// `eos_token` variables.
    pub bos_text: Option<String>,
    pub eos_text: Option<String>,
    pub model_path: PathBuf,
}

impl ModelProfile {
    /// Read everything from the files beside `model_path` and `tokenizer_path`.
    ///
    /// A tokenizer path that is given but cannot be loaded is an error; `None`
    /// builds a profile that serves token-id requests only.
    pub fn load(model_path: &Path, tokenizer_path: Option<&Path>) -> Result<Self> {
        let config = LlamaConfig::beside_checkpoint(model_path)?;
        Self::from_parts(config, model_path, tokenizer_path)
    }

    /// [`ModelProfile::load`] with an architecture the caller already has —
    /// a quantization choice, or a fixture config.
    pub fn from_parts(
        config: LlamaConfig,
        model_path: &Path,
        tokenizer_path: Option<&Path>,
    ) -> Result<Self> {
        let tokenizer = tokenizer_path
            .map(|p| {
                Tokenizer::from_file(p)
                    .map_err(|e| anyhow::anyhow!("failed to load tokenizer {}: {e}", p.display()))
            })
            .transpose()?;

        let tokenizer_config = tokenizer_path
            .map(|p| read_json_beside(p, "tokenizer_config.json"))
            .transpose()?
            .flatten();
        let generation_config = read_json_beside(model_path, "generation_config.json")?;

        let bos_text = tokenizer_config
            .as_ref()
            .and_then(|c| special_token_text(c.get("bos_token")));
        let eos_text = tokenizer_config
            .as_ref()
            .and_then(|c| special_token_text(c.get("eos_token")));
        let add_bos_token = tokenizer_config
            .as_ref()
            .and_then(|c| c.get("add_bos_token"))
            .and_then(Value::as_bool)
            .unwrap_or(true);
        let chat_template = tokenizer_config
            .as_ref()
            .and_then(|c| chat_template_of(c.get("chat_template")));

        let text_id = |text: &Option<String>| -> Option<u32> {
            let t = tokenizer.as_ref()?;
            t.token_to_id(text.as_deref()?)
        };

        // Precedence: the model's config, then its generation config, then
        // the tokenizer's own idea of its special tokens.
        let bos_token = config
            .bos_token_id
            .or_else(|| {
                generation_config
                    .as_ref()
                    .and_then(|g| g.get("bos_token_id"))
                    .and_then(Value::as_u64)
                    .and_then(|v| u32::try_from(v).ok())
            })
            .or_else(|| text_id(&bos_text));

        let mut eos_tokens = config.eos_token_ids.clone();
        if let Some(g) = &generation_config {
            for id in token_ids(g.get("eos_token_id")) {
                if !eos_tokens.contains(&id) {
                    eos_tokens.push(id);
                }
            }
        }
        if let Some(id) = text_id(&eos_text) {
            if !eos_tokens.contains(&id) {
                eos_tokens.push(id);
            }
        }
        anyhow::ensure!(
            !eos_tokens.is_empty(),
            "no end-of-sequence token declared in config.json, generation_config.json or the \
             tokenizer; the model could never stop"
        );

        Ok(Self {
            max_context: config.max_position_embeddings,
            config,
            tokenizer,
            bos_token,
            eos_tokens,
            add_bos_token,
            chat_template,
            bos_text,
            eos_text,
            model_path: model_path.to_path_buf(),
        })
    }

    /// Engine settings that are properties of the model. The caller layers
    /// pool size and sampling defaults on top.
    pub fn engine_config(&self) -> EngineConfig {
        let mut eos = self.eos_tokens.iter().copied();
        EngineConfig {
            eos_token: eos.next().expect("never empty"),
            extra_eos_tokens: eos.collect(),
            bos_token: self.add_bos_token.then_some(self.bos_token).flatten(),
            max_context: self.max_context,
            ..EngineConfig::default()
        }
    }

    pub fn tokenizer(&self) -> Result<&Tokenizer> {
        self.tokenizer.as_ref().ok_or_else(|| {
            anyhow::anyhow!("no tokenizer loaded; send prompt_tokens instead of text")
        })
    }

    pub fn has_chat_template(&self) -> bool {
        self.chat_template.is_some()
    }

    /// Tokenize a plain text prompt as a complete model input: the tokenizer's
    /// special tokens, plus exactly one BOS if the model expects one and the
    /// tokenizer did not already add it.
    pub fn encode_prompt(&self, text: &str) -> Result<Vec<u32>> {
        let tokenizer = self.tokenizer()?;
        let encoding = tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
        let mut ids = encoding.get_ids().to_vec();
        if let (true, Some(bos)) = (self.add_bos_token, self.bos_token) {
            if ids.first() != Some(&bos) {
                ids.insert(0, bos);
            }
        }
        Ok(ids)
    }

    /// Render a conversation through the checkpoint's chat template.
    pub fn render_chat(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        let template = self.chat_template.as_deref().ok_or_else(|| {
            anyhow::anyhow!(
                "this checkpoint's tokenizer_config.json has no chat_template; use /v1/completions \
                 with a prompt formatted for the model instead"
            )
        })?;
        anyhow::ensure!(!messages.is_empty(), "messages must not be empty");
        for (i, m) in messages.iter().enumerate() {
            anyhow::ensure!(
                SUPPORTED_ROLES.contains(&m.role.as_str()),
                "messages[{i}].role {:?} is not supported (expected one of {:?})",
                m.role,
                SUPPORTED_ROLES
            );
        }
        render_template(
            template,
            messages,
            self.bos_text.as_deref().unwrap_or(""),
            self.eos_text.as_deref().unwrap_or(""),
            add_generation_prompt,
        )
    }

    /// Tokenize a conversation exactly as `apply_chat_template` would: the
    /// rendered text, with no special tokens added by the tokenizer. Whether a
    /// BOS appears is the template's decision.
    pub fn encode_chat(&self, messages: &[ChatMessage]) -> Result<Vec<u32>> {
        let rendered = self.render_chat(messages, true)?;
        let tokenizer = self.tokenizer()?;
        let encoding = tokenizer
            .encode(rendered.as_str(), false)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode generated tokens to text, dropping special tokens.
    pub fn decode(&self, tokens: &[u32]) -> Option<String> {
        self.tokenizer.as_ref()?.decode(tokens, true).ok()
    }
}

/// Render one chat template. Stateless, so it is also usable to check a
/// template against a reference without a checkpoint.
pub fn render_template(
    template: &str,
    messages: &[ChatMessage],
    bos_token: &str,
    eos_token: &str,
    add_generation_prompt: bool,
) -> Result<String> {
    let mut env = minijinja::Environment::new();
    // transformers renders with `trim_blocks=True, lstrip_blocks=True`; the
    // newlines templates put after `{% %}` tags are not output.
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.set_unknown_method_callback(minijinja_contrib::pycompat::unknown_method_callback);
    minijinja_contrib::add_to_environment(&mut env);
    env.add_function(
        "raise_exception",
        |message: String| -> Result<(), minijinja::Error> {
            Err(minijinja::Error::new(
                minijinja::ErrorKind::InvalidOperation,
                message,
            ))
        },
    );
    env.add_function("strftime_now", |format: String| -> String {
        strftime_now(&format)
    });
    env.add_template("chat", template)
        .map_err(|e| anyhow::anyhow!("chat_template does not parse: {e}"))?;
    let tmpl = env.get_template("chat").expect("just added");
    tmpl.render(minijinja::context! {
        messages => messages,
        bos_token => bos_token,
        eos_token => eos_token,
        add_generation_prompt => add_generation_prompt,
    })
    .map_err(|e| anyhow::anyhow!("chat_template failed to render: {e:#}"))
}

/// A small `strftime` for the handful of directives chat templates use to
/// stamp today's date into a system prompt. Unknown directives are left as is.
fn strftime_now(format: &str) -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (year, month, day) = civil_from_days((secs / 86_400) as i64);
    const MONTHS: [&str; 12] = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ];
    let mut out = String::new();
    let mut chars = format.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('d') => out.push_str(&format!("{day:02}")),
            Some('m') => out.push_str(&format!("{month:02}")),
            Some('Y') => out.push_str(&format!("{year}")),
            Some('y') => out.push_str(&format!("{:02}", year % 100)),
            Some('b') => out.push_str(MONTHS[(month - 1) as usize]),
            Some('B') => out.push_str(
                [
                    "January",
                    "February",
                    "March",
                    "April",
                    "May",
                    "June",
                    "July",
                    "August",
                    "September",
                    "October",
                    "November",
                    "December",
                ][(month - 1) as usize],
            ),
            Some('%') => out.push('%'),
            Some(other) => {
                out.push('%');
                out.push(other);
            }
            None => out.push('%'),
        }
    }
    out
}

/// Days since 1970-01-01 to a proleptic Gregorian (year, month, day).
fn civil_from_days(z: i64) -> (i64, u32, u32) {
    let z = z + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    (if m <= 2 { y + 1 } else { y }, m, d)
}

fn read_json_beside(path: &Path, file_name: &str) -> Result<Option<Value>> {
    let candidate = path.with_file_name(file_name);
    if !candidate.exists() {
        return Ok(None);
    }
    let raw = std::fs::read_to_string(&candidate)
        .with_context(|| format!("reading {}", candidate.display()))?;
    let json =
        serde_json::from_str(&raw).with_context(|| format!("parsing {}", candidate.display()))?;
    Ok(Some(json))
}

/// `bos_token` in a tokenizer config is a string or an added-token object.
fn special_token_text(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(s) => Some(s.clone()),
        Value::Object(o) => o.get("content").and_then(Value::as_str).map(str::to_owned),
        _ => None,
    }
}

/// `chat_template` is a string, or a list of named templates in which case the
/// one called `default` (or the only one) is used.
fn chat_template_of(value: Option<&Value>) -> Option<String> {
    match value? {
        Value::String(s) => Some(s.clone()),
        Value::Array(items) => {
            let pick = |name: &str| {
                items.iter().find_map(|item| {
                    (item.get("name").and_then(Value::as_str) == Some(name))
                        .then(|| item.get("template").and_then(Value::as_str))
                        .flatten()
                        .map(str::to_owned)
                })
            };
            pick("default").or_else(|| {
                (items.len() == 1)
                    .then(|| items[0].get("template").and_then(Value::as_str))
                    .flatten()
                    .map(str::to_owned)
            })
        }
        _ => None,
    }
}

fn token_ids(value: Option<&Value>) -> Vec<u32> {
    let one = |v: &Value| v.as_u64().and_then(|n| u32::try_from(n).ok());
    match value {
        Some(Value::Array(items)) => items.iter().filter_map(one).collect(),
        Some(v) => one(v).into_iter().collect(),
        None => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msgs(pairs: &[(&str, &str)]) -> Vec<ChatMessage> {
        pairs
            .iter()
            .map(|(r, c)| ChatMessage {
                role: r.to_string(),
                content: c.to_string(),
            })
            .collect()
    }

    /// TinyLlama-1.1B-Chat's template, verbatim from its tokenizer_config.json.
    const TINYLLAMA: &str = "{% for message in messages %}\n{% if message['role'] == 'user' %}\n{{ '<|user|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'system' %}\n{{ '<|system|>\n' + message['content'] + eos_token }}\n{% elif message['role'] == 'assistant' %}\n{{ '<|assistant|>\n'  + message['content'] + eos_token }}\n{% endif %}\n{% if loop.last and add_generation_prompt %}\n{{ '<|assistant|>' }}\n{% endif %}\n{% endfor %}";

    #[test]
    fn tinyllama_template_renders_as_transformers_does() {
        // Reference output of `apply_chat_template(add_generation_prompt=True)`
        // for this conversation: per-message EOS, newline-separated markers,
        // and the assistant marker opening the generation.
        let out = render_template(
            TINYLLAMA,
            &msgs(&[("system", "You are a bot."), ("user", "Hello")]),
            "<s>",
            "</s>",
            true,
        )
        .unwrap();
        assert_eq!(
            out,
            "<|system|>\nYou are a bot.</s>\n<|user|>\nHello</s>\n<|assistant|>\n"
        );
        let no_prompt =
            render_template(TINYLLAMA, &msgs(&[("user", "Hello")]), "<s>", "</s>", false).unwrap();
        assert_eq!(no_prompt, "<|user|>\nHello</s>\n");
    }

    #[test]
    fn llama2_style_template_with_python_methods_and_exceptions() {
        let template = "{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'].strip() %}{% set loop_messages = messages[1:] %}{% else %}{% set system_message = '' %}{% set loop_messages = messages %}{% endif %}{{ bos_token }}{% for message in loop_messages %}{% if message['role'] == 'user' %}{{ '[INST] ' + system_message + message['content'].strip() + ' [/INST]' }}{% elif message['role'] == 'assistant' %}{{ ' ' + message['content'].strip() + ' ' + eos_token }}{% else %}{{ raise_exception('unexpected role') }}{% endif %}{% endfor %}";
        let out = render_template(
            template,
            &msgs(&[
                ("system", "  Be brief. "),
                ("user", " Hi "),
                ("assistant", "Hello!"),
            ]),
            "<s>",
            "</s>",
            false,
        )
        .unwrap();
        assert_eq!(out, "<s>[INST] Be brief.Hi [/INST] Hello! </s>");

        // The template's own guard fires as an error, not as output.
        let err = render_template(template, &msgs(&[("system", "x")]), "<s>", "</s>", false);
        let _ = err; // no user turn: nothing rendered but the bos, and no error
        let broken = "{{ raise_exception('nope') }}";
        assert!(
            render_template(broken, &msgs(&[("user", "x")]), "", "", false)
                .unwrap_err()
                .to_string()
                .contains("nope")
        );
    }

    #[test]
    fn a_template_that_does_not_parse_is_an_error() {
        assert!(render_template(
            "{% for m in messages %}",
            &msgs(&[("user", "x")]),
            "",
            "",
            true
        )
        .is_err());
    }

    #[test]
    fn named_template_lists_pick_the_default() {
        let list = serde_json::json!([
            {"name": "tool_use", "template": "T"},
            {"name": "default", "template": "D"}
        ]);
        assert_eq!(chat_template_of(Some(&list)).as_deref(), Some("D"));
        let single = serde_json::json!([{"name": "rag", "template": "R"}]);
        assert_eq!(chat_template_of(Some(&single)).as_deref(), Some("R"));
        assert_eq!(
            chat_template_of(Some(&serde_json::json!("S"))).as_deref(),
            Some("S")
        );
        assert_eq!(chat_template_of(None), None);
    }

    #[test]
    fn special_tokens_may_be_strings_or_objects() {
        assert_eq!(
            special_token_text(Some(&serde_json::json!("<s>"))).as_deref(),
            Some("<s>")
        );
        assert_eq!(
            special_token_text(Some(
                &serde_json::json!({"content": "<s>", "special": true})
            ))
            .as_deref(),
            Some("<s>")
        );
        assert_eq!(special_token_text(Some(&serde_json::json!(null))), None);
    }

    #[test]
    fn strftime_handles_the_directives_templates_use() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(19_935), (2024, 7, 31));
        let s = strftime_now("%d %b %Y");
        assert_eq!(s.len(), 11, "{s}");
        assert_eq!(strftime_now("100%%"), "100%");
    }
}
