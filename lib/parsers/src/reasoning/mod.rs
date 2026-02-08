// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
use std::collections::HashMap;
use std::sync::OnceLock;

mod base_parser;
mod gpt_oss_parser;
mod granite_parser;

// Re-export main types and functions for convenience
pub use base_parser::BasicReasoningParser;
pub use gpt_oss_parser::GptOssReasoningParser;
pub use granite_parser::GraniteReasoningParser;

static REASONING_PARSER_MAP: OnceLock<HashMap<&'static str, ReasoningParserType>> = OnceLock::new();

/// Initialize the global reasoning parser map
fn get_reasoning_parser_map() -> &'static HashMap<&'static str, ReasoningParserType> {
    REASONING_PARSER_MAP.get_or_init(|| {
        let mut map = HashMap::new();
        map.insert("deepseek_r1", ReasoningParserType::DeepseekR1);
        map.insert("basic", ReasoningParserType::Basic);
        map.insert("gpt_oss", ReasoningParserType::GptOss);
        map.insert("qwen3", ReasoningParserType::Qwen);
        map.insert("nemotron_deci", ReasoningParserType::NemotronDeci);
        map.insert("kimi", ReasoningParserType::Kimi);
        map.insert("kimi_k25", ReasoningParserType::KimiK25);
        map.insert("step3", ReasoningParserType::Step3);
        map.insert("mistral", ReasoningParserType::Mistral);
        map.insert("granite", ReasoningParserType::Granite);
        map.insert("nemotron_nano", ReasoningParserType::NemotronDeci); // nemotron nano is <think>...</think>
        map
    })
}

/// Get all available reasoning parser names
pub fn get_available_reasoning_parsers() -> Vec<&'static str> {
    get_reasoning_parser_map().keys().copied().collect()
}

#[derive(Debug, Clone, Default)]
pub struct ParserResult {
    /// The normal text outside of reasoning blocks.
    pub normal_text: String,

    /// The extracted reasoning text from within reasoning blocks.
    pub reasoning_text: String,
}

impl ParserResult {
    pub fn get_some_reasoning(&self) -> Option<String> {
        if self.reasoning_text.is_empty() {
            None
        } else {
            Some(self.reasoning_text.clone())
        }
    }

    pub fn get_some_normal_text(&self) -> Option<String> {
        if self.normal_text.is_empty() {
            None
        } else {
            Some(self.normal_text.clone())
        }
    }
}

pub trait ReasoningParser: Send + std::fmt::Debug {
    /// Parses a standalone, non-streaming input chunk. Implementations may reset or ignore
    /// internal streaming state and should return the split of normal vs reasoning text for
    /// this complete input. Marker tokens must not be included in either output.
    fn detect_and_parse_reasoning(&mut self, text: &str, token_ids: &[u32]) -> ParserResult;

    /// Parses a streaming chunk and updates internal state. The return value should be the
    /// delta: only the newly discovered normal and reasoning text attributable to this chunk
    /// (not the cumulative totals). Marker tokens must not be included in either output.
    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
        token_ids: &[u32],
    ) -> ParserResult;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum ReasoningParserType {
    DeepseekR1,
    Step3,
    Basic,
    GptOss,
    Qwen,
    NemotronDeci,
    Kimi,
    KimiK25,
    Mistral,
    Granite,
}

#[derive(std::fmt::Debug)]
pub struct ReasoningParserWrapper {
    parser: Box<dyn ReasoningParser>,
}

impl ReasoningParser for ReasoningParserWrapper {
    fn detect_and_parse_reasoning(&mut self, text: &str, token_ids: &[u32]) -> ParserResult {
        self.parser.detect_and_parse_reasoning(text, token_ids)
    }

    fn parse_reasoning_streaming_incremental(
        &mut self,
        text: &str,
        token_ids: &[u32],
    ) -> ParserResult {
        self.parser
            .parse_reasoning_streaming_incremental(text, token_ids)
    }
}

impl ReasoningParserType {
    pub fn get_reasoning_parser(self) -> ReasoningParserWrapper {
        let basic_parser =
            BasicReasoningParser::new("<think>".into(), "</think>".into(), false, true);
        let force_reasoning_basic_parser =
            BasicReasoningParser::new("<think>".into(), "</think>".into(), true, true);
        match self {
            ReasoningParserType::DeepseekR1 => ReasoningParserWrapper {
                parser: Box::new(force_reasoning_basic_parser),
            },
            ReasoningParserType::Step3 => ReasoningParserWrapper {
                parser: Box::new(force_reasoning_basic_parser),
            },
            ReasoningParserType::Basic => ReasoningParserWrapper {
                parser: Box::new(basic_parser),
            },
            ReasoningParserType::Qwen => ReasoningParserWrapper {
                parser: Box::new(basic_parser),
            },
            ReasoningParserType::NemotronDeci => ReasoningParserWrapper {
                parser: Box::new(basic_parser),
            },
            ReasoningParserType::Kimi => ReasoningParserWrapper {
                parser: Box::new(BasicReasoningParser::new(
                    "◁think▷".into(),
                    "◁/think▷".into(),
                    false,
                    true,
                )),
            },
            ReasoningParserType::KimiK25 => ReasoningParserWrapper {
                parser: Box::new(BasicReasoningParser::new(
                    "<think>".into(),
                    "</think>".into(),
                    true,
                    true,
                )),
            },
            ReasoningParserType::Mistral => ReasoningParserWrapper {
                parser: Box::new(BasicReasoningParser::new(
                    "[THINK]".into(),
                    "[/THINK]".into(),
                    true,
                    true,
                )),
            },
            ReasoningParserType::GptOss => match GptOssReasoningParser::new() {
                Ok(parser) => ReasoningParserWrapper {
                    parser: Box::new(parser),
                },
                Err(e) => {
                    tracing::warn!(
                        "GptOssReasoningParser could not be initialized, falling back to Basic Reasoning Parser: {e}"
                    );
                    ReasoningParserWrapper {
                        parser: Box::new(BasicReasoningParser::new(
                            "<think>".into(),
                            "</think>".into(),
                            false,
                            true,
                        )),
                    }
                }
            },
            ReasoningParserType::Granite => ReasoningParserWrapper {
                parser: Box::new(GraniteReasoningParser::new()),
            },
        }
    }

    pub fn get_reasoning_parser_from_name(name: &str) -> ReasoningParserWrapper {
        tracing::debug!("Selected reasoning parser: {}", name);

        let parser_map = get_reasoning_parser_map();
        let normalized_name = name.to_lowercase();

        match parser_map.get(normalized_name.as_str()) {
            Some(parser_type) => parser_type.get_reasoning_parser(),
            None => {
                tracing::warn!(
                    parser_name = name,
                    "Unknown reasoning parser type, falling back to Basic Reasoning Parser",
                );
                Self::Basic.get_reasoning_parser()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_available_reasoning_parsers() {
        let parsers = get_available_reasoning_parsers();
        assert!(!parsers.is_empty());
        // Update this list when adding a new parser
        let available_parsers = [
            "deepseek_r1",
            "basic",
            "gpt_oss",
            "qwen3",
            "nemotron_deci",
            "kimi",
            "kimi_k25",
            "step3",
            "mistral",
            "granite",
            "nemotron_nano",
        ];
        for parser in available_parsers {
            assert!(parsers.contains(&parser));
        }
    }

    #[test]
    fn test_kimi_k25_parser_is_force_reasoning() {
        // KimiK25 uses force_reasoning=true: output without <think> tags is still treated as reasoning
        let mut parser = ReasoningParserType::KimiK25.get_reasoning_parser();
        let result = parser.detect_and_parse_reasoning("no think tags here", &[]);
        assert_eq!(result.reasoning_text, "no think tags here");
        assert_eq!(result.normal_text, "");
    }

    #[test]
    fn test_kimi_k25_parser_with_think_tags() {
        // KimiK25 default: model generates <think>...</think> then content
        let mut parser = ReasoningParserType::KimiK25.get_reasoning_parser();
        let result = parser.detect_and_parse_reasoning(
            "<think>Let me reason about this.</think>Hello!",
            &[],
        );
        assert_eq!(result.reasoning_text, "Let me reason about this.");
        assert_eq!(result.normal_text, "Hello!");
    }

    #[test]
    fn test_kimi_k25_parser_empty_think_block() {
        // Instant mode: model generates <think></think> then content (thinking disabled)
        let mut parser = ReasoningParserType::KimiK25.get_reasoning_parser();
        let result =
            parser.detect_and_parse_reasoning("<think></think>Hello from instant mode!", &[]);
        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, "Hello from instant mode!");
    }

    #[test]
    fn test_kimi_k25_parser_empty_think_block_with_newline() {
        // Some models emit <think>\n</think> in instant mode
        let mut parser = ReasoningParserType::KimiK25.get_reasoning_parser();
        let result =
            parser.detect_and_parse_reasoning("<think>\n</think>Hello from instant mode!", &[]);
        assert_eq!(result.reasoning_text, "");
        assert_eq!(result.normal_text, "Hello from instant mode!");
    }

    #[test]
    fn test_kimi_k25_streaming_force_reasoning() {
        // Streaming: force_reasoning means tokens before <think> are treated as reasoning
        let mut parser = ReasoningParserType::KimiK25.get_reasoning_parser();

        // First chunk: partial think tag — buffered because it's a prefix of "<think>"
        let r1 = parser.parse_reasoning_streaming_incremental("<thi", &[]);
        assert_eq!(r1.reasoning_text, "");
        assert_eq!(r1.normal_text, "");

        // Second chunk: completes the think tag + reasoning content
        let r2 = parser.parse_reasoning_streaming_incremental("nk>reasoning here", &[]);
        assert_eq!(r2.reasoning_text, "reasoning here");
        assert_eq!(r2.normal_text, "");

        // Third chunk: close tag + normal content
        let r3 = parser.parse_reasoning_streaming_incremental("</think>Hello!", &[]);
        assert_eq!(r3.reasoning_text, "");
        assert_eq!(r3.normal_text, "Hello!");
    }

    #[test]
    fn test_kimi_k25_streaming_complete_response() {
        // Streaming token-by-token through a full KimiK25 response
        let mut parser = ReasoningParserType::KimiK25.get_reasoning_parser();
        let mut all_reasoning = String::new();
        let mut all_content = String::new();

        let tokens = [
            "<think>",
            "I need to",
            " think about",
            " this carefully.",
            "</think>",
            "Bonjour",
            "!",
        ];
        for token in tokens {
            let r = parser.parse_reasoning_streaming_incremental(token, &[]);
            all_reasoning.push_str(&r.reasoning_text);
            all_content.push_str(&r.normal_text);
        }

        assert_eq!(all_reasoning, "I need to think about this carefully.");
        assert_eq!(all_content, "Bonjour!");
    }

    #[test]
    fn test_kimi_k25_streaming_empty_think_instant_mode() {
        // Streaming: instant mode produces <think></think> then content
        let mut parser = ReasoningParserType::KimiK25.get_reasoning_parser();
        let mut all_reasoning = String::new();
        let mut all_content = String::new();

        let tokens = ["<think>", "</think>", "Direct answer."];
        for token in tokens {
            let r = parser.parse_reasoning_streaming_incremental(token, &[]);
            all_reasoning.push_str(&r.reasoning_text);
            all_content.push_str(&r.normal_text);
        }

        assert_eq!(all_reasoning, "");
        assert_eq!(all_content, "Direct answer.");
    }

    #[test]
    fn test_kimi_k25_parser_lookup_by_name() {
        // Verify the parser can be looked up by name
        let mut parser = ReasoningParserType::get_reasoning_parser_from_name("kimi_k25");
        let result = parser.detect_and_parse_reasoning(
            "<think>thinking</think>answer",
            &[],
        );
        assert_eq!(result.reasoning_text, "thinking");
        assert_eq!(result.normal_text, "answer");
    }

    #[test]
    fn test_kimi_vs_kimi_k25_different_tags() {
        // Kimi (original) uses ◁think▷/◁/think▷, KimiK25 uses <think>/</think>
        let mut kimi = ReasoningParserType::Kimi.get_reasoning_parser();
        let mut kimi_k25 = ReasoningParserType::KimiK25.get_reasoning_parser();

        // Kimi original does NOT parse <think> tags
        let r_kimi = kimi.detect_and_parse_reasoning("<think>reasoning</think>answer", &[]);
        assert_eq!(r_kimi.normal_text, "<think>reasoning</think>answer");
        assert_eq!(r_kimi.reasoning_text, "");

        // KimiK25 does parse <think> tags
        let r_k25 = kimi_k25.detect_and_parse_reasoning("<think>reasoning</think>answer", &[]);
        assert_eq!(r_k25.reasoning_text, "reasoning");
        assert_eq!(r_k25.normal_text, "answer");
    }
}
